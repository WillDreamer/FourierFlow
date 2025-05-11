# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np

from functools import partial
from .afno2d import AFNO2D, Block
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed


# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1,2).contiguous()
            k_xf = k.transpose(1,2).contiguous()
            v_xf = v.transpose(1,2).contiguous()
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DiffAttention(nn.Module):
    def __init__(self, num_hidden, num_heads, d_k=1) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_heads
        self.d_k = torch.tensor(d_k)

        self.W_q = nn.Linear(num_hidden, 2 * num_heads * num_hidden)
        self.W_k = nn.Linear(num_hidden, 2 * num_heads * num_hidden)
        self.W_v = nn.Linear(num_hidden, num_heads * num_hidden)
        self.W_o = nn.Linear(num_heads * num_hidden, num_hidden)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=num_heads)
    
    def get_mask(self, size):
        device = next(self.parameters()).device
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)  
        return mask.unsqueeze(0).unsqueeze(0)  

    def forward(self, query, dropout=0.1, mask=None):
        key = query
        values = query
        self.seq_len = query.shape[1]
        self.mask = self.get_mask(self.seq_len)
        _lambda_init = torch.rand(1,device=key.device)
        _lambda = nn.Parameter(_lambda_init.clone())

        query = self.W_q(query).view(-1, self.num_heads, self.seq_len, 2 * self.num_hidden)
        key = self.W_k(key).view(-1, self.num_heads, self.seq_len, 2 * self.num_hidden)
        values = self.W_v(values).view(-1, self.num_heads, self.seq_len, self.num_hidden)

        #split query into [q1;q2] and same for keys [k1;k2]
        query_1 = query[:, :, :, :self.num_hidden]
        query_2 = query[:, :, :, self.num_hidden:]

        key_1 = key[:, :, :, :self.num_hidden]
        key_2 = key[:, :, :, self.num_hidden:]

        QK_T_1 = torch.matmul(query_1, key_1.mT) / torch.sqrt(self.d_k)
        QK_T_2 = torch.matmul(query_2, key_2.mT) / torch.sqrt(self.d_k)

        QK_T_1_norm = self.softmax(QK_T_1)
        QK_T_2_norm = self.softmax(QK_T_2)

        #eq 1
        attention_scores = (QK_T_1_norm - _lambda * QK_T_2_norm)

        if mask:
            self.mask = self.mask.to(query.device)
            attention_scores = attention_scores.masked_fill(self.mask == 1, float('-inf'))

        attention_scores = self.dropout(attention_scores) 
        output = torch.matmul(attention_scores, values)  
        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads , self.seq_len, self.num_hidden)  
        
        output = self.group_norm(output)
        output = output * (1 - _lambda_init)
        output = torch.cat([output[:, i, :, :] for i in range(self.num_heads)], dim=-1)

        output = self.W_o(output)  
        return output 
    
#################################################################################
#               Multi-Head Cross Attention Layer                                #
#################################################################################

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, attention_mode='math'):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape
        
        q = rearrange(self.q_linear(x), "B N (H d) -> B H N d", H=self.num_heads)
        kv = rearrange(self.kv_linear(cond), "B N (kv H d) -> B H kv N d", kv=2, H=self.num_heads)
        k, v = kv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class condition_dropper(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(condition_dropper, self).__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.drop_prob = dropout_prob

    def null(self, y):
        # given a tensor y with last dimension hidden_size, return a tensor repeated with the same shape as y
        return self.mask_token.expand(y.shape)

    def forward(self, c, training, use_cfg=False):
        # inference mode, support both cfg and non-cfg
        if not training and use_cfg:
            half, rest = c.split(c.shape[0] // 2, dim=0)
            return torch.cat([half, self.null(rest)], dim=0)
        if not training and not use_cfg:
            return c
        
        # expected shape of c: (B (T S) C)
        B, N, C = c.shape
        # make sure the mask token has the same shape as the condition
        mask = (torch.rand(B, 1, 1, device=c.device) < self.drop_prob)
        c = torch.where(mask, self.mask_token.expand(B, N, -1), c)
        return c


#################################################################################
#                                 STDiT Block                                   #
#################################################################################
class STDiTBlock(nn.Module):
    """
    A Latte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, d_s=256, d_t=16, **block_kwargs):
        super().__init__()
        self.d_s = d_s
        self.d_t = d_t
        self.norm_c, self.norm_s, self.norm_t, self.norm_mlp = [nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) for _ in range(4)]
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.attn_s = DiffAttention(hidden_size, num_heads=num_heads)
        self.attn_t = DiffAttention(hidden_size, num_heads=num_heads)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, initial_steps=None, tpe=None):
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=-1)
        x_m = modulate(x, shift_msa, scale_msa)
        # spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)
        x_s = self.attn_s(self.norm_s(x_s))
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + gate_msa * x_s

        # temporal branch
        x_t = rearrange(x, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        if tpe is not None:
            x_t = x_t + tpe[:, -self.d_t:]
        x_t = self.attn_t(self.norm_t(x_t))
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + gate_msa * x_t

        # cross attn
        x = x + gate_msa * self.cross_attn(self.norm_c(x), self.norm_c(initial_steps))

        # mlp
        x = x + gate_mlp * self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))
        return x
    
#################################################################################
#                                 Core Latte Model                              #
#################################################################################

# class TransformerBlock(nn.Module):
#     """
#     A Latte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
#     """
#     def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         self.attn = DiffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
#         self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
#         mlp_hidden_dim = int(hidden_size * mlp_ratio)
#         approx_gelu = lambda: nn.GELU(approximate="tanh")
#         self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(hidden_size, 6 * hidden_size, bias=True)
#         )

#     def forward(self, x, c):
#         shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
#         x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
#         x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
#         return x

class FinalLayer(nn.Module):
    """
    The final layer of Latte.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )
    

class STDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=128,
        patch_size=2,
        in_channels=4,
        num_frames=8,
        num_initial_steps=4,
        hidden_size=512,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        initial_dropout_prob=0.0,
        embed_weights=False,
        attention_mode='math',
        projector_dim=2048,
        encoder_depth=6,
        afno_depth=8,
        **block_kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.encoder_depth = encoder_depth
        self.embed_weights = embed_weights
        self.initial_dropout_prob = initial_dropout_prob
        self.num_frames = num_frames
        self.num_initial_steps = num_initial_steps
        self.z_dims = [768]
        assert num_initial_steps > 0, "STDiT requires num_initial_steps > 0"

        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, self.num_frames, hidden_size), requires_grad=False)
        self.hidden_size =  hidden_size

        self.blocks = nn.ModuleList([
            STDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, d_s=self.num_patches,
                       d_t=self.num_frames-self.num_initial_steps, attention_mode=attention_mode) for _ in range(depth)
        ])

        ######## AFNO Branch
        uniform_drop = True
        drop_path_rate=0.
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        h = input_size // patch_size
        w = h // 2 + 1
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(afno_depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, afno_depth)]
        
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.afno_blocks = nn.ModuleList([
            Block(
                dim=hidden_size, mlp_ratio=mlp_ratio,
                drop=0., drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w, use_fno=False, use_blocks=False)
            for i in range(afno_depth)])
        self.conv_fuse = nn.Conv2d(in_channels=2 * num_initial_steps, out_channels=1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.norm = norm_layer(hidden_size)

        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in self.z_dims
            ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Latte blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, 
                x, 
                t, 
                condition=None):
        """
        Forward pass of Latte.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of equation labels
        initial_steps: (N, f, C, H, W) tensor of initial steps
        grid: (2, H, W) tensor of grid coordinates
        """
        
        t = self.t_embedder(t.squeeze())  

        batches, frames, channels, high, weight = x.shape 

        if self.num_initial_steps == 1:
            # b t c h w -> b (t s) d
            x = rearrange(x, 'b t c h w -> (b t) c h w')

            x = self.x_embedder(x, self.pos_embed)
            x = rearrange(x, '(b t) s c -> b (t s) c', b=batches)
            condition = rearrange(condition, 'b f c h w -> (b f) c h w')
            
            condition = self.x_embedder(condition, self.pos_embed)
            condition = rearrange(condition, '(b f) s c -> b (f s) c', b=batches)  
        
        if self.num_initial_steps > 1:
            # inconsistent with previous implementation due to the use of temporal embedding
            # b t c h w -> b (t s) d
            
            x = torch.cat((condition, x), dim=1)
            x = rearrange(x, 'b f c h w -> (b f) c h w')

            x = self.x_embedder(x, self.pos_embed)[:,1:,:]
            x = rearrange(x, '(b f) s c -> (b s) f c', b=batches)  
            x += self.temp_embed
            condition, x = x[:, :self.num_initial_steps], x[:, self.num_initial_steps:]
            x = rearrange(x, '(b s) f c -> b (f s) c', b=batches)
            condition = rearrange(condition, '(b s) f c -> b (f s) c', b=batches)   

        B, ST, D = x.shape

        afno_feat = x.detach().clone()
        for afno_blk in self.afno_blocks:
            afno_feat = afno_blk(afno_feat)
        afno_feat = self.norm(afno_feat)
        
        v_WFNO = self.final_layer(afno_feat, t) 
        v_WFNO = rearrange(v_WFNO, 'b (t s) d -> b t s d', s=self.num_patches)
        
        for i, block in enumerate(self.blocks):
            x = block(x, t, condition, self.temp_embed)
            if (i + 1) == self.encoder_depth:
                proj = self.projectors[0](x.reshape(-1, D)).reshape(B, ST, -1)
                zs = [rearrange(proj, 'b (t n) d -> b t n d', t=frames)] 
        
        x = self.final_layer(x, t)

        x = rearrange(x, 'b (t s) d -> b t s d', s=self.num_patches) 

        concatenated = torch.cat((v_WFNO, x), dim=1) 
        G = self.sigmoid(self.conv_fuse(concatenated))
        x = G * x + (1 - G) * v_WFNO 

        x = rearrange(x, 'b t s d -> (b t) s d') 
        x = self.unpatchify(x)                  
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x,zs

    def forward_with_cfg(self, x, t, y=None, initial_steps=None, grid=None, cfg_scale=7.0, use_fp16=False):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        assert grid is None, "grid is not supported in forward_with_cfg"
        # 虽然第一次迭代没用，但之后每次迭代需要同步有条件和无条件的输入需要用到。
        x = x.chunk(2, dim=0)[0].repeat(2, *[1]*(x.dim()-1))
        if use_fp16:
            x = x.to(dtype=torch.float16)
        model_out = self.forward(x, t, y=y, initial_steps=initial_steps, grid=grid, use_cfg=True, use_fp16=use_fp16)
        # 对于后一半channel（预测的方差）不使用CFG
        eps, rest = model_out[:, :, :self.in_channels], model_out[:, :, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # cfg_scale相当于classifier guidance中的gamma
        # cfg_scale=0.0无条件生成，cfg_scale=1.0标准的条件生成.
        # cfg_scale=7.0是CFG的默认值，引入一个通过低温聚焦的条件分布p(y|x).
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps) 
        eps = torch.cat([half_eps, half_eps], dim=0) 
        return torch.cat([eps, rest], dim=2)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   Latte Configs                                  #
#################################################################################

# def STDiT_S_2(**kwargs):
#     return STDiT(depth=8, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

# def STDiT_L_8(**kwargs):
#     return STDiT(depth=12, hidden_size=512, patch_size=8, num_heads=4, **kwargs)


# Latte_models = {
#     "STDiT-S/2":  STDiT_S_2,   'STDiT-L/8':  STDiT_L_8,
#     }

def SiT_XL_2(**kwargs):
    return STDiT(depth=8, patch_size=8,num_heads=4, **kwargs)

def SiT_XL_4(**kwargs):
    return STDiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return STDiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return STDiT(hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=8, **kwargs)

def SiT_L_4(**kwargs):
    return STDiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return STDiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return STDiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return STDiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return STDiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return STDiT(hidden_size=384,decoder_hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return STDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return STDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}

if __name__ == '__main__':

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(3, 4, 4, 128, 128).to(device)
    time_input = torch.rand((img.shape[0],1,1,1)).to(device)
    t = torch.randint(0, 100, (img.shape[0],), device=device)

    model = SiT_XL_2().to(device)

    print(model(img, time_input,img)[0].shape)
    print(f"STDiT Parameters: {sum(p.numel() for p in model.parameters())/1e9:,} GB")
