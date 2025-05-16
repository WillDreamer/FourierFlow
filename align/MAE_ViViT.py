import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch import nn, einsum
import torch.nn.functional as F
# from module import Attention, PreNorm, FeedForward

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class ReAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
class LeFF(nn.Module):
    
    def __init__(self, dim = 192, scale = 4, depth_kernel = 3):
        super().__init__()
        
        scale_dim = dim*scale
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    nn.GELU(),
                                    Rearrange('b c (h w) -> b c h w', h=14, w=14)
                                    )
        
        self.depth_conv =  nn.Sequential(nn.Conv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1, groups=scale_dim, bias=False),
                          nn.BatchNorm2d(scale_dim),
                          nn.GELU(),
                          Rearrange('b c h w -> b (h w) c', h=14, w=14)
                          )
        
        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c')
                                    )
        
    def forward(self, x):
        x = self.up_proj(x)
        x = self.depth_conv(x)
        x = self.down_proj(x)
        return x
    
    
class LCAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = q[:, :, -1, :].unsqueeze(2) # Only Lth element use as query

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        
        patches = take_indexes(patches, forward_indexes)
        
        patches = patches[:remain_T]
        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=128,
                 patch_size=8,
                 dim = 768,
                 depth=12,
                 heads=12,
                 mask_ratio=0.75,
                 in_channels = 4,
                 num_frames = 4,
                 emb_dropout = 0.,
                 dropout = 0.,
                 ) -> None:
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim, dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim, dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.shuffle = PatchShuffle(mask_ratio)
        self.mask_ratio = mask_ratio
        self.layer_norm = torch.nn.LayerNorm(dim)
        
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, t, n, _ = x.shape

        # cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        # x = torch.cat((cls_space_tokens, x), dim=2)
        
        x += self.pos_embedding[:, :, :(n)]
        x = self.dropout(x)  # [batch, num_frame, patch_num, dim]
        
        patches = rearrange(x, 'b t n d -> n (b t) d')
        # spatial masking
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        
        x = rearrange(patches, 'n (b t) d -> (b t) n d',t=t)
        x = self.layer_norm(self.space_transformer(x))
        patches = rearrange(x, '(b t) n d -> t (b n) d',t=t)

        # temporal masking
        patches, forward_indexes_t, backward_indexes_t = self.shuffle(patches)

        x = rearrange(patches, 't (b n) d -> (b n) t d',t=int(t*(1- self.mask_ratio)),b=b)
        x = self.layer_norm(self.temporal_transformer(x))
        x = rearrange(x, '(b n) t d -> b t n d',t=int(t*(1- self.mask_ratio)),b=b)

        return x, backward_indexes, backward_indexes_t

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=128,
                 patch_size=8,
                 dim=768,
                 depth=4,
                 heads=3,
                 out_channel = 4,
                 mask_ratio = 0.75,
                 num_frames = 4,
                 dropout = 0.,
                 ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, dim))
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim, dim, dropout)
    
        self.temporal_transformer = Transformer(dim, depth, heads, dim, dim, dropout)

        self.head = torch.nn.Linear(dim, out_channel * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        self.out_channel = out_channel

    def forward(self, features, backward_indexes, backward_indexes_t):
        B = features.shape[0]
        N = features.shape[2]
        features = rearrange(features, 'b t n d -> t (b n) d')

        T = features.shape[0]
        
        # backward_indexes_t = backward_indexes_t + 1
        # backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes_t.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes_t)
        
        mask = torch.zeros_like(features)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes_t)
        
        x = rearrange(features, 't (b n) d -> (b n) t d',b=B,t=int(T/(1-self.mask_ratio)))
        x = self.temporal_transformer(x)
        features = rearrange(x, '(b n) t d -> n (b t) d',b=B,t=int(T/(1-self.mask_ratio)))
        
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        
        mask = rearrange(mask, 't (b n) d -> n (b t) d',b=B,t=int(T/(1-self.mask_ratio)))
        mask = mask.repeat(int(1/(1-self.mask_ratio)), 1, 1)
        mask[N:] = 1
        mask = take_indexes(mask, backward_indexes)
        
        features = rearrange(features, 'n (b t) d -> b t n d',b=B,t=int(T/(1-self.mask_ratio)))
        
        features = features + self.pos_embedding
        
        x = rearrange(features, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        
        patches = self.head(x)
        patches = rearrange(patches, '(b t) n d -> n (b t) d',t=int(T/(1-self.mask_ratio)))
        mask = mask[:,:,:patches.shape[-1]]
        
        img = self.patch2img(patches)
        mask = self.patch2img(mask)
        img = rearrange(img, '(b t) ... -> b t ...',b=B)
        mask = rearrange(mask, '(b t) ... -> b t ...',b=B)

        return img, mask

class MAE_ViViT(torch.nn.Module):
    def __init__(self,
                image_size=128,
                patch_size=8,
                emb_dim=768,
                encoder_layer=12,
                encoder_head=12,
                decoder_layer=4,
                decoder_head=12,
                mask_ratio=0.75,
                in_out_channel = 4,
                num_frames = 4,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio, in_out_channel,num_frames)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head, in_out_channel,mask_ratio,num_frames)

    def forward(self, img):
        features, backward_indexes, backward_indexes_t = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes, backward_indexes_t)
        return predicted_img, mask

class ViViT_Encoder(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder) -> None:
        super().__init__()
        self.to_patch_embedding = encoder.to_patch_embedding
        self.pos_embedding = encoder.pos_embedding
        self.space_token = encoder.space_token
        self.space_transformer = encoder.space_transformer
        self.temporal_token = encoder.temporal_token
        self.temporal_transformer = encoder.temporal_transformer
        self.dropout = encoder.dropout
        self.layer_norm = encoder.layer_norm
        # self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, t, n, _ = x.shape
        
        x += self.pos_embedding[:, :, :(n)]
        x = self.dropout(x)  # [batch, num_frame, patch_num, dim]
        
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.layer_norm(self.space_transformer(x))

        x = rearrange(x, '(b t) n d -> (b n) t d',t=t,b=b)
        x = self.layer_norm(self.temporal_transformer(x))
        x = rearrange(x, '(b n) t d -> b t n d',t=t,b=b)
        
        return x

if __name__ == '__main__':

    img = torch.ones([1, 4, 4, 128, 128])
    # encoder = MAE_Encoder()
    # encoder_out, backward_indexes, backward_indexes_t = encoder(img)
    # print(encoder_out.shape)
    # decoder = MAE_Decoder()
    # print(decoder(encoder_out,backward_indexes,backward_indexes_t)[0].shape,'=======')
    model = MAE_ViViT()
    print(model(img)[0].shape)
