import argparse
import copy
from copy import deepcopy
import logging
from pathlib import Path
from collections import OrderedDict
import json
from data_utils import FNODatasetSingle, FNODatasetMult
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from fno import FNO1d, FNO2d, FNO3d
from ViT_MAE import *
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import pdb
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = get_logger(__name__)

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 



def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Testing Loop                                #
#################################################################################

def main(args):    
    os.makedirs(args.logging_dir, exist_ok=True)
    logger = create_logger(args.logging_dir)
    logger.info(f"Experiment directory created at {args.logging_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
 
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution 

    z_dims = [128]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )
    ckpt_name = str(args.ckpt_step).zfill(7) +'.pt'
    ckpt = torch.load(
        f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
        map_location='cpu',
        )
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    flnm = '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'
    base_path='/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/'

    #* 换成PDE数据集，先快速实验用reduced_batch 100
    train_dataset, test_dataset = FNODatasetSingle.get_train_test_datasets(
                                    flnm,
                                    reduced_resolution=1,
                                    reduced_resolution_t=1,
                                    reduced_batch=1,
                                    initial_step=0,
                                    saved_folder=base_path,
                                    logger=logger
                                )
    local_batch_size = 8
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    model.eval()  # important! This enables embedding dropout for classifier-free guidance
               
    from samplers import euler_sampler
    with torch.no_grad():
        test_iter = iter(test_dataloader)
        target_test, grid_test, raw_image_test = next(test_iter)
        raw_image_test = rearrange(raw_image_test, "B H W C -> B C H W").to(device)
        target_test = rearrange(target_test, "B H W C -> B C H W").to(device)
        sample_input = torch.randn_like(target_test, device=device)
        samples = euler_sampler(
            model, 
            sample_input, 
            raw_image_test,
            num_steps=50, 
            cfg_scale=4.0,
            guidance_low=0.,
            guidance_high=1.,
            path_type=args.path_type,
            heun=False,
        ).to(torch.float32)
    with PdfPages('output.pdf') as pdf:
        for i in range(samples.size(0)):  # 遍历每个样本
            fig, axes = plt.subplots(2, 4, figsize=(16, 4))  # 创建 1 行 4 列的子图
            for j in range(samples.size(1)):  # 遍历每个通道
                axes[j].imshow(samples[i, j].cpu().numpy(), cmap='warm')  # 显示图像
                axes[j+4].imshow(target_test[i, j].cpu().numpy(), cmap='warm')  # 显示图像
                axes[j].axis('off')  # 关闭坐标轴
                axes[j].set_title(f'Smaple {i+1}, Channel {j+1}')  # 设置标题
                axes[j+4].axis('off')  # 关闭坐标轴
                axes[j+4].set_title(f'GT {i+1}, Channel {j+1}')  # 设置标题
            plt.tight_layout()
            pdf.savefig(fig,args.exp_name+'.pdf')  # 保存当前图形到 PDF
            plt.close(fig)  # 关闭图形以释放内存
        
                   

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps-old")
    parser.add_argument("--exp-name", type=str, default="2d_cfd_mse_align_scheduler")
    parser.add_argument("--logging-dir", type=str, default="/wanghaixin/REPA-origin/logs/test")
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--ckpt-step", type=int, default=50000)

    # model
    parser.add_argument("--model", type=str,default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[128,256], default=128)
    parser.add_argument("--batch-size", type=int, default=64)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
