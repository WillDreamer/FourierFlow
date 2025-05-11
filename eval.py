import argparse
from copy import deepcopy
import logging
from data.CNS_data_utils import FNODatasetSingle, FNODatasetMultistep
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.metrics import *
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
from models.diff_afno_sit import SiT_models
# from models.diff_sit import SiT_models
import math
from torchvision.utils import make_grid
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
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="/wanghaixin/FourierFlow/exps")
    #* 替换为新的exp的name
    parser.add_argument("--exp-name", type=str, default="3d_cfd_0.001_align_difftrans_afno_cycle_SiT-small_0428-03:07")
    parser.add_argument("--flnm", type=str, default="2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5")
    parser.add_argument("--logging-dir", type=str, default="/wanghaixin/FourierFlow/logs/test")
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--ckpt-step", type=int, default=135000)

    # model
    parser.add_argument("--model", type=str,default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=3)
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
    
    args.logging_dir = os.path.join(args.logging_dir,args.exp_name)
        
    return args

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
        )["model"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in ckpt.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    # model.load_state_dict(ckpt['model'])
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # flnm = '2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
    flnm = args.flnm
    base_path='/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/'
    reduce_resolution = 4
    reduced_batch = 1

    # base_path = '/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/'
    # flnm = '2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
    
    # base_path = '/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/'
    # flnm = '2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5'
    # reduce_resolution = 1
    # reduced_batch = 10

    #* 换成PDE数据集，先快速实验用reduced_batch 100
    train_dataset, test_dataset,normalizer = FNODatasetMultistep.get_train_test_datasets(
                                    flnm,
                                    reduced_resolution=reduce_resolution,
                                    reduced_resolution_t=1,
                                    reduced_batch=reduced_batch,
                                    initial_step=0,
                                    saved_folder=base_path,
                                    if_eval_plot=True
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

    print(f'==== {next(iter(test_dataloader))[0].mean().item():.6f} ====')
    
    model.eval()  # important! This enables embedding dropout for classifier-free guidance
               
    from samplers import euler_sampler
    _err_RMSE_avg = 0
    _err_nRMSE_avg = 0
    _err_max_avg = 0
    with torch.no_grad():
        # test_iter = iter(test_dataloader)
        # target_test, grid_test, raw_image_test = next(test_iter)
        for target_test, grid_test, raw_image_test in test_dataloader:
            raw_image_test = rearrange(raw_image_test, "B H W T C -> B T C H W").to(device)
            target_test = rearrange(target_test, "B H W T C -> B T C H W").to(device)
            sample_input = torch.randn_like(target_test, device=device)
            samples = euler_sampler(
                model, 
                sample_input, 
                raw_image_test,
                num_steps=3, 
                cfg_scale=4.0,
                guidance_low=0.,
                guidance_high=1.,
                path_type=args.path_type,
                heun=False,
            ).to(torch.float32)
            Lx, Ly, Lz = 1., 1., 1.
            _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F \
            = metric_func(samples, target_test, if_mean=True, Lx=Lx, Ly=Ly, Lz=Lz)
            _err_RMSE_avg += _err_RMSE.item()
            _err_nRMSE_avg += _err_nRMSE.item()
            _err_max_avg += _err_Max.item()
        _err_RMSE_avg /= len(test_dataloader)
        _err_nRMSE_avg /= len(test_dataloader)
        _err_max_avg /= len(test_dataloader)
        
        logger.info(f'RMSE: {_err_RMSE_avg:.4f}, nRMSE: {_err_nRMSE_avg:.4f}, Max:{_err_max_avg:.4f}')
    # with PdfPages(os.path.join('/wanghaixin/FourierFlow/output',args.exp_name+'.pdf')) as pdf:
    #     print(samples.shape)
    #     samples = rearrange(samples, "B T C H W -> B H W T C")
    #     target_test = rearrange(target_test, "B T C H W -> B H W T C")
    #     samples = normalizer.decode(samples.cpu())
    #     target_test = normalizer.decode(target_test.cpu())
    #     for i in range(samples.size(0)):  
    #         fig, axes = plt.subplots(8, 4, figsize=(16, 9))  
    #         axes = axes.flatten()  # 将 axes 数组扁平化为一维数组
    #         for j in range(samples.size(-1)):
    #             T = samples.size(-2)
    #             for k in range(T):  
    #                 axes[j*T+k].imshow(samples[i,:,:,k,j].numpy(), cmap='coolwarm')  
    #                 axes[j*T+k+(T*samples.size(-1))].imshow(target_test[i,:,:,k,j].numpy(), cmap='coolwarm')  
    #                 axes[j*T+k].axis('off')  
    #                 axes[j*T+k].set_title(f'Smaple {i+1}, Step {k+1}, Channel {j+1}') 
    #                 axes[j*T+k+(T*samples.size(-1))].axis('off') 
    #                 axes[j*T+k+(T*samples.size(-1))].set_title(f'GT {i+1}, Step {k+1}, Channel {j+1}')  
    #         plt.tight_layout(pad=0.5, w_pad=2, h_pad=2)
    #         pdf.savefig(fig,dpi=300)  
    #         plt.close(fig)  

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
