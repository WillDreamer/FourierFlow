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
# from models.diff_sit_flash import SiT_models
import math
from torchvision.utils import make_grid
import os
from samplers import euler_sampler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.append('/wanghaixin/')
from FourierFlowSurrogate.models.diff_afno_sit import SiT_models
from models.diff_afno_sit import SiT_models as SiT_flow_models

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

def evaluate_surrogate_over_time(
    model, test_dataloader, metric_func, args, device, logger, normalizer=None
):

    model.eval()  # Enable classifier-free guidance

    _err_RMSE_list = []
    _err_nRMSE_list = []
    _err_max_list = []

    with torch.no_grad():
        for _, _, raw_video in test_dataloader:
            raw_video = rearrange(raw_video, "B H W T C -> B T C H W").to(device)
            
            B, T_total, C, H, W = raw_video.shape
            pred_len = 4
            num_rollouts = (T_total - pred_len) // pred_len

            model_input = raw_video[:, :pred_len]  
            pred_all = []

            for rollout_step in range(num_rollouts):
                # 构造时间引导 t
                B = model_input.shape[0]
                if args.weighting == "uniform":
                    time_input = torch.rand((B, 1, 1, 1, 1), device=device, dtype=raw_video.dtype)
                elif args.weighting == "lognormal":
                    rnd_normal = torch.randn((B, 1, 1, 1, 1), device=device, dtype=raw_video.dtype)
                    sigma = rnd_normal.exp()
                    if args.path_type == "linear":
                        time_input = sigma / (1 + sigma)
                    elif args.path_type == "cosine":
                        time_input = 2 / np.pi * torch.atan(sigma)
                    else:
                        raise ValueError(f"Unsupported path_type: {args.path_type}")
                else:
                    raise ValueError(f"Unsupported weighting: {args.weighting}")

                pred = model(x=model_input, t=time_input.flatten())  # [B, pred_len, C, H, W]
                pred_all.append(pred)

                model_input = torch.cat([model_input, pred], dim=1)[:, -pred_len:]

            # 拼接所有预测步
            pred_all = torch.cat(pred_all, dim=1)  # [B, T_total - input_len, C, H, W]
            target_eval = raw_video[:, pred_len:]  # [B, T_total - input_len, C, H, W]
            T = pred_all.shape[1]

            for t in range(T):
                output_t = pred_all[:, t].unsqueeze(1)  # [B, 1, C, H, W]
                target_t = target_eval[:, t].unsqueeze(1)

                _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F = \
                    metric_func(output_t, target_t, if_mean=True, Lx=1., Ly=1., Lz=1.)

                if len(_err_RMSE_list) <= t:
                    _err_RMSE_list.append(0.0)
                    _err_nRMSE_list.append(0.0)
                    _err_max_list.append(0.0)

                _err_RMSE_list[t] += _err_RMSE.item()
                _err_nRMSE_list[t] += _err_nRMSE.item()
                _err_max_list[t] += _err_Max.item()

    # Average over batches
    num_batches = len(test_dataloader)
    _err_RMSE_list = [v / num_batches for v in _err_RMSE_list]
    _err_nRMSE_list = [v / num_batches for v in _err_nRMSE_list]
    _err_max_list = [v / num_batches for v in _err_max_list]

    # Logging results
    for t in range(len(_err_RMSE_list)):
        logger.info(
            f'Surrogate Model Time Step {t}: '
            f'RMSE={_err_RMSE_list[t]:.4f}, '
            f'nRMSE={_err_nRMSE_list[t]:.4f}, '
            f'Max={_err_max_list[t]:.4f}'
        )

    return _err_RMSE_list, _err_nRMSE_list, _err_max_list

def evaluate_flow_model(
    model_flow,
    test_dataloader,
    euler_sampler,
    metric_func,
    args,
    device,
    logger,
    normalizer=None,  
    cfg_scale=4.0,
    num_steps=3,
    guidance_low=0.0,
    guidance_high=1.0,
    heun=False,
    zero_mask=False,
    zero_prob=0.1,
):

    model_flow.eval()
    _err_RMSE_list = []
    _err_nRMSE_list = []
    _err_max_list = []
    
    with torch.no_grad():
        for _, _, raw_video in test_dataloader:
            raw_video = rearrange(raw_video, "B H W T C -> B T C H W").to(device)
            
            
            B, T_total, C, H, W = raw_video.shape
            pred_len = 4
            num_rollouts = (T_total - pred_len) // pred_len

            model_input = raw_video[:, :pred_len]  
            sample_input = torch.randn_like(model_input, device=device)
            pred_all = []

            for rollout_step in range(num_rollouts):
                # 构造时间引导 t
                B = model_input.shape[0]
                if args.weighting == "uniform":
                    time_input = torch.rand((B, 1, 1, 1, 1), device=device, dtype=raw_video.dtype)
                elif args.weighting == "lognormal":
                    rnd_normal = torch.randn((B, 1, 1, 1, 1), device=device, dtype=raw_video.dtype)
                    sigma = rnd_normal.exp()
                    if args.path_type == "linear":
                        time_input = sigma / (1 + sigma)
                    elif args.path_type == "cosine":
                        time_input = 2 / np.pi * torch.atan(sigma)
                    else:
                        raise ValueError(f"Unsupported path_type: {args.path_type}")
                else:
                    raise ValueError(f"Unsupported weighting: {args.weighting}")

                pred = euler_sampler(
                    model_flow,
                    sample_input,
                    model_input,
                    num_steps=num_steps,
                    cfg_scale=cfg_scale,
                    guidance_low=guidance_low,
                    guidance_high=guidance_high,
                    path_type=args.path_type,
                    heun=heun,
                ).to(torch.float32)  # [B, T, C, H, W]
                pred_all.append(pred)

                model_input = torch.cat([model_input, pred], dim=1)[:, -pred_len:]

            pred_all = torch.cat(pred_all, dim=1)  # [B, T_total - input_len, C, H, W]
            target_eval = raw_video[:, pred_len:]  # [B, T_total - input_len, C, H, W]
            T = pred_all.shape[1]

            for t in range(T):
                output_t = pred_all[:, t].unsqueeze(1)  # [B, 1, C, H, W]
                target_t = target_eval[:, t].unsqueeze(1)

                _err_RMSE, _err_nRMSE, _err_CSV, _err_Max, _err_BD, _err_F = \
                    metric_func(output_t, target_t, if_mean=True, Lx=1., Ly=1., Lz=1.)

                if len(_err_RMSE_list) <= t:
                    _err_RMSE_list.append(0.0)
                    _err_nRMSE_list.append(0.0)
                    _err_max_list.append(0.0)

                _err_RMSE_list[t] += _err_RMSE.item()
                _err_nRMSE_list[t] += _err_nRMSE.item()
                _err_max_list[t] += _err_Max.item()

    # Average over batches
    num_batches = len(test_dataloader)
    _err_RMSE_list = [v / num_batches for v in _err_RMSE_list]
    _err_nRMSE_list = [v / num_batches for v in _err_nRMSE_list]
    _err_max_list = [v / num_batches for v in _err_max_list]

    # Logging results
    for t in range(len(_err_RMSE_list)):
        logger.info(
            f'Fourier Flow Time Step {t}: '
            f'RMSE={_err_RMSE_list[t]:.4f}, '
            f'nRMSE={_err_nRMSE_list[t]:.4f}, '
            f'Max={_err_max_list[t]:.4f}'
        )

    return _err_RMSE_list, _err_nRMSE_list, _err_max_list

            

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
    logger.info(f"SiT Surrogate Parameters: {sum(p.numel() for p in model.parameters()):,}")

    z_dims = [128]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model_flow = SiT_flow_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )
    ckpt_step_flow = 270000
    output_dir_flow = '/wanghaixin/FourierFlow/exps'
    exp_name_flow = "3d_cfd_mse_align_0.01_difftrans_afno_cycle_0220-00:48"
    ckpt_name_flow = str(ckpt_step_flow).zfill(7) +'.pt'
    ckpt_flow = torch.load(
        f'{os.path.join(output_dir_flow, exp_name_flow)}/checkpoints/{ckpt_name_flow}',
        map_location='cpu',
        )["model"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in ckpt_flow.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    model_flow.load_state_dict(new_state_dict)
    model_flow = model_flow.to(device)
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    flnm = '2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
    base_path='/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/'
    reduce_resolution = 4
    reduced_batch = 1

    # base_path = '/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/'
    # flnm = '2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
    
    # base_path = '/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/'
    # # flnm = '2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'
    # flnm='2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5'
    # reduce_resolution = 1
    # reduced_batch = 1

    #* 换成PDE数据集，先快速实验用reduced_batch 100
    train_dataset, test_dataset,normalizer = FNODatasetMultistep.get_train_test_datasets(
                                    flnm,
                                    reduced_resolution=reduce_resolution,
                                    reduced_resolution_t=1,
                                    reduced_batch=reduced_batch,
                                    initial_step=0,
                                    saved_folder=base_path,
                                    if_eval_plot=True,
                                    if_rollout=True
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
    
    rmse_list, nrmse_list, max_list = evaluate_surrogate_over_time(
            model=model,
            test_dataloader=test_dataloader,
            metric_func=metric_func,
            args=args,
            device=device,
            logger=logger,
            normalizer=normalizer,  # optional
        )

    rmse_list, nrmse_list, max_list = evaluate_flow_model(
        model_flow=model_flow,
        test_dataloader=test_dataloader,
        euler_sampler=euler_sampler,
        metric_func=metric_func,
        args=args,
        device=device,
        logger=logger,
        normalizer=normalizer,  # optional
        cfg_scale=4.0,
        num_steps=3,
        guidance_low=0.0,
        guidance_high=1.0,
        heun=False,
        zero_mask=True,
        zero_prob=0.1,
    )


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
        
                   

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="/wanghaixin/FourierFlowSurrogate/exps")
    #* 替换为新的exp的name
    parser.add_argument("--exp-name", type=str, default="3d_cfd_surrogate_predict_0319-11-s06")
    parser.add_argument("--logging-dir", type=str, default="/wanghaixin/FourierFlowSurrogate/logs/test")
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--ckpt-step", type=int, default=36000)

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
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
