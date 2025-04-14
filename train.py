import argparse
import copy
from copy import deepcopy
import logging
from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
import math
from torchvision.utils import make_grid
import random
import os

from align.MAE_ViViT import ViViT_Encoder, MAE_ViViT
from models.diff_afno_sit import SiT_models
from utils.loss import SILoss
from utils.metrics import *



logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")
    #* 每次试验前标注实验名称，
    parser.add_argument("--output-dir", type=str, default="/wanghaixin/FourierFlow/exps/")
    # parser.add_argument("--exp-name", type=str, \
    #                     default="3d_cfd_0.05_align_difftrans_afno_cycle")
    # parser.add_argument("--flnm", type=str, \
    #                     default="2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5")
    # parser.add_argument("--base-path", type=str, \
    #                     default="/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/")
    parser.add_argument("--exp-name", type=str, \
                        default="3d_CFD_M1.0_0.001_align_difftrans_afno_cycle")
    parser.add_argument("--flnm", type=str, \
                        default="2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5")
    parser.add_argument("--base-path", type=str, \
                        default="/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/")
    
    parser.add_argument("--reduced-resolution", type=int, default=4)
    
    parser.add_argument("--logging-dir", type=str, default="/wanghaixin/FourierFlow/logs")
    parser.add_argument("--report-to", type=str, default="tensorboard")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30001) # +1 for saving ckpts
    # (BS//len(loader)) iters for one epoch
    parser.add_argument("--sampling-steps", type=int, default=45000)
    parser.add_argument("--checkpointing-steps", type=int, default=45000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--proj-coeff", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=5e-4)

    # model
    parser.add_argument("--model", type=str,default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=4)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=True)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[128,256], default=128)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=42)

    # cpu
    parser.add_argument("--num-workers", type=int, default=16)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args


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


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def remove_module_prefix(state_dict):
    """
    仅在所有 key 都带有 'module.' 前缀的情况下，统一移除 'module.'。
    否则原样返回。
    """
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k[len("module."):]
            new_state_dict[new_key] = v
        return new_state_dict
    else:
        return state_dict  

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
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logger = get_logger(__name__)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    from datetime import datetime
    current_time = datetime.now().strftime("%m%d-%H:%M")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, (args.exp_name+'_'+current_time))
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
        torch.backends.cudnn.enabled = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution 

    # if args.enc_type != 'None':
    #     encoders, encoder_types, architectures = load_encoders(args.enc_type, device)
    # else:
    #     encoders, encoder_types, architectures = [None], [None], [None]
    #* FNO的embed dim应该是20
    z_dims = [128]
    # z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )
    if accelerator.is_main_process:
        logger.info(model)
    model = model.to(device)

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    
    # pretrained_mae_path = '/wanghaixin/MAE-ViViT/ckpt/vivit-t-mae_1999.pt'
    pretrained_mae_path = '/wanghaixin/MAE-ViViT/vivit-M1.0-1e-8-mask0.5-mae_1999.pt'
    ckpt = torch.load(pretrained_mae_path, map_location='cpu')
    ckpt = remove_module_prefix(ckpt['model_state_dict'])
    vit_model = MAE_ViViT()
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in ckpt.items() if 'encoder' in k}
    
    vit_model.encoder.load_state_dict(encoder_state_dict)
    vit_encoder = ViViT_Encoder(vit_model.encoder)
    vit_encoder.to(device)
    requires_grad(ema, False)
    
    #* 暂时来看SILoss没有用到encoder
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=[],
        accelerator=accelerator,
        latents_scale=None,
        latents_bias=None,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters())/1e9:,} GB")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.1)  # 根据需要调整参数
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs//3, T_mult=2, eta_min=1e-8)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate//5, max_lr=args.learning_rate,
                                              mode = 'triangular2', gamma = 0.95,
                                              step_size_up=10000, step_size_down=20000,cycle_momentum=False)  
    
    # flnm = '2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5'
    # base_path='/wanghaixin/PDEBench/data/2D/CFD/2D_Train_Rand/'

    #* 换成PDE数据集，先快速实验用reduced_batch 100
    if 'CFD' in args.flnm:
        from data.CNS_data_utils import FNODatasetMultistep
    elif 'react' in args.flnm:
        from data.DR_data_utils import FNODatasetMultistep
    train_dataset, test_dataset = FNODatasetMultistep.get_train_test_datasets(
                                    args.flnm,
                                    reduced_resolution=args.reduced_resolution,
                                    reduced_resolution_t=1,
                                    reduced_batch=1,
                                    initial_step=0,
                                    saved_folder=args.base_path,)
    
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images, {len(train_dataloader)} iters ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    accelerator.register_for_checkpointing(scheduler)

    if accelerator.is_main_process:
        # tracker_config = vars(copy.deepcopy(args))
        # from datetime import datetime
        # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        accelerator.init_trackers(
            project_name=args.exp_name+'_'+current_time,  
        #     config=tracker_config,
        #     init_kwargs={
        #     "wandb": {"name": f"{args.exp_name}"+f"{current_time}"}
        # },
        )
    max_train_steps = int(args.epochs * len(train_dataloader))
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        # disable=True,
    )
    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes

    for epoch in range(args.epochs):
        model.train()
        #* raw_image, x应该是VAE encoder输出, y是类别标签
        for target, grid, raw_video in train_dataloader:
            raw_video = rearrange(raw_video, "B H W T C -> B T C H W").to(device)
            target = rearrange(target, "B H W T C -> B T C H W").to(device)
            #* target size [bs, 4, 4, 128, 128]
            
            with torch.no_grad():
                zs = []
                with accelerator.autocast():
                    #* 需要选好用哪里的特征做output
                    z = vit_encoder(target)
                    #* z.shape [bs,4, 256, 768]
                    zs.append(z)

            with accelerator.accumulate(model):

                model_kwargs = dict()
                loss, proj_loss = loss_fn(model, target, raw_video, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                # loss = loss_mean
                loss = loss_mean + proj_loss_mean * args.proj_coeff
                # pdb.set_trace()
                    
                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                current_lr = optimizer.param_groups[0]['lr']

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > (max_train_steps//2):
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0):
                model.eval()  # important! This disables randomized embedding dropout
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
                            num_steps=50, 
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
                   
                    logger.info(f'RMSE: {_err_RMSE_avg:.4f}, nRMSE: {_err_nRMSE_avg:.4f}, MAX-ERR:{_err_max_avg:.4f}')
                    val_log = {"val_RMSE": _err_RMSE_avg, "val_nRMSE": _err_nRMSE_avg, 'MAX-ERR':_err_max_avg}
                    accelerator.log(val_log, step=global_step)

            logs = {
                "lr": current_lr,
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item(),
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item()
                
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        scheduler.step()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()



if __name__ == "__main__":
    args = parse_args()
    
    main(args)
