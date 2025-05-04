#!/bin/bash

cd /wanghaixin/FourierFlow

EXPNAME=3d_cfd_M1.0_0.001_align_difftrans_afno_cycle
FLNM=2D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
PRETRAINED_PATH=/wanghaixin/MAE-ViViT/ckpt_M1/vivit-M1.0-1e-8-mask0.5-mae_1999.pt

export WANDB_ENTITY="FourierFlow"
export WANDB_PROJECT="${WANDB_ENTITY}_${EXPNAME}"
export WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f"

/root/anaconda3/bin/accelerate launch /wanghaixin/FourierFlow/train.py \
    --allow-tf32 \
    --exp-name "$EXPNAME" \
    --flnm "$FLNM" \
    --pretrained-mae-path "$PRETRAINED_PATH" \
    --batch-size 120 \
    --epochs 40001