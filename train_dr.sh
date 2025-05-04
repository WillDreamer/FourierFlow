#!/bin/bash

cd /wanghaixin/FourierFlow

EXPNAME=variable
FLNM=2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_512_Train.hdf5
PRETRAINED_PATH=/wanghaixin/MAE-ViViT/ckpt/vivit-t-mae_1999.pt

/root/anaconda3/bin/accelerate launch /wanghaixin/FourierFlow-baseline/train.py \
    --allow-tf32 \
    --exp-name "$EXPNAME" \
    --flnm "$FLNM" \
    --pretrained-mae-path "$PRETRAINED_PATH"