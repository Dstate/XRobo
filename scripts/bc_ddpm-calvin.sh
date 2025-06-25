#!/bin/bash
export WANDB_API_KEY=a9abdbf33f19f62cbbd321e4498210cbaaf1efc0

PORT=29992
NUM_PROCS=4
CONFIG_NAME="bc_ddpm-calvin"
AVAILABLE_GPUS="0,1,2,3"

CUDA_VISIBLE_DEVICES=$AVAILABLE_GPUS torchrun \
    --nproc_per_node=$NUM_PROCS \
    --master_port=$PORT \
    train_vla.py \
        --config_name $CONFIG_NAME \
        --output_dir runnings/$CONFIG_NAME