#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python train.py \
        --epoch 30 \
        --lr 1e-3 \
        --batch_size 512 \
        --loss_weight 1.0 1.0 \
        --audio_input both \
        --text_input g2p_embed \
        --stack_extractor \
        --text_avgpool \
        --output_dir ./results/text_avgpool \
        --comment 'user comments for each experiment'