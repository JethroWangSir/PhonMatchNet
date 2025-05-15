#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

python train_ga.py \
        --epoch 30 \
        --lr 1e-3 \
        --batch_size 512 \
        --gradient_accumulation_steps 8 \
        --loss_weight 1.0 1.0 \
        --audio_input both \
        --text_input g2p_embed \
        --stack_extractor \
        --output_dir ./results/ga \
        --comment 'user comments for each experiment'