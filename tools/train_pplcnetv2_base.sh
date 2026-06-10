#!/usr/bin/env bash

# for single card train
# python tools/train.py -c ./ppcls/configs/_TYJT/PPLCNetV2_base.yaml

# for multi-cards train
export CUDA_VISIBLE_DEVICES=0,1
imgsize=224

# 初始命令
# python -m paddle.distributed.launch --gpus="0,1" tools/train.py -c ./ppcls/configs/_TYJT/PPLCNetV2_base.yaml

# 反光背心识别PPLCNetV2_base
python3 -m paddle.distributed.launch \
    --gpus="0,1" \
    tools/train.py \
        -c ./ppcls/configs/_TYJT/PPLCNetV2_base.yaml \
        -o Global.output_dir=./output_train/PPLCNetV2_base_${imgsize}X${imgsize}_$(date +%m%d) \
        -o Global.epochs=1