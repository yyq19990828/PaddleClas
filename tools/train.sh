#!/usr/bin/env bash

# for single card train
# python tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards train
export CUDA_VISIBLE_DEVICES=0,1

# 初始命令
# python -m paddle.distributed.launch --gpus="0,1" tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# # 车辆属性识别
# python3 -m paddle.distributed.launch \
#     --gpus="0,1" \
#     tools/train.py \
#         -c ./ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml


# # 车辆属性识别resnet_CSRA
# python3 -m paddle.distributed.launch \
#     --gpus="0, 1" \
#     tools/train.py \
#         -c ./ppcls/configs/PULC/vehicle_attribute/ResNet_CSRA.yaml \
#         -o Arch.pretrained=True \
#         -o Arch.name=ResNet101_vd_CSRA \
#         -o Arch.num_heads=1 \
#         -o Arch.lam=0.1  

# 车辆属性识别resnet_ori
python3 -m paddle.distributed.launch \
    --gpus="0, 1" \
    tools/train.py \
        -c ./ppcls/configs/PULC/vehicle_attribute/ResNet_CSRA.yaml \
        -o Arch.pretrained=True \
        -o Arch.name=ResNet101_vd \
        -o Global.output_dir=./output/ResNet_vd_ori \
 
