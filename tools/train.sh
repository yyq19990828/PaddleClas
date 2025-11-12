#!/usr/bin/env bash

# for single card train
# python tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# for multi-cards train
export CUDA_VISIBLE_DEVICES=0,1
imgsize=224

# 初始命令
# python -m paddle.distributed.launch --gpus="0,1" tools/train.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50.yaml

# 车辆属性识别PPLCNet
python3 -m paddle.distributed.launch \
    --gpus="0,1" \
    tools/train.py \
        -c ./ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0_tyjt.yaml \
        -o Global.output_dir=./output_train/LCNet_${imgsize}X${imgsize}_$(date +%m%d) \
        -o Global.epochs=100 \


# 车辆属性识别resnet_CSRA
# python3 -m paddle.distributed.launch \
#     --gpus="0, 1" \
#     tools/train.py \
#         -c ./ppcls/configs/PULC/vehicle_attribute/ResNet_CSRA.yaml \
#         -o Arch.pretrained=True \
#         -o Arch.name=ResNet101_vd_CSRA \
#         -o Arch.num_heads=4 \
#         -o Arch.lam=0.1 \
#         -o Global.output_dir=./output_train/CSRA_$(date +%m%d)_head4_lam0.1

# 车辆属性识别resnet_ori
# python3 -m paddle.distributed.launch \
#     --gpus="0, 1" \
#     tools/train.py \
#         -c ./ppcls/configs/PULC/vehicle_attribute/ResNet_CSRA.yaml \
#         -o Arch.pretrained=True \
#         -o Arch.name=ResNet101_vd \
#         -o Global.output_dir=./output/ResNet_vd_ori \
 

# # 测试代码是否改对
# python3 -m paddle.distributed.launch \
#     --gpus="0,1" \
#     tools/train.py \
#         -c ./ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0_tyjt.yaml \
#         -o Global.output_dir=./output_train/test \
#         -o Global.epochs=1 \

