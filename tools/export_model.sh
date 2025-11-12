#!/bin/bash

# 获取当前日期，格式为MMDD
date_suffix=$(date +%m%d)

# 获取输入参数
model_name=${1:-test}
config_file=${2:-/home/paddle_ws/PaddleClas/ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0_tyjt.yaml}

python ./tools/export_model.py \
    -c ${config_file} \
    -o Global.pretrained_model=./output_train/${model_name}/best_model \
    -o Global.save_inference_dir=./output_inference/${model_name} \
    -o Global.batch_infer=None
