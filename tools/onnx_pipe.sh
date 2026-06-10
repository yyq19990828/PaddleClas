#!/bin/bash

# 获取当前日期，格式为MMDD
date_suffix=$(date +%m%d)

# 获取输入参数
model_name=${1:-test}
config_file=${2:-ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0_tyjt.yaml}
metadata=${3:-None}
batch_size=${4:-None}
add_softmax=${5:-True}

# 导出模型
bash ./tools/export_model.sh ${model_name} ${config_file} ${batch_size} ${add_softmax}

# 转换为ONNX格式
bash ./tools/p2onnx.sh ${model_name}

# 重命名ONNX文件的输入输出
bash ./tools/onnx_rename_io.sh ${model_name} "${metadata}"
