#!/bin/bash

# 获取当前日期，格式为MMDD
date_suffix=$(date +%m%d)

# 获取输入参数
model_name=${1:-test}
# 默认的 class_name 列表
default_class_name='class_name=["car", "truck", "bus", "tanker", "slagcar", "fire engine", "mixer", "ambulance", "police car", "engineering truck", "hazardous_goods_vehicle", "manned_sweeping_vehicle", "school_bus", "black", "white", "gray", "red", "yellow", "green", "blue", "purple", "brown", "pink", "other"]'
metadata=${2:-$default_class_name}

# 如果 metadata 为 None，则使用默认值
if [ "$metadata" = "None" ]; then
    metadata=$default_class_name
fi

python3 "$(dirname "$0")/onnx_rename_io.py" \
    --onnx /home/paddle_ws/PaddleClas/output_inference/${model_name}/inference_dynamic.onnx \
    --output /home/paddle_ws/PaddleClas/output_inference/${model_name}/inference_dynamic_renamed.onnx \
    --input_names input \
    --output_names output \
    --enable_metadata \
    --add_metadata ${metadata}
