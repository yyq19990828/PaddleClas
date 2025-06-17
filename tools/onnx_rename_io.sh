#!/bin/bash

# 获取当前日期，格式为MMDD
date_suffix=$(date +%m%d)

# 获取输入参数，默认为test
model_name=${1:-test}

python3 "$(dirname "$0")/onnx_rename_io.py" \
    --onnx /home/paddle_ws/PaddleClas/output_inference/${model_name}_${date_suffix}/inference_dynamic_${date_suffix}.onnx \
    --output /home/paddle_ws/PaddleClas/output_inference/${model_name}_${date_suffix}/inference_dynamic_renamed_${date_suffix}.onnx \
    --input_names input \
    --output_names output \
    --enable_metadata \
    --add_metadata  class_name_ori=["car", "truck", "bus", "tanker", "slagcar", "fire engine", \
                    "mixer", "ambulance", "police car", "engineering truck", \
                    "black", "white", "gray", "red", "yellow", "green", \
                    "blue", "purple", "brown", "pink", "other"] \
                    class_name_new=["bus", "car", "engineering truck", "truck", "police car", \
                    "ambulance", "mixer", "tanker", "slagcar", "fire engine", \
                    "white", "gray", "red", "yellow", "brown", "blue", "black", \
                    "green", "purple", "pink", "other"]
