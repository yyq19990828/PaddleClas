#!/bin/bash

# 获取当前日期，格式为MMDD
date_suffix=$(date +%m%d)

# 获取输入参数，默认为test
model_name=${1:-test}

# 导出模型
bash ./tools/export_model.sh ${model_name}

# 转换为ONNX格式
bash ./tools/p2onnx.sh ${model_name}

# 重命名ONNX文件的输入输出
bash ./tools/onnx_rename_io.sh ${model_name}
