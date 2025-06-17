# 获取当前日期，格式为MMDD
date_suffix=$(date +%m%d)

# 获取输入参数，默认为test
model_name=${1:-test}

paddle2onnx \
    --model_dir=/home/paddle_ws/PaddleClas/output_inference/${model_name}_${date_suffix} \
    --model_filename=inference.pdmodel \
    --params_filename=inference.pdiparams \
    --save_file=/home/paddle_ws/PaddleClas/output_inference/${model_name}_${date_suffix}/inference_dynamic_${date_suffix}.onnx \
    --opset_version=13 \
    --enable_onnx_checker=True
