paddle2onnx \
    --model_dir=/home/paddle_ws/PaddleClas/output_inference/LCNet0326 \
    --model_filename=inference.pdmodel \
    --params_filename=inference.pdiparams \
    --save_file=/home/paddle_ws/PaddleClas/output_inference/LCNet0326/inference.onnx \
    --opset_version=13 \
    --enable_onnx_checker=True
