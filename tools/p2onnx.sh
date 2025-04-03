paddle2onnx \
    --model_dir=/home/paddle_ws/PaddleClas/output_inference/LCNet_0402 \
    --model_filename=inference.pdmodel \
    --params_filename=inference.pdiparams \
    --save_file=/home/paddle_ws/PaddleClas/output_inference/LCNet_0402/inference_dynamic.onnx \
    --opset_version=13 \
    --enable_onnx_checker=True
