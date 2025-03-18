paddle2onnx \
    --model_dir=/home/paddle_ws/PaddleClas/output_inference/CSRA_batch1 \
    --model_filename=inference.pdmodel \
    --params_filename=inference.pdiparams \
    --save_file=/home/paddle_ws/PaddleClas/output_inference/CSRA_batch1/inference_batch1_384x512.onnx \
    --opset_version=13 \
    --enable_onnx_checker=True