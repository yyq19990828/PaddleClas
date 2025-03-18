import onnx

# 加载旧的 ONNX 模型
model = onnx.load('/home/paddle_ws/PaddleClas/output_inference/inference.onnx')

# 更新模型的 IR 版本
model.ir_version = onnx.IR_VERSION

# 保存更新后的模型
onnx.save(model, '/home/paddle_ws/PaddleClas/output_inference/inference1.onnx')