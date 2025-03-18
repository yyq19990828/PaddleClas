import paddle
from ppcls.arch import ResNet101_CSRA
from ppcls.utils.logger import init_logger

init_logger()
model = ResNet101_CSRA(class_num=21)
# model = paddle.load('/home/paddle_ws/PaddleClas/output/CSRA/best_model.pdparams')
# print(model)

Flops = paddle.flops(model, input_size=(1, 3, 256, 192), print_detail=True)
print(Flops)