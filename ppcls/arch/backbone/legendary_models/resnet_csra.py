from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, BatchNorm2D, Softmax, LayerList
from paddle.nn import MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
from paddle.regularizer import L2Decay
import math

from .custom_devices_layers import AdaptiveAvgPool2D
from ....utils import logger
from ..base.theseus_layer import TheseusLayer
from ..base.dbb.dbb_block import DiverseBranchBlock
from ....utils.save_load import load_dygraph_pretrain
from .resnet import ResNet

MODEL_URLS = {
    "ResNet18":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_pretrained.pdparams",
    "ResNet18_dbb":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_dbb_pretrained.pdparams",
    "ResNet18_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet18_vd_pretrained.pdparams",
    "ResNet34":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_pretrained.pdparams",
    "ResNet34_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet34_vd_pretrained.pdparams",
    "ResNet50":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_pretrained.pdparams",
    "ResNet50_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams",
    "ResNet101":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_pretrained.pdparams",
    "ResNet101_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet101_vd_pretrained.pdparams",
    "ResNet152":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet152_pretrained.pdparams",
    "ResNet152_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet152_vd_pretrained.pdparams",
    "ResNet200_vd":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet200_vd_pretrained.pdparams",
}

MODEL_STAGES_PATTERN = {
    "ResNet18": ["blocks[1]", "blocks[3]", "blocks[5]", "blocks[7]"],
    "ResNet34": ["blocks[2]", "blocks[6]", "blocks[12]", "blocks[15]"],
    "ResNet50": ["blocks[2]", "blocks[6]", "blocks[12]", "blocks[15]"],
    "ResNet101": ["blocks[2]", "blocks[6]", "blocks[29]", "blocks[32]"],
    "ResNet152": ["blocks[2]", "blocks[10]", "blocks[46]", "blocks[49]"],
    "ResNet200": ["blocks[2]", "blocks[14]", "blocks[62]", "blocks[65]"]
}

__all__ = MODEL_URLS.keys()
'''
ResNet config: dict.
    key: depth of ResNet.
    values: config's dict of specific model.
        keys:
            block_type: Two different blocks in ResNet, BasicBlock and BottleneckBlock are optional.
            block_depth: The number of blocks in different stages in ResNet.
            num_channels: The number of channels to enter the next stage.
'''
NET_CONFIG = {
    "18": {
        "block_type": "BasicBlock",
        "block_depth": [2, 2, 2, 2],
        "num_channels": [64, 64, 128, 256]
    },
    "34": {
        "block_type": "BasicBlock",
        "block_depth": [3, 4, 6, 3],
        "num_channels": [64, 64, 128, 256]
    },
    "50": {
        "block_type": "BottleneckBlock",
        "block_depth": [3, 4, 6, 3],
        "num_channels": [64, 256, 512, 1024]
    },
    "101": {
        "block_type": "BottleneckBlock",
        "block_depth": [3, 4, 23, 3],
        "num_channels": [64, 256, 512, 1024]
    },
    "152": {
        "block_type": "BottleneckBlock",
        "block_depth": [3, 8, 36, 3],
        "num_channels": [64, 256, 512, 1024]
    },
    "200": {
        "block_type": "BottleneckBlock",
        "block_depth": [3, 12, 48, 3],
        "num_channels": [64, 256, 512, 1024]
    },
}

class CSRA(TheseusLayer):
    """
    http://arxiv.org/abs/2108.02456
    CSRA Layer for attention mechanism in ResNet architecture.
    """
    def __init__(self,
                 input_dim,
                 num_classes,
                 T,
                 lam):
        super().__init__()
        self.T = T
        self.lam = lam
        self.head = Conv2D(
            in_channels=input_dim,
            out_channels=num_classes,
            kernel_size=1,
            bias_attr=False
        )
        self.softmax = Softmax(axis=2)
    
    def forward(self, x):
        # print('head weight shape:',self.head.weight.shape)
        score = self.head(x) / paddle.norm(self.head.weight, axis=1, keepdim=True).transpose([1, 0, 2, 3])
        score = score.flatten(2)
        base_logit = paddle.mean(score, axis=2)

        if self.T == 99: # max-pooling
            att_logit = paddle.max(score, axis=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = paddle.sum(score * score_soft, axis=2)

        return base_logit + self.lam * att_logit


class MHA(TheseusLayer):  # multi-head attention
    temp_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA, self).__init__()
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = LayerList([
            CSRA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit

class ResNet_CSRA(ResNet):
    def __init__(self,
                 num_heads=1,
                 lam=0.1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = MHA(
            num_heads=num_heads,
            lam=lam,
            input_dim=self.num_channels[-1]*2,
            num_classes=self.class_num
        )

    def _forward(self, x):
        if self.data_format == "NHWC":
            x = paddle.transpose(x, [0, 2, 3, 1])
            x.stop_gradient = True
        x = self.stem(x)
        if self.max_pool:
            x = self.max_pool(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        # x = self.flatten(x)
        x = self.fc(x)
        return x

def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )

def ResNet18_CSRA(pretrained=False,
             use_ssld=False,
             layer_type="ConvBNLayer",
             **kwargs):
    """
    ResNet18
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet18` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["18"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet18"],
        version="vb",
        layer_type=layer_type,
        **kwargs)
    if layer_type == "DiverseBranchBlock":
        _load_pretrained(pretrained, model, MODEL_URLS["ResNet18_dbb"],
                         use_ssld)
    else:
        _load_pretrained(pretrained, model, MODEL_URLS["ResNet18"], use_ssld)
    return model


def ResNet18_vd_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet18_vd
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet18_vd` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["18"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet18"],
        version="vd",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet18_vd"], use_ssld)
    return model


def ResNet34_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet34
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet34` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["34"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet34"],
        version="vb",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet34"], use_ssld)
    return model


def ResNet34_vd_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet34_vd
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet34_vd` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["34"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet34"],
        version="vd",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet34_vd"], use_ssld)
    return model


def ResNet50_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet50
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet50` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["50"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet50"],
        version="vb",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet50"], use_ssld)
    return model


def ResNet50_vd_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet50_vd
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet50_vd` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["50"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet50"],
        version="vd",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet50_vd"], use_ssld)
    return model


def ResNet101_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet101
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet101` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["101"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet101"],
        version="vb",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet101"], use_ssld)
    return model


def ResNet101_vd_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet101_vd
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet101_vd` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["101"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet101"],
        version="vd",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet101_vd"], use_ssld)
    return model


def ResNet152_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet152
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet152` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["152"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet152"],
        version="vb",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet152"], use_ssld)
    return model


def ResNet152_vd_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet152_vd
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet152_vd` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["152"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet152"],
        version="vd",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet152_vd"], use_ssld)
    return model


def ResNet200_vd_CSRA(pretrained=False, use_ssld=False, **kwargs):
    """
    ResNet200_vd
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet200_vd` model depends on args.
    """
    model = ResNet_CSRA(
        config=NET_CONFIG["200"],
        stages_pattern=MODEL_STAGES_PATTERN["ResNet200"],
        version="vd",
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["ResNet200_vd"], use_ssld)
    return model
