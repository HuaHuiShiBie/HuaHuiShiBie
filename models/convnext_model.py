import torch.nn as nn
from timm import create_model

def build_convnext(num_classes, pretrained=True):
    """
    构建 ConvNeXt 模型
    :param num_classes: 分类类别数
    :param pretrained: 是否加载 ImageNet 预训练权重
    """
    model = create_model('convnext_small', pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model
