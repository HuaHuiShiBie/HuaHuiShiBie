# coding: utf-8
"""
model.py
模型定义 - 使用 timm 中的 convnext_small（可替换）。
"""
import torch

def create_model(num_classes, pretrained=True, model_name='convnext_small'):
    try:
        import timm
    except Exception as e:
        raise ImportError("timm is required. Install with `pip install timm`") from e
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model
