#
import torch
import torch.nn as nn
from torchvision import models

def build_model(arch: str = "densenet121", num_outputs: int = 14, pretrained: bool = True):
    if arch.lower() == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = m.classifier.in_features
        m.classifier = nn.Linear(in_feats, num_outputs)
    elif arch.lower() == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_outputs)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return m
