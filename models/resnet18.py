
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(
    num_classes=2,
    pretrained=True,
    freeze_backbone=True
):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
