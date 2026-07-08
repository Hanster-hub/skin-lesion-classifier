"""Step 6: MobileNetV2 model builder with three transfer-learning strategies."""
import torch.nn as nn
from torchvision import models


def build_mobilenet(strategy, num_classes, device):
    """strategy: 'scratch' (random init, train all), 'full_finetune' (pretrained,
    train all), or 'last_layer' (pretrained, freeze backbone, train classifier only)."""
    if strategy == "scratch":
        model = models.mobilenet_v2(weights=None)
    elif strategy in ("full_finetune", "last_layer"):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        raise ValueError("strategy must be one of: scratch, full_finetune, last_layer")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if strategy == "last_layer":
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model.to(device)
