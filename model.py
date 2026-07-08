"""Step 6: model builders with three transfer-learning strategies.

BACKBONES is the single source of truth for every architecture supported by
both `train.py` and the Gradio UI (`app.py`). To add a new architecture,
add one entry here — everything else (training, freezing, checkpoint
naming, and UI model detection) picks it up automatically.
"""
import torch.nn as nn
from torchvision import models

BACKBONES = {
    "mobilenet_v2": {
        "label": "MobileNetV2",
        "build": lambda weights: models.mobilenet_v2(weights=weights),
        "weights": models.MobileNet_V2_Weights.DEFAULT,
        "head_attr": "classifier",
        "head_index": 1,
    },
    "efficientnet_b0": {
        "label": "EfficientNet-B0",
        "build": lambda weights: models.efficientnet_b0(weights=weights),
        "weights": models.EfficientNet_B0_Weights.DEFAULT,
        "head_attr": "classifier",
        "head_index": 1,
    },
    "efficientnet_b1": {
        "label": "EfficientNet-B1",
        "build": lambda weights: models.efficientnet_b1(weights=weights),
        "weights": models.EfficientNet_B1_Weights.DEFAULT,
        "head_attr": "classifier",
        "head_index": 1,
    },
    "efficientnet_b2": {
        "label": "EfficientNet-B2",
        "build": lambda weights: models.efficientnet_b2(weights=weights),
        "weights": models.EfficientNet_B2_Weights.DEFAULT,
        "head_attr": "classifier",
        "head_index": 1,
    },
    "resnet18": {
        "label": "ResNet18",
        "build": lambda weights: models.resnet18(weights=weights),
        "weights": models.ResNet18_Weights.DEFAULT,
        "head_attr": "fc",
        "head_index": None,
    },
    "resnet50": {
        "label": "ResNet50",
        "build": lambda weights: models.resnet50(weights=weights),
        "weights": models.ResNet50_Weights.DEFAULT,
        "head_attr": "fc",
        "head_index": None,
    },
    "shufflenet_v2": {
        "label": "ShuffleNetV2",
        "build": lambda weights: models.shufflenet_v2_x2_0(weights=weights),
        "weights": models.ShuffleNet_V2_X2_0_Weights.DEFAULT,
        "head_attr": "fc",
        "head_index": None,
    },
}


def _replace_head(model, backbone, num_classes):
    spec = BACKBONES[backbone]
    head_attr, head_index = spec["head_attr"], spec["head_index"]

    if head_index is None:
        in_features = getattr(model, head_attr).in_features
        setattr(model, head_attr, nn.Linear(in_features, num_classes))
    else:
        head = getattr(model, head_attr)
        in_features = head[head_index].in_features
        head[head_index] = nn.Linear(in_features, num_classes)

    return model


def _non_head_params(model, backbone):
    """All params outside the classification head — frozen for the 'last_layer' strategy."""
    head_module = getattr(model, BACKBONES[backbone]["head_attr"])
    head_param_ids = {id(p) for p in head_module.parameters()}
    return [p for p in model.parameters() if id(p) not in head_param_ids]


def build_architecture(backbone, num_classes):
    """Build a model with the right output shape and no pretrained weights.

    Used to reconstruct a model's architecture before loading a trained
    state_dict (e.g. in the Gradio UI) — avoids re-downloading ImageNet
    weights just to immediately overwrite them.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choices: {sorted(BACKBONES)}")

    model = BACKBONES[backbone]["build"](None)
    return _replace_head(model, backbone, num_classes)


def build_model(backbone, strategy, num_classes, device):
    """strategy: 'scratch' (random init, train all), 'full_finetune' (pretrained,
    train all), or 'last_layer' (pretrained, freeze backbone, train head only)."""
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choices: {sorted(BACKBONES)}")
    if strategy not in ("scratch", "full_finetune", "last_layer"):
        raise ValueError("strategy must be one of: scratch, full_finetune, last_layer")

    spec = BACKBONES[backbone]
    weights = None if strategy == "scratch" else spec["weights"]
    model = spec["build"](weights)
    model = _replace_head(model, backbone, num_classes)

    if strategy == "last_layer":
        for param in _non_head_params(model, backbone):
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    return model.to(device)
