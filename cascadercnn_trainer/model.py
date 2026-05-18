from __future__ import annotations

from typing import Any, Dict

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from cascadercnn_trainer.presets import resolve_backbone, resolve_scheduler_preset


def build_model(model_cfg: Dict[str, Any], num_classes: int) -> FasterRCNN:
    backbone_name = resolve_backbone(model_cfg["backbone"])
    trainable_layers = int(model_cfg.get("trainable_backbone_layers", 3))

    try:
        backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            weights=None,
            trainable_layers=trainable_layers,
        )
    except TypeError:
        backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            pretrained=False,
            trainable_layers=trainable_layers,
        )

    return FasterRCNN(backbone=backbone, num_classes=num_classes)


def build_optimizer(model: torch.nn.Module, optim_cfg: Dict[str, Any]) -> Optimizer:
    name = optim_cfg.get("name", "SGD").lower()
    lr = float(optim_cfg.get("lr", 0.02))
    wd = float(optim_cfg.get("weight_decay", 0.0001))

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            weight_decay=wd,
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    raise ValueError(f"Unsupported optimizer: {optim_cfg.get('name')}")


def build_scheduler(optimizer: Optimizer, scheduler_cfg: Dict[str, Any]) -> MultiStepLR:
    preset = resolve_scheduler_preset(scheduler_cfg.get("preset", "1x"))
    return MultiStepLR(
        optimizer,
        milestones=list(preset["milestones"]),
        gamma=float(preset["gamma"]),
    )
