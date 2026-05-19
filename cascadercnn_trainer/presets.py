from __future__ import annotations

from typing import Dict, List

SUPPORTED_BACKBONES: Dict[str, str] = {
    "R-50-FPN": "resnet50",
    "R-101-FPN": "resnet101",
    "X-101-32x4d-FPN": "resnext101_32x4d",
}

SCHEDULER_PRESETS: Dict[str, Dict[str, List[int] | float]] = {
    "1x": {"milestones": [8, 11], "gamma": 0.1},
    "20e": {"milestones": [16, 19], "gamma": 0.1},
}


def resolve_backbone(name: str) -> str:
    try:
        return SUPPORTED_BACKBONES[name]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_BACKBONES)
        raise ValueError(f"Unsupported backbone '{name}'. Supported: {supported}") from exc


def resolve_scheduler_preset(name: str) -> Dict[str, List[int] | float]:
    try:
        return SCHEDULER_PRESETS[name]
    except KeyError as exc:
        supported = ", ".join(SCHEDULER_PRESETS)
        raise ValueError(f"Unsupported scheduler preset '{name}'. Supported: {supported}") from exc
