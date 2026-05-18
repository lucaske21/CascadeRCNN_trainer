from pathlib import Path

from cascadercnn_trainer.config import load_yaml_config
from cascadercnn_trainer.presets import resolve_backbone, resolve_scheduler_preset


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_base_override_works():
    cfg = load_yaml_config(REPO_ROOT / "configs" / "train_r101_20e.yaml")
    assert cfg["model"]["backbone"] == "R-101-FPN"
    assert cfg["scheduler"]["preset"] == "20e"
    assert cfg["training"]["epochs"] == 20
    assert cfg["optimizer"]["name"] == "SGD"


def test_required_backbones_resolve():
    assert resolve_backbone("R-50-FPN") == "resnet50"
    assert resolve_backbone("R-101-FPN") == "resnet101"
    assert resolve_backbone("X-101-32x4d-FPN") == "resnext101_32x4d"


def test_required_lr_presets_resolve():
    assert resolve_scheduler_preset("1x")["milestones"] == [8, 11]
    assert resolve_scheduler_preset("20e")["milestones"] == [16, 19]
