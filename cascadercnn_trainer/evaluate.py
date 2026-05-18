from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cascadercnn_trainer.config import load_yaml_config
from cascadercnn_trainer.engine import evaluate_loss
from cascadercnn_trainer.model import build_model
from cascadercnn_trainer.train import _make_loader
from cascadercnn_trainer.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg["model"], int(cfg["dataset"]["num_classes"]))
    model.to(device)
    load_checkpoint(Path(args.checkpoint), model, map_location=device)

    val_loader = _make_loader(cfg, "val")
    print(evaluate_loss(model, val_loader, device))


if __name__ == "__main__":
    main()
