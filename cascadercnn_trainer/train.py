from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch

from cascadercnn_trainer.config import load_yaml_config
from cascadercnn_trainer.dali import DALIUnavailableError, create_dali_dataloader
from cascadercnn_trainer.data import create_torch_dataloader
from cascadercnn_trainer.engine import evaluate_loss, train_one_epoch
from cascadercnn_trainer.model import build_model, build_optimizer, build_scheduler
from cascadercnn_trainer.tracker import ExperimentTracker
from cascadercnn_trainer.utils import load_checkpoint, save_checkpoint, setup_logging

logger = logging.getLogger(__name__)


def _make_loader(cfg: Dict[str, Any], split: str):
    if cfg.get("dali", {}).get("enabled", False):
        try:
            return create_dali_dataloader(cfg, split)
        except DALIUnavailableError as err:
            logger.warning("DALI unavailable for split=%s (%s). Falling back to PyTorch DataLoader.", split, err)

    data_cfg = cfg["dataset"][split]
    aug_cfg = cfg.get("augmentation", {}) if split == "train" else {}
    return create_torch_dataloader(
        images_dir=data_cfg["images"],
        annotation_file=data_cfg["annotations"],
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=(split == "train"),
        num_workers=int(cfg["training"].get("num_workers", 4)),
        augmentation_cfg=aug_cfg,
    )


def run_training(cfg: Dict[str, Any]) -> None:
    output_dir = Path(cfg["logging"].get("output_dir", "outputs"))
    setup_logging(output_dir)

    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Training on %s", device)

    model = build_model(cfg["model"], int(cfg["dataset"]["num_classes"]))
    model.to(device)

    optimizer = build_optimizer(model, cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"])

    train_loader = _make_loader(cfg, "train")
    val_loader = _make_loader(cfg, "val")

    tracker = ExperimentTracker(cfg, output_dir)
    tracker.start(cfg)

    start_epoch = 0
    resume_path = cfg["training"].get("resume_from")
    if resume_path:
        start_epoch = load_checkpoint(Path(resume_path), model, optimizer, scheduler, map_location=device)
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    epochs = int(cfg["training"]["epochs"])
    checkpoint_every = int(cfg["training"].get("checkpoint_every", 1))

    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(model, optimizer, train_loader, device)
        val_metrics = evaluate_loss(model, val_loader, device)
        scheduler.step()

        metrics = {**train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        logger.info("Epoch %d/%d - %s", epoch + 1, epochs, metrics)
        tracker.log_metrics(metrics, step=epoch)

        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, cfg)
            tracker.log_artifact(ckpt_path)

    final_ckpt = output_dir / "model_final.pt"
    save_checkpoint(final_ckpt, model, optimizer, scheduler, epochs - 1, cfg)
    tracker.log_artifact(final_ckpt)
    tracker.end()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CascadeRCNN HBB trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(load_yaml_config(args.config))
