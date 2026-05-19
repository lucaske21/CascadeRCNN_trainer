from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch


@torch.inference_mode()
def evaluate_loss(
    model: torch.nn.Module,
    data_loader: Iterable[Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]],
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    count = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values()).item()
        total += loss
        count += 1
    return {"val_loss": total / max(1, count)}


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable[Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]],
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        total_loss += float(losses.item())
        steps += 1

    return {"train_loss": total_loss / max(1, steps)}
