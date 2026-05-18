from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


class CocoHBBDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        annotation_file: str | Path,
        augmentation_cfg: Dict[str, Any] | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.augmentation_cfg = augmentation_cfg or {}

        with Path(annotation_file).open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.images = {img["id"]: img for img in raw.get("images", [])}
        self.ids = sorted(self.images)
        self.annotations: Dict[int, List[Dict[str, Any]]] = {i: [] for i in self.ids}
        for ann in raw.get("annotations", []):
            image_id = ann["image_id"]
            if image_id in self.annotations:
                self.annotations[image_id].append(ann)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.ids[index]
        image_info = self.images[image_id]
        image_path = self.images_dir / image_info["file_name"]

        image = Image.open(image_path).convert("RGB")
        width, _ = image.size

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ann in self.annotations.get(image_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            area.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        if self.augmentation_cfg.get("horizontal_flip_prob", 0.0) > 0:
            prob = float(self.augmentation_cfg["horizontal_flip_prob"])
            if torch.rand(1).item() < prob:
                image = F.hflip(image)
                if boxes_tensor.numel():
                    xmin = boxes_tensor[:, 0].clone()
                    xmax = boxes_tensor[:, 2].clone()
                    boxes_tensor[:, 0] = width - xmax
                    boxes_tensor[:, 2] = width - xmin

        image_tensor = F.to_tensor(image)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(area, dtype=torch.float32) if area else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))


def create_torch_dataloader(
    images_dir: str | Path,
    annotation_file: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    augmentation_cfg: Dict[str, Any] | None,
) -> DataLoader:
    dataset = CocoHBBDataset(images_dir, annotation_file, augmentation_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
