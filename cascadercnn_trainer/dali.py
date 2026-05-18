from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Tuple

import torch

logger = logging.getLogger(__name__)


class DALIUnavailableError(RuntimeError):
    pass


class DALIDetectionLoader:
    def __init__(self, iterator: Any) -> None:
        self.iterator = iterator

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]]:
        for batch in self.iterator:
            data = batch[0]
            images = data["images"]
            boxes = data["boxes"]
            labels = data["labels"]

            image_list: List[torch.Tensor] = []
            targets: List[Dict[str, torch.Tensor]] = []
            bs = images.shape[0]
            for i in range(bs):
                image_list.append(images[i])
                sample_boxes = boxes.at(i) if hasattr(boxes, "at") else boxes[i]
                sample_labels = labels.at(i) if hasattr(labels, "at") else labels[i]
                targets.append(
                    {
                        "boxes": sample_boxes.to(dtype=torch.float32),
                        "labels": sample_labels.to(dtype=torch.int64).flatten(),
                        "image_id": torch.tensor([i]),
                        "iscrowd": torch.zeros((sample_labels.shape[0],), dtype=torch.int64, device=sample_labels.device),
                        "area": (sample_boxes[:, 2] - sample_boxes[:, 0]) * (sample_boxes[:, 3] - sample_boxes[:, 1]),
                    }
                )
            yield image_list, targets


def create_dali_dataloader(cfg: Dict[str, Any], split: str) -> DALIDetectionLoader:
    dali_cfg = cfg.get("dali", {})
    if not dali_cfg.get("enabled", False):
        raise DALIUnavailableError("DALI is disabled in config")

    try:
        from nvidia.dali import fn, pipeline_def, types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
    except ImportError as exc:
        raise DALIUnavailableError("NVIDIA DALI is not installed") from exc

    dataset_cfg = cfg["dataset"][split]
    batch_size = int(cfg["training"]["batch_size"])
    num_threads = int(dali_cfg.get("num_threads", 2))
    device_id = int(dali_cfg.get("device_id", 0))
    image_size = dali_cfg.get("image_size", [800, 800])

    @pipeline_def
    def detection_pipe(image_dir: str, annotation_file: str):
        images, bboxes, labels = fn.readers.coco(
            file_root=image_dir,
            annotations_file=annotation_file,
            ratio=True,
            ltrb=True,
            random_shuffle=(split == "train"),
            name="Reader",
        )
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, resize_x=image_size[0], resize_y=image_size[1])
        images = fn.crop_mirror_normalize(
            images,
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
        )
        return images, bboxes, labels

    pipeline = detection_pipe(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        image_dir=dataset_cfg["images"],
        annotation_file=dataset_cfg["annotations"],
    )
    output_map = ["images", "boxes", "labels"]
    iterator = DALIGenericIterator([pipeline], output_map=output_map, auto_reset=True)
    logger.info("Using NVIDIA DALI data pipeline")
    return DALIDetectionLoader(iterator)
