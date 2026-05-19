from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Tuple

import torch

logger = logging.getLogger(__name__)


class DALIUnavailableError(RuntimeError):
    pass


def _to_absolute_xyxy(
    normalized_boxes: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    if normalized_boxes.numel() == 0:
        return normalized_boxes.to(dtype=torch.float32)

    scale = torch.tensor(
        [image_width, image_height, image_width, image_height],
        dtype=normalized_boxes.dtype,
        device=normalized_boxes.device,
    )
    return (normalized_boxes * scale).to(dtype=torch.float32)


class DALIDetectionLoader:
    def __init__(self, iterator: Any) -> None:
        self.iterator = iterator

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]]:
        for batch in self.iterator:
            data = batch[0]
            images = data["images"]
            boxes = data["boxes"]
            labels = data["labels"]
            image_ids = data.get("image_ids")
            if image_ids is None:
                raise RuntimeError("DALI pipeline output is missing 'image_ids'")

            image_list: List[torch.Tensor] = []
            targets: List[Dict[str, torch.Tensor]] = []
            bs = images.shape[0]
            for i in range(bs):
                image_list.append(images[i])
                sample_boxes = boxes.at(i) if hasattr(boxes, "at") else boxes[i]
                sample_labels = labels.at(i) if hasattr(labels, "at") else labels[i]
                sample_image_id = image_ids.at(i) if hasattr(image_ids, "at") else image_ids[i]

                if hasattr(sample_image_id, "item"):
                    sample_image_id = int(sample_image_id.item())
                else:
                    sample_image_id = int(sample_image_id)

                image_height, image_width = int(images[i].shape[-2]), int(images[i].shape[-1])
                abs_boxes = _to_absolute_xyxy(sample_boxes, image_width=image_width, image_height=image_height)
                targets.append(
                    {
                        "boxes": abs_boxes,
                        "labels": sample_labels.to(dtype=torch.int64).flatten(),
                        "image_id": torch.tensor([sample_image_id], dtype=torch.int64),
                        "iscrowd": torch.zeros((sample_labels.shape[0],), dtype=torch.int64, device=sample_labels.device),
                        "area": (abs_boxes[:, 2] - abs_boxes[:, 0]) * (abs_boxes[:, 3] - abs_boxes[:, 1]),
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
        images, bboxes, labels, image_ids = fn.readers.coco(
            file_root=image_dir,
            annotations_file=annotation_file,
            ratio=True,
            ltrb=True,
            random_shuffle=(split == "train"),
            image_ids=True,
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
        return images, bboxes, labels, image_ids

    pipeline = detection_pipe(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        image_dir=dataset_cfg["images"],
        annotation_file=dataset_cfg["annotations"],
    )
    output_map = ["images", "boxes", "labels", "image_ids"]
    iterator = DALIGenericIterator([pipeline], output_map=output_map, auto_reset=True)
    logger.info("Using NVIDIA DALI data pipeline")
    return DALIDetectionLoader(iterator)
