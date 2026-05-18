from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import onnx
import torch

from cascadercnn_trainer.config import load_yaml_config
from cascadercnn_trainer.model import build_model
from cascadercnn_trainer.utils import load_checkpoint


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(list(images))
        first = outputs[0]
        return first["boxes"], first["scores"], first["labels"]


def _fold_batchnorm_if_requested(onnx_model: onnx.ModelProto, enabled: bool) -> onnx.ModelProto:
    if not enabled:
        return onnx_model
    try:
        import onnxoptimizer  # type: ignore

        return onnxoptimizer.optimize(onnx_model, ["fuse_bn_into_conv"])
    except Exception:
        return onnx_model


def _simplify_if_requested(
    onnx_model: onnx.ModelProto,
    enabled: bool,
    input_shape: Tuple[int, int, int, int],
) -> onnx.ModelProto:
    if not enabled:
        return onnx_model
    try:
        from onnxsim import simplify  # type: ignore

        simplified, ok = simplify(
            onnx_model,
            input_shapes={"images": list(input_shape)},
            dynamic_input_shape=True,
        )
        return simplified if ok else onnx_model
    except Exception:
        return onnx_model


def export_onnx(cfg: Dict[str, Any], checkpoint: Path, output: Path) -> None:
    model = build_model(cfg["model"], int(cfg["dataset"]["num_classes"]))
    load_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()

    wrapper = ONNXWrapper(model)
    export_cfg = cfg.get("export", {}).get("onnx", {})

    image_size = export_cfg.get("input_size", [3, 800, 800])
    dummy = torch.randn(1, image_size[0], image_size[1], image_size[2], dtype=torch.float32)

    dynamic_axes = None
    if export_cfg.get("dynamic_batch", True):
        dynamic_axes = {
            "images": {0: "batch"},
            "boxes": {0: "num_boxes"},
            "scores": {0: "num_boxes"},
            "labels": {0: "num_boxes"},
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(output),
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes=dynamic_axes,
        opset_version=int(export_cfg.get("opset", 12)),
    )

    onnx_model = onnx.load(str(output))
    onnx_model = _fold_batchnorm_if_requested(onnx_model, bool(export_cfg.get("fold_batchnorm", False)))
    onnx_model = _simplify_if_requested(onnx_model, bool(export_cfg.get("simplify", False)), tuple(dummy.shape))

    metadata = export_cfg.get("metadata", {})
    if metadata:
        onnx_model.metadata_props.clear()
        for key, value in metadata.items():
            prop = onnx_model.metadata_props.add()
            prop.key = str(key)
            prop.value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

    onnx.save(onnx_model, str(output))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export checkpoint to ONNX")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml_config(args.config)
    export_onnx(cfg, Path(args.checkpoint), Path(args.output))
