# CascadeRCNN_trainer

PyTorch-based HBB object detection training framework with YAML-driven experiments, optional MLflow, optional NVIDIA DALI input pipeline, ONNX export, and Docker support.

## Features

- HBB detection training/evaluation with PyTorch (`torchvision` FasterRCNN + FPN)
- YAML config system with `base_config` override support
- Backbone switching via config only:
  - `R-50-FPN`
  - `R-101-FPN`
  - `X-101-32x4d-FPN`
- LR scheduler presets via config only:
  - `1x`
  - `20e`
- Optional MLflow logging (params/metrics/artifacts)
- Optional NVIDIA DALI dataloader path (with automatic fallback)
- ONNX export utility:
  - dynamic batch axis
  - metadata writing
  - optional BatchNorm folding
  - optional graph simplification
- Dockerized runtime

## Install

```bash
pip install -r requirements.txt
# optional features
pip install -r requirements-optional.txt
```

## Train

```bash
python -m cascadercnn_trainer.train --config configs/train_r50_1x.yaml
```

To switch backbone/scheduler, only change YAML:

```yaml
model:
  backbone: X-101-32x4d-FPN
scheduler:
  preset: 20e
```

## Evaluate

```bash
python -m cascadercnn_trainer.evaluate \
  --config configs/train_r50_1x.yaml \
  --checkpoint outputs/r50_1x/model_final.pt
```

## Export ONNX

```bash
python -m cascadercnn_trainer.export_onnx \
  --config configs/train_r50_1x.yaml \
  --checkpoint outputs/r50_1x/model_final.pt \
  --output outputs/r50_1x/model.onnx
```

ONNX options are controlled in YAML under `export.onnx`.

## Optional MLflow

Enable in YAML:

```yaml
logging:
  mlflow:
    enabled: true
    tracking_uri: http://localhost:5000
    experiment_name: CascadeRCNN_trainer
```

If MLflow is disabled (or not installed), training still runs normally.

## Optional NVIDIA DALI

Enable in YAML:

```yaml
dali:
  enabled: true
  image_size: [800, 800]
```

When disabled or unavailable, the system automatically falls back to standard PyTorch DataLoader.

## Docker

```bash
docker build -t cascadercnn-trainer .
docker run --gpus all --rm -it \
  -v /path/to/data:/data \
  cascadercnn-trainer
```

## Config files

- `configs/base.yaml`: base defaults
- `configs/train_r50_1x.yaml`: required `R-50-FPN` + `1x`
- `configs/train_r101_20e.yaml`: required `R-101-FPN` + `20e`
- `configs/train_x101_20e.yaml`: required `X-101-32x4d-FPN` + `20e`
