from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, cfg: Dict[str, Any], output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = output_dir
        self.enabled = bool(cfg.get("logging", {}).get("mlflow", {}).get("enabled", False))
        self._mlflow = None
        self._active = False

        if self.enabled:
            try:
                import mlflow  # type: ignore

                self._mlflow = mlflow
            except ImportError:
                logger.warning("MLflow is enabled in config but not installed. Disabling MLflow.")
                self.enabled = False

    def start(self, params: Dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        params_file = self.output_dir / "run_params.json"
        params_file.write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")

        if not self.enabled or self._mlflow is None:
            return

        tracking_cfg = self.cfg.get("logging", {}).get("mlflow", {})
        uri = tracking_cfg.get("tracking_uri")
        if uri:
            self._mlflow.set_tracking_uri(uri)

        exp_name = tracking_cfg.get("experiment_name", "CascadeRCNN_trainer")
        run_name = tracking_cfg.get("run_name")
        self._mlflow.set_experiment(exp_name)
        self._mlflow.start_run(run_name=run_name)
        flat_params = _flatten_dict(params)
        self._mlflow.log_params(flat_params)
        self._active = True

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.enabled and self._mlflow is not None and self._active:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        if self.enabled and self._mlflow is not None and self._active and path.exists():
            self._mlflow.log_artifact(str(path))

    def end(self) -> None:
        if self.enabled and self._mlflow is not None and self._active:
            self._mlflow.end_run()
            self._active = False


def _flatten_dict(input_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in input_dict.items():
        composite_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, composite_key))
        else:
            flat[composite_key] = value
    return flat
