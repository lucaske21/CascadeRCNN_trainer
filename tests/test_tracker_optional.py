from pathlib import Path

from cascadercnn_trainer.tracker import ExperimentTracker


def test_tracker_disabled_works_without_mlflow(tmp_path: Path):
    cfg = {
        "logging": {
            "mlflow": {
                "enabled": False,
            }
        }
    }

    tracker = ExperimentTracker(cfg, tmp_path)
    tracker.start({"foo": "bar"})
    tracker.log_metrics({"loss": 1.0}, step=0)
    tracker.end()

    assert (tmp_path / "run_params.json").exists()
