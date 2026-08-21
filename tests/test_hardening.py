import pytest
import os
import subprocess
import sys
from src.ml.rdd2022 import convert_to_yolo

def test_convert_to_yolo_raw_negative():
    boxes = [{"class": 0, "xmin": -10, "ymin": 10, "xmax": 20, "ymax": 20}]
    with pytest.raises(ValueError, match="out of image bounds"):
        convert_to_yolo(boxes, 100, 100)

def test_train_pothole_missing_weights_no_flag(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  base_weights: non_existent_weights_123.pt")

    monkeypatch.setattr(sys, "argv", ["train_pothole.py", "--config", str(cfg), "--dataset", "fake_dir"])

    # Mock validation so we reach the weights check
    def mock_val(d): return True, ""
    from scripts import train_pothole
    monkeypatch.setattr("scripts.train_pothole.validate_prepared_yolo_dataset", mock_val)

    with pytest.raises(SystemExit) as exc:
        train_pothole.main()

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "not found locally" in captured.out or "not found locally" in captured.err
