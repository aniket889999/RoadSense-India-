import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.ml.local_model_runtime import (
    LocalModelRuntimeError,
    load_verified_yolo_model,
    predict_yolo_frame,
    validate_requested_device,
)


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)


class _FakeModel:
    def __init__(self):
        self.predict_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        boxes = types.SimpleNamespace(
            xyxy=_FakeTensor(
                [
                    [1, 2, 30, 40],
                    [2, 3, 20, 30],
                    [0, 0, 1, 1],
                ]
            ),
            conf=_FakeTensor([0.8, 0.7, float("nan")]),
            cls=_FakeTensor([0, 3, 0]),
        )
        return [types.SimpleNamespace(boxes=boxes)]

    def train(self, *args, **kwargs):
        raise AssertionError("Inference must never call train().")

    def val(self, *args, **kwargs):
        raise AssertionError("Inference must never call val().")


def test_predict_yolo_frame_calls_only_predict_and_normalizes_target_class():
    model = _FakeModel()

    detections = predict_yolo_frame(
        model,
        frame=np.zeros((50, 50, 3), dtype=np.uint8),
        device="mps",
        image_size=640,
        confidence_threshold=0.25,
        iou_threshold=0.7,
        max_detections_per_frame=100,
    )

    assert detections == [
        {
            "class_id": 0,
            "confidence": 0.8,
            "x_min": 1.0,
            "y_min": 2.0,
            "x_max": 30.0,
            "y_max": 40.0,
        }
    ]
    assert len(model.predict_calls) == 1
    assert model.predict_calls[0]["classes"] == [0]
    assert model.predict_calls[0]["device"] == "mps"
    assert model.predict_calls[0]["imgsz"] == 640


def test_predict_yolo_frame_rejects_invalid_runtime_parameters():
    model = _FakeModel()
    with pytest.raises(LocalModelRuntimeError, match="image_size"):
        predict_yolo_frame(
            model,
            np.zeros((10, 10, 3), dtype=np.uint8),
            device="cpu",
            image_size=0,
            confidence_threshold=0.25,
            iou_threshold=0.7,
            max_detections_per_frame=1,
        )


def test_load_verified_yolo_model_rejects_missing_before_import(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)

    with pytest.raises(LocalModelRuntimeError, match="unavailable"):
        load_verified_yolo_model(tmp_path / "missing.pt", "a" * 64)

    assert "ultralytics" not in sys.modules


def test_load_verified_yolo_model_uses_lazy_ultralytics_import(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint_bytes = b"trusted local placeholder"
    checkpoint.write_bytes(checkpoint_bytes)
    created_paths = []

    class FakeYOLO:
        def __init__(self, path):
            created_paths.append(path)
            assert Path(path).read_bytes() == checkpoint_bytes

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    model = load_verified_yolo_model(
        checkpoint.resolve(), hashlib.sha256(checkpoint_bytes).hexdigest()
    )

    assert isinstance(model, FakeYOLO)
    assert len(created_paths) == 1
    assert created_paths[0] != str(checkpoint.resolve())
    assert not Path(created_paths[0]).exists()


def test_load_verified_yolo_model_rejects_changed_checkpoint_before_yolo_import(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"original trusted checkpoint")
    pinned_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.write_bytes(b"replacement checkpoint must never reach YOLO")
    yolo_calls = []

    class FakeYOLO:
        def __init__(self, path):
            yolo_calls.append(path)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    with pytest.raises(LocalModelRuntimeError, match="does not match the pinned SHA-256"):
        load_verified_yolo_model(checkpoint.resolve(), pinned_sha256)

    assert yolo_calls == []


def test_load_verified_yolo_model_rejects_symlinked_parent_before_yolo_import(monkeypatch, tmp_path):
    real_weights = tmp_path / "real-weights"
    real_weights.mkdir()
    checkpoint_bytes = b"trusted checkpoint"
    (real_weights / "best.pt").write_bytes(checkpoint_bytes)
    training_run = tmp_path / "training-run"
    training_run.mkdir()
    (training_run / "weights").symlink_to(real_weights, target_is_directory=True)
    yolo_calls = []

    class FakeYOLO:
        def __init__(self, path):
            yolo_calls.append(path)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    with pytest.raises(LocalModelRuntimeError, match="unsafe path component"):
        load_verified_yolo_model(
            training_run / "weights" / "best.pt", hashlib.sha256(checkpoint_bytes).hexdigest()
        )

    assert yolo_calls == []


def test_validate_requested_device_never_silently_falls_back(monkeypatch):
    fake_torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(LocalModelRuntimeError, match="MPS"):
        validate_requested_device("mps")
    with pytest.raises(LocalModelRuntimeError, match="CUDA"):
        validate_requested_device("cuda")
    assert validate_requested_device("cpu") == "cpu"
