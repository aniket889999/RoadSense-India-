from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.ml.experimental_inference import (
    ExperimentalInferenceSettings,
    build_public_inference_metadata,
    build_verified_frame_predictor,
    run_verified_sampled_video_inference,
)
from src.ml.model_provenance import FrozenModelInfo


def _model_info():
    digest = "a" * 64
    return FrozenModelInfo(
        run_id="pothole_yolov8n_rdd2022_india_mps_baseline_v1",
        training_run_directory=Path("/private/training-run"),
        checkpoint_path=Path("/private/training-run/weights/best.pt"),
        checkpoint_sha256=digest,
        model_metadata_path=Path("/private/training-run/model_metadata.json"),
        model_metadata_sha256="b" * 64,
        base_weights_path=Path("/private/models/yolov8n.pt"),
        base_weights_sha256="c" * 64,
        git_sha="d" * 40,
        dataset_fingerprint="e" * 64,
        task="detection",
        class_mapping={"0": "pothole"},
        artifact_hashes={"weights/best.pt": digest},
    )


def test_public_metadata_keeps_provenance_but_never_paths():
    settings = ExperimentalInferenceSettings(
        device="mps",
        image_size=640,
        confidence_threshold=0.25,
        iou_threshold=0.7,
        max_detections_per_frame=100,
        output_fps=5.0,
    )

    metadata = build_public_inference_metadata(_model_info(), settings)

    assert metadata["baseline_run_id"] == "pothole_yolov8n_rdd2022_india_mps_baseline_v1"
    assert metadata["checkpoint_sha256"] == "a" * 64
    assert metadata["held_out_test_reused"] is False
    assert all("path" not in key and "directory" not in key for key in metadata)
    assert "/private" not in repr(metadata)


def test_settings_reject_invalid_runtime_values():
    with pytest.raises(ValueError, match="confidence_threshold"):
        ExperimentalInferenceSettings(
            device="mps",
            image_size=640,
            confidence_threshold=1.1,
            iou_threshold=0.7,
            max_detections_per_frame=100,
            output_fps=5.0,
        )
    with pytest.raises(ValueError, match="output_fps"):
        ExperimentalInferenceSettings(
            device="mps",
            image_size=640,
            confidence_threshold=0.25,
            iou_threshold=0.7,
            max_detections_per_frame=100,
            output_fps=0.0,
        )


def test_verified_predictor_forwards_only_predict_parameters():
    settings = ExperimentalInferenceSettings(
        device="cpu",
        image_size=320,
        confidence_threshold=0.5,
        iou_threshold=0.65,
        max_detections_per_frame=5,
        output_fps=5.0,
    )
    calls = []

    class FakeTensor:
        def __init__(self, data):
            self.data = data

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.data)

    class FakeModel:
        def predict(self, **kwargs):
            calls.append(kwargs)
            boxes = SimpleNamespace(
                xyxy=FakeTensor([[1, 2, 10, 20]]),
                conf=FakeTensor([0.9]),
                cls=FakeTensor([0]),
            )
            return [SimpleNamespace(boxes=boxes)]

        def train(self, **_kwargs):
            raise AssertionError("Experimental inference must never train.")

        def val(self, **_kwargs):
            raise AssertionError("Experimental inference must never evaluate.")

    predictor = build_verified_frame_predictor(FakeModel(), settings)
    detections = predictor(np.zeros((32, 32, 3), dtype=np.uint8))

    assert detections[0]["class_id"] == 0
    assert calls[0]["device"] == "cpu"
    assert calls[0]["imgsz"] == 320
    assert calls[0]["conf"] == 0.5
    assert calls[0]["iou"] == 0.65
    assert calls[0]["max_det"] == 5


def test_rejects_malformed_optional_input_hash_before_video_work():
    settings = ExperimentalInferenceSettings(
        device="cpu",
        image_size=320,
        confidence_threshold=0.5,
        iou_threshold=0.65,
        max_detections_per_frame=5,
        output_fps=5.0,
    )
    with pytest.raises(ValueError, match="input_video_sha256"):
        run_verified_sampled_video_inference(
            video_bytes=b"not opened because hash is invalid",
            video_filename="input.mp4",
            sampled_frame_indices=[0],
            model=object(),
            model_info=_model_info(),
            settings=settings,
            input_video_sha256="not-a-sha",
        )
