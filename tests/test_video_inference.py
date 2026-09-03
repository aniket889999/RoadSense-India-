import csv
import json
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.ml.video_inference as video_inference
from src.ml.video_inference import ExperimentalDetection, run_sampled_video_inference


def create_marker_video(frame_count=8, width=96, height=72):
    """Return an MP4 whose uniformly coloured frames identify their index."""

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            marker = index * 25
            frame = np.full((height, width, 3), marker, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    try:
        return Path(path).read_bytes(), width, height
    finally:
        os.remove(path)


def archive_member_frame_count(report_zip, tmp_path):
    with zipfile.ZipFile(BytesIO(report_zip), "r") as archive:
        output_path = tmp_path / "annotated.mp4"
        output_path.write_bytes(archive.read("annotated_experimental_predictions.mp4"))

    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def test_uses_exact_explicit_frames_and_packages_a_path_free_report(tmp_path):
    video_bytes, width, height = create_marker_video()
    observed_markers = []

    def predictor(frame):
        observed_markers.append(int(frame[0, 0, 0]))
        return {
            "class_id": 0,
            "confidence": 0.90,
            "x_min": 5,
            "y_min": 6,
            "x_max": width - 5,
            "y_max": height - 6,
        }

    result = run_sampled_video_inference(
        video_bytes,
        "/private/secret/source-video.mp4",
        [1, 4, 7],
        predictor,
        confidence_threshold=0.5,
        model_metadata={
            "model_source": "local_frozen_baseline",
            "baseline_run_id": "synthetic-callback",
            "checkpoint_sha256": "a" * 64,
            "model_metadata_sha256": "b" * 64,
            "training_git_sha": "c" * 40,
            "dataset_fingerprint": "d" * 64,
            "task": "detection",
            "class_mapping": {"0": "pothole"},
            "device": "cpu",
            "image_size": 320,
            "iou_threshold": 0.7,
            "max_detections_per_frame": 5,
            "local_only": True,
            "held_out_test_reused": False,
            "weights_path": "/private/secret/weights.pt",
            "uri": "file:///private/secret/weights.pt",
            "remote_like_uri": "file://localhost/private/secret/weights.pt",
            "nested": {"dataset_directory": "/private/secret/data", "uri": "file:///private/secret/data"},
        },
    )

    assert result.total_sampled_frames == 3
    assert result.frames_with_detections == 3
    assert [detection.frame_index for detection in result.detections] == [1, 4, 7]
    assert len(observed_markers) == 3  # no inference happened on unsampled frames
    assert all(abs(actual - expected) <= 10 for actual, expected in zip(observed_markers, [25, 100, 175]))

    with zipfile.ZipFile(BytesIO(result.report_zip), "r") as archive:
        assert set(archive.namelist()) == {
            "model_predictions.csv",
            "inference_metadata.json",
            "annotated_experimental_predictions.mp4",
        }
        metadata_text = archive.read("inference_metadata.json").decode("utf-8")
        metadata = json.loads(metadata_text)
        assert metadata["sampled_frame_indices"] == [1, 4, 7]
        assert metadata["total_sampled_frames"] == 3
        assert metadata["annotated_video_frame_count"] == 3
        assert metadata["human_verification_status"] == "not_human_verified"
        assert metadata["model_metadata"] == {
            "model_source": "local_frozen_baseline",
            "baseline_run_id": "synthetic-callback",
            "checkpoint_sha256": "a" * 64,
            "model_metadata_sha256": "b" * 64,
            "training_git_sha": "c" * 40,
            "dataset_fingerprint": "d" * 64,
            "task": "detection",
            "class_mapping": {"0": "pothole"},
            "device": "cpu",
            "image_size": 320,
            "iou_threshold": 0.7,
            "max_detections_per_frame": 5,
            "local_only": True,
            "held_out_test_reused": False,
        }
        assert "/private/secret" not in metadata_text
        assert "source-video.mp4" not in metadata_text
        assert "file:" not in metadata_text
        assert "nested" not in metadata_text

        records = list(csv.DictReader(archive.read("model_predictions.csv").decode("utf-8").splitlines()))
        assert [int(record["frame_index"]) for record in records] == [1, 4, 7]
        assert all(int(record["class_id"]) == 0 for record in records)

    assert archive_member_frame_count(result.report_zip, tmp_path) == 3


def test_empty_predictions_are_valid_and_still_render_all_sampled_frames(tmp_path):
    video_bytes, _, _ = create_marker_video(frame_count=5)
    calls = []

    def predictor(frame):
        calls.append(frame.shape)
        return []

    result = run_sampled_video_inference(
        video_bytes,
        "input.mp4",
        [0, 3],
        predictor,
        confidence_threshold=0.25,
    )

    assert len(calls) == 2
    assert result.detections == []
    assert result.total_sampled_frames == 2
    assert result.frames_with_detections == 0
    with zipfile.ZipFile(BytesIO(result.report_zip), "r") as archive:
        assert archive.read("model_predictions.csv").decode("utf-8").splitlines() == [
            "frame_index,timestamp_seconds,class_id,confidence,x_min,y_min,x_max,y_max"
        ]
    assert archive_member_frame_count(result.report_zip, tmp_path) == 2


def test_filters_invalid_or_non_pothole_predictions_without_clamping(tmp_path):
    video_bytes, width, height = create_marker_video(frame_count=3)

    def predictor(_frame):
        return [
            {"class_id": 0, "confidence": 0.8, "x_min": 1, "y_min": 2, "x_max": 40, "y_max": 50},
            {"class_id": 0, "confidence": 0.7, "x1": 2, "y1": 3, "x2": width, "y2": height},
            {"class_id": 1, "confidence": 0.9, "x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2},
            {"class_id": 0, "confidence": 0.49, "x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2},
            {"class_id": 0, "confidence": float("nan"), "x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2},
            {"class_id": 0, "confidence": 1.1, "x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2},
            {"class_id": 0, "confidence": 0.8, "x_min": -1, "y_min": 1, "x_max": 2, "y_max": 2},
            {"class_id": 0, "confidence": 0.8, "x_min": 2, "y_min": 1, "x_max": 2, "y_max": 2},
            {"class_id": 0, "confidence": 0.8, "x_min": 1, "y_min": 1, "x_max": width + 1, "y_max": 2},
            {"class_id": 0, "confidence": 0.8, "xyxy": [1, 1, float("inf"), 2]},
            {"class_id": False, "confidence": 0.8, "x_min": 1, "y_min": 1, "x_max": 2, "y_max": 2},
            {"class_id": 0, "confidence": 0.8, "xyxy": [1, 1, 3]},
        ]

    result = run_sampled_video_inference(
        video_bytes,
        "input.mp4",
        [1],
        predictor,
        confidence_threshold=0.5,
    )

    assert result.frames_with_detections == 1
    assert len(result.detections) == 2
    assert all(0.0 <= detection.confidence <= 1.0 for detection in result.detections)
    assert all(0 <= detection.x_min < detection.x_max <= width for detection in result.detections)
    assert all(0 <= detection.y_min < detection.y_max <= height for detection in result.detections)
    assert archive_member_frame_count(result.report_zip, tmp_path) == 1


def test_unreadable_video_and_writer_failure_clean_temp_resources(tmp_path, monkeypatch):
    real_named_temporary_file = tempfile.NamedTemporaryFile
    real_mkstemp = tempfile.mkstemp

    def local_named_temporary_file(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        return real_named_temporary_file(*args, **kwargs)

    def local_mkstemp(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(video_inference.tempfile, "NamedTemporaryFile", local_named_temporary_file)
    monkeypatch.setattr(video_inference.tempfile, "mkstemp", local_mkstemp)

    with pytest.raises(ValueError, match="Unreadable video file"):
        run_sampled_video_inference(b"not a video", "bad.mp4", [0], lambda _frame: [])
    assert list(tmp_path.iterdir()) == []

    video_bytes, _, _ = create_marker_video(frame_count=2)

    class ClosedWriter:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.released = False
            self.__class__.instances.append(self)

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    monkeypatch.setattr(video_inference.cv2, "VideoWriter", ClosedWriter)
    with pytest.raises(RuntimeError, match="Failed to open VideoWriter"):
        run_sampled_video_inference(video_bytes, "input.mp4", [0], lambda _frame: [])
    assert ClosedWriter.instances[0].released is True
    assert list(tmp_path.iterdir()) == []


def test_module_does_not_import_training_frameworks():
    source = Path(video_inference.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import ultralytics" not in source


def test_rejects_non_monotonic_sampled_frame_plan():
    video_bytes, _, _ = create_marker_video(frame_count=4)

    with pytest.raises(ValueError, match="strictly increasing"):
        run_sampled_video_inference(video_bytes, "input.mp4", [2, 1], lambda _frame: [])


def test_drive_review_circle_marker_is_drawn_only_for_raw_suggestions(monkeypatch):
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    detection = ExperimentalDetection(
        frame_index=3,
        timestamp_seconds=0.3,
        class_id=0,
        confidence=0.75,
        x_min=20.0,
        y_min=10.0,
        x_max=60.0,
        y_max=50.0,
    )
    calls = []
    real_circle = video_inference.cv2.circle

    def spy_circle(*args, **kwargs):
        calls.append((args, kwargs))
        return real_circle(*args, **kwargs)

    monkeypatch.setattr(video_inference.cv2, "circle", spy_circle)
    video_inference._draw_experimental_overlays(
        frame,
        [detection],
        frame_index=3,
        timestamp_seconds=0.3,
        render_mode="drive_review",
    )
    assert calls
    assert calls[0][0][1] == (40, 30)
    assert calls[0][0][3] == (0, 220, 0)

    calls.clear()
    video_inference._draw_experimental_overlays(
        frame,
        [],
        frame_index=4,
        timestamp_seconds=0.4,
        render_mode="drive_review",
    )
    assert calls == []


def test_rejects_unknown_render_mode_before_video_processing():
    with pytest.raises(ValueError, match="render_mode"):
        run_sampled_video_inference(
            b"not-opened",
            "input.mp4",
            [0],
            lambda _frame: [],
            render_mode="live_alert",
        )
