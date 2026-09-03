from pathlib import Path

import pytest

import src.ml.drive_review as drive_review
from src.ml.drive_review import DriveReviewPlan, build_drive_review_plan, run_verified_drive_review
from src.ml.experimental_inference import ExperimentalInferenceSettings


def test_builds_a_contiguous_bounded_window_plan():
    plan = build_drive_review_plan(
        frame_count=300,
        source_fps=10.0,
        window_start_seconds=5.0,
        window_duration_seconds=10.0,
        sampling_fps=2.0,
        max_frames=120,
    )

    assert plan.frame_indices[0] == 50
    assert plan.frame_indices[-1] == 149
    assert plan.window_start_seconds == 5.0
    assert plan.window_end_seconds == 15.0
    assert plan.sampled_frame_count == 21
    assert all(50 <= index <= 149 for index in plan.frame_indices)
    assert list(plan.frame_indices) == sorted(set(plan.frame_indices))


def test_plan_never_exceeds_cap_and_spreads_across_the_same_window():
    plan = build_drive_review_plan(
        frame_count=600,
        source_fps=10.0,
        window_start_seconds=0.0,
        window_duration_seconds=60.0,
        sampling_fps=10.0,
        max_frames=5,
    )

    assert plan.frame_indices == (0, 149, 299, 449, 599)
    assert plan.sampled_frame_count == 5
    assert plan.max_frames == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 0}, "frame_count"),
        ({"source_fps": 0.0}, "source_fps"),
        ({"window_start_seconds": -1.0}, "window_start_seconds"),
        ({"window_duration_seconds": 0.0}, "window_duration_seconds"),
        ({"sampling_fps": 0.0}, "sampling_fps"),
        ({"max_frames": 0}, "max_frames"),
        ({"window_start_seconds": 10.0}, "inside the uploaded video duration"),
    ],
)
def test_rejects_invalid_review_windows(kwargs, message):
    values = {
        "frame_count": 100,
        "source_fps": 10.0,
        "window_start_seconds": 0.0,
        "window_duration_seconds": 5.0,
        "sampling_fps": 2.0,
        "max_frames": 10,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        build_drive_review_plan(**values)


def test_runs_only_explicit_plan_with_drive_review_render_mode(monkeypatch):
    plan = DriveReviewPlan(
        frame_indices=(10, 15, 20),
        source_fps=5.0,
        source_duration_seconds=10.0,
        window_start_seconds=2.0,
        window_end_seconds=5.0,
        requested_sampling_fps=1.0,
        max_frames=20,
    )
    settings = ExperimentalInferenceSettings(
        device="cpu",
        image_size=320,
        confidence_threshold=0.25,
        iou_threshold=0.7,
        max_detections_per_frame=10,
        output_fps=5.0,
    )
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return "synthetic-result"

    monkeypatch.setattr(drive_review, "run_verified_sampled_video_inference", fake_run)
    result = run_verified_drive_review(
        video_bytes=b"video",
        video_filename="review.mp4",
        plan=plan,
        model=object(),
        model_info=object(),
        settings=settings,
        input_video_sha256="a" * 64,
    )

    assert result == "synthetic-result"
    assert captured["sampled_frame_indices"] == (10, 15, 20)
    assert captured["render_mode"] == "drive_review"
    assert captured["input_video_sha256"] == "a" * 64


def test_module_has_no_training_or_framework_imports():
    source = Path(drive_review.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import ultralytics" not in source
    assert "train_pothole" not in source
    assert "evaluate_pothole" not in source
