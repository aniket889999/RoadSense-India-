"""Unit tests for bounded OpenCV streaming frame pipeline and overlays."""

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.media.frame_pipeline import (
    FrameProcessingOptions,
    extract_evidence_crop,
    render_detection_overlays,
    rotate_frame,
    stream_video_frames,
)
from src.media.inspector import inspect_media_file
from tests.test_media_inspector import _create_synthetic_test_video


def test_frame_rotation():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    r90 = rotate_frame(img, 90)
    assert r90.shape[:2] == (200, 100)

    r180 = rotate_frame(img, 180)
    assert r180.shape[:2] == (100, 200)

    r270 = rotate_frame(img, 270)
    assert r270.shape[:2] == (200, 100)


def test_stream_video_frames_bounded():
    with tempfile.TemporaryDirectory() as tmpdir:
        vid_p = Path(tmpdir) / "synth.mp4"
        _create_synthetic_test_video(vid_p, num_frames=30, fps=15.0, width=160, height=120)

        meta = inspect_media_file(vid_p)
        opts = FrameProcessingOptions(target_fps=5.0, max_frames=5)

        frames = list(stream_video_frames(vid_p, meta, opts))
        assert len(frames) <= 5
        for idx, ts, f in frames:
            assert f.shape[:2] == (120, 160)
            assert ts >= 0.0


def test_render_detection_overlays_and_clipping():
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    dets = [
        {"x_min": 100.0, "y_min": 150.0, "x_max": 200.0, "y_max": 220.0, "confidence": 0.88, "track_id": 1},
        {"x_min": -50.0, "y_min": -20.0, "x_max": 900.0, "y_max": 700.0, "confidence": 0.95, "track_id": 2},  # Oversized boundary test
    ]

    annotated = render_detection_overlays(
        frame,
        dets,
        frame_index=1,
        timestamp_seconds=0.33,
        confidence_threshold=0.25,
        apply_privacy_mask=True,
    )

    assert annotated.shape == (480, 640, 3)
    # Original frame must be unmodified
    assert not np.array_equal(frame, annotated)


def test_extract_evidence_crop():
    frame = np.full((300, 400, 3), 120, dtype=np.uint8)
    bbox = (50.0, 50.0, 150.0, 120.0)

    crop_bytes = extract_evidence_crop(frame, bbox)
    assert len(crop_bytes) > 0
    assert crop_bytes.startswith(b"\xff\xd8")  # JPEG SOI marker
