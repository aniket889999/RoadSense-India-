"""Unit tests for FFprobe and OpenCV media inspection."""

import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.media.inspector import MediaInspectionError, inspect_media_file


def _create_synthetic_test_video(path: Path, num_frames: int = 15, fps: float = 10.0, width: int = 320, height: int = 240) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 15) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_inspect_valid_synthetic_video():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_synth.mp4"
        _create_synthetic_test_video(video_path, num_frames=20, fps=10.0, width=320, height=240)

        meta = inspect_media_file(video_path)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.duration_seconds > 0
        assert meta.file_size_bytes > 0
        assert len(meta.sha256) == 64
        assert meta.rotation_degrees in (0, 90, 180, 270)


def test_inspect_rejects_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_file = Path(tmpdir) / "empty.mp4"
        empty_file.write_bytes(b"")

        with pytest.raises(MediaInspectionError, match="empty"):
            inspect_media_file(empty_file)


def test_inspect_rejects_non_video_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        text_file = Path(tmpdir) / "notes.txt"
        text_file.write_text("Not a video file")

        with pytest.raises(MediaInspectionError):
            inspect_media_file(text_file)


def test_inspect_path_confinement_rejects_traversal():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "allowed"
        base.mkdir()
        outside = Path(tmpdir) / "outside.mp4"
        _create_synthetic_test_video(outside)

        with pytest.raises(MediaInspectionError, match="Path traversal detected"):
            inspect_media_file(outside, base_dir=base)
