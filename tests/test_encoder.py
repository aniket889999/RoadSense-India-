"""Unit tests for FFmpeg / OpenCV video encoding."""

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.media.encoder import encode_frames_to_mp4
from src.media.inspector import inspect_media_file


def test_encode_frames_to_mp4():
    frames = [
        np.full((240, 320, 3), (i * 20) % 255, dtype=np.uint8)
        for i in range(15)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out_p = Path(tmpdir) / "output.mp4"
        res = encode_frames_to_mp4(frames, out_p, fps=10.0)

        assert res.output_path.is_file()
        assert res.file_size_bytes > 0
        assert len(res.sha256) == 64
        assert res.total_frames == 15
        assert res.width == 320
        assert res.height == 240

        # Validate with inspect_media_file
        meta = inspect_media_file(out_p)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.duration_seconds > 0
