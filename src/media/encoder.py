"""FFmpeg H.264 browser-compatible video encoding engine.

Encodes processed BGR frames into web-optimized H.264 / yuv420p / +faststart MP4 files
using direct subprocess pipes, with optional audio stream remuxing and error containment.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple
import cv2
import numpy as np


class MediaEncodingError(RuntimeError):
    """Raised when FFmpeg or OpenCV fails to produce a valid browser-compatible video."""


@dataclass(frozen=True)
class EncodingResult:
    output_path: Path
    file_size_bytes: int
    sha256: str
    total_frames: int
    fps: float
    width: int
    height: int
    encoder_used: str  # "ffmpeg_libx264" | "opencv_mp4v"


def _compute_sha256(filepath: Path) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_frames_to_mp4(
    frames: Sequence[np.ndarray] | Generator[np.ndarray, None, None],
    output_path: Path | str,
    *,
    fps: float = 5.0,
    source_audio_path: Optional[Path | str] = None,
    crf: int = 23,
    preset: str = "fast",
) -> EncodingResult:
    """Encode a sequence or generator of OpenCV BGR frames into browser-ready MP4.

    Parameters:
        frames: Iterable or generator yielding np.ndarray BGR frames.
        output_path: Target .mp4 file destination.
        fps: Target playback framerate.
        source_audio_path: Optional original video path to remux audio from.
        crf: Constant Rate Factor (18-28, default 23).
        preset: x264 preset ('ultrafast', 'fast', 'medium').

    Returns:
        EncodingResult with output path, size, and cryptographic checksum.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg" or "/usr/local/bin/ffmpeg"
    has_ffmpeg = os.path.exists(ffmpeg_bin) and os.access(ffmpeg_bin, os.X_OK)

    frame_list: List[np.ndarray] = list(frames) if isinstance(frames, (list, tuple)) else []
    first_frame = frame_list[0] if frame_list else None

    if first_frame is None and not isinstance(frames, (list, tuple)):
        # Inspect first frame from generator
        try:
            first_frame = next(frames)
            frame_list.append(first_frame)
        except StopIteration:
            raise MediaEncodingError("No frames provided for video encoding.")

    if first_frame is None:
        raise MediaEncodingError("No frames provided for video encoding.")

    height, width = first_frame.shape[:2]
    total_frames = len(frame_list)

    if has_ffmpeg:
        # Build FFmpeg command array
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(float(fps)),
            "-i", "-",  # stdin pipe
        ]

        # Audio stream inclusion if present
        include_audio = False
        if source_audio_path and os.path.isfile(str(source_audio_path)):
            cmd.extend(["-i", str(source_audio_path), "-c:a", "aac", "-b:a", "128k", "-shortest"])
            include_audio = True

        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", preset,
            "-crf", str(crf),
            str(out_p),
        ])

        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Stream frames into FFmpeg stdin
            for f in frame_list:
                if f.shape[:2] != (height, width):
                    f = cv2.resize(f, (width, height))
                proc.stdin.write(f.tobytes())

            if not isinstance(frames, (list, tuple)):
                for f in frames:
                    total_frames += 1
                    if f.shape[:2] != (height, width):
                        f = cv2.resize(f, (width, height))
                    proc.stdin.write(f.tobytes())

            stdout_data, stderr_data = proc.communicate(timeout=60.0)
            if proc.returncode != 0:
                err_msg = stderr_data.decode("utf-8", errors="replace")[:300]
                raise MediaEncodingError(f"FFmpeg encoding failed: {err_msg}")

            if not out_p.is_file() or out_p.stat().st_size <= 0:
                raise MediaEncodingError("FFmpeg produced empty output file.")

            return EncodingResult(
                output_path=out_p,
                file_size_bytes=out_p.stat().st_size,
                sha256=_compute_sha256(out_p),
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height,
                encoder_used="ffmpeg_libx264",
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            if proc:
                proc.kill()
            raise MediaEncodingError(f"FFmpeg process error: {exc}") from exc

    # Fallback to OpenCV VideoWriter if FFmpeg is unavailable
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise MediaEncodingError("OpenCV failed to open VideoWriter for output MP4.")

    try:
        for f in frame_list:
            if f.shape[:2] != (height, width):
                f = cv2.resize(f, (width, height))
            writer.write(f)

        if not isinstance(frames, (list, tuple)):
            for f in frames:
                total_frames += 1
                if f.shape[:2] != (height, width):
                    f = cv2.resize(f, (width, height))
                writer.write(f)
    finally:
        writer.release()

    if not out_p.is_file() or out_p.stat().st_size <= 0:
        raise MediaEncodingError("OpenCV produced empty output file.")

    return EncodingResult(
        output_path=out_p,
        file_size_bytes=out_p.stat().st_size,
        sha256=_compute_sha256(out_p),
        total_frames=total_frames,
        fps=fps,
        width=width,
        height=height,
        encoder_used="opencv_mp4v",
    )
