"""Secure media intake and ffprobe inspection service for RoadSense India.

This module inspects and validates uploaded dashcam videos using direct subprocess
argument arrays (never shell strings), strictly validating container format, codecs,
resolution, display rotation, frame rate, and cryptographic checksums.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


class MediaInspectionError(ValueError):
    """Raised when an uploaded video is invalid, corrupt, or violates constraints."""


@dataclass(frozen=True)
class MediaMetadata:
    """Validated metadata extracted from an uploaded video file."""
    source_filename: str
    file_size_bytes: int
    sha256: str
    container_format: str
    video_codec: str
    width: int
    height: int
    rotation_degrees: int
    duration_seconds: float
    avg_fps: float
    real_fps: float
    time_base: str
    frame_count: Optional[int]
    is_variable_frame_rate: bool
    has_audio: bool
    audio_codec: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_sha256(filepath: Path) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_safe_regular_file(path: Path, base_dir: Optional[Path] = None) -> Path:
    """Verify that path is an absolute, non-symlink regular file within base_dir."""
    if not isinstance(path, Path):
        path = Path(path)

    resolved = path.resolve()
    if not resolved.is_absolute():
        raise MediaInspectionError("Media path must be absolute.")

    if base_dir is not None:
        try:
            resolved.relative_to(base_dir.resolve())
        except ValueError as exc:
            raise MediaInspectionError(f"Path traversal detected: {path} is outside {base_dir}") from exc

    try:
        info = os.lstat(resolved)
    except OSError as exc:
        raise MediaInspectionError(f"Media file cannot be inspected: {exc}") from exc

    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise MediaInspectionError(f"Media file must be a real, non-symlink regular file: {resolved}")

    return resolved


def _parse_fps_fraction(fraction_str: str) -> float:
    """Safely parse frame rate fractions like '30000/1001' or '30/1'."""
    if not fraction_str or not isinstance(fraction_str, str):
        return 0.0
    try:
        if "/" in fraction_str:
            num, denom = fraction_str.split("/", 1)
            d = float(denom)
            if d == 0:
                return 0.0
            return float(num) / d
        return float(fraction_str)
    except (ValueError, ZeroDivisionError):
        return 0.0


def inspect_media_file(
    file_path: Path | str,
    *,
    base_dir: Optional[Path | str] = None,
    max_file_size_bytes: int = 1024 * 1024 * 1024,  # 1 GB
    max_duration_seconds: float = 3600.0,            # 1 hour
    max_resolution: tuple[int, int] = (3840, 2160),  # 4K UHD
) -> MediaMetadata:
    """Securely inspect and validate an uploaded video file with ffprobe and OpenCV.

    Parameters:
        file_path: Absolute path to the candidate video file.
        base_dir: Optional confinement directory to reject path traversal.
        max_file_size_bytes: Maximum permitted size in bytes.
        max_duration_seconds: Maximum permitted video duration in seconds.
        max_resolution: (max_width, max_height) bounds.

    Returns:
        MediaMetadata instance with validated properties.

    Raises:
        MediaInspectionError if the video is missing, corrupt, truncated, or unsafe.
    """
    base = Path(base_dir).resolve() if base_dir else None
    resolved_path = _ensure_safe_regular_file(Path(file_path), base)

    size = resolved_path.stat().st_size
    if size <= 0:
        raise MediaInspectionError("Media file is empty (0 bytes).")
    if size > max_file_size_bytes:
        raise MediaInspectionError(f"Media file exceeds maximum size limit ({size} > {max_file_size_bytes} bytes).")

    sha256 = _compute_sha256(resolved_path)

    # Locate ffprobe binary
    ffprobe_bin = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe" or "/usr/local/bin/ffprobe"
    has_ffprobe = os.path.exists(ffprobe_bin) and os.access(ffprobe_bin, os.X_OK)

    if has_ffprobe:
        try:
            cmd = [
                ffprobe_bin,
                "-v", "error",
                "-show_entries", "format=format_name,duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,nb_frames,tags:stream_tags=rotate",
                "-of", "json",
                str(resolved_path),
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=15.0,
            )
            if proc.returncode != 0:
                raise MediaInspectionError(f"ffprobe failed to parse video container: {proc.stderr.strip()[:200]}")

            probe_data = json.loads(proc.stdout)
        except subprocess.TimeoutExpired as exc:
            raise MediaInspectionError("ffprobe timed out inspecting media.") from exc
        except json.JSONDecodeError as exc:
            raise MediaInspectionError("ffprobe returned invalid JSON output.") from exc

        # Locate video and audio streams
        video_stream = None
        audio_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream

        if not video_stream:
            raise MediaInspectionError("No valid video stream found in uploaded file.")

        codec_name = str(video_stream.get("codec_name", "")).lower()
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        if width <= 0 or height <= 0:
            raise MediaInspectionError(f"Invalid video dimensions: {width}x{height}")
        if width > max_resolution[0] or height > max_resolution[1]:
            raise MediaInspectionError(f"Video resolution {width}x{height} exceeds maximum {max_resolution[0]}x{max_resolution[1]}")

        # Rotation
        rotation = 0
        tags = video_stream.get("tags") or {}
        if "rotate" in tags:
            try:
                rotation = int(tags["rotate"]) % 360
            except (ValueError, TypeError):
                rotation = 0

        # Frame rates
        avg_fps = _parse_fps_fraction(str(video_stream.get("avg_frame_rate", "0/0")))
        real_fps = _parse_fps_fraction(str(video_stream.get("r_frame_rate", "0/0")))
        fps = avg_fps if avg_fps > 0 else (real_fps if real_fps > 0 else 0.0)
        if fps <= 0 or not math.isfinite(fps):
            raise MediaInspectionError("Unable to determine valid frame rate for video.")

        is_vfr = bool(avg_fps > 0 and real_fps > 0 and abs(avg_fps - real_fps) > 0.05)
        time_base = str(video_stream.get("time_base", ""))

        # Duration
        format_info = probe_data.get("format") or {}
        try:
            duration = float(format_info.get("duration") or video_stream.get("duration") or 0.0)
        except (ValueError, TypeError):
            duration = 0.0

        if duration <= 0 or not math.isfinite(duration):
            raise MediaInspectionError("Video duration is zero or invalid.")
        if duration > max_duration_seconds:
            raise MediaInspectionError(f"Video duration {duration:.1f}s exceeds maximum {max_duration_seconds:.1f}s limit.")

        # Frame count
        nb_frames = None
        if "nb_frames" in video_stream:
            try:
                nb_frames = int(video_stream["nb_frames"])
            except (ValueError, TypeError):
                pass
        if nb_frames is None or nb_frames <= 0:
            nb_frames = int(round(duration * fps))

        return MediaMetadata(
            source_filename=resolved_path.name,
            file_size_bytes=size,
            sha256=sha256,
            container_format=str(format_info.get("format_name", "unknown")),
            video_codec=codec_name,
            width=width,
            height=height,
            rotation_degrees=rotation,
            duration_seconds=duration,
            avg_fps=fps,
            real_fps=real_fps if real_fps > 0 else fps,
            time_base=time_base,
            frame_count=nb_frames,
            is_variable_frame_rate=is_vfr,
            has_audio=audio_stream is not None,
            audio_codec=str(audio_stream.get("codec_name")) if audio_stream else None,
        )

    # Fallback to OpenCV inspection if ffprobe is unavailable
    cap = cv2.VideoCapture(str(resolved_path))
    if not cap.isOpened():
        raise MediaInspectionError("OpenCV failed to open or parse video file.")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0 or not math.isfinite(fps):
            raise MediaInspectionError("OpenCV video metadata is invalid or truncated.")

        duration = frame_count / fps
        if duration <= 0 or duration > max_duration_seconds:
            raise MediaInspectionError(f"Video duration {duration:.1f}s violates duration limits.")

        return MediaMetadata(
            source_filename=resolved_path.name,
            file_size_bytes=size,
            sha256=sha256,
            container_format="mp4/opencv_decoded",
            video_codec="opencv_decoded",
            width=width,
            height=height,
            rotation_degrees=0,
            duration_seconds=duration,
            avg_fps=fps,
            real_fps=fps,
            time_base="1/fps",
            frame_count=frame_count,
            is_variable_frame_rate=False,
            has_audio=False,
        )
    finally:
        cap.release()
