"""Security helpers, path confinement, and input sanitation for RoadSense India."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from fastapi import HTTPException, status
from src.ml.model_provenance import REPO_ROOT

_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Common video magic signatures
_VIDEO_SIGNATURES = [
    b"\x00\x00\x00\x18ftyp", # MP4
    b"\x00\x00\x00\x1cftyp", # MP4
    b"\x00\x00\x00\x20ftyp", # MP4
    b"ftyp",                 # Generic ISO/MP4
    b"moov",                 # QuickTime / MP4
    b"RIFF",                 # AVI
    b"\x1a\x45\xdf\xa3",     # Matroska / MKV
]


def sanitize_filename(filename: str) -> str:
    """Sanitize and return only the base filename, rejecting path traversal."""
    if not isinstance(filename, str) or not filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must be a non-empty string."
        )

    base = os.path.basename(filename.strip())
    if not _SAFE_FILENAME_RE.match(base) or ".." in base:
        # Fallback sanitize to safe alphanumeric
        safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(base).stem)[:64]
        suffix = Path(base).suffix.lower()
        if suffix not in _ALLOWED_VIDEO_EXTENSIONS:
            suffix = ".mp4"
        base = f"{safe_stem}{suffix}"

    return base


def validate_file_signature(content: bytes) -> bool:
    """Validate that uploaded bytes match a supported video container signature."""
    if len(content) < 16:
        return False

    header_sample = content[:128]
    for sig in _VIDEO_SIGNATURES:
        if sig in header_sample:
            return True
    return False


def validate_safe_storage_path(path: Path, base_dir: Path) -> Path:
    """Ensure path is within base_dir and contains no symlink components."""
    try:
        resolved_base = base_dir.resolve()
        resolved_target = path.resolve()
        resolved_target.relative_to(resolved_base)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or unsafe file path."
        ) from exc

    # Ensure no symlinks in path
    curr = base_dir
    try:
        for part in path.relative_to(base_dir).parts:
            curr = curr / part
            if curr.is_symlink():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Symlinked path components are strictly forbidden."
                )
    except ValueError:
        pass

    return resolved_target
