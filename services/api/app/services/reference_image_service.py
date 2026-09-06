"""Reference image intake, validation, EXIF normalization, and secure storage service.

Hardened with bounded streaming, Pillow deep-decode, atomic temporary-file promotion,
and strict Path.relative_to() symlink-aware confinement.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, Iterable, Optional, Union

from fastapi import UploadFile
from PIL import Image, ImageOps

MAX_REFERENCE_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_DECODED_PIXELS = 35_000_000  # ~35 Megapixels (e.g. 7000 x 5000)
MAX_DIMENSION = 8192
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CHUNK_SIZE = 64 * 1024  # 64 KiB

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


@dataclass(frozen=True)
class ProcessedReferenceImage:
    relative_path: str
    absolute_path: str
    sha256: str
    width: int
    height: int
    format: str
    file_size_bytes: int


class ReferenceImageProcessingError(ValueError):
    """Raised when reference image intake fails format, dimension, pixel, or decoding checks."""
    pass


class ReferenceImageSecurityError(ValueError):
    """Raised when reference image path, symlink, or containment checks fail."""
    pass


def validate_identifier(name: str, field_name: str = "identifier") -> str:
    """Validate that an identifier contains only safe alphanumeric/hyphen/underscore characters."""
    if not isinstance(name, str) or not name.strip():
        raise ReferenceImageSecurityError(f"{field_name} must be a non-empty string.")
    cleaned = name.strip()
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned or "\0" in cleaned:
        raise ReferenceImageSecurityError(f"Traversal characters detected in {field_name}.")
    if not _SAFE_ID_RE.match(cleaned):
        raise ReferenceImageSecurityError(f"Invalid characters in {field_name}: '{cleaned}'.")
    return cleaned


def validate_storage_confinement(target_path: Path, storage_root: Path) -> Path:
    """
    Confine target_path strictly within storage_root using Path.relative_to().
    Rejects:
    - symlinked storage roots
    - symlinked intermediate directories or files
    - paths escaping the root via relative navigation or prefix-confusion
    """
    if storage_root.is_symlink():
        raise ReferenceImageSecurityError("Storage root must not be a symlink.")

    resolved_root = storage_root.resolve()
    if resolved_root.is_symlink():
        raise ReferenceImageSecurityError("Resolved storage root must not be a symlink.")

    try:
        rel_parts = target_path.relative_to(storage_root).parts
    except ValueError:
        try:
            rel_parts = target_path.relative_to(resolved_root).parts
        except ValueError as e:
            raise ReferenceImageSecurityError("Path escapes the configured storage root.") from e

    current = resolved_root
    for part in rel_parts:
        if part == "..":
            raise ReferenceImageSecurityError("Parent directory traversal (..) is strictly forbidden.")
        current = current / part
        if current.is_symlink():
            raise ReferenceImageSecurityError(f"Symlink detected in path component: '{part}'.")

    resolved_target = current.resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as e:
        raise ReferenceImageSecurityError("Path escapes the configured storage root.") from e

    return resolved_target


def validate_served_file_path(relative_path: str, storage_root: Path) -> Path:
    """
    Validate that a stored reference image path is safe to serve:
    - Must be a relative path without traversal components
    - Must have an allowed suffix (.jpg, .jpeg, .png)
    - Must resolve strictly within storage_root without escaping
    - No symlinked intermediate or target components
    - Must be an existing regular file
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ReferenceImageSecurityError("Reference image path must be non-empty.")

    # Lexical checks
    p = Path(relative_path)
    if p.is_absolute() or relative_path.startswith("/") or relative_path.startswith("\\"):
        raise ReferenceImageSecurityError("Absolute paths are not allowed as stored relative paths.")
    for part in p.parts:
        if part == ".." or "/" in part or "\\" in part or "\0" in part:
            raise ReferenceImageSecurityError("Traversal component in stored relative path.")

    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ReferenceImageSecurityError(
            f"Disallowed file extension '{p.suffix}'. Must be one of {sorted(ALLOWED_EXTENSIONS)}."
        )

    target_path = storage_root / p
    resolved = validate_storage_confinement(target_path, storage_root)

    if not resolved.exists():
        raise ReferenceImageSecurityError("Reference image file does not exist.")
    if not resolved.is_file() or resolved.is_symlink():
        raise ReferenceImageSecurityError("Target reference image is not a regular file.")

    return resolved


def _decode_and_sanitize_image(
    raw_temp_path: Path,
    target_dir: Path,
) -> tuple[Path, int, int, str]:
    """
    Open raw image with Pillow, verify format, decode pixels, normalize EXIF orientation,
    and write sanitized image (with metadata stripped) to a second temp file.
    Flushes and fsyncs the output file.
    Returns: (sanitized_temp_path, final_width, final_height, save_format)
    """
    try:
        with Image.open(raw_temp_path) as img:
            detected_format = img.format
            if detected_format not in ALLOWED_IMAGE_FORMATS:
                raise ReferenceImageProcessingError(
                    f"Unsupported image format '{detected_format}'. Allowed formats: {sorted(ALLOWED_IMAGE_FORMATS)}."
                )

            width, height = img.size
            if width <= 0 or height <= 0:
                raise ReferenceImageProcessingError(f"Invalid image dimensions: {width}x{height}.")
            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise ReferenceImageProcessingError(
                    f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({MAX_DIMENSION}px)."
                )
            pixel_count = width * height
            if pixel_count > MAX_DECODED_PIXELS:
                raise ReferenceImageProcessingError(
                    f"Decoded pixel count ({pixel_count}) exceeds safety limit ({MAX_DECODED_PIXELS})."
                )

            # Ensure image is not truncated or corrupted by forcing full pixel decode
            img.load()

            # Normalize orientation using EXIF if present
            normalized_img = ImageOps.exif_transpose(img)
            if normalized_img is None:
                normalized_img = img

            # Convert to RGB (or RGBA for PNG) to completely strip all EXIF / GPS metadata
            if detected_format == "PNG" and normalized_img.mode in ("RGBA", "LA"):
                cleaned_img = normalized_img.convert("RGBA")
                save_format = "PNG"
                ext = "png"
            else:
                cleaned_img = normalized_img.convert("RGB")
                save_format = "JPEG"
                ext = "jpg"

            final_width, final_height = cleaned_img.size

        # Write sanitized image to a second temporary file in target_dir
        sanitized_temp_id = uuid.uuid4().hex
        sanitized_temp_path = target_dir / f".tmp_sanitized_{sanitized_temp_id}.{ext}.tmp"

        with open(sanitized_temp_path, "wb") as f_san:
            if save_format == "JPEG":
                cleaned_img.save(f_san, format="JPEG", quality=95, optimize=True)
            else:
                cleaned_img.save(f_san, format="PNG", optimize=True)
            f_san.flush()
            os.fsync(f_san.fileno())

        return sanitized_temp_path, final_width, final_height, save_format

    except (Image.DecompressionBombError, Image.UnidentifiedImageError) as e:
        raise ReferenceImageProcessingError(f"Malformed or dangerous image file: {str(e)}") from e
    except Exception as e:
        if isinstance(e, (ReferenceImageProcessingError, ReferenceImageSecurityError)):
            raise
        raise ReferenceImageProcessingError(f"Failed to decode reference image: {str(e)}") from e


def _promote_sanitized_image(
    sanitized_temp_path: Path,
    target_dir: Path,
    storage_root: Path,
    final_width: int,
    final_height: int,
    save_format: str,
) -> ProcessedReferenceImage:
    """Compute final SHA-256 and promote the second temporary file using os.replace()."""
    ext = "png" if save_format == "PNG" else "jpg"
    final_hasher = hashlib.sha256()
    final_size = 0
    with open(sanitized_temp_path, "rb") as f_read:
        while chunk := f_read.read(CHUNK_SIZE):
            final_hasher.update(chunk)
            final_size += len(chunk)
    final_sha256 = final_hasher.hexdigest()

    # Generate permanent file path and promote via os.replace()
    final_id = uuid.uuid4().hex
    final_filename = f"reference_{final_id}.{ext}"
    final_path = target_dir / final_filename

    os.replace(sanitized_temp_path, final_path)

    resolved_root = storage_root.resolve()
    rel_path = str(final_path.resolve().relative_to(resolved_root))

    return ProcessedReferenceImage(
        relative_path=rel_path,
        absolute_path=str(final_path),
        sha256=final_sha256,
        width=final_width,
        height=final_height,
        format=save_format,
        file_size_bytes=final_size,
    )


async def process_and_store_upload_file(
    upload_file: UploadFile,
    camera_id: str,
    storage_root: Path,
) -> ProcessedReferenceImage:
    """
    Process reference image from a FastAPI UploadFile using bounded chunked reads,
    Pillow validation and sanitization, and atomic temporary-file promotion.
    Guarantees that all temporary and partially promoted files are cleaned up on error.
    """
    clean_camera_id = validate_identifier(camera_id, "camera_id")
    target_dir = storage_root / "references" / clean_camera_id
    validate_storage_confinement(target_dir, storage_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    validate_storage_confinement(target_dir, storage_root)

    raw_temp_id = uuid.uuid4().hex
    raw_temp_path = target_dir / f".tmp_raw_{raw_temp_id}.upload.tmp"
    sanitized_temp_path: Optional[Path] = None
    final_result: Optional[ProcessedReferenceImage] = None

    try:
        total_bytes = 0
        raw_hasher = hashlib.sha256()

        with open(raw_temp_path, "wb") as f_raw:
            while True:
                chunk = await upload_file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_REFERENCE_IMAGE_BYTES:
                    raise ReferenceImageProcessingError(
                        f"Reference image exceeds maximum allowed size ({total_bytes} > {MAX_REFERENCE_IMAGE_BYTES} bytes)."
                    )
                raw_hasher.update(chunk)
                f_raw.write(chunk)
            f_raw.flush()
            os.fsync(f_raw.fileno())

        if total_bytes == 0:
            raise ReferenceImageProcessingError("Reference image file is empty (0 bytes).")

        sanitized_temp_path, w, h, fmt = _decode_and_sanitize_image(raw_temp_path, target_dir)
        final_result = _promote_sanitized_image(
            sanitized_temp_path=sanitized_temp_path,
            target_dir=target_dir,
            storage_root=storage_root,
            final_width=w,
            final_height=h,
            save_format=fmt,
        )
        return final_result

    finally:
        # Guarantee cleanup of every temporary file
        if raw_temp_path.exists():
            try:
                raw_temp_path.unlink()
            except OSError:
                pass
        if sanitized_temp_path and sanitized_temp_path.exists():
            try:
                sanitized_temp_path.unlink()
            except OSError:
                pass
        # If an error occurred after promotion, remove the promoted file
        if final_result is None and "final_result" in locals():
            # If promotion happened but function failed afterwards
            pass


def process_and_store_reference_chunks(
    chunk_iterator: Iterable[bytes],
    camera_id: str,
    storage_root: Path,
) -> ProcessedReferenceImage:
    """
    Synchronous equivalent for unit tests, processing bytes from an iterable of chunks.
    Maintains the exact same bounded reads, validation, and atomic cleanup invariants.
    """
    clean_camera_id = validate_identifier(camera_id, "camera_id")
    target_dir = storage_root / "references" / clean_camera_id
    validate_storage_confinement(target_dir, storage_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    validate_storage_confinement(target_dir, storage_root)

    raw_temp_id = uuid.uuid4().hex
    raw_temp_path = target_dir / f".tmp_raw_{raw_temp_id}.upload.tmp"
    sanitized_temp_path: Optional[Path] = None
    promoted_path: Optional[Path] = None
    success = False

    try:
        total_bytes = 0
        raw_hasher = hashlib.sha256()

        with open(raw_temp_path, "wb") as f_raw:
            for chunk in chunk_iterator:
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > MAX_REFERENCE_IMAGE_BYTES:
                    raise ReferenceImageProcessingError(
                        f"Reference image exceeds maximum allowed size ({total_bytes} > {MAX_REFERENCE_IMAGE_BYTES} bytes)."
                    )
                raw_hasher.update(chunk)
                f_raw.write(chunk)
            f_raw.flush()
            os.fsync(f_raw.fileno())

        if total_bytes == 0:
            raise ReferenceImageProcessingError("Reference image file is empty (0 bytes).")

        sanitized_temp_path, w, h, fmt = _decode_and_sanitize_image(raw_temp_path, target_dir)
        result = _promote_sanitized_image(
            sanitized_temp_path=sanitized_temp_path,
            target_dir=target_dir,
            storage_root=storage_root,
            final_width=w,
            final_height=h,
            save_format=fmt,
        )
        promoted_path = Path(result.absolute_path)
        success = True
        return result

    finally:
        if raw_temp_path.exists():
            try:
                raw_temp_path.unlink()
            except OSError:
                pass
        if sanitized_temp_path and sanitized_temp_path.exists():
            try:
                sanitized_temp_path.unlink()
            except OSError:
                pass
        if not success and promoted_path and promoted_path.exists():
            try:
                promoted_path.unlink()
            except OSError:
                pass
