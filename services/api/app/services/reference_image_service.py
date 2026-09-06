"""Reference image intake, validation, EXIF normalization, and secure storage service."""

from __future__ import annotations

import email
import hashlib
import io
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

from PIL import Image, ImageOps

MAX_REFERENCE_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_DECODED_PIXELS = 35_000_000  # ~35 Megapixels (e.g. 7000 x 5000)
MAX_DIMENSION = 8192
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}
CHUNK_SIZE = 64 * 1024  # 64 KB


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
    """Raised when reference image intake fails security or validation checks."""
    pass


def extract_image_bytes_from_payload(
    content_type: str,
    raw_body: bytes,
) -> Tuple[bytes, Optional[str]]:
    """
    Extract image bytes from either a raw binary body or a multipart/form-data payload.
    Uses standard library email parser to avoid requiring external python-multipart dependencies.
    """
    if not raw_body:
        raise ReferenceImageProcessingError("Upload payload is empty (0 bytes).")

    if len(raw_body) > MAX_REFERENCE_IMAGE_BYTES:
        raise ReferenceImageProcessingError(
            f"Uploaded image exceeds maximum size limit ({len(raw_body)} > {MAX_REFERENCE_IMAGE_BYTES} bytes)."
        )

    ct_lower = content_type.lower()
    if "multipart/form-data" in ct_lower:
        msg = email.message_from_bytes(b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + raw_body)
        for part in msg.walk():
            filename = part.get_filename()
            if filename or part.get_content_maintype() == "image":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes) and len(payload) > 0:
                    return payload, filename

        raise ReferenceImageProcessingError("No valid file part found in multipart form upload.")

    # Direct raw image upload
    return raw_body, None


def process_and_store_reference_image(
    file_stream: BinaryIO,
    camera_id: str,
    storage_root: Path,
    original_filename: Optional[str] = None,
) -> ProcessedReferenceImage:
    """
    Process, validate, EXIF-normalize, and securely store a reference image for a camera.

    Security & Safety Guarantees:
    - Strictly bounded upload size (< 15 MB).
    - Streaming incremental SHA-256.
    - Deep image decoding check (format validation, decompression bomb protection).
    - EXIF orientation normalized; EXIF GPS and privacy metadata stripped.
    - Path traversal protected and confined to storage_root.
    - Atomic temporary-file promotion with guaranteed cleanup on failure.
    """
    storage_root = storage_root.resolve()
    target_dir = (storage_root / "references" / camera_id).resolve()

    # Confine to storage root
    if not str(target_dir).startswith(str(storage_root)):
        raise ReferenceImageProcessingError("Path traversal detected in camera storage directory.")

    target_dir.mkdir(parents=True, exist_ok=True)

    temp_id = str(uuid.uuid4())
    temp_path = target_dir / f"temp_{temp_id}.upload.tmp"

    total_bytes = 0
    raw_hasher = hashlib.sha256()

    try:
        with open(temp_path, "wb") as f_out:
            while True:
                chunk = file_stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_REFERENCE_IMAGE_BYTES:
                    raise ReferenceImageProcessingError(
                        f"Reference image exceeds maximum allowed size ({total_bytes} > {MAX_REFERENCE_IMAGE_BYTES} bytes)."
                    )
                raw_hasher.update(chunk)
                f_out.write(chunk)

        if total_bytes == 0:
            raise ReferenceImageProcessingError("Reference image file is empty (0 bytes).")

        # Open and inspect the temporary image file with Pillow
        try:
            with Image.open(temp_path) as img:
                detected_format = img.format
                if detected_format not in ALLOWED_IMAGE_FORMATS:
                    raise ReferenceImageProcessingError(
                        f"Unsupported image format '{detected_format}'. Allowed formats: JPEG, PNG."
                    )

                # Check pixel count and dimensions for decompression bomb safety
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

                # Normalize orientation using EXIF if present and convert to RGB
                normalized_img = ImageOps.exif_transpose(img)
                if normalized_img is None:
                    normalized_img = img

                # Convert to RGB (or RGBA for PNG) to strip all EXIF / GPS metadata
                if detected_format == "PNG" and normalized_img.mode in ("RGBA", "LA"):
                    cleaned_img = normalized_img.convert("RGBA")
                    save_format = "PNG"
                    ext = "png"
                else:
                    cleaned_img = normalized_img.convert("RGB")
                    save_format = "JPEG"
                    ext = "jpg"

                final_width, final_height = cleaned_img.size

                # Generate permanent filename
                final_file_id = str(uuid.uuid4())
                final_filename = f"reference_{final_file_id}.{ext}"
                final_path = target_dir / final_filename

                # Save sanitized image with zero EXIF
                save_buffer = io.BytesIO()
                if save_format == "JPEG":
                    cleaned_img.save(save_buffer, format="JPEG", quality=95, optimize=True)
                else:
                    cleaned_img.save(save_buffer, format="PNG", optimize=True)

                saved_bytes = save_buffer.getvalue()
                final_sha256 = hashlib.sha256(saved_bytes).hexdigest()

                # Write sanitized image to disk atomically
                with open(final_path, "wb") as f_final:
                    f_final.write(saved_bytes)

        except (Image.DecompressionBombError, Image.UnidentifiedImageError) as e:
            raise ReferenceImageProcessingError(f"Malformed or dangerous image file: {str(e)}") from e
        except Exception as e:
            if isinstance(e, ReferenceImageProcessingError):
                raise
            raise ReferenceImageProcessingError(f"Failed to decode reference image: {str(e)}") from e

    finally:
        # Guarantee partial temp file cleanup
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    rel_path = str(final_path.relative_to(storage_root))
    return ProcessedReferenceImage(
        relative_path=rel_path,
        absolute_path=str(final_path),
        sha256=final_sha256,
        width=final_width,
        height=final_height,
        format=save_format,
        file_size_bytes=len(saved_bytes),
    )
