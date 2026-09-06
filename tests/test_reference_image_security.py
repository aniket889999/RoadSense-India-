"""Unit tests for bounded, atomic reference image intake and symlink/path confinement."""

import io
import os
import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from services.api.app.services.reference_image_service import (
    MAX_DECODED_PIXELS,
    MAX_DIMENSION,
    MAX_REFERENCE_IMAGE_BYTES,
    ReferenceImageProcessingError,
    ReferenceImageSecurityError,
    process_and_store_reference_chunks,
    validate_identifier,
    validate_served_file_path,
    validate_storage_confinement,
)


@pytest.fixture
def temp_storage():
    """Create a temporary real directory to serve as storage root."""
    tmp = tempfile.mkdtemp(prefix="roadsense_test_storage_")
    storage_root = Path(tmp).resolve()
    yield storage_root
    shutil.rmtree(tmp, ignore_errors=True)


def _generate_valid_jpeg_bytes(width: int = 200, height: int = 150) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _generate_valid_png_bytes(width: int = 200, height: int = 150) -> bytes:
    img = Image.new("RGBA", (width, height), color=(100, 150, 200, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================================
# STEP 1: Bounded and Atomic Intake Tests
# ============================================================================

def test_valid_jpeg_upload_and_atomic_promotion(temp_storage):
    jpeg_bytes = _generate_valid_jpeg_bytes(320, 240)
    result = process_and_store_reference_chunks(
        chunk_iterator=[jpeg_bytes],
        camera_id="cam_valid_01",
        storage_root=temp_storage,
    )

    assert result.width == 320
    assert result.height == 240
    assert result.format == "JPEG"
    assert len(result.sha256) == 64
    assert Path(result.absolute_path).is_file()
    assert not result.relative_path.startswith("/")

    # Verify no leftover temporary files in camera dir
    cam_dir = temp_storage / "references" / "cam_valid_01"
    files = list(cam_dir.iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("reference_")
    assert not any(f.name.startswith(".tmp_") for f in files)


def test_valid_png_upload(temp_storage):
    png_bytes = _generate_valid_png_bytes(100, 100)
    result = process_and_store_reference_chunks(
        chunk_iterator=[png_bytes],
        camera_id="cam_png_01",
        storage_root=temp_storage,
    )
    assert result.format == "PNG"
    assert result.width == 100
    assert result.height == 100
    assert Path(result.absolute_path).is_file()


def test_empty_image_upload_fails_without_leftovers(temp_storage):
    cam_id = "cam_empty"
    with pytest.raises(ReferenceImageProcessingError, match="empty"):
        process_and_store_reference_chunks(
            chunk_iterator=[],
            camera_id=cam_id,
            storage_root=temp_storage,
        )

    cam_dir = temp_storage / "references" / cam_id
    if cam_dir.exists():
        assert list(cam_dir.iterdir()) == []


def test_oversized_stream_fails_incrementally_without_leftovers(temp_storage):
    cam_id = "cam_oversized"
    chunk_64k = b"X" * (64 * 1024)

    def chunk_generator():
        # Yield more than 15 MB
        for _ in range((MAX_REFERENCE_IMAGE_BYTES // len(chunk_64k)) + 2):
            yield chunk_64k

    with pytest.raises(ReferenceImageProcessingError, match="exceeds maximum allowed size"):
        process_and_store_reference_chunks(
            chunk_iterator=chunk_generator(),
            camera_id=cam_id,
            storage_root=temp_storage,
        )

    # Verify all partial files are cleaned up
    cam_dir = temp_storage / "references" / cam_id
    if cam_dir.exists():
        assert list(cam_dir.iterdir()) == []


def test_malformed_bytes_fail_without_leftovers(temp_storage):
    cam_id = "cam_malformed"
    garbage_bytes = b"NOT_A_REAL_IMAGE_DATA_CORRUPT" * 100

    with pytest.raises(ReferenceImageProcessingError, match="Malformed or dangerous|Failed to decode"):
        process_and_store_reference_chunks(
            chunk_iterator=[garbage_bytes],
            camera_id=cam_id,
            storage_root=temp_storage,
        )

    cam_dir = temp_storage / "references" / cam_id
    if cam_dir.exists():
        assert list(cam_dir.iterdir()) == []


def test_unsupported_format_fails_without_leftovers(temp_storage):
    cam_id = "cam_gif"
    # Create a GIF image
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    gif_bytes = buf.getvalue()

    with pytest.raises(ReferenceImageProcessingError, match="Unsupported image format"):
        process_and_store_reference_chunks(
            chunk_iterator=[gif_bytes],
            camera_id=cam_id,
            storage_root=temp_storage,
        )

    cam_dir = temp_storage / "references" / cam_id
    if cam_dir.exists():
        assert list(cam_dir.iterdir()) == []


def test_oversized_dimensions_fail_without_leftovers(temp_storage):
    cam_id = "cam_too_wide"
    # Create image exceeding MAX_DIMENSION (8192)
    # Note: Pillow will raise or we reject dimensions
    img = Image.new("RGB", (MAX_DIMENSION + 1, 10), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    too_wide_bytes = buf.getvalue()

    with pytest.raises(ReferenceImageProcessingError, match="exceed maximum allowed dimension"):
        process_and_store_reference_chunks(
            chunk_iterator=[too_wide_bytes],
            camera_id=cam_id,
            storage_root=temp_storage,
        )

    cam_dir = temp_storage / "references" / cam_id
    if cam_dir.exists():
        assert list(cam_dir.iterdir()) == []


# ============================================================================
# STEP 2: Path, Traversal, and Symlink Confinement Tests
# ============================================================================

def test_identifier_validation():
    assert validate_identifier("cam_123-abc") == "cam_123-abc"
    assert validate_identifier("pole-camera-04") == "pole-camera-04"

    with pytest.raises(ReferenceImageSecurityError):
        validate_identifier("../evil")
    with pytest.raises(ReferenceImageSecurityError):
        validate_identifier("cam/sub")
    with pytest.raises(ReferenceImageSecurityError):
        validate_identifier("cam\\sub")
    with pytest.raises(ReferenceImageSecurityError):
        validate_identifier("cam\0null")
    with pytest.raises(ReferenceImageSecurityError):
        validate_identifier("")
    with pytest.raises(ReferenceImageSecurityError):
        validate_identifier("   ")


def test_prefix_confusion_rejection(temp_storage):
    # Simulate prefix confusion:
    # If storage_root is /tmp/roadsense_test_storage_123,
    # an attacker target might be /tmp/roadsense_test_storage_123_evil
    evil_sibling = temp_storage.parent / f"{temp_storage.name}_evil"
    evil_sibling.mkdir(exist_ok=True)
    try:
        evil_file = evil_sibling / "test.jpg"
        evil_file.write_bytes(b"data")

        # Must raise because evil_sibling is not relative to temp_storage
        with pytest.raises(ReferenceImageSecurityError, match="escapes the configured storage root"):
            validate_storage_confinement(evil_file, temp_storage)
    finally:
        shutil.rmtree(evil_sibling, ignore_errors=True)


def test_parent_traversal_rejection(temp_storage):
    traversal_path = temp_storage / "references" / ".." / "outside.jpg"
    with pytest.raises(ReferenceImageSecurityError, match="traversal"):
        validate_storage_confinement(traversal_path, temp_storage)


def test_symlinked_root_rejection(temp_storage):
    symlink_root = temp_storage.parent / f"symlink_root_{temp_storage.name}"
    try:
        os.symlink(temp_storage, symlink_root)
        target = symlink_root / "references" / "cam1"
        with pytest.raises(ReferenceImageSecurityError, match="must not be a symlink"):
            validate_storage_confinement(target, symlink_root)
    finally:
        if symlink_root.is_symlink():
            symlink_root.unlink()


def test_symlinked_camera_directory_rejection(temp_storage):
    outside_dir = temp_storage.parent / f"outside_cam_dir_{temp_storage.name}"
    outside_dir.mkdir(exist_ok=True)
    symlink_dir = temp_storage / "references" / "cam_symlink"
    (temp_storage / "references").mkdir(parents=True, exist_ok=True)

    try:
        os.symlink(outside_dir, symlink_dir)
        target_file = symlink_dir / "image.jpg"
        with pytest.raises(ReferenceImageSecurityError, match="Symlink detected"):
            validate_storage_confinement(target_file, temp_storage)
    finally:
        if symlink_dir.is_symlink():
            symlink_dir.unlink()
        shutil.rmtree(outside_dir, ignore_errors=True)


def test_symlinked_file_serving_rejection(temp_storage):
    # Create real image in camera dir
    real_img = _generate_valid_jpeg_bytes(100, 100)
    ref = process_and_store_reference_chunks(
        chunk_iterator=[real_img],
        camera_id="cam_legit",
        storage_root=temp_storage,
    )

    # Validate served file works on real file
    served = validate_served_file_path(ref.relative_path, temp_storage)
    assert served.is_file()

    # Now create a symlink pointing to /etc/passwd or a file outside
    fake_target = temp_storage / "references" / "cam_legit" / "symlink_attack.jpg"
    secret_file = temp_storage.parent / "secret.txt"
    secret_file.write_text("SUPER_SECRET")
    try:
        os.symlink(secret_file, fake_target)
        rel_attack_path = str(fake_target.relative_to(temp_storage))

        with pytest.raises(ReferenceImageSecurityError, match="Symlink detected|not a regular file"):
            validate_served_file_path(rel_attack_path, temp_storage)
    finally:
        if fake_target.is_symlink():
            fake_target.unlink()
        if secret_file.exists():
            secret_file.unlink()


def test_disallowed_extension_serving_rejection(temp_storage):
    with pytest.raises(ReferenceImageSecurityError, match="Disallowed file extension"):
        validate_served_file_path("references/cam1/script.py", temp_storage)

    with pytest.raises(ReferenceImageSecurityError, match="Disallowed file extension"):
        validate_served_file_path("references/cam1/malicious.exe", temp_storage)


def test_absolute_path_serving_rejection(temp_storage):
    with pytest.raises(ReferenceImageSecurityError, match="Absolute paths"):
        validate_served_file_path("/etc/passwd.jpg", temp_storage)


def test_missing_file_serving_rejection(temp_storage):
    with pytest.raises(ReferenceImageSecurityError, match="does not exist"):
        validate_served_file_path("references/cam1/non_existent.jpg", temp_storage)
