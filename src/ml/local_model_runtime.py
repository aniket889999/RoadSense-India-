"""Local-only runtime adapter for the pinned experimental YOLO baseline.

This module intentionally has no top-level Torch or Ultralytics dependency.  The
manual annotation application can therefore continue to start in its lightweight
environment even when the optional model runtime is not installed.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Dict, List


class LocalModelRuntimeError(RuntimeError):
    """Raised when the optional local model runtime cannot run safely."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def validate_requested_device(device: str) -> str:
    """Validate an explicit device without silently falling back to another one."""

    if not isinstance(device, str) or not device.strip():
        raise LocalModelRuntimeError("Inference device must be a non-empty value such as 'mps' or 'cpu'.")

    normalized = device.strip().lower()
    if normalized not in {"mps", "cpu", "cuda"}:
        raise LocalModelRuntimeError(
            "Unsupported inference device. Choose an explicit supported device: mps, cpu, or cuda."
        )

    try:
        import torch
    except ImportError as exc:
        raise LocalModelRuntimeError(
            "Experimental inference requires the local training dependencies. "
            "Install requirements-training.txt in the environment running Streamlit."
        ) from exc

    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise LocalModelRuntimeError(
                "Apple Metal (MPS) is not available in this environment. "
                "Choose cpu explicitly in the frozen inference configuration if that is intended."
            )
    elif normalized == "cuda" and not torch.cuda.is_available():
        raise LocalModelRuntimeError(
            "CUDA is not available in this environment. Choose cpu explicitly if that is intended."
        )

    return normalized


def _open_regular_file_without_symlinks(path: Path) -> int:
    """Open an absolute file through descriptor-relative, no-follow traversal.

    This avoids a check-then-open race in a parent directory such as ``weights``:
    every component is opened from an already-held directory descriptor and is
    rejected if it is a symlink.  A platform without ``O_NOFOLLOW`` is refused
    rather than silently weakening the frozen-checkpoint boundary.
    """

    if not isinstance(path, Path) or not path.is_absolute():
        raise LocalModelRuntimeError("The verified checkpoint path must be an absolute local path.")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise LocalModelRuntimeError(
            "This platform cannot safely open a provenance-pinned checkpoint without following links."
        )

    try:
        current_fd = os.open(path.anchor, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow)
    except OSError as exc:
        raise LocalModelRuntimeError(f"Unable to open the local filesystem root safely: {exc}") from exc

    try:
        parts = path.relative_to(path.anchor).parts
        if not parts:
            raise LocalModelRuntimeError("The verified checkpoint path must name a regular file.")

        for index, part in enumerate(parts):
            is_final = index == len(parts) - 1
            flags = os.O_RDONLY | no_follow
            if not is_final:
                flags |= getattr(os, "O_DIRECTORY", 0)
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except OSError as exc:
                raise LocalModelRuntimeError(
                    f"The verified checkpoint is unavailable or contains an unsafe path component: {exc}"
                ) from exc
            os.close(current_fd)
            current_fd = next_fd

            info = os.fstat(current_fd)
            if not is_final and not stat.S_ISDIR(info.st_mode):
                raise LocalModelRuntimeError("The verified checkpoint path contains a non-directory component.")
            if is_final and not stat.S_ISREG(info.st_mode):
                raise LocalModelRuntimeError("The verified local checkpoint is not a regular file.")
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _write_all(file_fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(file_fd, view)
        if written <= 0:
            raise OSError("unable to write a complete verified checkpoint snapshot")
        view = view[written:]


def _snapshot_checkpoint_from_verified_fd(checkpoint_fd: int, expected_sha256: str) -> Path:
    """Hash and copy the exact opened file descriptor into a private snapshot."""

    digest = hashlib.sha256()
    snapshot_fd: int | None = None
    snapshot_path: str | None = None
    try:
        snapshot_fd, snapshot_path = tempfile.mkstemp(
            prefix="roadsense-verified-checkpoint-", suffix=".pt"
        )
        os.fchmod(snapshot_fd, 0o600)
        while True:
            chunk = os.read(checkpoint_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            _write_all(snapshot_fd, chunk)
        os.fsync(snapshot_fd)
    except OSError as exc:
        raise LocalModelRuntimeError(f"Unable to create a private verified checkpoint snapshot: {exc}") from exc
    finally:
        if snapshot_fd is not None:
            os.close(snapshot_fd)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        if snapshot_path is not None:
            try:
                os.unlink(snapshot_path)
            except OSError:
                pass
        raise LocalModelRuntimeError(
            "The checkpoint changed or does not match the pinned SHA-256; it was not loaded."
        )
    if snapshot_path is None:
        raise LocalModelRuntimeError("Unable to create a private verified checkpoint snapshot.")
    return Path(snapshot_path)


def load_verified_yolo_model(checkpoint_path: Path, expected_sha256: str) -> Any:
    """Load exactly a hash-pinned local checkpoint, without downloads.

    The mutable on-disk pathname is never passed directly to Ultralytics.  The
    loader opens it through a no-follow descriptor chain, hashes the exact bytes
    while copying them into a private 0600 snapshot, verifies the expected
    digest, and only then constructs ``YOLO`` from that snapshot.
    """

    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise LocalModelRuntimeError("The expected checkpoint SHA-256 must be a lowercase 64-character digest.")

    checkpoint_fd = _open_regular_file_without_symlinks(checkpoint_path)
    try:
        snapshot_path = _snapshot_checkpoint_from_verified_fd(checkpoint_fd, expected_sha256)
    finally:
        os.close(checkpoint_fd)

    try:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise LocalModelRuntimeError(
                "Experimental inference requires Ultralytics locally. "
                "Install requirements-training.txt in the environment running Streamlit."
            ) from exc

        try:
            return YOLO(str(snapshot_path))
        except Exception as exc:
            raise LocalModelRuntimeError(f"Unable to load the verified local checkpoint: {exc}") from exc
    finally:
        # YOLO's constructor reads the checkpoint synchronously.  Delete the
        # private snapshot only after construction completes or fails.
        try:
            os.unlink(snapshot_path)
        except OSError:
            pass


def _as_rows(value: Any) -> List[Any]:
    """Convert common Torch/NumPy result containers to a plain Python list."""

    current = value
    if hasattr(current, "detach"):
        current = current.detach()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "numpy"):
        current = current.numpy()
    if hasattr(current, "tolist"):
        current = current.tolist()
    if isinstance(current, (list, tuple)):
        return list(current)
    raise LocalModelRuntimeError("The local model returned an unsupported detection container.")


def predict_yolo_frame(
    model: Any,
    frame: Any,
    *,
    device: str,
    image_size: int,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections_per_frame: int,
) -> List[Dict[str, float | int]]:
    """Return raw target-class detections from one decoded video frame.

    This adapter only invokes ``predict``.  It never trains, validates, downloads
    weights, or accesses a dataset.
    """

    if not isinstance(image_size, int) or image_size <= 0:
        raise LocalModelRuntimeError("Inference image_size must be a positive integer.")
    if not isinstance(max_detections_per_frame, int) or max_detections_per_frame <= 0:
        raise LocalModelRuntimeError("max_detections_per_frame must be a positive integer.")
    for name, value in {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise LocalModelRuntimeError(f"{name} must be a finite numeric value.")
        if not 0.0 <= float(value) <= 1.0:
            raise LocalModelRuntimeError(f"{name} must be between 0 and 1.")

    try:
        results = model.predict(
            source=frame,
            device=device,
            imgsz=image_size,
            conf=float(confidence_threshold),
            iou=float(iou_threshold),
            classes=[0],
            max_det=max_detections_per_frame,
            verbose=False,
        )
    except Exception as exc:
        raise LocalModelRuntimeError(f"Local model prediction failed: {exc}") from exc

    if not results:
        return []

    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return []

    try:
        xyxy_rows = _as_rows(getattr(boxes, "xyxy"))
        confidence_rows = _as_rows(getattr(boxes, "conf"))
        class_rows = _as_rows(getattr(boxes, "cls"))
    except AttributeError as exc:
        raise LocalModelRuntimeError("The local model returned boxes without xyxy, confidence, or class values.") from exc

    if not (len(xyxy_rows) == len(confidence_rows) == len(class_rows)):
        raise LocalModelRuntimeError("The local model returned inconsistent detection array lengths.")

    detections: List[Dict[str, float | int]] = []
    for xyxy, confidence, class_id in zip(xyxy_rows, confidence_rows, class_rows):
        try:
            x_min, y_min, x_max, y_max = [float(value) for value in xyxy]
            score = float(confidence)
            raw_class_id = float(class_id)
        except (TypeError, ValueError):
            continue

        if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max, score, raw_class_id)):
            continue
        if raw_class_id != 0.0 or not 0.0 <= score <= 1.0:
            continue

        detections.append(
            {
                "class_id": 0,
                "confidence": score,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        )

    return detections
