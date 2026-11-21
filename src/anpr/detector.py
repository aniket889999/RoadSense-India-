"""Dedicated license-plate detector adapter using verified local Ultralytics weights.

Importing this module does not import Torch/Ultralytics, load weights, or make
network requests. The checkpoint must be configured via
``ROADSENSE_ANPR_PLATE_WEIGHTS`` (repository-relative path under ``models/``)
and ``ROADSENSE_ANPR_PLATE_SHA256`` (64-character lowercase SHA-256 hex digest),
confined without symlinks, and expose exactly one class named ``license_plate``.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from src.anpr.contracts import PlateBox
from src.anpr.preprocessing import validate_bgr_image
from src.anpr.runtime_contracts import PlateDetection, RuntimeUnavailable

REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_PLATE_WEIGHTS = "ROADSENSE_ANPR_PLATE_WEIGHTS"
ENV_PLATE_SHA256 = "ROADSENSE_ANPR_PLATE_SHA256"
ENV_ANPR_DEVICE = "ROADSENSE_ANPR_DEVICE"

REQUIRED_CLASS_NAME = "license_plate"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_DEVICES = {"cpu", "mps", "cuda"}


def _normalize_repo_root(repo_root: Optional[Path] = None) -> Path:
    root = Path(os.path.abspath(os.fspath(repo_root or REPO_ROOT)))
    try:
        info = os.lstat(root)
    except OSError as exc:
        raise RuntimeUnavailable(f"Repository root is not accessible: {root} ({exc})") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeUnavailable(f"Repository root must be a real, non-symlink directory: {root}")
    return root


def verify_confinement_under_models(
    relative_path_str: str,
    *,
    repo_root: Optional[Path] = None,
    expect_directory: bool = False,
    setting_name: str = ENV_PLATE_WEIGHTS,
) -> Path:
    """Verify a repository-relative path stays under ``<repo>/models`` without symlinks."""
    if not isinstance(relative_path_str, str) or not relative_path_str.strip():
        raise RuntimeUnavailable(
            f"{setting_name} is not configured. Provide a repository-relative path under models/."
        )
    if relative_path_str != relative_path_str.strip():
        raise RuntimeUnavailable(f"{setting_name} must not contain leading or trailing whitespace.")

    root = _normalize_repo_root(repo_root)
    rel = Path(relative_path_str)
    if rel.is_absolute():
        raise RuntimeUnavailable(
            f"{setting_name} must be repository-relative, not an absolute path: {relative_path_str!r}"
        )
    if ".." in rel.parts:
        raise RuntimeUnavailable(
            f"{setting_name} must not contain parent traversal ('..'): {relative_path_str!r}"
        )

    models_dir = root / "models"
    try:
        models_info = os.lstat(models_dir)
    except OSError as exc:
        raise RuntimeUnavailable(f"models/ directory is unavailable: {models_dir} ({exc})") from exc
    if stat.S_ISLNK(models_info.st_mode) or not stat.S_ISDIR(models_info.st_mode):
        raise RuntimeUnavailable(f"models/ must be a real, non-symlink directory: {models_dir}")

    candidate = Path(os.path.abspath(os.fspath(root / rel)))
    try:
        rel_to_models = candidate.relative_to(models_dir)
    except ValueError as exc:
        raise RuntimeUnavailable(
            f"{setting_name} must resolve inside models/: {relative_path_str!r}"
        ) from exc
    if not rel_to_models.parts:
        raise RuntimeUnavailable(
            f"{setting_name} must name a target inside models/, not models/ itself."
        )

    current = root
    parts = candidate.relative_to(root).parts
    for idx, part in enumerate(parts):
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise RuntimeUnavailable(
                f"{setting_name} target does not exist or is unreadable: {current} ({exc})"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeUnavailable(
                f"{setting_name} contains a forbidden symlink component: {current}"
            )
        is_last = idx == len(parts) - 1
        if not is_last and not stat.S_ISDIR(info.st_mode):
            raise RuntimeUnavailable(
                f"{setting_name} contains a non-directory parent component: {current}"
            )
        if is_last:
            if expect_directory and not stat.S_ISDIR(info.st_mode):
                raise RuntimeUnavailable(f"{setting_name} must be a real directory: {current}")
            if not expect_directory and not stat.S_ISREG(info.st_mode):
                raise RuntimeUnavailable(f"{setting_name} must be a regular file: {current}")

    return candidate


def _compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(65536):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeUnavailable(f"Unable to read checkpoint for hashing: {path} ({exc})") from exc
    return digest.hexdigest()


def _validate_device(device: Optional[str]) -> str:
    raw = "cpu" if device is None else device
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeUnavailable(
            f"{ENV_ANPR_DEVICE} must be a non-empty device name ('cpu', 'mps', or 'cuda')."
        )
    normalized = raw.strip().lower()
    if normalized not in _SUPPORTED_DEVICES:
        raise RuntimeUnavailable(
            f"Unsupported {ENV_ANPR_DEVICE} {raw!r}. Supported values: cpu, mps, cuda."
        )
    if normalized in {"mps", "cuda"}:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeUnavailable(
                f"PyTorch is required to validate {ENV_ANPR_DEVICE}={normalized!r}, "
                "but torch is not installed."
            ) from exc
        if normalized == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                raise RuntimeUnavailable(
                    "Requested ROADSENSE_ANPR_DEVICE='mps', but Apple Metal (MPS) is unavailable."
                )
        elif normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeUnavailable(
                "Requested ROADSENSE_ANPR_DEVICE='cuda', but CUDA is unavailable."
            )
    return normalized


def _validate_single_plate_class(model: Any) -> None:
    """Require the loaded model to declare exactly one class named ``license_plate``."""
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)

    normalized_names: dict[int, str] = {}
    if isinstance(names, Mapping):
        for key, value in names.items():
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise RuntimeUnavailable(
                    f"Detector class mapping contains a non-integer key: {key!r}"
                )
            if isinstance(key, str) and key != "0":
                raise RuntimeUnavailable("Detector class mapping requires class ID 0.")
            idx = int(key)
            if idx in normalized_names:
                raise RuntimeUnavailable("Detector class mapping contains duplicate class IDs.")
            if not isinstance(value, str):
                raise RuntimeUnavailable("Detector class name must be a string.")
            normalized_names[idx] = value.strip()
    elif isinstance(names, (list, tuple)):
        for idx, value in enumerate(names):
            if not isinstance(value, str):
                raise RuntimeUnavailable("Detector class name must be a string.")
            normalized_names[idx] = value.strip()
    else:
        raise RuntimeUnavailable(
            "Loaded YOLO checkpoint does not expose a verifiable class names mapping."
        )

    if normalized_names != {0: REQUIRED_CLASS_NAME}:
        raise RuntimeUnavailable(
            "Detector checkpoint class mismatch: expected single-class "
            f"{{0: {REQUIRED_CLASS_NAME!r}}}, got {normalized_names!r}. "
            "Refusing to substitute pothole or general vehicle checkpoints."
        )


def _as_list(container: Any) -> List[Any]:
    current = container
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
    raise RuntimeUnavailable("Detector returned an unsupported prediction tensor container.")


class UltralyticsPlateDetector:
    """Dedicated license-plate detector backed by a verified local YOLO checkpoint."""

    def __init__(
        self,
        *,
        weights_path: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        device: Optional[str] = None,
        repo_root: Optional[Path] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        max_detections: int = 16,
    ) -> None:
        if (
            isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not math.isfinite(float(confidence_threshold))
            or not (0.0 <= float(confidence_threshold) <= 1.0)
        ):
            raise ValueError("confidence_threshold must be a finite number between 0 and 1.")
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, (int, float))
            or not math.isfinite(float(iou_threshold))
            or not (0.0 <= float(iou_threshold) <= 1.0)
        ):
            raise ValueError("iou_threshold must be a finite number between 0 and 1.")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if (
            isinstance(max_detections, bool)
            or not isinstance(max_detections, int)
            or max_detections <= 0
        ):
            raise ValueError("max_detections must be a positive integer.")

        raw_weights = (
            os.environ.get(ENV_PLATE_WEIGHTS, "") if weights_path is None else weights_path
        )
        raw_sha256 = (
            os.environ.get(ENV_PLATE_SHA256, "") if expected_sha256 is None else expected_sha256
        )
        raw_device = os.environ.get(ENV_ANPR_DEVICE, "cpu") if device is None else device

        if not isinstance(raw_sha256, str) or not raw_sha256.strip():
            raise RuntimeUnavailable(
                f"{ENV_PLATE_SHA256} is not configured. Provide a 64-character lowercase SHA-256 digest."
            )
        normalized_sha256 = raw_sha256.strip()
        if not _SHA256_RE.fullmatch(normalized_sha256):
            raise RuntimeUnavailable(
                f"{ENV_PLATE_SHA256} must be a 64-character lowercase hex SHA-256 digest."
            )

        verified_path = verify_confinement_under_models(
            raw_weights,
            repo_root=repo_root,
            expect_directory=False,
            setting_name=ENV_PLATE_WEIGHTS,
        )

        actual_sha256 = _compute_file_sha256(verified_path)
        if actual_sha256 != normalized_sha256:
            raise RuntimeUnavailable(
                f"Plate detector checkpoint SHA-256 mismatch for {verified_path}: "
                f"expected {normalized_sha256}, actual {actual_sha256}."
            )

        validated_device = _validate_device(raw_device)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeUnavailable(
                "Ultralytics is not installed in the local Python environment."
            ) from exc

        try:
            model = YOLO(str(verified_path))
        except Exception as exc:
            raise RuntimeUnavailable(
                f"Unable to load plate detector checkpoint {verified_path}: {exc}"
            ) from exc

        _validate_single_plate_class(model)

        self.checkpoint_path = verified_path
        self.checkpoint_sha256 = actual_sha256
        self.device = validated_device
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.image_size = image_size
        self.max_detections = max_detections
        self._model = model

    def detect(self, image_bgr: np.ndarray) -> Sequence[PlateDetection]:
        """Detect license plate bounding boxes in a BGR image."""
        validate_bgr_image(image_bgr, label="image_bgr")
        img_h, img_w = image_bgr.shape[:2]

        try:
            results = self._model.predict(
                source=image_bgr,
                device=self.device,
                imgsz=self.image_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[0],
                max_det=self.max_detections,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeUnavailable(f"Plate detector inference failed: {exc}") from exc

        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        xyxy_rows = _as_list(getattr(boxes, "xyxy", []))
        conf_rows = _as_list(getattr(boxes, "conf", []))
        cls_rows = _as_list(getattr(boxes, "cls", []))

        if not (len(xyxy_rows) == len(conf_rows) == len(cls_rows)):
            raise RuntimeUnavailable("Plate detector returned mismatched box/confidence arrays.")

        detections: List[PlateDetection] = []
        for xyxy, conf_val, cls_val in zip(xyxy_rows, conf_rows, cls_rows):
            try:
                x1, y1, x2, y2 = [float(coord) for coord in xyxy]
                score = float(conf_val)
                class_id = float(cls_val)
            except (TypeError, ValueError):
                continue

            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score, class_id)):
                continue
            if class_id != 0.0 or not (0.0 <= score <= 1.0):
                continue

            clamped_x1 = max(0.0, min(float(img_w), x1))
            clamped_y1 = max(0.0, min(float(img_h), y1))
            clamped_x2 = max(0.0, min(float(img_w), x2))
            clamped_y2 = max(0.0, min(float(img_h), y2))

            if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
                continue

            detections.append(
                PlateDetection(
                    bbox=PlateBox(
                        x1=clamped_x1,
                        y1=clamped_y1,
                        x2=clamped_x2,
                        y2=clamped_y2,
                    ),
                    confidence=score,
                )
            )

        # Deterministic ordering: top-to-bottom, left-to-right, then descending confidence
        detections.sort(key=lambda d: (d.bbox.y1, d.bbox.x1, -d.confidence))
        return detections
