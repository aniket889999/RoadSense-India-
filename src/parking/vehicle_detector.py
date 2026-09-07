"""Local vehicle detector adapter using verified YOLO COCO checkpoints."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from src.parking.occupancy_config import DetectorConfig
from src.parking.occupancy_contracts import VehicleDetection

logger = logging.getLogger(__name__)


class LocalVehicleDetector:
    """
    Fail-closed local vehicle detector wrapping verified YOLO weights.
    Never downloads weights at runtime; requires pre-verified local checkpoint file.
    """

    def __init__(self, config: DetectorConfig, root_dir: Optional[Path] = None) -> None:
        self.config = config
        self.root_dir = (root_dir or Path(__file__).resolve().parent.parent.parent).resolve()
        self._model = None
        self._checkpoint_sha256 = ""
        self.model_path = self._verify_and_load_checkpoint()

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    def _verify_and_load_checkpoint(self) -> Path:
        """
        Verify checkpoint is lexically valid, confined beneath <repo>/models,
        contains no symlink in any path component, and matches expected SHA-256 before importing ultralytics.
        """
        model_path_str = str(self.config.model_path).strip()
        if Path(model_path_str).is_absolute():
            raise ValueError(f"Absolute model paths are prohibited: {model_path_str}")

        if ".." in Path(model_path_str).parts:
            raise ValueError(f"Path traversal ('..') is prohibited in model path: {model_path_str}")

        raw_path = self.root_dir / model_path_str

        # Check each lexical path component for symlinks
        curr = raw_path
        while curr != self.root_dir and curr != curr.parent:
            if curr.is_symlink():
                raise ValueError(f"Symlink found in model path component: {curr}")
            curr = curr.parent

        models_dir = self.root_dir / "models"
        if models_dir.is_symlink():
            raise ValueError(f"models directory cannot be a symlink: {models_dir}")

        resolved_path = raw_path.resolve()
        resolved_models_dir = models_dir.resolve()

        try:
            resolved_path.relative_to(resolved_models_dir)
        except ValueError:
            raise ValueError(f"Model path {resolved_path} escapes models directory {resolved_models_dir}")

        if not resolved_path.exists() or not resolved_path.is_file():
            raise FileNotFoundError(
                f"Vehicle detector checkpoint not found at: {resolved_path}. "
                "Runtime checkpoint downloads are strictly prohibited."
            )

        if resolved_path.is_symlink():
            raise ValueError(f"Vehicle detector checkpoint at {resolved_path} is a symlink, which is prohibited.")

        # Hash checkpoint file
        h = hashlib.sha256()
        with open(resolved_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        self._checkpoint_sha256 = h.hexdigest()

        if self.config.expected_model_sha256 and self._checkpoint_sha256 != self.config.expected_model_sha256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {resolved_path}. "
                f"Expected: {self.config.expected_model_sha256}, Actual: {self._checkpoint_sha256}"
            )

        # Import ultralytics fail-closed only after all path and hash validations pass
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(resolved_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {resolved_path}: {e}")

        logger.info(f"Loaded local vehicle detector from {resolved_path} (SHA: {self._checkpoint_sha256[:12]})")
        return resolved_path

    def detect_vehicles(self, frame_bgr: np.ndarray) -> List[VehicleDetection]:
        """
        Run vehicle inference on a single BGR image.
        Filters strictly by allowed_classes (e.g. car, motorcycle, bus, truck) and confidence threshold.
        """
        if self._model is None:
            raise RuntimeError("Vehicle detector model is not loaded.")

        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]
        results = self._model.predict(
            source=frame_bgr,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.allowed_classes,
            device=self.config.device,
            verbose=False,
        )

        detections: List[VehicleDetection] = []
        if not results:
            return detections

        first_res = results[0]
        if first_res.boxes is None or len(first_res.boxes) == 0:
            return detections

        boxes = first_res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            cls_id = int(classes[i])
            if cls_id not in self.config.allowed_classes:
                continue

            conf = float(confs[i])
            if conf < self.config.confidence_threshold:
                continue

            box = (
                float(max(0.0, min(float(w), xyxy[i][0]))),
                float(max(0.0, min(float(h), xyxy[i][1]))),
                float(max(0.0, min(float(w), xyxy[i][2]))),
                float(max(0.0, min(float(h), xyxy[i][3]))),
            )
            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            class_name = self.config.class_names.get(cls_id, f"vehicle_{cls_id}")

            detections.append(
                VehicleDetection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=box,
                    track_id=None,
                    center_xy=center,
                )
            )

        return detections


class DeterministicVehicleDetectorDouble:
    """
    Test double vehicle detector for deterministic automated tests and validation harnesses.
    Replays pre-computed or synthetic frame detections without loading PyTorch or YOLO checkpoints.
    Never used in production runs.
    """

    def __init__(
        self,
        frame_detections: Optional[Dict[int, List[VehicleDetection]]] = None,
        checkpoint_sha256: str = "0" * 64,
    ) -> None:
        self.frame_detections = frame_detections or {}
        self._checkpoint_sha256 = checkpoint_sha256 or ("0" * 64)
        self._current_frame_idx = 0

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def config(self) -> Any:
        class _MockConfig:
            device = "cpu"
            model_path = "deterministic_double"
        return _MockConfig()

    def set_frame_detections(self, frame_detections: Dict[int, List[VehicleDetection]]) -> None:
        """Set or update the deterministic frame-level detections map."""
        self.frame_detections = frame_detections
        self._current_frame_idx = 0

    def detect_vehicles(self, frame_bgr: np.ndarray) -> List[VehicleDetection]:
        """Returns deterministic detections for the current frame index, incrementing internal counter."""
        idx = self._current_frame_idx
        self._current_frame_idx += 1
        dets = self.frame_detections.get(idx, [])
        # Return copies with unassigned track_id so ByteTrack performs tracking
        return [
            VehicleDetection(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox_xyxy=d.bbox_xyxy,
                track_id=None,
                center_xy=d.center_xy,
            )
            for d in dets
        ]
