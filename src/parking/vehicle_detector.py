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
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent.parent
        self.model_path = (self.root_dir / self.config.model_path).resolve()
        self._model = None
        self._checkpoint_sha256 = ""
        self._verify_and_load_checkpoint()

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    def _verify_and_load_checkpoint(self) -> None:
        """Verify checkpoint is a regular non-symlink file matching the expected SHA-256."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Vehicle detector checkpoint not found at: {self.model_path}. "
                "Runtime checkpoint downloads are strictly prohibited."
            )

        if self.model_path.is_symlink():
            raise ValueError(f"Vehicle detector checkpoint at {self.model_path} is a symlink, which is prohibited.")

        if not self.model_path.is_file():
            raise ValueError(f"Vehicle detector checkpoint at {self.model_path} is not a regular file.")

        # Hash checkpoint file
        h = hashlib.sha256()
        with open(self.model_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        self._checkpoint_sha256 = h.hexdigest()

        if self.config.expected_model_sha256 and self._checkpoint_sha256 != self.config.expected_model_sha256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {self.model_path}. "
                f"Expected: {self.config.expected_model_sha256}, Actual: {self._checkpoint_sha256}"
            )

        # Import ultralytics fail-closed
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {e}")

        logger.info(f"Loaded local vehicle detector from {self.model_path} (SHA: {self._checkpoint_sha256[:12]})")

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
