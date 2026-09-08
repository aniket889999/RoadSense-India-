"""Test double and synthetic testing support for parking validation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from src.parking.occupancy_contracts import VehicleDetection


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
