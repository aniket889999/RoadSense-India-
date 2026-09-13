"""Test double and synthetic testing support for parking validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional
import numpy as np

from src.parking.occupancy_contracts import VehicleDetection


def compute_detection_schedule_sha256(frame_detections: Dict[int, List[VehicleDetection]]) -> str:
    """Computes deterministic canonical SHA-256 over synthetic vehicle detection schedule."""
    serialized = []
    for frame_idx in sorted(frame_detections.keys()):
        dets_for_frame = []
        for d in frame_detections[frame_idx]:
            dets_for_frame.append({
                "class_id": int(d.class_id),
                "class_name": str(d.class_name),
                "confidence": round(float(d.confidence), 4),
                "bbox_xyxy": [round(float(c), 2) for c in d.bbox_xyxy],
            })
        serialized.append({"frame_idx": frame_idx, "detections": dets_for_frame})
    canonical_json = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class DeterministicVehicleDetectorDouble:
    """
    Test double vehicle detector for deterministic automated tests and validation harnesses.
    Replays pre-computed or synthetic frame detections without loading PyTorch or YOLO checkpoints.
    Never used in production runs.
    """

    is_synthetic_test_double: bool = True

    def __init__(
        self,
        frame_detections: Optional[Dict[int, List[VehicleDetection]]] = None,
        checkpoint_sha256: Optional[str] = None,
    ) -> None:
        self.frame_detections = frame_detections or {}
        if checkpoint_sha256 and checkpoint_sha256 != "0" * 64:
            self._checkpoint_sha256 = checkpoint_sha256
        else:
            self._checkpoint_sha256 = compute_detection_schedule_sha256(self.frame_detections)
        self._current_frame_idx = 0

    @property
    def checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    @property
    def config(self) -> Any:
        class _MockConfig:
            device = "cpu"
            model_path = "deterministic_synthetic_double"
        return _MockConfig()

    def set_frame_detections(self, frame_detections: Dict[int, List[VehicleDetection]]) -> None:
        """Set or update the deterministic frame-level detections map."""
        self.frame_detections = frame_detections
        self._checkpoint_sha256 = compute_detection_schedule_sha256(self.frame_detections)
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
