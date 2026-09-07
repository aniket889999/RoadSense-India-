"""Session-local ByteTrack vehicle tracking adapter for RoadSense parking occupancy."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import yaml

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from src.parking.occupancy_contracts import VehicleDetection

logger = logging.getLogger(__name__)


def _compute_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


class ParkingByteTracker:
    """
    Session-local ByteTrack wrapper for multi-vehicle tracking across parking video frames.
    - Strictly session-local: fresh instance created per job, tracks never leak across sessions.
    - Preserves raw vehicle detections while mapping stable integer track IDs.
    - Tolerates brief detector misses without dropping track identities.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.30,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
        fps: int = 30,
        min_hits: int = 2,
    ) -> None:
        self.config_sha256 = ""
        if config_path and Path(config_path).is_file():
            p = Path(config_path).resolve()
            content = p.read_text(encoding="utf-8")
            self.config_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
            raw = yaml.safe_load(content) or {}
            track_high_thresh = float(raw.get("track_high_thresh", track_high_thresh))
            track_low_thresh = float(raw.get("track_low_thresh", track_low_thresh))
            new_track_thresh = float(raw.get("new_track_thresh", new_track_thresh))
            track_buffer = int(raw.get("track_buffer", track_buffer))
            match_thresh = float(raw.get("match_thresh", match_thresh))
            fps = int(raw.get("fps", fps))
            min_hits = int(raw.get("min_hits", min_hits))

        import types
        self._args = types.SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
            gmc_method="none",
            proximity_thresh=0.5,
            appearance_thresh=0.25,
            with_reid=False,
        )
        self._tracker = BYTETracker(self._args)
        self._last_frame_idx: int = -1

    def reset(self) -> None:
        """Reset tracker state completely between jobs."""
        self._tracker = BYTETracker(self._args)
        self._last_frame_idx = -1

    def update_tracks(
        self,
        detections: List[VehicleDetection],
        frame_idx: int,
        frame_shape: Tuple[int, int] = (1080, 1920),
    ) -> List[VehicleDetection]:
        """
        Update ByteTrack with detections from the current frame and assign stable track IDs.
        Returns the detections updated with track_id fields.
        """
        self._last_frame_idx = frame_idx

        if not detections:
            empty_tensor = torch.empty((0, 6), dtype=torch.float32)
            boxes = Boxes(empty_tensor, orig_shape=frame_shape)
            try:
                self._tracker.update(boxes)
            except Exception as e:
                logger.debug(f"Tracker empty update exception: {e}")
            return []

        rows = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            if x2 > x1 and y2 > y1 and 0.0 <= det.confidence <= 1.0:
                rows.append([float(x1), float(y1), float(x2), float(y2), float(det.confidence), float(det.class_id)])

        if not rows:
            return list(detections)

        dets_tensor = torch.tensor(rows, dtype=torch.float32)
        boxes = Boxes(dets_tensor, orig_shape=frame_shape)

        try:
            online_targets = self._tracker.update(boxes)
        except Exception as e:
            logger.warning(f"ByteTrack update failed on frame {frame_idx}: {e}")
            return list(detections)

        # Extract online targets (tlbr and track_id)
        target_tracks: List[Tuple[int, Tuple[float, float, float, float]]] = []
        for target in online_targets:
            try:
                # Target can be STrack object with .tlbr and .track_id or ndarray
                if hasattr(target, "tlbr") and hasattr(target, "track_id"):
                    tlbr = target.tlbr
                    tid = int(target.track_id)
                    box = (float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3]))
                    target_tracks.append((tid, box))
                elif isinstance(target, (list, tuple, np.ndarray)) and len(target) >= 5:
                    box = (float(target[0]), float(target[1]), float(target[2]), float(target[3]))
                    tid = int(target[4])
                    target_tracks.append((tid, box))
            except Exception:
                continue

        # Match detections to target tracks by highest IoU >= 0.3
        updated_detections: List[VehicleDetection] = []
        matched_track_ids = set()

        for det in detections:
            best_iou = 0.3  # minimum matching IoU
            matched_tid: Optional[int] = None

            for tid, tbox in target_tracks:
                if tid in matched_track_ids:
                    continue
                iou = _compute_iou(det.bbox_xyxy, tbox)
                if iou > best_iou:
                    best_iou = iou
                    matched_tid = tid

            if matched_tid is not None:
                matched_track_ids.add(matched_tid)
                updated_det = VehicleDetection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=det.bbox_xyxy,
                    track_id=matched_tid,
                    center_xy=det.center_xy,
                )
            else:
                updated_det = VehicleDetection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=det.bbox_xyxy,
                    track_id=None,
                    center_xy=det.center_xy,
                )
            updated_detections.append(updated_det)

        return updated_detections
