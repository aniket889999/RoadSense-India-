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


def _validate_float(val: Any, name: str, min_v: float = 0.0, max_v: float = 1.0) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"ByteTrack field '{name}' must be a float, got {type(val).__name__} ({val})")
    f_v = float(val)
    import math
    if not math.isfinite(f_v) or f_v < min_v or f_v > max_v:
        raise ValueError(f"ByteTrack field '{name}' ({f_v}) out of valid range [{min_v}, {max_v}]")
    return f_v


def _validate_int(val: Any, name: str, min_v: int = 1, max_v: int = 1000) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"ByteTrack field '{name}' must be an int, got {type(val).__name__} ({val})")
    if val < min_v or val > max_v:
        raise ValueError(f"ByteTrack field '{name}' ({val}) out of valid range [{min_v}, {max_v}]")
    return int(val)


class ParkingByteTracker:
    """
    Session-local ByteTrack wrapper for multi-vehicle tracking across parking video frames.
    - Strictly session-local: fresh instance created per job, tracks never leak across sessions.
    - Preserves raw vehicle detections while mapping stable integer track IDs.
    - Tolerates brief detector misses without dropping track identities.
    """

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.30,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
        fps: int = 30,
        min_hits: int = 2,
        root_dir: Optional[Path] = None,
    ) -> None:
        self.root_dir = (root_dir or Path(__file__).resolve().parent.parent.parent).resolve()
        tracking_dir = self.root_dir / "configs" / "tracking"
        self.config_sha256 = ""

        if config_path is not None:
            config_path_str = str(config_path).strip()
            p_cand = Path(config_path_str) if Path(config_path_str).is_absolute() else (self.root_dir / config_path_str)

            if ".." in p_cand.parts:
                raise ValueError(f"Path traversal ('..') prohibited in tracking config path: {config_path_str}")

            curr = p_cand
            while curr != self.root_dir and curr != curr.parent:
                if curr.is_symlink():
                    raise ValueError(f"Symlink found in tracking config path component: {curr}")
                curr = curr.parent

            if tracking_dir.is_symlink():
                raise ValueError(f"Tracking configs directory cannot be a symlink: {tracking_dir}")

            resolved_path = p_cand.resolve()
            resolved_tracking_dir = tracking_dir.resolve()

            try:
                resolved_path.relative_to(resolved_tracking_dir)
            except ValueError:
                raise ValueError(f"Tracking config path {resolved_path} escapes tracking configs directory {resolved_tracking_dir}")

            if not resolved_path.exists() or not resolved_path.is_file():
                raise FileNotFoundError(f"ByteTrack config file not found at: {resolved_path}")
            if resolved_path.is_symlink():
                raise ValueError(f"ByteTrack config file {resolved_path} cannot be a symlink.")

            content_bytes = resolved_path.read_bytes()
            self.config_sha256 = hashlib.sha256(content_bytes).hexdigest()
            raw = yaml.safe_load(content_bytes.decode("utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"ByteTrack YAML content must be a dictionary, got {type(raw).__name__}")

            ALLOWED_KEYS = {
                "schema_version", "tracker_type", "track_high_thresh",
                "track_low_thresh", "new_track_thresh", "track_buffer",
                "match_thresh", "fps", "min_hits"
            }
            for k in raw.keys():
                if k not in ALLOWED_KEYS:
                    raise ValueError(f"Unknown field in ByteTrack YAML config: '{k}'")

            track_high_thresh = _validate_float(raw.get("track_high_thresh", track_high_thresh), "track_high_thresh", 0.01, 1.0)
            track_low_thresh = _validate_float(raw.get("track_low_thresh", track_low_thresh), "track_low_thresh", 0.01, 1.0)
            new_track_thresh = _validate_float(raw.get("new_track_thresh", new_track_thresh), "new_track_thresh", 0.01, 1.0)
            track_buffer = _validate_int(raw.get("track_buffer", track_buffer), "track_buffer", 1, 1000)
            match_thresh = _validate_float(raw.get("match_thresh", match_thresh), "match_thresh", 0.01, 1.0)
            fps = _validate_int(raw.get("fps", fps), "fps", 1, 240)
            if "min_hits" in raw:
                min_hits = _validate_int(raw["min_hits"], "min_hits", 1, 100)

        self.min_hits = min_hits
        self._track_hit_counts: Dict[int, int] = {}
        import types
        self._args = types.SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fps=fps,
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
        self._track_hit_counts.clear()

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
                self._track_hit_counts[matched_tid] = self._track_hit_counts.get(matched_tid, 0) + 1
                assigned_tid = matched_tid if self._track_hit_counts[matched_tid] >= self.min_hits else None
                updated_det = VehicleDetection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=det.bbox_xyxy,
                    track_id=assigned_tid,
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
