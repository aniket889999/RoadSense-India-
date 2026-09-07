"""Session-local ByteTrack vehicle tracking adapter for RoadSense parking occupancy."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TrackerUpdateResult:
    """Result of vehicle tracking step on a single frame."""
    detections: List[VehicleDetection]
    is_healthy: bool = True
    failure_reason: Optional[str] = None


class ParkingByteTracker:
    """
    Session-local ByteTrack wrapper for multi-vehicle tracking across parking video frames.
    - Strictly session-local: fresh instance created per job, tracks never leak across sessions.
    - Preserves raw vehicle detections while mapping stable integer track IDs.
    - Uses runtime validated video FPS for BYTETracker dynamics.
    - Enforces consecutive min_hits tracking and bounded track memory.
    """

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        fps: int = 30,
        root_dir: Optional[Path] = None,
    ) -> None:
        self.root_dir = (root_dir or Path(__file__).resolve().parent.parent.parent).resolve()
        tracking_dir = self.root_dir / "configs" / "tracking"
        self.config_sha256 = ""
        self.fps = max(1, int(round(fps)))

        track_high_thresh = 0.25
        track_low_thresh = 0.10
        new_track_thresh = 0.30
        track_buffer = 30
        match_thresh = 0.80
        min_hits = 2

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

            REQUIRED_KEYS = {
                "schema_version", "tracker_type", "track_high_thresh",
                "track_low_thresh", "new_track_thresh", "track_buffer",
                "match_thresh", "min_hits"
            }
            if set(raw.keys()) != REQUIRED_KEYS:
                missing = REQUIRED_KEYS - set(raw.keys())
                extra = set(raw.keys()) - REQUIRED_KEYS
                if missing:
                    raise ValueError(f"ByteTrack YAML missing required fields: {missing}")
                if extra:
                    raise ValueError(f"Unknown field in ByteTrack YAML config: {extra}")

            # Validate schema_version
            s_ver = raw["schema_version"]
            if isinstance(s_ver, bool) or not isinstance(s_ver, int) or s_ver != 1:
                raise ValueError(f"Unsupported ByteTrack schema_version: {s_ver} (must be integer 1)")

            # Validate tracker_type
            t_type = raw["tracker_type"]
            if not isinstance(t_type, str) or t_type != "bytetrack":
                raise ValueError(f"Unsupported tracker_type: '{t_type}' (must be 'bytetrack')")

            track_high_thresh = _validate_float(raw["track_high_thresh"], "track_high_thresh", 0.01, 1.0)
            track_low_thresh = _validate_float(raw["track_low_thresh"], "track_low_thresh", 0.01, 1.0)
            new_track_thresh = _validate_float(raw["new_track_thresh"], "new_track_thresh", 0.01, 1.0)
            track_buffer = _validate_int(raw["track_buffer"], "track_buffer", 1, 1000)
            match_thresh = _validate_float(raw["match_thresh"], "match_thresh", 0.01, 1.0)
            min_hits = _validate_int(raw["min_hits"], "min_hits", 1, 100)

            # Threshold ordering validation
            if not (track_low_thresh < track_high_thresh):
                raise ValueError(
                    f"Incoherent threshold ordering: track_low_thresh ({track_low_thresh}) must be < track_high_thresh ({track_high_thresh})"
                )
            if not (track_low_thresh <= new_track_thresh <= 1.0):
                raise ValueError(
                    f"Incoherent threshold ordering: new_track_thresh ({new_track_thresh}) must be between track_low_thresh ({track_low_thresh}) and 1.0"
                )

        self.min_hits = min_hits
        self.track_buffer = track_buffer
        self._track_consecutive_hits: Dict[int, int] = {}
        self._track_last_seen_frame: Dict[int, int] = {}

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
            fps=self.fps,
            frame_rate=self.fps,
        )
        try:
            self._tracker = BYTETracker(self._args, frame_rate=self.fps)
        except TypeError:
            self._tracker = BYTETracker(self._args)
        self._last_frame_idx: int = -1

    def reset(self) -> None:
        """Reset tracker state completely between jobs."""
        try:
            self._tracker = BYTETracker(self._args, frame_rate=self.fps)
        except TypeError:
            self._tracker = BYTETracker(self._args)
        self._last_frame_idx = -1
        self._track_consecutive_hits.clear()
        self._track_last_seen_frame.clear()

    def update_tracks(
        self,
        detections: List[VehicleDetection],
        frame_idx: int,
        frame_shape: Tuple[int, int] = (1080, 1920),
    ) -> TrackerUpdateResult:
        """
        Update ByteTrack with detections from the current frame and assign stable track IDs.
        Returns a TrackerUpdateResult containing updated detections and tracker health.
        """
        self._last_frame_idx = frame_idx

        # Prune expired track IDs to bound memory
        expired = [tid for tid, last_f in self._track_last_seen_frame.items() if (frame_idx - last_f) > (self.track_buffer * 2)]
        for tid in expired:
            self._track_consecutive_hits.pop(tid, None)
            self._track_last_seen_frame.pop(tid, None)

        if not detections:
            empty_tensor = torch.empty((0, 6), dtype=torch.float32)
            boxes = Boxes(empty_tensor, orig_shape=frame_shape)
            try:
                self._tracker.update(boxes)
                return TrackerUpdateResult(detections=[], is_healthy=True)
            except Exception as e:
                logger.warning(f"Tracker empty update exception on frame {frame_idx}: {e}")
                return TrackerUpdateResult(detections=[], is_healthy=False, failure_reason=str(e))

        rows = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            if x2 > x1 and y2 > y1 and 0.0 <= det.confidence <= 1.0:
                rows.append([float(x1), float(y1), float(x2), float(y2), float(det.confidence), float(det.class_id)])

        if not rows:
            return TrackerUpdateResult(detections=list(detections), is_healthy=True)

        dets_tensor = torch.tensor(rows, dtype=torch.float32)
        boxes = Boxes(dets_tensor, orig_shape=frame_shape)

        try:
            online_targets = self._tracker.update(boxes)
        except Exception as e:
            logger.warning(f"ByteTrack update failed on frame {frame_idx}: {e}")
            return TrackerUpdateResult(
                detections=list(detections),
                is_healthy=False,
                failure_reason=f"ByteTrack update exception: {e}",
            )

        # Extract online targets (tlbr and track_id)
        target_tracks: List[Tuple[int, Tuple[float, float, float, float]]] = []
        for target in online_targets:
            try:
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

                # Check consecutive hits
                last_seen = self._track_last_seen_frame.get(matched_tid, -2)
                if last_seen == frame_idx - 1:
                    self._track_consecutive_hits[matched_tid] = self._track_consecutive_hits.get(matched_tid, 0) + 1
                else:
                    self._track_consecutive_hits[matched_tid] = 1

                self._track_last_seen_frame[matched_tid] = frame_idx
                consecutive_count = self._track_consecutive_hits[matched_tid]

                assigned_tid = matched_tid if consecutive_count >= self.min_hits else None
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

        return TrackerUpdateResult(detections=updated_detections, is_healthy=True)
