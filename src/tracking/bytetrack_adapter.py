"""ByteTrack adapter for RoadSense India temporal pothole tracking.

This module encapsulates ByteTrack multi-object tracking behind an injectable,
typesafe interface. It maps per-frame candidate bounding boxes into stable,
session-local tracks while strictly separating low-confidence association detections
from verified human incident candidates.
"""

from __future__ import annotations

import hashlib
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import yaml

from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


@dataclass(frozen=True)
class ByteTrackConfig:
    """Validated parameters for ByteTrack association."""
    config_path: Path
    config_sha256: str
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.30
    track_buffer: int = 30
    match_thresh: float = 0.80
    fps: int = 30
    min_hits: int = 2


@dataclass(frozen=True)
class TrackObservation:
    """A single tracking observation on one frame."""
    session_id: str
    track_id: int
    frame_number: int
    timestamp_seconds: float
    bbox: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    confidence: float
    class_id: int = 0
    model_sha256: str = ""
    processing_status: str = "active"


@dataclass
class TrackSummary:
    """Aggregated lifecycle of a session-local track."""
    track_id: int
    session_id: str
    first_seen_seconds: float
    last_seen_seconds: float
    first_frame: int
    last_frame: int
    observation_count: int
    max_confidence: float
    avg_confidence: float
    representative_frame: int
    representative_bbox: tuple[float, float, float, float]
    observations: List[TrackObservation] = field(default_factory=list)
    is_stable: bool = False
    termination_reason: str = "active"


def _create_args_namespace(cfg: ByteTrackConfig) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=cfg.track_high_thresh,
        track_low_thresh=cfg.track_low_thresh,
        new_track_thresh=cfg.new_track_thresh,
        track_buffer=cfg.track_buffer,
        match_thresh=cfg.match_thresh,
        fuse_score=True,
        gmc_method="none",
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,
    )


def load_bytetrack_config(config_path: Path | str) -> ByteTrackConfig:
    """Load and hash the pinned ByteTrack configuration file."""
    p = Path(config_path).resolve()
    if not p.is_file() or p.is_symlink():
        raise FileNotFoundError(f"ByteTrack config file missing or invalid: {p}")

    content = p.read_text(encoding="utf-8")
    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    raw = yaml.safe_load(content)

    return ByteTrackConfig(
        config_path=p,
        config_sha256=sha256,
        track_high_thresh=float(raw.get("track_high_thresh", 0.25)),
        track_low_thresh=float(raw.get("track_low_thresh", 0.10)),
        new_track_thresh=float(raw.get("new_track_thresh", 0.30)),
        track_buffer=int(raw.get("track_buffer", 30)),
        match_thresh=float(raw.get("match_thresh", 0.80)),
        fps=int(raw.get("fps", 30)),
        min_hits=int(raw.get("min_hits", 2)),
    )


class ByteTrackAdapter:
    """Stateful adapter wrapping Ultralytics BYTETracker for a single drive session."""

    def __init__(self, config: ByteTrackConfig):
        self.config = config
        self._args = _create_args_namespace(config)
        self._tracker = BYTETracker(self._args)
        self._tracks: Dict[int, TrackSummary] = {}
        self._last_timestamp: float = -1.0
        self._last_frame_number: int = -1

    def reset(self) -> None:
        """Reset internal tracker state when a new session starts or discontinuity occurs."""
        self._tracker = BYTETracker(self._args)
        self._tracks.clear()
        self._last_timestamp = -1.0
        self._last_frame_number = -1

    def update(
        self,
        *,
        frame_number: int,
        timestamp_seconds: float,
        detections: Sequence[Dict[str, Any]],
        session_id: str,
        model_sha256: str = "",
        frame_shape: tuple[int, int] = (480, 640),
    ) -> List[TrackObservation]:
        """Update tracker with raw detections for the current frame."""
        # Discontinuity or rewind detection -> reset tracking
        if self._last_timestamp >= 0 and (timestamp_seconds < self._last_timestamp or (timestamp_seconds - self._last_timestamp) > 3.0):
            self.reset()

        self._last_timestamp = timestamp_seconds
        self._last_frame_number = frame_number

        if not detections:
            empty_tensor = torch.empty((0, 6), dtype=torch.float32)
            boxes = Boxes(empty_tensor, orig_shape=frame_shape)
            try:
                self._tracker.update(boxes)
            except Exception:
                pass
            return []

        rows = []
        for d in detections:
            x1 = float(d.get("x_min", d.get("x1", 0.0)))
            y1 = float(d.get("y_min", d.get("y1", 0.0)))
            x2 = float(d.get("x_max", d.get("x2", 0.0)))
            y2 = float(d.get("y_max", d.get("y2", 0.0)))
            conf = float(d.get("confidence", d.get("score", 0.0)))
            cls_id = float(d.get("class_id", 0.0))
            if x2 > x1 and y2 > y1 and 0.0 <= conf <= 1.0:
                rows.append([x1, y1, x2, y2, conf, cls_id])

        if not rows:
            return []

        dets_tensor = torch.tensor(rows, dtype=torch.float32)
        boxes = Boxes(dets_tensor, orig_shape=frame_shape)

        try:
            online_targets = self._tracker.update(boxes)
        except Exception:
            return []

        current_observations: List[TrackObservation] = []

        for target in online_targets:
            if isinstance(target, (list, tuple, np.ndarray)) and len(target) >= 6:
                x1, y1, x2, y2 = float(target[0]), float(target[1]), float(target[2]), float(target[3])
                track_id = int(target[4])
                conf = float(target[5])
                cls_id = int(target[6]) if len(target) > 6 else 0
            else:
                continue

            obs = TrackObservation(
                session_id=session_id,
                track_id=track_id,
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_id=cls_id,
                model_sha256=model_sha256,
                processing_status="active",
            )
            current_observations.append(obs)

            if track_id not in self._tracks:
                self._tracks[track_id] = TrackSummary(
                    track_id=track_id,
                    session_id=session_id,
                    first_seen_seconds=timestamp_seconds,
                    last_seen_seconds=timestamp_seconds,
                    first_frame=frame_number,
                    last_frame=frame_number,
                    observation_count=1,
                    max_confidence=conf,
                    avg_confidence=conf,
                    representative_frame=frame_number,
                    representative_bbox=(x1, y1, x2, y2),
                    observations=[obs],
                    is_stable=1 >= self.config.min_hits,
                )
            else:
                summary = self._tracks[track_id]
                summary.last_seen_seconds = timestamp_seconds
                summary.last_frame = frame_number
                summary.observation_count += 1
                summary.observations.append(obs)
                summary.avg_confidence = sum(o.confidence for o in summary.observations) / summary.observation_count
                if conf > summary.max_confidence:
                    summary.max_confidence = conf
                    summary.representative_frame = frame_number
                    summary.representative_bbox = (x1, y1, x2, y2)
                if summary.observation_count >= self.config.min_hits:
                    summary.is_stable = True

        return current_observations

    def get_all_tracks(self) -> List[TrackSummary]:
        """Return all tracks recorded in this session."""
        return list(self._tracks.values())

    def get_stable_tracks(self) -> List[TrackSummary]:
        """Return only tracks with observation count >= min_hits."""
        return [t for t in self._tracks.values() if t.is_stable]
