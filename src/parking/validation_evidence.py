"""Machine-readable evidence reporting for Phase 2C stationary parking occupancy validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ValidationCheckItem:
    """Individual validation check item outcome."""
    check_name: str
    category: str
    expected: Any
    observed: Any
    passed: bool
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParkingValidationEvidenceReport:
    """
    Comprehensive machine-readable report documenting Phase 2C stable camera validation run.
    Contains cryptographic provenance, algorithm versions, timeline checks, color compliance, and pass/fail assertion.
    """
    run_id: str
    is_synthetic_fixture: bool
    disclaimer: str
    camera_id: str
    site_id: str
    job_id: str
    input_video_sha256: str
    reference_image_sha256: str
    layout_canonical_sha256: str
    stability_assessment_id: str
    stability_config_sha256: str
    occupancy_config_sha256: str
    detector_mode: str
    detector_checkpoint_sha256: str
    bytetrack_config_sha256: str
    bytetrack_frame_rate: float
    output_video_sha256: str
    timeline_sha256: str
    summary_sha256: str
    manifest_sha256: str
    operational_gate: str
    gate_reasons: List[str]
    total_frames: int
    processed_frames: int
    fps: float
    duration_seconds: float
    video_width: int
    video_height: int
    total_bays: int
    final_occupied_count: int
    final_vacant_count: int
    final_unknown_count: int
    final_occluded_count: int
    total_state_transitions: int
    software_versions: Dict[str, str]
    git_commit_sha: str
    created_at_utc: str
    checks: List[ValidationCheckItem] = field(default_factory=list)
    passed: bool = True
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "is_synthetic_fixture": self.is_synthetic_fixture,
            "disclaimer": self.disclaimer,
            "camera_id": self.camera_id,
            "site_id": self.site_id,
            "job_id": self.job_id,
            "input_video_sha256": self.input_video_sha256,
            "reference_image_sha256": self.reference_image_sha256,
            "layout_canonical_sha256": self.layout_canonical_sha256,
            "stability_assessment_id": self.stability_assessment_id,
            "stability_config_sha256": self.stability_config_sha256,
            "occupancy_config_sha256": self.occupancy_config_sha256,
            "detector_mode": self.detector_mode,
            "detector_checkpoint_sha256": self.detector_checkpoint_sha256,
            "bytetrack_config_sha256": self.bytetrack_config_sha256,
            "bytetrack_frame_rate": self.bytetrack_frame_rate,
            "output_video_sha256": self.output_video_sha256,
            "timeline_sha256": self.timeline_sha256,
            "summary_sha256": self.summary_sha256,
            "manifest_sha256": self.manifest_sha256,
            "operational_gate": self.operational_gate,
            "gate_reasons": self.gate_reasons,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "total_bays": self.total_bays,
            "final_occupied_count": self.final_occupied_count,
            "final_vacant_count": self.final_vacant_count,
            "final_unknown_count": self.final_unknown_count,
            "final_occluded_count": self.final_occluded_count,
            "total_state_transitions": self.total_state_transitions,
            "software_versions": self.software_versions,
            "git_commit_sha": self.git_commit_sha,
            "created_at_utc": self.created_at_utc,
            "checks": [c.to_dict() for c in self.checks],
            "passed": self.passed,
            "failure_reasons": self.failure_reasons,
        }

    def save_to_file(self, path: Path) -> Path:
        """Saves report to a formatted JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
