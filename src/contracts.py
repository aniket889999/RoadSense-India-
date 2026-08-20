from pydantic import BaseModel
from typing import Optional

class ManualAnnotationRow(BaseModel):
    incident_id: str
    frame_index: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str
    note: Optional[str] = None

class ManualIncident(BaseModel):
    incident_id: str
    label: str
    observation_count: int
    first_seen_frame: int
    first_seen_seconds: float
    last_seen_frame: int
    last_seen_seconds: float
    representative_frame: int
    representative_seconds: float
    representative_bbox_area_px: float
    evidence_file: str
    review_status: str = "unreviewed"
    aggregation_method: str = "manual_incident_id"

class ReportingSummary(BaseModel):
    analysis_mode: str = "manual_annotation_baseline"
    model_used: bool = False
    total_observations: int
    total_incidents: int
