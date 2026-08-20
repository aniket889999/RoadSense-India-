from typing import Tuple, List
from src.contracts import ManualAnnotationRow, ManualIncident, ReportingSummary
from src.manual_annotations import parse_manual_csv
from src.aggregation import aggregate_incidents
from src.reporting import generate_report

def run_analysis(csv_bytes: bytes, video_bytes: bytes, video_name: str, width: int, height: int, fps: float, allowed_frames: List[int]) -> Tuple[List[ManualIncident], List[str], bytes, ReportingSummary]:
    rows, errors = parse_manual_csv(csv_bytes, width, height, allowed_frames)
    if errors:
        return [], errors, b"", None

    incidents = aggregate_incidents(rows, fps)

    summary = ReportingSummary(
        total_observations=len(rows),
        total_incidents=len(incidents)
    )

    report_zip = generate_report(rows, incidents, summary, video_bytes, video_name, allowed_frames)

    return incidents, [], report_zip, summary
