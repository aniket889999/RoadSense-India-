from typing import List, Dict
from src.contracts import ManualAnnotationRow, ManualIncident

def aggregate_incidents(annotations: List[ManualAnnotationRow], fps: float) -> List[ManualIncident]:
    groups: Dict[str, List[ManualAnnotationRow]] = {}
    for ann in annotations:
        groups.setdefault(ann.incident_id, []).append(ann)

    incidents = []
    for incident_id, group in groups.items():
        group.sort(key=lambda x: x.frame_index)

        first_frame = group[0].frame_index
        last_frame = group[-1].frame_index

        rep_ann = group[0]
        max_area = 0
        for ann in group:
            area = (ann.x_max - ann.x_min) * (ann.y_max - ann.y_min)
            if area > max_area:
                max_area = area
                rep_ann = ann

        incidents.append(ManualIncident(
            incident_id=incident_id,
            label=group[0].label,
            observation_count=len(group),
            first_seen_frame=first_frame,
            first_seen_seconds=first_frame / fps if fps > 0 else 0,
            last_seen_frame=last_frame,
            last_seen_seconds=last_frame / fps if fps > 0 else 0,
            representative_frame=rep_ann.frame_index,
            representative_seconds=rep_ann.frame_index / fps if fps > 0 else 0,
            representative_bbox_area_px=max_area,
            evidence_file=f"{incident_id}_rep_frame.jpg",
            review_status="unreviewed",
            aggregation_method="manual_incident_id"
        ))

    return incidents
