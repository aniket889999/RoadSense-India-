from src.contracts import ManualAnnotationRow
from src.aggregation import aggregate_incidents

def test_aggregate_incidents():
    rows = [
        ManualAnnotationRow(incident_id="POT-001", frame_index=1, x_min=10, y_min=10, x_max=20, y_max=20, label="pothole"),
        ManualAnnotationRow(incident_id="POT-001", frame_index=2, x_min=5, y_min=5, x_max=50, y_max=50, label="pothole"),
        ManualAnnotationRow(incident_id="POT-002", frame_index=5, x_min=10, y_min=10, x_max=20, y_max=20, label="pothole"),
    ]

    incidents = aggregate_incidents(rows, fps=10.0)
    assert len(incidents) == 2

    pot1 = next(i for i in incidents if i.incident_id == "POT-001")
    assert pot1.observation_count == 2
    assert pot1.representative_frame == 2
