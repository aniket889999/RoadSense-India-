import os
import cv2
import tempfile
import json
import zipfile
from io import BytesIO
from unittest.mock import patch
from src.contracts import ManualAnnotationRow, ManualIncident, ReportingSummary
from src.reporting import generate_report
from src.analysis_runner import run_analysis
from tests.test_video_io import create_synthetic_video

def test_generate_report_and_verify_mp4():
    video_bytes = create_synthetic_video(10)

    rows = [
        ManualAnnotationRow(incident_id="POT-001", frame_index=0, x_min=10, y_min=10, x_max=20, y_max=20, label="pothole")
    ]
    incidents = [
        ManualIncident(
            incident_id="POT-001", label="pothole", observation_count=1,
            first_seen_frame=0, first_seen_seconds=0.0, last_seen_frame=0, last_seen_seconds=0.0,
            representative_frame=0, representative_seconds=0.0, representative_bbox_area_px=100,
            evidence_file="POT-001_rep_frame.jpg"
        )
    ]
    summary = ReportingSummary(total_observations=1, total_incidents=1)

    report_zip = generate_report(rows, incidents, summary, video_bytes, "test.mp4", sampled_frames=[0, 5, 9])

    with zipfile.ZipFile(BytesIO(report_zip), 'r') as zf:
        namelist = zf.namelist()
        assert "evidence/POT-001_rep_frame.jpg" in namelist

        summary_data = json.loads(zf.read("summary.json"))
        assert summary_data["analysis_mode"] == "manual_annotation_baseline"
        assert summary_data["model_used"] is False

        fd, temp_mp4 = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        with open(temp_mp4, "wb") as f:
            f.write(zf.read("annotated_manual_samples.mp4"))

        cap = cv2.VideoCapture(temp_mp4)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        os.remove(temp_mp4)

        assert frame_count == 3

def test_videowriter_failure():
    video_bytes = create_synthetic_video(5)
    rows = []
    incidents = []
    summary = ReportingSummary(total_observations=0, total_incidents=0)

    with patch('cv2.VideoWriter.isOpened', return_value=False):
        try:
            generate_report(rows, incidents, summary, video_bytes, "test.mp4", sampled_frames=[0])
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "Failed to open VideoWriter" in str(e)

def test_end_to_end_analysis_runner():
    video_bytes = create_synthetic_video(10)
    csv_bytes = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\nPOT-001,0,10,10,20,20,pothole,\n"

    incidents, errors, report_zip, summary = run_analysis(
        csv_bytes, video_bytes, "test.mp4", 100, 100, 10.0, allowed_frames=[0, 5, 9]
    )

    assert not errors
    assert len(incidents) == 1
    assert incidents[0].incident_id == "POT-001"

    assert summary.total_observations == 1
    assert summary.total_incidents == 1

    with zipfile.ZipFile(BytesIO(report_zip), 'r') as zf:
        namelist = zf.namelist()
        assert "raw_manual_annotations.csv" in namelist
        assert "manual_incidents.csv" in namelist
        assert "summary.json" in namelist
        assert "annotated_manual_samples.mp4" in namelist
        assert "evidence/POT-001_rep_frame.jpg" in namelist

def test_end_to_end_malformed_csv():
    video_bytes = create_synthetic_video(5)
    csv_bytes = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\n POT-001 ,0,10,10,20,20,pothole,\n"
    incidents, errors, report_zip, summary = run_analysis(
        csv_bytes, video_bytes, "test.mp4", 100, 100, 10.0, allowed_frames=[0, 4]
    )

    assert errors
    assert len(incidents) == 0
    assert report_zip == b""
    assert summary is None

def test_run_analysis_header_plus_blank_line():
    video_bytes = create_synthetic_video(5)
    csv_bytes = b"incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\n\n"
    incidents, errors, report_zip, summary = run_analysis(
        csv_bytes, video_bytes, "test.mp4", 100, 100, 10.0, allowed_frames=[0, 4]
    )

    assert errors
    assert "no data rows" in errors[0]
    assert len(incidents) == 0
    assert report_zip == b""
    assert summary is None
