import os
import cv2
import tempfile
import zipfile
import csv
from io import BytesIO, StringIO
from typing import List
from src.contracts import ManualAnnotationRow, ManualIncident, ReportingSummary

def generate_report(annotations: List[ManualAnnotationRow], incidents: List[ManualIncident], summary: ReportingSummary, video_bytes: bytes, file_name: str, sampled_frames: List[int]) -> bytes:
    ext = os.path.splitext(file_name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tfile:
        tfile.write(video_bytes)
        temp_path = tfile.name

    zip_buffer = BytesIO()

    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Unreadable video file for reporting.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("summary.json", summary.model_dump_json(indent=2))

            csv_buf = StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=list(ManualAnnotationRow.model_fields.keys()))
            writer.writeheader()
            for ann in annotations:
                writer.writerow(ann.model_dump())
            zf.writestr("raw_manual_annotations.csv", csv_buf.getvalue())

            csv_buf = StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=list(ManualIncident.model_fields.keys()))
            writer.writeheader()
            for inc in incidents:
                writer.writerow(inc.model_dump())
            zf.writestr("manual_incidents.csv", csv_buf.getvalue())

            anns_by_frame = {}
            for ann in annotations:
                anns_by_frame.setdefault(ann.frame_index, []).append(ann)

            rep_frames = {inc.representative_frame: [] for inc in incidents}
            for inc in incidents:
                rep_frames[inc.representative_frame].append(inc)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fd, video_temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)

            try:
                out_fps = 5.0
                out = cv2.VideoWriter(video_temp_path, fourcc, out_fps, (width, height))

                if not out.isOpened():
                    raise RuntimeError("Failed to open VideoWriter for generating MP4 report.")

                try:
                    for current_frame in sampled_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if current_frame in anns_by_frame:
                            for ann in anns_by_frame[current_frame]:
                                cv2.rectangle(frame, (ann.x_min, ann.y_min), (ann.x_max, ann.y_max), (0, 0, 255), 2)
                                cv2.putText(frame, ann.incident_id, (ann.x_min, max(0, ann.y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                        # Watermarks
                        cv2.putText(frame, "MANUAL ANNOTATION BASELINE - NOT ML INFERENCE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        timestamp = current_frame / fps if fps > 0 else 0
                        cv2.putText(frame, f"Frame: {current_frame} | Original time: {timestamp:.2f} s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                        if current_frame in rep_frames:
                            for inc in rep_frames[current_frame]:
                                ret_img, buffer = cv2.imencode('.jpg', frame)
                                if ret_img:
                                    # Ensure safe path without directory traversal just in case, though regex handles it
                                    safe_name = "".join(c for c in inc.evidence_file if c.isalnum() or c in ('-', '_', '.'))
                                    zf.writestr(f"evidence/{safe_name}", buffer.tobytes())

                        out.write(frame)
                finally:
                    out.release()

                if os.path.exists(video_temp_path) and os.path.getsize(video_temp_path) > 0:
                    with open(video_temp_path, "rb") as vf:
                        zf.writestr("annotated_manual_samples.mp4", vf.read())
                else:
                    raise RuntimeError("Generated MP4 file is empty or does not exist.")
            finally:
                if os.path.exists(video_temp_path):
                    os.remove(video_temp_path)

    finally:
        if 'cap' in locals():
            cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()
