import os
import cv2
import tempfile
import zipfile
import numpy as np
from io import BytesIO
from pydantic import BaseModel
from typing import Tuple, List
from src.config import Config

class VideoMetadata(BaseModel):
    filename: str
    size_mb: float
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    sampled_frame_count: int = 0
    sampled_frame_indices: List[int] = []

def validate_video(file_name: str, file_size: int, config: Config) -> Tuple[bool, str]:
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in config.video.allowed_extensions:
        return False, f"Unsupported format. Allowed: {', '.join(config.video.allowed_extensions)}"

    size_mb = file_size / (1024 * 1024)
    if size_mb > config.app.max_upload_mb:
        return False, f"Oversized upload: {size_mb:.2f} MB exceeds maximum of {config.app.max_upload_mb} MB"

    return True, ""

def process_and_create_kit(file_bytes: bytes, file_name: str, config: Config) -> Tuple[VideoMetadata, bytes]:
    ext = os.path.splitext(file_name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tfile:
        tfile.write(file_bytes)
        temp_path = tfile.name

    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Unreadable video file.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            raise ValueError("Video has zero FPS.")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        size_mb = len(file_bytes) / (1024 * 1024)

        if config.video.max_sampled_frames < 1:
            raise ValueError("max_sampled_frames must be >= 1")

        target_frames = int(duration * config.video.sampling_fps)
        num_samples = min(target_frames, config.video.max_sampled_frames)

        if frame_count > 0:
            if frame_count >= 2 and config.video.max_sampled_frames >= 2:
                num_samples = max(2, num_samples)
            else:
                num_samples = max(1, num_samples)

            num_samples = min(num_samples, frame_count)
            num_samples = min(num_samples, config.video.max_sampled_frames)

            if num_samples == 1:
                sampled_indices = [0]
            else:
                sampled_indices = np.linspace(0, frame_count - 1, num_samples, dtype=int)
                sampled_indices = sorted(list(set(sampled_indices)))
        else:
            sampled_indices = []

        zip_buffer = BytesIO()
        sampled_count = 0
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            readme_text = (
                "Manual Annotation Kit for RoadSense India\n"
                "----------------------------------------\n\n"
                "Rules:\n"
                "- `incident_id` is required, for example `POT-001` (alphanumeric and dashes only)\n"
                "- `frame_index` must exist in `frame_manifest.csv`\n"
                "- box coordinates use the original video's pixel size\n"
                "- `x_min < x_max` and `y_min < y_max`\n"
                "- all coordinates must be within the frame\n"
                "- `label` must be exactly `pothole`\n"
                "- `note` is optional\n"
                "- do not accept confidence, severity, risk, priority, traffic, GPS, or model fields\n"
            )
            zf.writestr('README.txt', readme_text)

            template_header = "incident_id,frame_index,x_min,y_min,x_max,y_max,label,note\n"
            zf.writestr('manual_potholes_template.csv', template_header)

            manifest_lines = ["frame_index,timestamp_seconds,frame_file,width,height\n"]

            for current_frame in sampled_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = current_frame / fps
                frame_filename = f"frame_{current_frame:05d}.jpg"

                ret_img, buffer = cv2.imencode('.jpg', frame)
                if ret_img:
                    zf.writestr(f"frames/{frame_filename}", buffer.tobytes())
                    manifest_lines.append(f"{current_frame},{timestamp:.3f},{frame_filename},{width},{height}\n")
                    sampled_count += 1

            zf.writestr('frame_manifest.csv', "".join(manifest_lines))

        metadata = VideoMetadata(
            filename=file_name,
            size_mb=size_mb,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
            sampled_frame_count=sampled_count,
            sampled_frame_indices=sampled_indices[:sampled_count]
        )

    finally:
        if 'cap' in locals():
            cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    zip_buffer.seek(0)
    return metadata, zip_buffer.getvalue()
