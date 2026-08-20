import os
import cv2
import tempfile
import numpy as np
from src.video_io import validate_video, process_and_create_kit
from src.config import Config, AppConfig, VideoConfig

def create_synthetic_video(num_frames: int = 20) -> bytes:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fd, path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    out = cv2.VideoWriter(path, fourcc, 10.0, (100, 100))
    for i in range(num_frames):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data

def test_process_and_create_kit_cap():
    video_bytes = create_synthetic_video(num_frames=50)
    config = Config(
        app=AppConfig(name="Test", max_upload_mb=10),
        video=VideoConfig(allowed_extensions=[".mp4"], sampling_fps=10, max_sampled_frames=5)
    )
    metadata, kit_zip = process_and_create_kit(video_bytes, "test.mp4", config)
    assert metadata.sampled_frame_count <= 5
    assert len(metadata.sampled_frame_indices) <= 5
    assert metadata.sampled_frame_indices[0] == 0
    assert metadata.sampled_frame_indices[-1] == 49

def test_short_video_sampling_cap_1():
    video_bytes = create_synthetic_video(num_frames=5)
    config = Config(
        app=AppConfig(name="Test", max_upload_mb=10),
        video=VideoConfig(allowed_extensions=[".mp4"], sampling_fps=1, max_sampled_frames=1)
    )
    metadata, kit_zip = process_and_create_kit(video_bytes, "test.mp4", config)
    assert metadata.sampled_frame_count == 1
    assert metadata.sampled_frame_indices == [0]

def test_short_video_sampling():
    video_bytes = create_synthetic_video(num_frames=5)
    config = Config(
        app=AppConfig(name="Test", max_upload_mb=10),
        video=VideoConfig(allowed_extensions=[".mp4"], sampling_fps=1, max_sampled_frames=2)
    )
    # Target frames is 5/10 * 1 = 0.5 -> 0. But num_samples = min(0, 2) -> 0.
    # Then max(1, 0) -> 1, max(2, 1) -> 2. So it should sample exactly 2 frames!
    metadata, kit_zip = process_and_create_kit(video_bytes, "test.mp4", config)
    assert metadata.sampled_frame_count == 2
    assert metadata.sampled_frame_indices == [0, 4]

def test_validate_video():
    config = Config(
        app=AppConfig(name="Test", max_upload_mb=10),
        video=VideoConfig(allowed_extensions=[".mp4"], sampling_fps=2, max_sampled_frames=5)
    )
    is_valid, _ = validate_video("test.mp4", 1024 * 1024 * 5, config)
    assert is_valid

    is_valid, _ = validate_video("test.txt", 1024 * 1024 * 5, config)
    assert not is_valid
