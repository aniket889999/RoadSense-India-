import yaml
from pydantic import BaseModel, Field
from typing import List

class AppConfig(BaseModel):
    name: str
    max_upload_mb: int

class VideoConfig(BaseModel):
    allowed_extensions: List[str]
    sampling_fps: int
    max_sampled_frames: int


class ExperimentalInferenceConfig(BaseModel):
    """Configuration for the opt-in, local-only experimental model panel."""

    enabled: bool = False
    frozen_baseline_config_path: str = "configs/inference/frozen_baseline.yaml"
    device: str = "mps"
    image_size: int = Field(default=640, gt=0)
    default_confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_detections_per_frame: int = Field(default=100, gt=0)
    output_fps: float = Field(default=5.0, gt=0.0)


class DriveReviewConfig(BaseModel):
    """Bounded settings for an upload-backed experimental Drive Review replay."""

    enabled: bool = True
    default_window_seconds: int = Field(default=30, gt=0, le=60)
    max_window_seconds: int = Field(default=60, gt=0, le=60)
    sampling_fps: float = Field(default=2.0, gt=0.0, le=10.0)
    max_sampled_frames: int = Field(default=120, gt=0, le=180)
    output_fps: float = Field(default=5.0, gt=0.0, le=30.0)


class Config(BaseModel):
    app: AppConfig
    video: VideoConfig
    experimental_inference: ExperimentalInferenceConfig = Field(
        default_factory=ExperimentalInferenceConfig
    )
    drive_review: DriveReviewConfig = Field(default_factory=DriveReviewConfig)

def load_config(path: str = "configs/default.yaml") -> Config:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)
