import yaml
from pydantic import BaseModel
from typing import List

class AppConfig(BaseModel):
    name: str
    max_upload_mb: int

class VideoConfig(BaseModel):
    allowed_extensions: List[str]
    sampling_fps: int
    max_sampled_frames: int

class Config(BaseModel):
    app: AppConfig
    video: VideoConfig

def load_config(path: str = "configs/default.yaml") -> Config:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)
