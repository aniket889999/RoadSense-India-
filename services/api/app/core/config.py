"""Configuration and environment management for RoadSense India API."""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseModel):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RoadSense India Operations API"
    VERSION: str = "1.0.0"

    # Host and CORS
    ROADSENSE_API_HOST: str = os.getenv("ROADSENSE_API_HOST", "127.0.0.1")
    ROADSENSE_API_PORT: int = int(os.getenv("ROADSENSE_API_PORT", "8000"))
    ROADSENSE_CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv("ROADSENSE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ]

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///outputs/roadsense.db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "roadsense")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "roadsense_local_dev")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "roadsense_india")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))

    # Inference defaults
    ROADSENSE_DEVICE: str = os.getenv("ROADSENSE_DEVICE", "mps")
    ROADSENSE_CONFIDENCE_THRESHOLD: float = float(os.getenv("ROADSENSE_CONFIDENCE_THRESHOLD", "0.25"))
    ROADSENSE_IOU_THRESHOLD: float = float(os.getenv("ROADSENSE_IOU_THRESHOLD", "0.45"))
    ROADSENSE_MAX_FRAMES_PER_SESSION: int = int(os.getenv("ROADSENSE_MAX_FRAMES_PER_SESSION", "150"))
    ROADSENSE_SAMPLING_FPS: float = float(os.getenv("ROADSENSE_SAMPLING_FPS", "5.0"))

    # Storage paths (relative to repo root)
    ROADSENSE_SESSION_STORAGE: str = os.getenv("ROADSENSE_SESSION_STORAGE", "outputs/sessions")
    FROZEN_BASELINE_CONFIG: str = os.getenv("FROZEN_BASELINE_CONFIG", "configs/inference/frozen_baseline.yaml")

    @property
    def session_dir(self) -> Path:
        p = REPO_ROOT / self.ROADSENSE_SESSION_STORAGE
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def frozen_config_path(self) -> Path:
        return REPO_ROOT / self.FROZEN_BASELINE_CONFIG


settings = Settings()
