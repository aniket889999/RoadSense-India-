"""Pydantic schemas for Artifacts, System Health, and Live Camera."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict


class ArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    artifact_type: str
    relative_path: str
    sha256: str
    file_size_bytes: int
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    service: str
    version: str


class SystemHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    api_version: str
    database_connected: bool
    database_type: str
    model_verified: bool
    model_hash_prefix: Optional[str] = None
    model_run_id: Optional[str] = None
    mps_available: bool
    cuda_available: bool
    active_jobs: int = 0
    disk_free_gb: float
    last_error: Optional[str] = None


class LiveOfferRequest(BaseModel):
    sdp: str
    type: str = "offer"


class LiveOfferResponse(BaseModel):
    status: str # "not_connected" | "connected"
    sdp: Optional[str] = None
    type: Optional[str] = None
    message: str
    supported: bool = False
