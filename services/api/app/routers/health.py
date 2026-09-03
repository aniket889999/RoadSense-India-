"""Health check and system observability router."""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.app.core.config import settings
from services.api.app.db.session import get_db
from services.api.app.schemas.health import HealthResponse, SystemHealthResponse
from services.api.app.services.model_loader import get_verified_model_info

router = APIRouter(tags=["Health & Observability"])


@router.get("/health/live", response_model=HealthResponse)
async def health_live():
    """Liveness probe to check if the API process is running."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
    )


@router.get("/health/ready", response_model=HealthResponse)
async def health_ready(db: AsyncSession = Depends(get_db)):
    """Readiness probe checking database connectivity and model configuration."""
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database is not ready: {exc}"
        ) from exc

    try:
        get_verified_model_info()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model configuration is not ready: {exc}"
        ) from exc

    return HealthResponse(
        status="ready",
        timestamp=datetime.now(timezone.utc),
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
    )


@router.get("/health/system", response_model=SystemHealthResponse)
async def health_system(db: AsyncSession = Depends(get_db)):
    """Detailed system telemetry: hardware acceleration, storage, and model provenance."""
    db_connected = False
    db_type = "sqlite" if "sqlite" in settings.DATABASE_URL else "postgresql"
    try:
        await db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    model_verified = False
    model_hash_prefix = None
    model_run_id = None
    try:
        config, model_info = get_verified_model_info()
        model_verified = True
        model_hash_prefix = model_info.checkpoint_sha256[:8]
        model_run_id = model_info.run_id
    except Exception:
        model_verified = False

    # Check hardware backends
    mps_available = False
    cuda_available = False
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            mps_available = True
        if torch.cuda.is_available():
            cuda_available = True
    except Exception:
        pass

    # Disk space
    disk_free_gb = 0.0
    try:
        stat = shutil.disk_usage(str(settings.session_dir.resolve()))
        disk_free_gb = round(stat.free / (1024 ** 3), 2)
    except Exception:
        pass

    return SystemHealthResponse(
        status="operational" if (db_connected and model_verified) else "degraded",
        timestamp=datetime.now(timezone.utc),
        api_version=settings.VERSION,
        database_connected=db_connected,
        database_type=db_type,
        model_verified=model_verified,
        model_hash_prefix=model_hash_prefix,
        model_run_id=model_run_id,
        mps_available=mps_available,
        cuda_available=cuda_available,
        active_jobs=0,
        disk_free_gb=disk_free_gb,
    )
