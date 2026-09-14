"""FastAPI Application Entrypoint for RoadSense India Operations Dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from services.api.app.core.config import settings
from services.api.app.core.logging import logger
from services.api.app.db.base import Base
from services.api.app.db.session import engine
from services.api.app.routers import health, live, parking, road_events, sessions


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces strict Content-Security-Policy and security headers."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "connect-src 'self' ws://localhost:* ws://127.0.0.1:* http://localhost:* http://127.0.0.1:*;"
        )
        return response


async def reconcile_startup_parking_jobs(db_session: Any = None) -> None:
    """Reconcile non-terminal persisted jobs and clean abandoned upload staging on server startup."""
    import shutil
    import tempfile
    from datetime import datetime, timezone
    from pathlib import Path
    from sqlalchemy import select
    from services.api.app.db.session import async_session_factory
    from services.api.app.models.entities import ParkingOccupancyJob

    media_root = Path(settings.MEDIA_ROOT) if hasattr(settings, "MEDIA_ROOT") else Path(tempfile.gettempdir()) / "roadsense_media"

    # 1. Clean abandoned upload staging files
    upload_staging_dir = media_root / "upload_staging"
    if upload_staging_dir.exists() and not upload_staging_dir.is_symlink():
        for item in upload_staging_dir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to clean upload staging item {item}: {e}")

    # Clean leftover staging_ directories in jobs_root
    jobs_root = media_root / "parking_jobs"
    if jobs_root.exists() and not jobs_root.is_symlink():
        for item in jobs_root.iterdir():
            if item.is_dir() and item.name.startswith("staging_"):
                try:
                    shutil.rmtree(item, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to clean abandoned job staging directory {item}: {e}")

    # 2. Reconcile non-terminal jobs in DB
    NON_TERMINAL_STATES = ("QUEUED", "VALIDATING", "DETECTING", "CLASSIFYING_OCCUPANCY", "ENCODING", "PUBLISHING", "RUNNING")

    async def _reconcile_in_session(db):
        res = await db.execute(select(ParkingOccupancyJob).where(ParkingOccupancyJob.status.in_(NON_TERMINAL_STATES)))
        interrupted_jobs = res.scalars().all()
        if interrupted_jobs:
            now = datetime.now(timezone.utc)
            for job in interrupted_jobs:
                job.status = "FAILED"
                job.failure_code = "SERVER_RESTART_RECOVERY"
                job.failure_message = "Job was interrupted by a server restart and cannot be resumed."
                job.stage_message = "Job failed due to server restart recovery."
                job.completed_at = now
                logger.info(f"Reconciled interrupted parking job {job.id} to FAILED (SERVER_RESTART_RECOVERY)")
            await db.commit()

    try:
        if db_session is not None:
            await _reconcile_in_session(db_session)
        else:
            async with async_session_factory() as db:
                await _reconcile_in_session(db)
    except Exception as e:
        logger.error(f"Error during startup parking jobs reconciliation: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables
    logger.info("Initializing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")

    # Reconcile interrupted jobs and clean abandoned upload staging
    await reconcile_startup_parking_jobs()

    yield
    # Shutdown: dispose database engine
    logger.info("Shutting down database connection pool...")
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Local operations API for RoadSense India dashcam review, fixed-camera "
        "parking occupancy, human-reviewed pavement hazards, and fail-closed "
        "safe usable capacity."
    ),
    lifespan=lifespan,
)

# CORS Middleware (confined to local frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ROADSENSE_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Headers Middleware
app.add_middleware(SecurityHeadersMiddleware)

# Include Routers
app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(road_events.router)
app.include_router(live.router)
app.include_router(parking.router)
