"""Drive Sessions router: Uploads, background processing, telemetry, and artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import List, Optional
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from services.api.app.core.config import settings
from services.api.app.core.logging import logger
from services.api.app.core.security import (
    sanitize_filename,
    validate_file_signature,
    validate_safe_storage_path,
)
from services.api.app.db.session import get_db
from services.api.app.models.entities import (
    Artifact,
    DriveSession,
    RawDetection,
    RoadEvent,
)
from services.api.app.schemas.health import ArtifactResponse
from services.api.app.schemas.road_event import (
    RawDetectionResponse,
    RoadEventResponse,
)
from services.api.app.schemas.session import (
    SessionProcessRequest,
    SessionResponse,
)
from services.api.app.services.live_manager import ws_manager
from services.api.app.services.video_processor import process_video_session

router = APIRouter(prefix="/api/v1/sessions", tags=["Drive Sessions"])


@router.post("/upload", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def upload_session(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a local dashcam recording without cloud transmission."""
    safe_name = sanitize_filename(file.filename or "video.mp4")

    # Read uploaded content with size bounds check (max 500 MB)
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file exceeds the maximum permitted size (500 MB)."
        )

    if not validate_file_signature(content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file signature does not match a supported video container."
        )

    file_hash = hashlib.sha256(content).hexdigest()

    # Create session entity
    session = DriveSession(
        mode="upload",
        source_filename=safe_name,
        source_hash=file_hash,
        processing_state="queued",
    )
    db.add(session)
    await db.flush()

    # Store file in private session spool
    session_dir = settings.session_dir / session.id
    session_dir.mkdir(parents=True, exist_ok=True)
    raw_video_path = session_dir / "raw_video.mp4"
    raw_video_path.write_bytes(content)

    await db.commit()
    await db.refresh(session)
    return session


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List recent drive sessions ordered by creation date."""
    stmt = (
        select(DriveSession)
        .order_by(desc(DriveSession.started_at))
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve details and processing telemetry for a specific session."""
    stmt = select(DriveSession).where(DriveSession.id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return session


@router.post("/{session_id}/process", status_code=status.HTTP_202_ACCEPTED)
async def trigger_processing(
    session_id: str,
    request: SessionProcessRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Trigger background inference and Road Event fusion on an uploaded session."""
    stmt = select(DriveSession).where(DriveSession.id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    if session.processing_state in {"processing", "validating", "sampling", "model_loading"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already actively processing."
        )

    # Launch background job
    background_tasks.add_task(process_video_session, session_id, request)
    return {"message": "Processing job queued successfully.", "session_id": session_id}


@router.get("/{session_id}/detections", response_model=List[RawDetectionResponse])
async def get_session_detections(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve raw, frame-by-frame model detections for a session."""
    stmt = (
        select(RawDetection)
        .where(RawDetection.session_id == session_id)
        .order_by(RawDetection.frame_index)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{session_id}/road-events", response_model=List[RoadEventResponse])
async def get_session_road_events(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve clustered Road Events proposed for inspector review."""
    stmt = (
        select(RoadEvent)
        .where(RoadEvent.session_id == session_id)
        .options(selectinload(RoadEvent.review_actions))
        .order_by(RoadEvent.first_seen_seconds)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{session_id}/artifacts", response_model=List[ArtifactResponse])
async def get_session_artifacts(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve generated output artifacts (video replay, zip dossier, csv)."""
    stmt = select(Artifact).where(Artifact.session_id == session_id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{session_id}/artifacts/{artifact_type}/download")
async def download_artifact(
    session_id: str,
    artifact_type: str,
    db: AsyncSession = Depends(get_db),
):
    """Download a generated artifact safely without path traversal."""
    stmt = select(Artifact).where(
        Artifact.session_id == session_id,
        Artifact.artifact_type == artifact_type
    )
    result = await db.execute(stmt)
    art = result.scalar_one_or_none()
    if not art:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.")

    file_path = settings.session_dir / session_id / Path(art.relative_path).name
    validated_path = validate_safe_storage_path(file_path, settings.session_dir)

    if not validated_path.exists() or not validated_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact file missing on disk.")

    media_type = "application/octet-stream"
    if artifact_type == "annotated_video" or artifact_type == "raw_video":
        media_type = "video/mp4"
    elif artifact_type == "report_zip":
        media_type = "application/zip"

    return FileResponse(
        path=str(validated_path),
        media_type=media_type,
        filename=f"roadsense_{session_id[:8]}_{Path(art.relative_path).name}"
    )


@router.websocket("/{session_id}/events")
async def session_events_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress events."""
    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            # Keep connection alive; client can send pings
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(session_id, websocket)
    except Exception:
        await ws_manager.disconnect(session_id, websocket)
