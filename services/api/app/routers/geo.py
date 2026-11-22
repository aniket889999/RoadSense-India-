"""Geolocation router: GPS track import per drive session and GeoJSON exports for the map."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.app.db.session import get_db
from services.api.app.models.entities import DriveSession, RoadEvent
from services.api.app.services import gps_track as gt

router = APIRouter(prefix="/api/v1", tags=["Map & GPS"])

_REVIEW_STATUSES = {"PENDING_REVIEW", "CONFIRMED", "REJECTED", "NEEDS_REVISIT", "SPLITED", "MERGED"}


async def _get_session(db: AsyncSession, session_id: str) -> DriveSession:
    result = await db.execute(select(DriveSession).where(DriveSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return session


async def _session_events(db: AsyncSession, session_id: str) -> list[RoadEvent]:
    result = await db.execute(
        select(RoadEvent).where(RoadEvent.session_id == session_id).order_by(RoadEvent.first_seen_seconds)
    )
    return list(result.scalars().all())


@router.post("/sessions/{session_id}/gps-track")
async def upload_gps_track(
    session_id: str,
    file: UploadFile = File(...),
    offset_seconds: float = Form(0.0),
    db: AsyncSession = Depends(get_db),
):
    """Attach a GPS track (GPX, NMEA or CSV) to a session and place its road events on the map.

    ``offset_seconds`` is the video time at which the track's first point was recorded.
    """
    session = await _get_session(db, session_id)
    raw = await file.read(gt.MAX_TRACK_BYTES + 1)
    try:
        parsed = gt.parse_track(raw, file.filename or "", offset_seconds)
    except gt.GpsTrackError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session.route_telemetry = [p.to_json() for p in parsed.points]
    events = await _session_events(db, session_id)
    placed = 0
    for ev in events:
        pos = gt.locate(parsed.points, gt.event_time(ev.first_seen_seconds, ev.last_seen_seconds))
        ev.latitude, ev.longitude = (pos if pos else (None, None))
        placed += 1 if pos else 0
    await db.commit()

    duration = session.source_duration_seconds
    summary = gt.track_summary(parsed.points)
    warnings: list[str] = []
    if duration and summary["end_seconds"] is not None:
        if summary["end_seconds"] < 0 or summary["start_seconds"] > duration:
            warnings.append("The track does not overlap the video timeline; check offset_seconds.")
    if events and placed < len(events):
        warnings.append(f"{len(events) - placed} of {len(events)} road events fall outside the track or in a GPS gap and were left unplaced.")
    return {
        "session_id": session_id,
        "source_format": parsed.source_format,
        "absolute_time": parsed.has_absolute_time,
        "dropped_points": parsed.dropped_points,
        "events_total": len(events),
        "events_placed": placed,
        "location_method": gt.LOCATION_METHOD,
        "warnings": warnings,
        **summary,
    }


@router.delete("/sessions/{session_id}/gps-track")
async def delete_gps_track(session_id: str, db: AsyncSession = Depends(get_db)):
    """Remove a session's GPS track and the event positions derived from it."""
    session = await _get_session(db, session_id)
    session.route_telemetry = None
    for ev in await _session_events(db, session_id):
        ev.latitude = None
        ev.longitude = None
    await db.commit()
    return {"session_id": session_id, "message": "GPS track and derived event positions removed."}


@router.get("/sessions/{session_id}/geojson")
async def session_geojson(session_id: str, db: AsyncSession = Depends(get_db)):
    """GeoJSON FeatureCollection: the session route plus every placed road event."""
    session = await _get_session(db, session_id)
    points = gt.points_from_json(session.route_telemetry)
    features: list[dict] = []
    if len(points) >= 2:
        features.append(gt.route_feature(session_id, points))
    unplaced = 0
    for ev in await _session_events(db, session_id):
        if ev.latitude is not None and ev.longitude is not None:
            features.append(gt.event_feature(ev, ev.latitude, ev.longitude))
        else:
            unplaced += 1
    body = {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "session_id": session_id,
            "source_filename": session.source_filename,
            "unplaced_events": unplaced,
            "crs": "EPSG:4326",
        },
    }
    return JSONResponse(
        body,
        media_type="application/geo+json",
        headers={"Content-Disposition": f'inline; filename="roadsense_{session_id[:8]}.geojson"'},
    )


@router.get("/geo/road-events")
async def all_located_events(
    review_status: Optional[str] = Query(None, description="Filter by review status"),
    limit: int = Query(5000, ge=1, le=20000),
    db: AsyncSession = Depends(get_db),
):
    """All placed road events across sessions, as GeoJSON, for the overview map."""
    stmt = select(RoadEvent).where(RoadEvent.latitude.is_not(None), RoadEvent.longitude.is_not(None))
    if review_status:
        if review_status not in _REVIEW_STATUSES:
            raise HTTPException(status_code=422, detail="Unknown review status.")
        stmt = stmt.where(RoadEvent.review_status == review_status)
    stmt = stmt.order_by(RoadEvent.created_at.desc()).limit(limit)
    events = list((await db.execute(stmt)).scalars().all())
    return JSONResponse(
        {
            "type": "FeatureCollection",
            "features": [gt.event_feature(ev, ev.latitude, ev.longitude) for ev in events],
            "properties": {"count": len(events), "crs": "EPSG:4326"},
        },
        media_type="application/geo+json",
    )
