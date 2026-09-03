"""Road Event review router: Human verification, decisions, and audit logging."""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from services.api.app.db.session import get_db
from services.api.app.models.entities import ReviewAction, RoadEvent
from services.api.app.schemas.road_event import (
    RoadEventResponse,
    RoadEventReviewRequest,
)

router = APIRouter(prefix="/api/v1/road-events", tags=["Road Events & Review"])


@router.patch("/{event_id}/review", response_model=RoadEventResponse)
async def review_road_event(
    event_id: str,
    request: RoadEventReviewRequest,
    db: AsyncSession = Depends(get_db),
):
    """Apply an inspector review decision (CONFIRM, REJECT, NEEDS_REVISIT, SPLIT, MERGE)."""
    stmt = (
        select(RoadEvent)
        .where(RoadEvent.id == event_id)
        .options(selectinload(RoadEvent.review_actions))
    )
    result = await db.execute(stmt)
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Road Event not found.")

    prev_status = event.review_status

    if request.action == "CONFIRM":
        new_status = "CONFIRMED"
    elif request.action == "REJECT":
        new_status = "REJECTED"
    elif request.action == "NEEDS_REVISIT":
        new_status = "NEEDS_REVISIT"
    elif request.action in {"SPLIT", "MERGE"}:
        new_status = f"{request.action}ED"
    else:
        new_status = request.action

    event.review_status = new_status
    event.reviewer_note = request.reviewer_note
    event.reviewed_at = datetime.now(timezone.utc)

    # Append immutable audit trail
    action_log = ReviewAction(
        event_id=event.id,
        action=request.action,
        previous_status=prev_status,
        new_status=new_status,
        reviewer_note=request.reviewer_note,
    )
    event.review_actions.append(action_log)
    db.add(action_log)
    await db.commit()

    refreshed_stmt = (
        select(RoadEvent)
        .where(RoadEvent.id == event_id)
        .options(selectinload(RoadEvent.review_actions))
    )
    refreshed_result = await db.execute(refreshed_stmt)
    return refreshed_result.scalar_one()
