"""Live Camera ingest & WebRTC boundary router."""

from __future__ import annotations

from fastapi import APIRouter, status
from services.api.app.schemas.health import LiveOfferRequest, LiveOfferResponse

router = APIRouter(prefix="/api/v1/live", tags=["Live Camera / WebRTC"])


@router.post("/offer", response_model=LiveOfferResponse)
async def handle_webrtc_offer(offer: LiveOfferRequest):
    """Handle a WebRTC offer for live dashcam streaming.

    In this milestone, camera preview runs locally via browser MediaDevices.
    Server-side WebRTC inference is marked disconnected by default to ensure zero
    untested synthetic simulations.
    """
    return LiveOfferResponse(
        status="not_connected",
        message="Live WebRTC inference pipeline boundary is active. Direct browser MediaDevices camera preview is enabled; server-side WebRTC stream decoding is currently offline.",
        supported=False,
    )
