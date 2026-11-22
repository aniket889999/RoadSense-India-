"""Local, non-persisting still-image ANPR endpoint."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from services.api.app.core.auth import require_anpr_access
from src.anpr.contracts import AnprImageResult
from src.anpr.pipeline import (
    ALLOWED_MEDIA_TYPES,
    MAX_IMAGE_BYTES,
    AnprImageService,
    ImageLimitExceeded,
    InvalidImage,
    RecognitionBusy,
    RecognitionFailed,
    UnsupportedImage,
)


router = APIRouter(prefix="/api/v1/anpr", tags=["ANPR Image Inspection"])
logger = logging.getLogger(__name__)
_image_service = AnprImageService()


def get_anpr_service() -> AnprImageService:
    return _image_service


@router.post(
    "/images/recognize",
    response_model=AnprImageResult,
    dependencies=[Depends(require_anpr_access)],
    responses={
        400: {"description": "Empty, corrupt or invalid image."},
        401: {"description": "Only when ROADSENSE_API_KEY is configured: missing or wrong X-API-Key."},
        413: {"description": "Encoded bytes or decoded dimensions exceed the image limits."},
        415: {"description": "Unsupported image type or mismatched Content-Type."},
        429: {"description": "Only when ROADSENSE_ANPR_RATE_LIMIT_PER_MINUTE is configured: limit exceeded."},
        503: {"description": "Recognition is busy or an adapter failed to complete."},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "description": "Raw JPEG/PNG bytes, at most 10 MiB. This endpoint does not use multipart.",
            "content": {
                media_type: {"schema": {"type": "string", "format": "binary"}}
                for media_type in ALLOWED_MEDIA_TYPES
            },
        }
    },
)
async def recognize_image(
    request: Request,
    service: AnprImageService = Depends(get_anpr_service),
) -> AnprImageResult:
    """Read one bounded image without saving source bytes, OCR text or crops.

    A valid image with missing local assets returns UNAVAILABLE and no detections.
    Input errors use HTTP error responses. Model execution runs off the event loop.
    """
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=415, detail="Send raw JPEG/PNG bytes with the matching Content-Type.")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Content-Length must be a nonnegative integer.") from exc
        if declared_bytes < 0:
            raise HTTPException(status_code=400, detail="Content-Length must be a nonnegative integer.")
        if declared_bytes > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="The uploaded image exceeds 10 MiB.")

    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="The uploaded image exceeds 10 MiB.")
        payload.extend(chunk)

    request_id = str(uuid.uuid4())
    try:
        return await run_in_threadpool(service.recognize_bytes, bytes(payload), media_type, request_id)
    except ImageLimitExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except UnsupportedImage as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RecognitionBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
    except RecognitionFailed as exc:
        logger.error("ANPR adapter failure for request %s", request_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
