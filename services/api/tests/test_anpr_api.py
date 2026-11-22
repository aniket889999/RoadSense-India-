"""Exercise image upload without starting the app database or loading models."""

from io import BytesIO
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from services.api.app.routers.anpr import get_anpr_service, router
from src.anpr.contracts import PlateBox
from src.anpr.pipeline import AnprImageService, RecognitionBusy
from src.anpr.runtime_contracts import AnprRuntime, OcrRead, PlateDetection, RuntimeUnavailable


def vehicle_image():
    buffer = BytesIO()
    Image.new("RGB", (80, 40), (15, 25, 35)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest_asyncio.fixture
async def anpr_client():
    app = FastAPI()
    app.include_router(router)
    runtime = AnprRuntime(
        detector=SimpleNamespace(detect=lambda image: [
            PlateDetection(PlateBox(x1=5, y1=5, x2=40, y2=20), 0.96)
        ]),
        recognizer=SimpleNamespace(recognize=lambda crop: OcrRead("MH12AB1234", 0.9)),
    )
    app.dependency_overrides[get_anpr_service] = lambda: AnprImageService(lambda: runtime)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, app


@pytest.mark.asyncio
async def test_real_decode_and_injected_recognition_return_versioned_result(anpr_client):
    client, _ = anpr_client
    response = await client.post(
        "/api/v1/anpr/images/recognize", content=vehicle_image(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["schema_version"] == "1.3"
    assert result["status"] == "COMPLETE"
    assert (result["image_width"], result["image_height"]) == (80, 40)
    assert result["plates"][0]["normalized_text"] == "MH12AB1234"
    assert result["request_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type, body, expected", [
    ("application/octet-stream", b"not-an-image", 415),
    ("image/png", b"not-an-image", 400),
    ("image/png", b"", 400),
    ("image/jpeg", vehicle_image(), 415),
])
async def test_invalid_input_is_rejected(anpr_client, content_type, body, expected):
    client, _ = anpr_client
    response = await client.post(
        "/api/v1/anpr/images/recognize", content=body, headers={"Content-Type": content_type}
    )
    assert response.status_code == expected


@pytest.mark.asyncio
async def test_actual_stream_size_is_enforced_without_content_length(anpr_client, monkeypatch):
    client, _ = anpr_client
    monkeypatch.setattr("services.api.app.routers.anpr.MAX_IMAGE_BYTES", 64)

    async def chunks():
        yield b"a" * 40
        yield b"b" * 40

    response = await client.post(
        "/api/v1/anpr/images/recognize", content=chunks(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_unavailable_assets_are_explicit_in_response(anpr_client):
    client, app = anpr_client

    def unavailable():
        raise RuntimeUnavailable("Required local plate weights are not configured.")

    app.dependency_overrides[get_anpr_service] = lambda: AnprImageService(unavailable)
    response = await client.post(
        "/api/v1/anpr/images/recognize", content=vehicle_image(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "UNAVAILABLE"
    assert response.json()["plates"] == []
    assert response.json()["warnings"]


@pytest.mark.asyncio
async def test_busy_worker_returns_retry_response(anpr_client):
    client, app = anpr_client

    def busy(*args):
        raise RecognitionBusy("Recognition is busy.")

    app.dependency_overrides[get_anpr_service] = lambda: SimpleNamespace(recognize_bytes=busy)
    response = await client.post(
        "/api/v1/anpr/images/recognize", content=vehicle_image(), headers={"Content-Type": "image/png"}
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
