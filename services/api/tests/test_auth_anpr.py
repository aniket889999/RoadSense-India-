"""Optional ANPR API key and rate limit: off by default, enforced only when configured."""
from io import BytesIO
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from PIL import Image

from services.api.app.core import auth
from services.api.app.routers.anpr import get_anpr_service, router
from src.anpr.contracts import PlateBox
from src.anpr.pipeline import AnprImageService
from src.anpr.runtime_contracts import AnprRuntime, OcrRead, PlateDetection

URL = "/api/v1/anpr/images/recognize"


def png():
    buffer = BytesIO()
    Image.new("RGB", (80, 40), (15, 25, 35)).save(buffer, "PNG")
    return buffer.getvalue()


@pytest_asyncio.fixture
async def client(monkeypatch):
    monkeypatch.delenv(auth.ENV_API_KEY, raising=False)
    monkeypatch.delenv(auth.ENV_RATE_LIMIT, raising=False)
    auth.limiter.reset()
    app = FastAPI()
    app.include_router(router)
    runtime = AnprRuntime(
        detector=SimpleNamespace(detect=lambda image: [PlateDetection(PlateBox(x1=5, y1=5, x2=40, y2=20), 0.96)]),
        recognizer=SimpleNamespace(recognize=lambda crop: OcrRead("MH12AB1234", 0.9)),
    )
    app.dependency_overrides[get_anpr_service] = lambda: AnprImageService(lambda: runtime)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
    auth.limiter.reset()


async def post(http, **headers):
    return await http.post(URL, content=png(), headers={"Content-Type": "image/png", **headers})


@pytest.mark.asyncio
async def test_unconfigured_behaviour_is_unchanged(client):
    for _ in range(5):
        assert (await post(client)).status_code == 200


@pytest.mark.asyncio
async def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setenv(auth.ENV_API_KEY, "s3cret-key")
    missing = await post(client)
    assert missing.status_code == 401 and missing.headers["www-authenticate"] == "ApiKey"
    assert (await post(client, **{"X-API-Key": "wrong"})).status_code == 401
    assert (await post(client, **{"X-API-Key": "s3cret-key"})).status_code == 200
    assert "s3cret-key" not in missing.text


@pytest.mark.asyncio
async def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setenv(auth.ENV_RATE_LIMIT, "2")
    assert [(await post(client)).status_code for _ in range(2)] == [200, 200]
    limited = await post(client)
    assert limited.status_code == 429 and 1 <= int(limited.headers["retry-after"]) <= 60


@pytest.mark.asyncio
async def test_invalid_rate_limit_config_fails_closed(client, monkeypatch):
    monkeypatch.setenv(auth.ENV_RATE_LIMIT, "lots")
    assert (await post(client)).status_code == 500


def test_window_expires_and_memory_is_bounded(monkeypatch):
    now = [0.0]
    limiter = auth.SlidingWindowLimiter(clock=lambda: now[0])
    assert limiter.check("a", 1) is None
    assert limiter.check("a", 1) == pytest.approx(60.0)
    now[0] = 60.0
    assert limiter.check("a", 1) is None
    monkeypatch.setattr(auth, "MAX_TRACKED_CLIENTS", 3)
    for name in ("b", "c", "d", "e"):
        limiter.check(name, 5)
    assert len(limiter._hits) <= 3
