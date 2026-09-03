"""Tests for health & system telemetry endpoints."""

import pytest


@pytest.mark.asyncio
async def test_health_live(client):
    response = await client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_health_ready(client):
    response = await client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


@pytest.mark.asyncio
async def test_health_system(client):
    response = await client.get("/health/system")
    assert response.status_code == 200
    data = response.json()
    assert data["database_connected"] is True
    assert data["model_verified"] is True
    assert data["model_hash_prefix"] is not None
    assert "disk_free_gb" in data
