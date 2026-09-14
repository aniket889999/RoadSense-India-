"""Regression checks for the tracked API contract."""

from __future__ import annotations

import json
from pathlib import Path

from services.api.app.main import app


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tracked_openapi_matches_fastapi_application() -> None:
    tracked_path = REPO_ROOT / "docs" / "api" / "openapi.json"
    tracked = json.loads(tracked_path.read_text(encoding="utf-8"))
    generated = app.openapi()

    assert tracked == generated


def test_safe_capacity_operations_are_in_openapi() -> None:
    paths = app.openapi()["paths"]

    assert "/api/v1/cameras/{camera_id}/capacity/snapshot" in paths
    assert "/api/v1/cameras/{camera_id}/hazards/associations" in paths
    assert "/api/v1/hazards/associations/{association_id}/review" in paths
    assert "/api/v1/hazards/associations/{association_id}/lifecycle" in paths
    assert "/api/v1/cameras/{camera_id}/inspection/record" in paths
