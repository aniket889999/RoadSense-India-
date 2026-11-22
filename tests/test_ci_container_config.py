"""Container configuration invariants (no Docker needed): no baked assets, non-root, local-only."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_dockerfile_runs_non_root_on_cpu_without_baking_models_or_data():
    text = (REPO / "Dockerfile").read_text()
    copies = [line.split()[1] for line in text.splitlines() if line.startswith("COPY ")]
    assert not any(src.startswith(("models", "outputs", "data/processed", "data/raw", ".env")) for src in copies)
    assert "data/reference" in copies  # public RTO directory only
    assert "USER roadsense" in text and "ROADSENSE_ANPR_DEVICE=cpu" in text
    assert "ROADSENSE_ANPR_OCR_ENGINE=rapidocr" in text
    assert "37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db" in text


def test_dockerignore_excludes_private_material():
    ignored = set((REPO / ".dockerignore").read_text().split())
    assert {".venv", "models", "outputs", "data/processed", "data/raw", ".env", ".git"} <= ignored


def test_compose_api_is_opt_in_local_only_and_mounts_models_read_only():
    compose = yaml.safe_load((REPO / "infra" / "docker-compose.yml").read_text())
    db, api = compose["services"]["db"], compose["services"]["api"]
    assert db["image"] == "postgis/postgis:16-3.4-alpine" and "profiles" not in db
    assert api["profiles"] == ["api"]
    assert api["ports"] == ["127.0.0.1:8000:8000"]
    assert "../models:/app/models:ro" in api["volumes"]
