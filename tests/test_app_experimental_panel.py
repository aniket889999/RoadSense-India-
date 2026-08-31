from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

import src.ml.model_provenance as model_provenance


REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_video_bytes(tmp_path: Path) -> bytes:
    """Create a tiny local MP4 for the app test without using project data."""

    video_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64)
    )
    assert writer.isOpened()
    try:
        for _ in range(4):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    return video_path.read_bytes()


def test_experimental_model_cache_is_session_scoped_not_process_global():
    """The optional Ultralytics object must not be shared across browser sessions."""

    app_source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "@st.cache_resource" not in app_source
    assert "st.session_state.experimental_verified_model" in app_source


def test_optional_model_failure_keeps_manual_csv_uploader_available(monkeypatch, tmp_path):
    """A broken optional model must never block the primary manual workflow."""

    def unavailable_model(*_args, **_kwargs):
        raise OSError("simulated unreadable pinned artifact")

    monkeypatch.setattr(model_provenance, "verify_frozen_baseline", unavailable_model)
    app = AppTest.from_file(str(REPO_ROOT / "app.py"))
    app.run(timeout=30)

    app.file_uploader[0].set_value(("tiny.mp4", _synthetic_video_bytes(tmp_path), "video/mp4"))
    app.run(timeout=30)
    create_kit = next(button for button in app.button if button.label == "Create annotation kit")
    create_kit.click()
    app.run(timeout=30)

    assert not app.exception
    assert any("Experimental local model is unavailable" in error.value for error in app.error)
    assert "Upload your completed manual annotations CSV" in [
        uploader.label for uploader in app.file_uploader
    ]
