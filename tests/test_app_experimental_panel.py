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


def test_drive_review_is_upload_backed_and_truthfully_not_live():
    """The first web milestone must not present sampled replay as a live alert."""

    app_source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert "build_drive_review_plan" in app_source
    assert "run_verified_drive_review" in app_source
    assert "Dashcam Drive Review" in app_source
    assert "Recorded-video review" in app_source
    assert "not a continuous live camera feed" in app_source
    assert "Green means “model suggestion”, not “verified”" in app_source
    assert "st.camera_input" not in app_source


def test_landing_page_explains_the_three_stage_review_flow():
    app = AppTest.from_file(str(REPO_ROOT / "app.py"))
    app.run(timeout=30)

    assert not app.exception
    assert [uploader.label for uploader in app.file_uploader] == ["Upload a road video"]
    page_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "01 · Upload" in page_text
    assert "02 · Review" in page_text
    assert "03 · Confirm" in page_text
    assert "Direct live-camera alerts are a later" in page_text


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
