"""Opt-in rapidocr_gated_v1: who can be auto-accepted, and that the default stays review-only."""
import importlib.util
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from src.anpr.acceptance_policy import (
    RAPIDOCR_GATED_V1, RAPIDOCR_GATED_V1_MIN_SCORE, REVIEW_ONLY, resolve_acceptance,
)
from src.anpr.contracts import PlateBox, ReadStatus, ReasonCode
from src.anpr.pipeline import AnprImageService
from src.anpr.runtime_contracts import AnprRuntime, OcrRead, PlateDetection, RuntimeUnavailable

REPO = Path(__file__).resolve().parents[1]


def test_default_policy_keeps_rapidocr_review_only_and_easyocr_unchanged():
    assert resolve_acceptance("rapidocr", None) == resolve_acceptance("rapidocr", REVIEW_ONLY)
    assert resolve_acceptance("rapidocr", None).review_only is True
    easy = resolve_acceptance("easyocr", None)
    assert easy.review_only is False and easy.minimum_confidence is None


def test_gated_policy_settings_and_unsafe_combinations():
    gated = resolve_acceptance("rapidocr", RAPIDOCR_GATED_V1)
    assert (gated.review_only, gated.minimum_confidence) == (False, RAPIDOCR_GATED_V1_MIN_SCORE) == (False, 0.90)
    with pytest.raises(RuntimeUnavailable):
        resolve_acceptance("easyocr", RAPIDOCR_GATED_V1)
    with pytest.raises(RuntimeUnavailable):
        resolve_acceptance("rapidocr", "accept_everything")


def _read(text, score, runtime_threshold=0.90):
    payload = BytesIO()
    Image.new("RGB", (100, 60), (40, 40, 40)).save(payload, "PNG")
    runtime = AnprRuntime(
        SimpleNamespace(detect=lambda _: [PlateDetection(PlateBox(x1=10, y1=10, x2=90, y2=50), 0.9)]),
        SimpleNamespace(recognize=lambda _: OcrRead(text, score)),
        minimum_confidence=runtime_threshold,
    )
    return AnprImageService(lambda: runtime).recognize_bytes(payload.getvalue(), "image/png", "policy").plates[0]


@pytest.mark.parametrize("text, score, status, reasons", [
    ("MH12AB1234", 0.95, ReadStatus.READ, []),
    ("MH12AB1234", 0.90, ReadStatus.READ, []),                                     # threshold inclusive
    ("MH12AB1234", 0.89, ReadStatus.NEEDS_REVIEW, [ReasonCode.LOW_OCR_CONFIDENCE]),
    ("IND MH12AB1234", 0.99, ReadStatus.NEEDS_REVIEW,                              # cleaned: always reviewed
     [ReasonCode.TEXT_CLEANED, ReasonCode.FORMAT_NOT_SUPPORTED]),
    ("XY12AB1234", 0.99, ReadStatus.NEEDS_REVIEW, [ReasonCode.UNKNOWN_STATE_CODE]),
    ("22BH1234AA", 0.99, ReadStatus.NEEDS_REVIEW, [ReasonCode.FORMAT_NOT_SUPPORTED]),
    ("MH12AB12O4", 0.99, ReadStatus.NEEDS_REVIEW, [ReasonCode.FORMAT_NOT_SUPPORTED]),
])
def test_gated_decisions(text, score, status, reasons):
    plate = _read(text, score)
    assert plate.recognition_status == status and plate.reason_codes == reasons
    assert plate.raw_text == text


def test_runtime_without_policy_threshold_uses_service_default():
    assert _read("MH12AB1234", 0.80, runtime_threshold=None).recognition_status == ReadStatus.READ  # 0.75 default


def _launcher():
    spec = importlib.util.spec_from_file_location("serve_anpr_local_policy", REPO / "scripts" / "serve_anpr_local.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_auto_accept_is_explicit_and_never_inherited():
    launcher = _launcher()
    inherited = {"ROADSENSE_ANPR_ACCEPTANCE": RAPIDOCR_GATED_V1}
    assert "ROADSENSE_ANPR_ACCEPTANCE" not in launcher.build_environment("rapidocr", "mps", inherited=inherited)
    env = launcher.build_environment("rapidocr", "mps", inherited={}, auto_accept=True)
    assert env["ROADSENSE_ANPR_ACCEPTANCE"] == RAPIDOCR_GATED_V1
    with pytest.raises(ValueError):
        launcher.build_environment("easyocr", "mps", inherited={}, auto_accept=True)
