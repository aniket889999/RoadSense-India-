"""Controlled image degradation and honest review-policy integration."""
from io import BytesIO
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from src.anpr.contracts import AnprImageResult, PlateBox, QualityFlag
from src.anpr.pipeline import AnprImageService
from src.anpr.quality import assess_plate_quality
from src.anpr.runtime_contracts import AnprRuntime, OcrRead, PlateDetection
from src.anpr.validation import assess_ocr_read


def rendered_plate():
    image = np.full((160, 360, 3), 210, np.uint8)
    cv2.rectangle(image, (16, 40), (344, 120), (15, 15, 15), 3)
    cv2.putText(image, "MH12AB1234", (30, 94), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (15, 15, 15), 2)
    return image


def test_blur_reduces_detail_measurement_without_changing_original_pixels():
    image = rendered_plate()
    original = image.copy()
    blurred = cv2.GaussianBlur(image, (25, 25), 7)
    sharp = assess_plate_quality(image)
    soft = assess_plate_quality(blurred)
    assert soft.sharpness_score < sharp.sharpness_score * 0.2
    assert QualityFlag.LOW_SHARPNESS in soft.warnings
    assert np.array_equal(original, image)


def test_visible_rotated_edges_get_tilt_cue_without_claiming_perspective_pose():
    image = rendered_plate()
    matrix = cv2.getRotationMatrix2D((180, 80), 15, 1)
    rotated = cv2.warpAffine(image, matrix, (360, 160), borderValue=(210, 210, 210))
    quality = assess_plate_quality(rotated)
    assert quality.observed_line_tilt_degrees is not None
    assert 10 <= abs(quality.observed_line_tilt_degrees) <= 20
    assert QualityFlag.TILT_OBSERVED in quality.warnings


def test_small_dark_and_uniform_crops_report_evidence_not_invented_angle():
    quality = assess_plate_quality(np.zeros((12, 40, 3), np.uint8))
    assert set(quality.warnings) == {QualityFlag.SMALL_CROP, QualityFlag.DARK_CROP, QualityFlag.LOW_SHARPNESS}
    assert quality.observed_line_tilt_degrees is None
    assert quality.crop_width == 40 and quality.crop_height == 12
    assert QualityFlag.CLIPPED_HIGHLIGHTS in assess_plate_quality(np.full((60, 200, 3), 255, np.uint8)).warnings


def test_review_only_policy_is_not_misreported_as_missing_confidence():
    for confidence in (None, 0.99):
        read = assess_ocr_read("MH12AB1234", confidence, review_only=True)
        assert read.recognition_status == "NEEDS_REVIEW"
        assert read.reason_codes == ["OCR_REVIEW_ONLY"]
        assert read.ocr_confidence == confidence
    assert assess_ocr_read("", None, review_only=True).reason_codes == ["NO_TEXT"]


def test_pipeline_returns_quality_and_explicit_review_mode_and_accepts_legacy_results():
    payload = BytesIO()
    Image.new("RGB", (100, 60), (50, 50, 50)).save(payload, "PNG")
    runtime = AnprRuntime(
        SimpleNamespace(detect=lambda _: [PlateDetection(PlateBox(x1=10, y1=10, x2=90, y2=50), 0.9)]),
        SimpleNamespace(recognize=lambda _: OcrRead("MH12AB1234", None, review_only=True)),
    )
    result = AnprImageService(lambda: runtime).recognize_bytes(payload.getvalue(), "image/png", "quality-check")
    assert result.schema_version == "1.1"
    assert result.plates[0].image_quality.crop_width == 80
    assert result.plates[0].reason_codes == ["OCR_REVIEW_ONLY"]
    legacy = result.model_dump(mode="json")
    legacy["schema_version"] = "1.0"
    legacy["plates"][0].pop("image_quality")
    assert AnprImageResult.model_validate(legacy).plates[0].image_quality is None
