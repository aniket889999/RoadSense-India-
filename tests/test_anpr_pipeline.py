"""Image and orchestration tests using injected adapters, not real model outputs."""

from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.anpr.contracts import PlateBox
from src.anpr.pipeline import (
    AnprImageService,
    ImageLimitExceeded,
    InvalidImage,
    RecognitionFailed,
    UnsupportedImage,
    decode_image,
)
from src.anpr.runtime_contracts import AnprRuntime, OcrRead, PlateDetection, RuntimeUnavailable


def image_bytes(format="PNG", size=(80, 40), **save_kwargs):
    output = BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(output, format=format, **save_kwargs)
    return output.getvalue()


def detection(x1=2, y1=3, x2=22, y2=13):
    return PlateDetection(PlateBox(x1=x1, y1=y1, x2=x2, y2=y2), 0.96)


def service_with(detections, read=None):
    return AnprImageService(lambda: AnprRuntime(
        detector=SimpleNamespace(detect=lambda image: detections),
        recognizer=SimpleNamespace(recognize=read or (lambda crop: OcrRead("MH12AB1234", 0.92))),
    ))


def test_decode_converts_to_contiguous_bgr():
    image = decode_image(image_bytes(), "image/png")
    assert image.shape == (40, 80, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    assert image[0, 0].tolist() == [30, 20, 10]


def test_jpeg_exif_orientation_is_applied_before_returning_coordinates():
    exif = Image.Exif()
    exif[274] = 6
    image = decode_image(image_bytes("JPEG", exif=exif), "image/jpeg")
    assert image.shape == (80, 40, 3)


@pytest.mark.parametrize("payload", [b"", b"not an image", image_bytes()[:20]])
def test_invalid_image_does_not_initialize_runtime(payload):
    loads = []
    service = AnprImageService(lambda: loads.append("loaded"))
    with pytest.raises(InvalidImage):
        service.recognize_bytes(payload, "image/png", "test-request")
    assert loads == []


def test_declared_type_must_match_decoded_format():
    with pytest.raises(UnsupportedImage):
        decode_image(image_bytes(), "image/jpeg")


def test_pixel_cap_is_checked_before_full_decode(monkeypatch):
    monkeypatch.setattr("src.anpr.pipeline.MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ImageLimitExceeded):
        decode_image(image_bytes(), "image/png")


def test_runtime_is_cached_and_crop_has_expected_geometry():
    loads = []
    crops = []

    def read(crop):
        crops.append(crop)
        return OcrRead(" mh 12-ab 1234 ", 0.92)

    def build():
        loads.append("loaded")
        return AnprRuntime(
            SimpleNamespace(detect=lambda image: [detection()]),
            SimpleNamespace(recognize=read),
        )

    service = AnprImageService(build)
    first = service.recognize_bytes(image_bytes(), "image/png", "first")
    second = service.recognize_bytes(image_bytes(), "image/png", "second")
    assert first.status == "COMPLETE"
    assert first.plates[0].raw_text == " mh 12-ab 1234 "
    assert first.plates[0].normalized_text == "MH12AB1234"
    assert crops[0].shape == (10, 20, 3)
    assert crops[0].flags.c_contiguous
    assert loads == ["loaded"]
    assert second.request_id == "second"


def test_no_detection_does_not_invent_a_plate_or_call_ocr():
    def unexpected_ocr(crop):
        raise AssertionError("OCR cannot run without a detection.")

    result = service_with([], unexpected_ocr).recognize_bytes(image_bytes(), "image/png", "no-plate")
    assert result.status == "NO_PLATE"
    assert result.plates == []


def test_uncertain_or_empty_ocr_keeps_detection_for_review():
    service = service_with([detection()], lambda crop: OcrRead("", None))
    result = service.recognize_bytes(image_bytes(), "image/png", "empty-ocr")
    assert result.status == "NEEDS_REVIEW"
    assert len(result.plates) == 1
    assert result.plates[0].recognition_status == "UNREADABLE"


def test_missing_runtime_is_unavailable_not_no_plate():
    def unavailable():
        raise RuntimeUnavailable("Local OCR models are missing.")

    result = AnprImageService(unavailable).recognize_bytes(image_bytes(), "image/png", "missing-assets")
    assert result.status == "UNAVAILABLE"
    assert result.plates == []
    assert result.warnings == ["Local OCR models are missing."]


def test_invalid_detection_cannot_be_published_and_does_not_reach_ocr():
    ocr_calls = []
    service = service_with([detection(x2=90)], lambda crop: ocr_calls.append(crop))
    with pytest.raises(RecognitionFailed):
        service.recognize_bytes(image_bytes(), "image/png", "invalid-box")
    assert ocr_calls == []


def test_service_recovers_from_an_adapter_exception():
    reads = iter([RuntimeError("test failure"), OcrRead("MH12AB1234", 0.92)])

    def read(crop):
        next_result = next(reads)
        if isinstance(next_result, Exception):
            raise next_result
        return next_result

    service = service_with([detection()], read)
    with pytest.raises(RecognitionFailed):
        service.recognize_bytes(image_bytes(), "image/png", "failed-request")
    assert service.recognize_bytes(image_bytes(), "image/png", "retry").status == "COMPLETE"
