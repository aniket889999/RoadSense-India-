"""Bounded still-image decoding and injectable ANPR pipeline orchestration."""

from __future__ import annotations

import io
import math
import threading
import time
import warnings
from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.anpr.contracts import AnprImageResult, PlateBox, PlateResult, ReadStatus, ResultStatus
from src.anpr.runtime_contracts import AnprRuntime, OcrRead, PlateDetection, RuntimeUnavailable
from src.anpr.validation import assess_ocr_read


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_IMAGE_DIMENSION = 8192
MAX_PLATES_PER_IMAGE = 16
ALLOWED_MEDIA_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}


class InvalidImage(ValueError):
    """The input cannot be decoded as the requested image format."""


class ImageLimitExceeded(InvalidImage):
    """The encoded image or its decoded dimensions exceed the configured cap."""


class UnsupportedImage(InvalidImage):
    """The image format is unsupported or mismatches its declared media type."""


class RecognitionBusy(RuntimeError):
    """This worker already has an active image decode/inference call."""


class RecognitionFailed(RuntimeError):
    """An adapter failed or returned invalid evidence; no successful result exists."""


def decode_image(payload: bytes, media_type: str) -> np.ndarray:
    """Decode a bounded JPEG/PNG, normalize EXIF orientation and return BGR pixels."""
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise UnsupportedImage("Use image/jpeg or image/png.")
    if not payload:
        raise InvalidImage("The uploaded image is empty.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ImageLimitExceeded("The uploaded image exceeds 10 MiB.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                if source.format != ALLOWED_MEDIA_TYPES[media_type]:
                    raise UnsupportedImage("The decoded image does not match its Content-Type.")
                width, height = source.size
                if (
                    width <= 0 or height <= 0
                    or max(width, height) > MAX_IMAGE_DIMENSION
                    or width * height > MAX_IMAGE_PIXELS
                ):
                    raise ImageLimitExceeded("The image exceeds the decoded dimension/pixel limit.")
                if getattr(source, "n_frames", 1) != 1:
                    raise UnsupportedImage("Upload a still image, not an animated image.")
                source.load()
                normalized = ImageOps.exif_transpose(source)
                if "A" in normalized.getbands() or "transparency" in normalized.info:
                    rgba = normalized.convert("RGBA")
                    rgb = Image.new("RGB", rgba.size, "white")
                    rgb.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    rgb = normalized.convert("RGB")
                return np.array(rgb, dtype=np.uint8)[:, :, ::-1].copy()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageLimitExceeded("The image exceeds the decoded pixel limit.") from exc
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise InvalidImage("The uploaded image cannot be decoded as a complete JPEG/PNG.") from exc


def build_default_runtime() -> AnprRuntime:
    """Import the optional adapters only when a valid image is submitted."""
    try:
        from src.anpr.local_runtime import build_local_runtime
    except ModuleNotFoundError as exc:
        if exc.name == "src.anpr.local_runtime":
            raise RuntimeUnavailable("The local ANPR model adapters are not connected yet.") from exc
        raise
    return build_local_runtime()


class AnprImageService:
    """Serialize this process's decode/model calls, caching only a loaded runtime."""

    def __init__(
        self,
        runtime_factory: Callable[[], AnprRuntime] = build_default_runtime,
        *,
        minimum_confidence: float = 0.75,
    ) -> None:
        # Validate the threshold using the same decision policy as real reads.
        assess_ocr_read("", None, minimum_confidence=minimum_confidence)
        self._runtime_factory = runtime_factory
        self._runtime: AnprRuntime | None = None
        self._lock = threading.Lock()
        self.minimum_confidence = minimum_confidence

    def recognize_bytes(self, payload: bytes, media_type: str, request_id: str) -> AnprImageResult:
        if not self._lock.acquire(blocking=False):
            raise RecognitionBusy("Recognition is busy; retry after the current image completes.")
        try:
            started = time.perf_counter()
            image = decode_image(payload, media_type)
            height, width = image.shape[:2]
            try:
                if self._runtime is None:
                    self._runtime = self._runtime_factory()
                plates = self._recognize_plates(image)
            except RuntimeUnavailable as exc:
                return AnprImageResult(
                    request_id=request_id,
                    status=ResultStatus.UNAVAILABLE,
                    image_width=width,
                    image_height=height,
                    processing_ms=(time.perf_counter() - started) * 1000,
                    warnings=[str(exc)],
                )
            except Exception as exc:
                raise RecognitionFailed("The local ANPR adapters could not complete recognition.") from exc

            if not plates:
                status = ResultStatus.NO_PLATE
            elif all(p.recognition_status == ReadStatus.READ for p in plates):
                status = ResultStatus.COMPLETE
            else:
                status = ResultStatus.NEEDS_REVIEW
            return AnprImageResult(
                request_id=request_id,
                status=status,
                image_width=width,
                image_height=height,
                plates=plates,
                processing_ms=(time.perf_counter() - started) * 1000,
            )
        finally:
            self._lock.release()

    def _recognize_plates(self, image: np.ndarray) -> list[PlateResult]:
        assert self._runtime is not None
        detections = self._runtime.detector.detect(image)
        if not isinstance(detections, Sequence) or len(detections) > MAX_PLATES_PER_IMAGE:
            raise ValueError("Detector must return at most 16 plate candidates.")
        height, width = image.shape[:2]
        results: list[PlateResult] = []
        for index, detection in enumerate(detections, start=1):
            if not isinstance(detection, PlateDetection):
                raise TypeError("Detector returned an incompatible plate detection.")
            box = PlateBox.model_validate(detection.bbox)
            if box.x2 > width or box.y2 > height:
                raise ValueError("Detector returned a plate outside source-image bounds.")
            # Copy the crop so adapter preprocessing cannot mutate the source.
            crop = image[
                math.floor(box.y1):math.ceil(box.y2),
                math.floor(box.x1):math.ceil(box.x2),
            ].copy()
            ocr = self._runtime.recognizer.recognize(crop)
            if not isinstance(ocr, OcrRead):
                raise TypeError("Recognizer returned an incompatible OCR result.")
            assessment = assess_ocr_read(
                ocr.raw_text, ocr.confidence, minimum_confidence=self.minimum_confidence,
                review_only=ocr.review_only,
            )
            from src.anpr.quality import assess_plate_quality
            results.append(PlateResult(
                **assessment.model_dump(),
                plate_id=f"plate-{index}",
                bbox=box,
                detection_confidence=detection.confidence,
                image_quality=assess_plate_quality(crop),
            ))
        return results
