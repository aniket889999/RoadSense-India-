"""Versioned results shared by the ANPR backend and inspection UI."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Confidence = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
PixelCoordinate = Annotated[float, Field(strict=True, ge=0.0)]
Identifier = Annotated[str, Field(min_length=1, max_length=64, pattern=r"\S")]


class FormatStatus(str, Enum):
    MATCHED = "MATCHED"
    NO_MATCH = "NO_MATCH"
    EMPTY = "EMPTY"


class ReadStatus(str, Enum):
    READ = "READ"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    UNREADABLE = "UNREADABLE"


class ResultStatus(str, Enum):
    COMPLETE = "COMPLETE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NO_PLATE = "NO_PLATE"
    UNAVAILABLE = "UNAVAILABLE"


class ReasonCode(str, Enum):
    NO_TEXT = "NO_TEXT"
    OCR_CONFIDENCE_MISSING = "OCR_CONFIDENCE_MISSING"
    LOW_OCR_CONFIDENCE = "LOW_OCR_CONFIDENCE"
    FORMAT_NOT_SUPPORTED = "FORMAT_NOT_SUPPORTED"


class AnprContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PlateBox(AnprContract):
    """Half-open x/y pixel bounds in the EXIF-normalized source image."""

    x1: PixelCoordinate
    y1: PixelCoordinate
    x2: PixelCoordinate
    y2: PixelCoordinate

    @model_validator(mode="after")
    def require_positive_area(self) -> "PlateBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Plate box must have positive width and height.")
        return self


class OcrAssessment(AnprContract):
    """An OCR candidate and its review decision, never registry verification."""

    raw_text: str = Field(max_length=256)
    normalized_text: str = Field(max_length=256)
    ocr_confidence: Confidence | None
    format_profile: Literal["IN_COMMON_V1"] = "IN_COMMON_V1"
    format_status: FormatStatus
    recognition_status: ReadStatus
    reason_codes: list[ReasonCode] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_consistent_read_state(self) -> "OcrAssessment":
        if self.format_status == FormatStatus.EMPTY and self.normalized_text:
            raise ValueError("EMPTY format requires empty normalized text.")
        if self.format_status != FormatStatus.EMPTY and not self.normalized_text:
            raise ValueError("Nonempty format requires normalized text.")
        if self.recognition_status == ReadStatus.READ:
            if (
                self.format_status != FormatStatus.MATCHED
                or self.ocr_confidence is None
                or self.reason_codes
            ):
                raise ValueError("READ requires a format match, confidence and no review reasons.")
        elif not self.reason_codes:
            raise ValueError("A non-READ result requires a review reason.")
        if self.recognition_status == ReadStatus.UNREADABLE and self.normalized_text:
            raise ValueError("UNREADABLE requires empty normalized text.")
        return self


class PlateResult(OcrAssessment):
    plate_id: Identifier
    bbox: PlateBox
    detection_confidence: Confidence


class AnprImageResult(AnprContract):
    schema_version: Literal["1.0"] = "1.0"
    request_id: Identifier
    status: ResultStatus
    image_width: int = Field(strict=True, gt=0)
    image_height: int = Field(strict=True, gt=0)
    plates: list[PlateResult] = Field(default_factory=list)
    processing_ms: float = Field(strict=True, ge=0.0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_consistent_image_result(self) -> "AnprImageResult":
        if self.status in {ResultStatus.NO_PLATE, ResultStatus.UNAVAILABLE}:
            if self.plates:
                raise ValueError("NO_PLATE and UNAVAILABLE cannot contain plate results.")
        else:
            if not self.plates:
                raise ValueError("COMPLETE and NEEDS_REVIEW require plate results.")
            all_read = all(p.recognition_status == ReadStatus.READ for p in self.plates)
            if (self.status == ResultStatus.COMPLETE) != all_read:
                raise ValueError("COMPLETE requires every plate to be READ; otherwise use NEEDS_REVIEW.")

        plate_ids = [plate.plate_id for plate in self.plates]
        if len(plate_ids) != len(set(plate_ids)):
            raise ValueError("Plate identifiers must be unique within an image.")
        for plate in self.plates:
            if plate.bbox.x2 > self.image_width or plate.bbox.y2 > self.image_height:
                raise ValueError("Plate bounds must remain inside the source image.")
        return self
