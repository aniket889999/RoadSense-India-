/**
 * Versioned ANPR result contracts mirroring `src/anpr/contracts.py`.
 *
 * All coordinate bounds use EXIF-normalized source-image pixel coordinates.
 * OCR confidence scores are uncalibrated engine scores, not correctness
 * probabilities, and `IN_COMMON_V1` format checks never imply official
 * registration verification.
 */

export type FormatProfile = 'IN_COMMON_V1';

export type FormatStatus = 'MATCHED' | 'NO_MATCH' | 'EMPTY';

export type ReadStatus = 'READ' | 'NEEDS_REVIEW' | 'UNREADABLE';

export type ResultStatus = 'COMPLETE' | 'NEEDS_REVIEW' | 'NO_PLATE' | 'UNAVAILABLE';

export type ReasonCode =
  | 'NO_TEXT'
  | 'OCR_CONFIDENCE_MISSING'
  | 'LOW_OCR_CONFIDENCE'
  | 'FORMAT_NOT_SUPPORTED'
  | 'OCR_REVIEW_ONLY';

export type QualityFlag = 'SMALL_CROP' | 'LOW_SHARPNESS' | 'DARK_CROP' | 'CLIPPED_HIGHLIGHTS' | 'TILT_OBSERVED';

export interface PlateImageQuality {
  method: 'opencv_review_v1';
  crop_width: number;
  crop_height: number;
  sharpness_score: number;
  mean_brightness: number;
  clipped_highlight_fraction: number;
  observed_line_tilt_degrees: number | null;
  warnings: QualityFlag[];
}

export const QUALITY_GUIDANCE: Record<QualityFlag, string> = {
  SMALL_CROP: 'Small plate crop: move closer or use a higher-resolution source.',
  LOW_SHARPNESS: 'Low detail, possibly blur: hold the camera steady and check focus.',
  DARK_CROP: 'Dark crop: improve lighting and check that characters remain visible.',
  CLIPPED_HIGHLIGHTS: 'Many bright clipped pixels: check for glare hiding characters.',
  TILT_OBSERVED: 'Slanted edges observed: try a straight-on capture and inspect both text lines.',
};

/**
 * Half-open x/y pixel bounds in the EXIF-normalized source image.
 */
export interface PlateBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/**
 * An OCR candidate and its review decision, never registry verification.
 */
export interface OcrAssessment {
  raw_text: string;
  normalized_text: string;
  ocr_confidence: number | null;
  format_profile: FormatProfile;
  format_status: FormatStatus;
  recognition_status: ReadStatus;
  reason_codes: ReasonCode[];
}

/**
 * Detected plate box and its OCR candidate assessment.
 */
export interface PlateResult extends OcrAssessment {
  plate_id: string;
  bbox: PlateBox;
  detection_confidence: number;
  image_quality?: PlateImageQuality | null;
}

/**
 * Complete single-image ANPR inspection result.
 */
export interface AnprImageResult {
  schema_version: '1.0' | '1.1';
  request_id: string;
  status: ResultStatus;
  image_width: number;
  image_height: number;
  plates: PlateResult[];
  processing_ms: number;
  warnings: string[];
}

export const ANPR_MAX_IMAGE_BYTES = 10 * 1024 * 1024; // 10 MiB

export const ANPR_ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png'] as const;

export const FORMAT_STATUS_LABELS: Record<FormatStatus, string> = {
  MATCHED: 'Supported format matched',
  NO_MATCH: 'Unsupported format',
  EMPTY: 'Empty text',
};

export const READ_STATUS_LABELS: Record<ReadStatus, string> = {
  READ: 'Read',
  NEEDS_REVIEW: 'Needs review',
  UNREADABLE: 'Unreadable',
};

export const RESULT_STATUS_LABELS: Record<ResultStatus, string> = {
  COMPLETE: 'Complete',
  NEEDS_REVIEW: 'Needs review',
  NO_PLATE: 'No plate detected',
  UNAVAILABLE: 'Recognition unavailable',
};

export const REASON_CODE_LABELS: Record<ReasonCode, string> = {
  NO_TEXT: 'No readable characters returned by OCR',
  OCR_CONFIDENCE_MISSING: 'OCR confidence score missing',
  LOW_OCR_CONFIDENCE: 'OCR score below review threshold',
  FORMAT_NOT_SUPPORTED: 'Normalized text does not match IN_COMMON_V1 shape',
  OCR_REVIEW_ONLY: 'This engine provides review suggestions. Confirm the characters against the crop.',
};
