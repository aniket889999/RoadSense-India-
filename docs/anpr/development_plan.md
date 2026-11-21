# RoadSense ANPR: shared development plan

ANPR is the primary academic deliverable: plate detection, OCR, supported-format
validation, accuracy measurement, and analysis of blur/angle failures. Existing
parking occupancy and human-reviewed hazards provide the application context.

## Collaboration and Git

- Shared branch: `feat/anpr-foundation`, based on `f2c9a30`.
- Codex sends Gemini a bounded prompt before starting each paired module.
- Each agent edits only its assigned files. Codex integrates and creates commits
  after reviewing the pair's actual results. Gemini does not switch branches,
  stage, commit, push, install packages, or edit Codex-owned paths.
- Run focused checks after a module. Run the complete test suite and frontend
  production build at integration milestones, without duplicate concurrent runs.
- Preserve `.agents/`, `.claude/`, existing stashes, media, datasets and weights.
- Keep models and sample images local. No dependency downloads or training are
  part of Module 1. Training/download needs are assessed in the model module.

## Paired modules and completion criteria

| Module | Codex owns | Gemini owns | Completion evidence |
| --- | --- | --- | --- |
| 1: contracts and inspection view | `src/anpr/__init__.py`, `src/anpr/contracts.py`, `src/anpr/validation.py`, `tests/test_anpr_foundation.py`, this plan | `apps/web/src/components/anpr/AnprInspectionPanel.tsx`, `apps/web/src/lib/anprTypes.ts` | Backend contract tests; frontend typecheck; exact field/enum agreement |
| 2: image pipeline | Bounded upload endpoint, pipeline orchestration, API tests | Detector, OpenCV preprocessing, OCR adapters and focused adapter tests in agreed new files | Real image to real text; missing local runtime explicitly unavailable |
| 3: model making and evaluation | Dataset split audit, end-to-end metrics and provenance | Dedicated plate-detector training and blur/angle experiments | Trained detector, held-out results, observed failures, reproducible settings |
| 4: application integration | Endpoint registration, local evidence records, OpenAPI | Dashboard navigation, connected UI and annotated-image export | Uploaded image, visible plate boxes, raw/read text, honest review states |
| 5: optional video | Session lifecycle, bounded results and cancellation | Session-local tracking, repeated-read consensus, video rendering | Annotated short video after the image deliverable passes |

Modules 2 onward receive exact file boundaries and adapter signatures in their
own handoff prompts after Module 1 is reviewed. Do not start later modules early.

## Module 1 shared result contract

Source of truth: `src/anpr/contracts.py`. `AnprImageResult` serializes to:

```json
{
  "schema_version": "1.0",
  "request_id": "request-1",
  "status": "COMPLETE",
  "image_width": 640,
  "image_height": 480,
  "plates": [{
    "plate_id": "plate-1",
    "bbox": {"x1": 10, "y1": 20, "x2": 180, "y2": 75},
    "detection_confidence": 0.95,
    "raw_text": "mh 12-ab 1234",
    "normalized_text": "MH12AB1234",
    "ocr_confidence": 0.92,
    "format_profile": "IN_COMMON_V1",
    "format_status": "MATCHED",
    "recognition_status": "READ",
    "reason_codes": []
  }],
  "processing_ms": 12.5,
  "warnings": []
}
```

The JSON above is a synthetic contract illustration, not an inference result.
Boxes use EXIF-normalized source-image pixel coordinates. They must be scaled
with the displayed image and account for any letterboxing. `COMPLETE` requires
every plate to be `READ`; otherwise a nonempty result uses `NEEDS_REVIEW`.
`NO_PLATE` and `UNAVAILABLE` carry no plate results. The UI must never substitute
invented plates when the runtime is unavailable.

`IN_COMMON_V1` checks only this explicitly limited shape:
two ASCII letters, one or two digits, one to three letters, then four digits.
It is not a complete catalogue of Indian registrations. Special formats such as
BH remain unsupported/reviewable for this module. A shape match neither verifies
registry existence nor proves the OCR is correct. Normalization removes only
ASCII whitespace/hyphens and changes ASCII lowercase to uppercase; it does not
guess `O/0`, `I/1`, or `B/8`. Preserve raw OCR separately.

The initial OCR score threshold is 0.75, to be selected on validation data later.
OCR scores must not be displayed as calibrated correctness probabilities.

Gemini's component accepts:
`onAnalyze?: (file: File) => Promise<AnprImageResult>`.
Without that prop the button is disabled and states that the recognition service
is not connected. Module 1 creates no endpoint, model execution or fabricated demo.

## Dataset and evaluation decisions

- Use permitted vehicle images with plate bounding boxes and verified plate text.
- Split by vehicle/plate identity and capture session before augmentation.
- Prevent source and augmented copies from crossing split boundaries. Avoid
  text-mirroring augmentation in the OCR task.
- Report detector precision/recall, end-to-end exact plate match, character error
  rate, read coverage, wrong accepted reads and measured per-image latency.
- Count missed detections and unreadable results in end-to-end evaluation.
- Freeze thresholds on validation data before final held-out evaluation.
- Evaluate rectification only where plate-corner evidence supports it. Compare
  original/processed crops and retain unreadable outcomes for lost detail.

## Application boundary

Use close-range vehicle or gate-camera images for ANPR. Existing overhead lot
footage remains an occupancy input, not evidence of readable plate characters.
Keep gate observations independent of bay assignments unless a real association
is available. The old ANPR exclusion in privacy/product documents must be revised
when the feature is integrated, with explicit local storage/retention choices.

The research contribution to measure is whether quality assessment and
rectification reduce wrong accepted reads at a useful coverage level. Neither
academic novelty nor production accuracy is established by adding OCR alone.
