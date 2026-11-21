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
- After reviewing a completed module, Codex pushes its verified commits to
  `https://github.com/aniket889999/RoadSense-India-` on the feature branch.
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

### Module 2 file ownership and upload contract

Codex owns `src/anpr/runtime_contracts.py`, `src/anpr/pipeline.py`,
`services/api/app/routers/anpr.py`, application registration, API documentation,
`tests/test_anpr_pipeline.py` and `services/api/tests/test_anpr_api.py`.
Gemini owns `src/anpr/detector.py`, `src/anpr/preprocessing.py`,
`src/anpr/ocr.py`, `src/anpr/local_runtime.py` and
`tests/test_anpr_adapters.py`.

Runtime interfaces are defined in `src/anpr/runtime_contracts.py`. Both adapters
receive contiguous uint8 BGR arrays. Detector bounds must refer to the original,
EXIF-normalized image; the pipeline passes a copied bounding-box crop to OCR.
`build_local_runtime()` in `local_runtime.py` composes both adapters lazily.
Missing local assets raise `RuntimeUnavailable`. Importing the API does not load
weights or start OCR. These adapters do not install or download dependencies.

`POST /api/v1/anpr/images/recognize` accepts **raw image bytes**, with
`Content-Type: image/jpeg` or `image/png`. It deliberately does not use multipart,
so the endpoint can enforce its streaming byte cap before decoding. The future
frontend callback should use `fetch(url, {method: 'POST', headers:
{'Content-Type': file.type}, body: file})`, not `FormData`.

Encoded images are limited to 10 MiB, decoded images to 12 million pixels and
8192 pixels per dimension, and detections to 16 per image. Decoding/model calls
are serialized within a service instance; a busy instance rejects another call.
Model execution runs off the API event loop. Images, crops and OCR text are not
persisted by this endpoint. It returns the existing `AnprImageResult` contract.
Corrupt/empty input returns HTTP 400, over-limit input 413, unsupported/mismatched
type 415, and busy/failed recognition 503. Missing runtime assets return HTTP 200
with `status: UNAVAILABLE`, warnings and no fabricated plate results.

Local adapter configuration agreed with Gemini:

- `ROADSENSE_ANPR_PLATE_WEIGHTS`: repository-relative checkpoint under `models/`.
- `ROADSENSE_ANPR_PLATE_SHA256`: expected SHA-256 of the configured checkpoint.
- `ROADSENSE_ANPR_OCR_MODEL_DIR`: repository-relative OCR assets under `models/`.
- `ROADSENSE_ANPR_DEVICE`: explicit detector device; default CPU.

The plate checkpoint must contain exactly one class named `license_plate`.
The existing pothole/vehicle models are not substituted for missing plate weights.
EasyOCR must disable automatic downloads. Actual OCR/plate detection and latency
remain unverified until the required local packages and assets are available.

## Verified progress

- Module 1 backend: `aaf9162`, 18 focused tests passed.
- Module 1 frontend: reviewed against the Python contract and independently
  typechecked; committed as `6e4893f`.
- Module 2 Codex work: endpoint, runtime interfaces and injected pipeline covered
  by 42 passing focused tests (including the Module 1 and OpenAPI checks).
- Module 2 Gemini adapters are reviewed and integrated. The combined ANPR and
  OpenAPI checks pass: 61 tests, including a composed adapter-to-pipeline test
  with fake ML engines. Review corrections prevent OCR evidence truncation and
  invalid class IDs being coerced into a valid plate-detector schema.
- TypeScript checking and Python compilation pass. The frontend production build
  is blocked by an absent `next-swc.darwin-arm64.node` binary in the installed
  `@next/swc-darwin-arm64` package. No compiler/dependency download was authorized
  or completed to repair that local environment.
- Real-image verification remains pending. The local audit found EasyOCR absent
  and no configured plate-detector checkpoint. Tests with fake ML engines
  establish orchestration correctness, not ANPR accuracy or a completed real-image
  recognition demonstration.
- Module 3 implementation and usage are documented in
  [Data, training and evaluation](module3_data_and_training.md). Codex implements
  grouped dataset preparation and offline metrics; Gemini supplies the dedicated
  detector training wrapper. Actual training waits for a labelled ANPR dataset.
- Module 3 combined verification passes 93 focused tests and Python compilation.
  The local base checkpoint SHA matches its training config. A real project
  dry-run stops before training because `anpr_india_v1` has not been prepared.

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
