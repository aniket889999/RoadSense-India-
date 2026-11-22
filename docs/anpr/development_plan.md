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

Latest OCR updates:
- [Demonstration and jury guide](demo_and_jury_guide.md): crop-quality review cues,
  enlarged plate preview, annotated PNG and review JSON downloads, and explicit
  review-only policy. ANPR schema 1.1 adds advisory measurements while the reader
  still accepts saved 1.0 results. The held-out test split remains untouched.
- [Recognition model comparison (EasyOCR vs. RapidOCR PP-OCRv4 ONNX)](ocr_model_comparison_results.md):
  Off-the-shelf `rapidocr_onnxruntime==1.4.4` (`PP-OCRv4` ONNX staged offline
  under `models/rapidocr/`, selected on the 40-image training subset where exact
  reads rose from `5/50` to `19/50`) achieved **`37/94` (`39.36%`)** exact
  full-plate agreement and **`24.07%` CER** (`220/914` character edits; `15/31`
  = `48.39%` exact on `single_line`, `22/63` = `34.92%` on `two_line`, `30/65` =
  `46.15%` on `other`/day, `7/29` = `24.14%` on `low_light`) across all 71
  validation images in review-only mode (`confidence=None`, `0` wrong accepted
  reads).
- [Local recognizer adaptation](ocr_adaptation_results.md):
  Fine-tuning the existing EasyOCR `english_g2` checkpoint on single-line
  training crops increased validation complete-read agreement from `3/94` to
  `22/94` (`23.40%`) and reduced CER from `45.19%` to `30.09%`. The detector is
  unchanged; the held-out test split remains untouched. Use
  `scripts/serve_anpr_local.py --engine easyocr` to launch the API with the pinned adapted
  EasyOCR checkpoint already staged locally. The launcher defaults to review-only
  RapidOCR with `resized_bgr` crops; manual configuration can also set
  `ROADSENSE_ANPR_OCR_ENGINE=rapidocr` with
  `ROADSENSE_ANPR_OCR_MODEL_DIR=models/rapidocr` to use the review-only
  RapidOCR adapter.

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
  detector training wrapper.
- Module 3 combined verification passes 93 focused tests and Python compilation.
  The local base checkpoint SHA matches its training config. A real project
  dry-run initially stopped before training because `anpr_india_v1` had not been
  prepared.
- Witvix intake now prepares a real, limited truck-plate pilot:
  329 train / 71 val / 70 test images, with 408 / 94 / 87 plate annotations.
  Real training preflight passes using the pinned local base checkpoint. Dataset
  schema 1.1 records mixed publisher labels and unverified capture-session
  independence explicitly; these are not certified ground-truth labels or capture
  groups. See [Witvix intake results](witvix_intake_results.md).
- Combined ANPR/intake/API/OpenAPI verification after this intake: 102 tests
  passed, changed Python modules compiled, and whitespace checks passed.
- Module 3B real plate-detector baseline training completed on Apple M4 MPS
  (`configs/training/anpr_witvix_mps_baseline_v1.yaml`, run
  `outputs/training/anpr/anpr_witvix_detector_mps_v1`, 50/50 epochs in
  `2107.23s`). The trained single-class `license_plate` checkpoint
  `weights/best.pt` (SHA-256
  `37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db`, peak
  epoch 36) recorded `val` box precision `0.99752` (`0.998` fused), recall
  `0.94681` (`0.947` fused), `mAP50` `0.99217` (`0.992` fused) and `mAP50-95`
  `0.84655` (`0.847` fused); final epoch 50 (`weights/last.pt`) recorded `val`
  precision `0.97894`, recall `0.98893`, `mAP50` `0.99347` and `mAP50-95`
  `0.83607`. See
  [Witvix detector baseline results](witvix_detector_baseline_results.md).
- Module 4 offline runtime provisioning and validation-split OCR evaluation
  completed: `easyocr==1.7.2` installed in `.venv` alongside unmodified
  `torch==2.13.0`, `torchvision==0.28.0`, `ultralytics==8.4.124` and
  `opencv-python-headless==5.0.0.93`; trained detector staged at
  `models/anpr/anpr_witvix_detector_mps_v1_best.pt` and official EasyOCR English
  assets staged at `models/easyocr/craft_mlt_25k.pth` and
  `models/easyocr/english_g2.pth` (`download_enabled=False`). Across all 71
  validation images (`94` `publisher_mixed` plates), the composed pipeline
  (`scripts/generate_anpr_predictions.py` + `scripts/evaluate_anpr_predictions.py`)
  achieved detection precision `0.9681` (`91/94`), detection recall `0.9681`
  (`91/94`), read coverage `0.9574` (`90/94`), end-to-end exact plate agreement
  `0.0319` (`3/94`), character error rate `0.4584` (`419/914`), `0` wrong
  accepted reads (`91` plates routed to `NEEDS_REVIEW`, `3` `UNREADABLE`), and
  mean latency `324.61 ms/image`. Held-out `test` split remains untouched. See
  [Witvix validation OCR results](witvix_validation_ocr_results.md).
- Module 4 web dashboard integration and corrective OCR audit:
  `AnprInspectionPanel` is connected to `POST /api/v1/anpr/images/recognize` via
  `recognizeAnprImage` in `apps/web/src/lib/api.ts` (raw JPEG/PNG bytes with
  matching `Content-Type`) and exposed under the `ANPR Inspection` tab in
  `Sidebar.tsx` and `page.tsx`. The missing local
  `apps/web/node_modules/@next/swc-darwin-arm64/next-swc.darwin-arm64.node`
  binary (`@next/swc-darwin-arm64@14.2.18`) was restored from the lockfile's
  exact URL and verified against its lockfile SHA-512 integrity hash; both
  `npm run typecheck` and `npm run build` pass, and the live browser workflow
  (`http://localhost:3000` -> `http://127.0.0.1:8000`) was verified end-to-end
  with real validation images (`witvix-img_0137.png` and `witvix-img_0003.png`).
  A 40-image (`50`-plate) training-subset diagnosis showed that detector box
  crops are not the primary bottleneck (reference boxes scored identically to
  predicted boxes at `2/50` exact match under `enhanced_grayscale`), while
  pre-applying bilateral filtering + CLAHE amplified border/bolt noise. Switching
  EasyOCR's default view to `original_bgr` (`C4_original_bgr`, selected on the
  training subset where CER dropped from `0.5052` to `0.4062`) reduced
  validation-split (`71` images, `94` plates) CER from `0.4584` (`419/914`) to
  `0.4519` (`413/914`) and produced `1` accepted `READ` (`0` wrong accepted
  reads), while overall validation exact plate agreement remained `3/94`
  (`0.0319`) due to generic `english_g2` character confusions on two-line
  commercial truck plates. See
  [Witvix validation OCR results](witvix_validation_ocr_results.md).

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
