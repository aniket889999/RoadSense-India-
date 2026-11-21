# RoadSense ANPR Module 2 — Gemini Handoff

## Git State

- **Branch**: `feat/anpr-foundation`
- **Full HEAD SHA**: `a1fad77651f2a91d55accca9e80bedec58a9a517`
- **Whitespace / Patch Check (`git diff --check`)**: Clean (exit code `0`, no output)

## Preserved Implementation Files & Git Status

All five Module 2 files exist on disk unmodified since test verification and are currently untracked (`??`) for Codex review and integration:

| File | Git Status | Role |
| --- | --- | --- |
| `src/anpr/detector.py` | `??` (untracked) | Dedicated single-class (`license_plate`) Ultralytics YOLO detector adapter with `models/` confinement, no-symlink verification, and SHA-256 verification before loading |
| `src/anpr/preprocessing.py` | `??` (untracked) | Bounded OpenCV crop extraction and preprocessing (`PreprocessedPlateCrop`) preserving `original_bgr` alongside bounded resize, grayscale, bilateral filter, and CLAHE views |
| `src/anpr/ocr.py` | `??` (untracked) | Lazy-imported EasyOCR recognizer adapter (`download_enabled=False`) with `models/` confinement, local `.pth`/`.pt` weight checks, two-line row ordering, and conservative aggregate score |
| `src/anpr/local_runtime.py` | `??` (untracked) | Exports `build_local_runtime() -> AnprRuntime` composing `UltralyticsPlateDetector` and `EasyOcrPlateRecognizer` |
| `tests/test_anpr_adapters.py` | `??` (untracked) | Focused boundary, confinement, class-guard, preprocessing, row-ordering, and offline-runtime unit tests |

## Runtime Interfaces & Required Environment Variables

### Shared Interfaces (`src/anpr/runtime_contracts.py`)
- `PlateDetection(bbox: PlateBox, confidence: float)`
- `OcrRead(raw_text: str, confidence: float | None)`
- `detector.detect(image_bgr: np.ndarray) -> Sequence[PlateDetection]` (`UltralyticsPlateDetector.detect`)
- `recognizer.recognize(plate_bgr: np.ndarray) -> OcrRead` (`EasyOcrPlateRecognizer.recognize`)
- `AnprRuntime(detector=..., recognizer=...)`
- `RuntimeUnavailable` raised whenever a required package, environment variable, confined file/directory, SHA-256 digest, or single-class schema (`{0: "license_plate"}`) is missing or incompatible.
- `build_local_runtime() -> AnprRuntime` exported by `src/anpr/local_runtime.py`.

### Required & Optional Environment Variables
- `ROADSENSE_ANPR_PLATE_WEIGHTS` *(required)*: Repository-relative path under `models/` pointing to a regular, non-symlink YOLO checkpoint file whose class map is strictly `{0: "license_plate"}`.
- `ROADSENSE_ANPR_PLATE_SHA256` *(required)*: Expected 64-character lowercase hex SHA-256 digest verified before importing/constructing `YOLO`.
- `ROADSENSE_ANPR_OCR_MODEL_DIR` *(required)*: Repository-relative directory under `models/` containing non-empty, non-symlink EasyOCR weight files (`.pth`/`.pt`), loaded with `download_enabled=False`.
- `ROADSENSE_ANPR_DEVICE` *(optional, default `"cpu"`)*: Explicit inference device (`"cpu"`, `"mps"`, or `"cuda"`), validated without silent fallback.

## Verification Record

### Previously Executed Checks (During Module 2 Implementation Turn)
- `.venv/bin/pytest -q tests/test_anpr_adapters.py`: **14 passed in 0.77s**.

### New Checks Executed in This Handoff Turn
- `git branch --show-current`: Verified active branch is `feat/anpr-foundation`.
- `git rev-parse HEAD`: Verified HEAD is `a1fad77651f2a91d55accca9e80bedec58a9a517`.
- `git status --short` & `ls -la`: Confirmed all five implementation files exist on disk and are preserved without modification.
- `git diff --check`: Passed with zero whitespace or formatting errors.
- *(Per handoff instructions, completed pytest runs were not re-executed.)*

## Remaining Blockers & Real-Image Measurement Status

1. **Dedicated Plate-Detector Checkpoint**: No single-class `{0: "license_plate"}` YOLO checkpoint is staged under `models/` yet, and `ROADSENSE_ANPR_PLATE_WEIGHTS` / `ROADSENSE_ANPR_PLATE_SHA256` are not configured (pothole and general vehicle checkpoints are explicitly rejected).
2. **EasyOCR Package**: `easyocr` is not installed in the local `.venv`.
3. **Offline EasyOCR Weights**: Local EasyOCR detector/recognizer `.pth` weight files are not yet staged under `models/` (`ROADSENSE_ANPR_OCR_MODEL_DIR`).
4. **Accuracy Status**: Real-image detection/OCR accuracy has **not** yet been measured; all Module 2 verification used deterministic lightweight fakes to validate adapter boundaries and fail-closed `RuntimeUnavailable` behavior.

## Codex Integration Review — 2026-11-21

The report above preserves Gemini's original handoff at `a1fad77`. Codex then
reviewed the five implementation files and made these focused corrections:

- Reject fractional/boolean or duplicate class-map IDs instead of coercing them
  to an apparently valid single-class schema.
- Reject OCR text longer than the shared 256-character contract instead of
  truncating raw evidence and potentially hiding trailing characters.
- Remove a duplicate OCR model-directory scan and close its iterator explicitly.
- Add regressions for these failures and a composed image-decoding, detector,
  preprocessing, OCR and validation test using fake ML engines.

Independent combined verification: **61 tests passed in 1.17s** across the ANPR
foundation, pipeline, adapters, API and OpenAPI tests. Python compilation and
`npm run typecheck` passed. This is not a measurement of real model accuracy.

The frontend production build was attempted but could not complete: the local
`@next/swc-darwin-arm64` installation contains its manifest and README but is
missing the native `.node` compiler binary. Next.js attempted a download and
stopped at a sandbox cache-write restriction. No new compiler or package was
installed to repair it. Production-build verification remains pending that fix.

The original model/package blockers remain unchanged. Source media, weights,
datasets, stashes and IDE configuration directories were preserved.
