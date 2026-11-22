# Module 4: Offline EasyOCR Runtime Provisioning and Witvix Validation Split Evaluation

## Scope and Provenance Boundary

This document records the provisioning of the local offline ANPR runtime
(single-class YOLO plate detector + offline EasyOCR English text detector and
recognizer) and its measured end-to-end evaluation across all `71` images (`94`
ground-truth plate annotations) of the prepared Witvix validation split (`val`
in `data/processed/witvix_anpr_pilot_v1`).

- **Metric interpretation**: `agreement_with_supplied_labels`. All `94`
  validation plate annotations carry `label_provenance: "publisher_mixed"`
  (`all_plate_texts_human_verified: false`), and grouping uses
  `grouping_basis: "publisher_vehicle_proxy"`
  (`capture_session_disjointness: "not_established_vehicle_proxy"`). These
  measurements represent agreement with the publisher's supplied labels on the
  71-image validation split, **not** certified real-world Indian ANPR accuracy.
- **Held-out test split preserved**: The held-out `test` split (`70` images,
  `87` plates) was **not** inspected, run through inference, or evaluated.
- **No registration numbers printed**: To comply with privacy and repository
  hygiene rules, actual license-plate strings from the dataset or OCR predictions
  are never committed to Git. Frozen predictions and metric JSON outputs remain
  in ignored `outputs/anpr/` files.

---

## 1. Local Runtime Assets, Licences, and Integrity Hashes

All model weights are stored under ignored `models/` subdirectories and verified
locally before inference. Normal recognition runs strictly with
`download_enabled=False`.

### A. Dedicated Single-Class Plate Detector (`models/anpr/`)
- **Staged inference path**: `models/anpr/anpr_witvix_detector_mps_v1_best.pt`
- **Source artifact**: Copied unmodified from
  `outputs/training/anpr/anpr_witvix_detector_mps_v1/weights/best.pt` (Module 3B
  50-epoch Apple M4 MPS baseline run).
- **File size**: `6,221,034` bytes
- **Locally calculated SHA-256 (and required `ROADSENSE_ANPR_PLATE_SHA256`)**:
  `37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db`
- **Verified class schema**: `{0: "license_plate"}`

### B. Installed EasyOCR Package and Added Python Dependencies
Installed into `.venv` (`Python 3.12.7`) without modifying or replacing existing
`torch==2.13.0`, `torchvision==0.28.0`, `ultralytics==8.4.124`,
`opencv-python==5.0.0.93`, or `opencv-python-headless==5.0.0.93`:
- `easyocr==1.7.2` (Apache-2.0 licence)
- `scipy==1.18.1`
- `scikit-image==0.26.0`
- `Shapely==2.2.0`
- `pyclipper==1.4.0`
- `python-bidi==0.6.12`
- `imageio==2.38.2`
- `tifffile==2026.11.16`
- `lazy-loader==0.6`
- `ninja==1.13.2`

### C. Offline EasyOCR English Weights (`models/easyocr/`)

EasyOCR `1.7.2` configured for English (`languages=("en",)`, `recog_network="english_g2"`)
requires two `.pth` files inside `ROADSENSE_ANPR_OCR_MODEL_DIR`:

| Role | Filename (`models/easyocr/`) | Official Source URL (EasyOCR GitHub Releases) | Licence | Publisher Integrity Check (MD5 in `easyocr.config`) | Locally Calculated `.pth` MD5 | Locally Calculated `.pth` SHA-256 | `.pth` Size (Bytes) | Locally Calculated `.zip` SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| Text detector (CRAFT) | `craft_mlt_25k.pth` | `https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip` | EasyOCR Apache-2.0 / Clova AI CRAFT MIT | `2f8227d2def4037cdb3b34389dcf9ec1` | `2f8227d2def4037cdb3b34389dcf9ec1` (exact match) | `4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17` | `83,152,330` | `8dc6a1c703a89ed56308ef742d26ebd45c656248cbbbda6e7fe60e569f873e65` |
| English Gen2 recognizer | `english_g2.pth` | `https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip` | EasyOCR Apache-2.0 | `5864788e1821be9e454ec108d61b887d` | `5864788e1821be9e454ec108d61b887d` (exact match) | `e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8` | `15,143,997` | `1b5eaebf1c062de6205560c97ffcfa8dc0e6f413c340e8adc5cfc57e159f61ff` |

---

## 2. Reproducible Environment Configuration and Launch Commands

### Required Environment Variables
```bash
export ROADSENSE_ANPR_PLATE_WEIGHTS="models/anpr/anpr_witvix_detector_mps_v1_best.pt"
export ROADSENSE_ANPR_PLATE_SHA256="37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db"
export ROADSENSE_ANPR_OCR_MODEL_DIR="models/easyocr"
export ROADSENSE_ANPR_DEVICE="mps"   # or "cpu"
```

### Reproducible FastAPI Launch Command
```bash
ROADSENSE_ANPR_PLATE_WEIGHTS=models/anpr/anpr_witvix_detector_mps_v1_best.pt \
ROADSENSE_ANPR_PLATE_SHA256=37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db \
ROADSENSE_ANPR_OCR_MODEL_DIR=models/easyocr \
ROADSENSE_ANPR_DEVICE=mps \
.venv/bin/uvicorn services.api.app.main:app --host 127.0.0.1 --port 8000
```

### Reproducible Validation Batch Prediction & Evaluation Commands
```bash
# 1. Generate frozen predictions across all 71 validation images
ROADSENSE_ANPR_PLATE_WEIGHTS=models/anpr/anpr_witvix_detector_mps_v1_best.pt \
ROADSENSE_ANPR_PLATE_SHA256=37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db \
ROADSENSE_ANPR_OCR_MODEL_DIR=models/easyocr \
ROADSENSE_ANPR_DEVICE=mps \
.venv/bin/python scripts/generate_anpr_predictions.py \
  --dataset data/processed/witvix_anpr_pilot_v1 \
  --split val \
  --output outputs/anpr/witvix_val_predictions_mps_v1.jsonl

# 2. Score frozen validation predictions against prepared manifest
.venv/bin/python scripts/evaluate_anpr_predictions.py \
  --dataset data/processed/witvix_anpr_pilot_v1 \
  --predictions outputs/anpr/witvix_val_predictions_mps_v1.jsonl \
  --split val \
  --output outputs/anpr/witvix_val_metrics_mps_v1.json
```

---

## 3. Measured Validation Split (`val`) End-to-End Results

- **Dataset fingerprint**:
  `2c33c828642ba484c4698e561df47193975f491291718a44c2d8d853e0c94a4a`
- **Local predictions file (ignored by Git)**:
  `outputs/anpr/witvix_val_predictions_mps_v1.jsonl`
  (SHA-256: `44288878c4ed4faff74053f4924d658a56caf6e66f6c53849d8100e90653a24c`)
- **Local provenance file (ignored by Git)**:
  `outputs/anpr/witvix_val_predictions_mps_v1.provenance.json`
- **Local metrics report (ignored by Git)**:
  `outputs/anpr/witvix_val_metrics_mps_v1.json`
  (evaluation config SHA-256:
  `4d4483288193d0dba0b1e34ae25275953f57fdf5d0d87c084e6e8ac0260b9194`)
- **Minimum OCR confidence threshold**: `0.75` (uncalibrated engine score policy)
- **Box matching policy**: `greedy_descending_iou_without_text_v1` at IoU `0.5`

### Overall and Per-Condition Validation Metrics (Agreement with `publisher_mixed` Labels)

| Metric | Overall (`val`) | Condition: `other` (Day) | Condition: `low_light` (Night) |
| --- | ---: | ---: | ---: |
| **Images in denominator** | `71` | `47` | `24` |
| **Ground-truth plates** | `94` | `65` | `29` |
| **Predicted plate boxes** | `94` | `66` | `28` |
| **Matched detections (IoU ≥ 0.5)** | `91` | `64` | `27` |
| **False-positive detections** | `3` | `2` | `1` |
| **Missed ground-truth plates** | `3` | `1` | `2` |
| **Detection precision** | `0.9681` (`91 / 94`) | `0.9697` (`64 / 66`) | `0.9643` (`27 / 28`) |
| **Detection recall** | `0.9681` (`91 / 94`) | `0.9846` (`64 / 65`) | `0.9310` (`27 / 29`) |
| **Plates with non-empty OCR text** | `90` | `64` | `26` |
| **Read coverage (non-empty text / GT)** | `0.9574` (`90 / 94`) | `0.9846` (`64 / 65`) | `0.8966` (`26 / 29`) |
| **Exact plate match count (raw/norm text == GT)** | `3` | `3` | `0` |
| **End-to-end exact plate match rate** | `0.0319` (`3 / 94`) | `0.0462` (`3 / 65`) | `0.0000` (`0 / 29`) |
| **Expected characters / character edits** | `914` / `419` | `637` / `257` | `277` / `162` |
| **Character error rate (CER)** | `0.4584` (`419 / 914`) | `0.4035` (`257 / 637`) | `0.5848` (`162 / 277`) |
| **Accepted `READ` plates (`conf ≥ 0.75` & `MATCHED`)** | `0` | `0` | `0` |
| **Acceptance coverage (`accepted_matched / GT`)** | `0.0000` (`0 / 94`) | `0.0000` (`0 / 65`) | `0.0000` (`0 / 29`) |
| **Wrong accepted reads** | `0` | `0` | `0` |
| **Wrong accepted read rate** | `null` (`0 / 0`) | `null` (`0 / 0`) | `null` (`0 / 0`) |
| **Missing / unavailable images** | `0` / `0` | `0` / `0` | `0` / `0` |
| **Mean per-image processing latency** | `324.61 ms` (`n=71`) | `372.04 ms` (`n=47`) | `231.75 ms` (`n=24`) |

### Breakdown of Validation Image and Plate Statuses
- **Image-level `ResultStatus` (`71` images)**:
  - `COMPLETE`: `0` images
  - `NEEDS_REVIEW`: `69` images
  - `NO_PLATE`: `2` images
  - `UNAVAILABLE`: `0` images
- **Plate-level `ReadStatus` (`94` detected plates)**:
  - `READ`: `0` plates
  - `NEEDS_REVIEW`: `91` plates
  - `UNREADABLE`: `3` plates
- **Plate-level `FormatStatus` (`94` detected plates)**:
  - `MATCHED` (`IN_COMMON_V1`): `5` plates (all `5` had conservative `ocr_confidence < 0.75`, so they were held in `NEEDS_REVIEW` with `LOW_OCR_CONFIDENCE`)
  - `NO_MATCH`: `86` plates
  - `EMPTY`: `3` plates
- **Review `ReasonCode` occurrences across the `94` detected plates**:
  - `LOW_OCR_CONFIDENCE`: `89` plates
  - `FORMAT_NOT_SUPPORTED`: `86` plates
  - `NO_TEXT`: `3` plates

### Metadata-Selected Validation Examples (Selected by First Sorted `image_id` per Category)
To avoid cherry-picking successful reads, three representative validation images
were selected deterministically as the earliest sorted `image_id` matching
documented metadata criteria (`conditions` and `plates.csv` `layout`), with
registration strings omitted:

1. **Daylight multi-plate (`two_line` + `single_line`) — `witvix-img_0003`**
   (`conditions: ["other"]`, `2` ground-truth plates):
   - Detector found both plates (`detection_confidence`: `0.7737` and `0.9102`).
   - Preprocessed crops produced 10-character normalized OCR strings on both
     plates with uncalibrated OCR scores `0.1111` and `0.5959`.
   - Both plates were flagged `NEEDS_REVIEW` with
     `reason_codes: ["LOW_OCR_CONFIDENCE", "FORMAT_NOT_SUPPORTED"]` (overall
     image status `NEEDS_REVIEW`).
2. **Low-light `two_line` plate — `witvix-img_0128`**
   (`conditions: ["low_light"]`, `1` ground-truth `two_line` plate):
   - Detector found `plate-1` (`detection_confidence: 0.9402`, processing time
     `124.4 ms`).
   - EasyOCR returned a two-line candidate read (`ocr_confidence: 0.6837`),
     flagged `NEEDS_REVIEW` (`LOW_OCR_CONFIDENCE`, `FORMAT_NOT_SUPPORTED`).
3. **Low-light `single_line` plate — `witvix-img_0137`**
   (`conditions: ["low_light"]`, `1` ground-truth `single_line` plate):
   - Detector found `plate-1` (`detection_confidence: 0.8959`, processing time
     `339.2 ms`).
   - EasyOCR returned a candidate read (`ocr_confidence: 0.4888`), flagged
     `NEEDS_REVIEW` (`LOW_OCR_CONFIDENCE`, `FORMAT_NOT_SUPPORTED`).

### Diagnostic Observation
While the fine-tuned YOLOv8n detector achieves `96.81%` precision and `96.81%`
recall (`91 / 94` plates matched at IoU ≥ `0.5`), out-of-the-box English
EasyOCR (`english_g2`) without plate-specific character fine-tuning or corner
rectification frequently confuses visually similar characters on two-line yellow
commercial truck plates (`O/0`, `I/1`, `B/8`, `S/5`, `Z/2`) or includes bracket/border
artefacts. The conservative aggregate confidence policy (`min` across spans ≥
`0.75`) combined with the strict `IN_COMMON_V1` shape check routes `91 / 94`
detected plates to `NEEDS_REVIEW` and `3 / 94` to `UNREADABLE` (`0` wrong
accepted reads), preventing noisy OCR outputs from being silently accepted as
`COMPLETE`.

---

## 4. Corrective Module: Frontend SWC Restoration, OCR Root-Cause Diagnosis, and Candidate Comparison

### A. Narrow Frontend SWC Environment Repair
- **Observed cause**: `apps/web/node_modules/@next/swc-darwin-arm64/` had its
  `package.json` and `README.md` from `14.2.18`, but the native binary
  `next-swc.darwin-arm64.node` was missing on disk.
- **Repair action**: Fetched the exact lockfile-pinned tarball from
  `apps/web/package-lock.json`
  (`https://registry.npmjs.org/@next/swc-darwin-arm64/-/swc-darwin-arm64-14.2.18.tgz`,
  `37,194,831` bytes), verified its exact SHA-512 integrity against the lockfile
  (`sha512-tOBlDHCjGdyLf0ube/rDUs6VtwNOajaWV+5FV/ajPgrvHeisllEdymY/oDgv2cx561+gJksfMUtqf8crug7sbA==`,
  tarball SHA-256 `2fdcb35a1ed35944f09a13897b978a9451e14a8fac3c9f857b91dc5eb968aac1`),
  and extracted `next-swc.darwin-arm64.node` (`114,963,832` bytes, SHA-256
  `8bdfd15750f62783dd8371f784b90a417750d471f02f9762ac9d73b0c2413477`) into
  `apps/web/node_modules/@next/swc-darwin-arm64/next-swc.darwin-arm64.node`
  without modifying `package.json`, `package-lock.json`, or any other package.
- **Verification**:
  - `cd apps/web && npm run typecheck` (`tsc --noEmit`): **Passed** (`0` errors).
  - `cd apps/web && npm run build` (`next build` 14.2.18): **Passed** (compiled
    and generated all static routes cleanly).
  - `cd apps/web && npm run start` (`next start -p 3000`): **Ready in 315 ms** on
    `http://localhost:3000`.

### B. Training-Subset OCR Root-Cause Diagnosis (40 Training Images / 50 Plates)

A deterministic development subset of `40` training images (`50` ground-truth
plates) was selected by taking the first `10` sorted `image_id` records from
each of the four `train`-split metadata buckets:
- `('low_light', 'single_line')`: `witvix-img_0011` .. `witvix-img_0110` (`10` images)
- `('low_light', 'two_line')`: `witvix-img_0001` .. `witvix-img_0038` (`10` images)
- `('other', 'single_line')`: `witvix-img_0022` .. `witvix-img_0256` (`10` images)
- `('other', 'two_line')`: `witvix-img_0002` .. `witvix-img_0029` (`10` images)

#### Diagnostic Findings on the 50 Training Subset Plates
1. **Detector bounding boxes vs. reference boxes**:
   - The trained YOLOv8n detector matched all `50 / 50` reference plates at
     IoU ≥ `0.5`.
   - Running the baseline `enhanced_grayscale` OCR on **ground-truth reference
     boxes** yielded the exact same low exact-match count (`2 / 50` = `4.0%`,
     CER `0.4351`) as running on **predicted detector boxes** (`2 / 50` = `4.0%`,
     CER `0.5052`). Therefore, **detector box tightness is not the primary cause
     of low exact plate agreement**.
2. **Preprocessing degradation (`enhanced_grayscale` vs. un-filtered `original_bgr`)**:
   - EasyOCR's internal CRAFT detector and CRNN (`english_g2`) perform their own
     internal resizing and contrast normalization.
   - Pre-applying `cv2.bilateralFilter` + `CLAHE` (` clipLimit=2.0,
     tileGridSize=(4,4)`) on already small/compressed commercial-truck plate
     crops (median crop height `60 px`, 25th percentile `44 px`) amplifies plate
     bolts, frame borders, and background noise into false character strokes:
     `20 / 50` training plates contained non-alphanumeric punctuation noise under
     `enhanced_grayscale`.
3. **Out-of-the-box `english_g2` character recognition error taxonomy (on `original_bgr`)**:
   - `3 / 50` (`6.0%`): Exact plate match.
   - `15 / 50` (`30.0%`): Near-miss `1` or `2` character substitutions at the
     correct string length (`M->N`, `1->I`, `G->6`, `B->8`, `5->S`, `0->O`,
     `T->7`, `Q->0`) caused by hand-painted/weathered Indian commercial truck
     fonts and perspective skew.
   - `22 / 50` (`44.0%`): `3+` character errors at roughly the target length on
     low-resolution, motion-blurred, or angled plates.
   - `10 / 50` (`20.0%`): Length mismatch (`|len(obs) - len(exp)| >= 3`), split
     between CRAFT missing one line of a two-line plate (`5` plates) and CRAFT
     picking up extra frame/chassis markings or `IND` prefix text (`5` plates).
4. **Row grouping & scorer audit**:
   - On two-line plates, `order_and_merge_ocr_detections` grouped detected spans
     in top-to-bottom, left-to-right reading order without row-inversion bugs.
   - The evaluation scorer (`src/anpr/evaluation.py`) matched boxes and
     normalized text accurately.

### C. Predeclared Bounded OCR Candidate Experiments on the 40-Image Training Subset

Keeping the detector checkpoint, `IN_COMMON_V1` normalization rules, and `0.75`
acceptance threshold fixed, four predeclared `PreprocessedPlateCrop` views were
evaluated on the `40`-image (`50`-plate) training subset:

| Predeclared Candidate | Training Exact Match (`50` Plates) | Training CER (`485` Expected Chars) | Training `single_line` CER (`24` Plates) | Training `two_line` CER (`26` Plates) | Training `other` (Day) CER (`27` Plates) | Training `low_light` CER (`23` Plates) | Training Wrong Accepted Reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `C1_baseline_enhanced_gray` (`enhanced_grayscale`) | `2 / 50` (`4.0%`) | `0.5052` (`245 / 485`) | `0.4310` (`100 / 232`) | `0.5731` (`145 / 253`) | `0.4940` (`123 / 249`) | `0.5169` (`122 / 236`) | `0` |
| `C2_plain_grayscale` (`grayscale`) | `4 / 50` (`8.0%`) | `0.4515` (`219 / 485`) | `0.3836` (`89 / 232`) | `0.5138` (`130 / 253`) | `0.4458` (`111 / 249`) | `0.4576` (`108 / 236`) | `0` |
| `C3_resized_bgr` (`resized_bgr`) | `5 / 50` (`10.0%`) | `0.4392` (`213 / 485`) | `0.3879` (`90 / 232`) | `0.4862` (`123 / 253`) | `0.4297` (`107 / 249`) | `0.4492` (`106 / 236`) | `0` |
| **`C4_original_bgr` (`original_bgr`) — Selected** | **`5 / 50` (`10.0%`)** | **`0.4062` (`197 / 485`)** | **`0.3319` (`77 / 232`)** | **`0.4743` (`120 / 253`)** | **`0.4016` (`100 / 249`)** | **`0.4110` (`97 / 236`)** | **`0`** |

- **Selection decision (from training subset only)**: `C4_original_bgr` was
  selected because it achieved the lowest Character Error Rate across every
  training subgroup (`0.4062` overall vs. `0.5052` baseline) and the highest
  exact-match count (`5 / 50` vs. `2 / 50` baseline) with `0` wrong accepted
  reads. `EasyOcrPlateRecognizer` (`src/anpr/ocr.py`) was updated to use
  `crop_view="original_bgr"` by default (configurable via `crop_view` or
  `ROADSENSE_ANPR_OCR_CROP_VIEW`), preserving `preprocess_plate_crop` and falling
  back once to `enhanced_grayscale` only if `original_bgr` returns empty text.

### D. Validation Split (`71` Images, `94` Plates): Baseline (`v1`) vs. Selected Candidate (`v2`)

The selected `C4_original_bgr` configuration was run once across all `71`
validation images, writing new ignored files while preserving the v1 baseline:
- **New predictions**:
  `outputs/anpr/witvix_val_predictions_mps_v2_original_bgr.jsonl`
  (SHA-256: `334b5f2b5af82f42c8f437be4a03dd8c71875c15ea5550b591e18968ff62d1f8`)
- **New provenance**:
  `outputs/anpr/witvix_val_predictions_mps_v2_original_bgr.provenance.json`
- **New evaluation report**:
  `outputs/anpr/witvix_val_metrics_mps_v2_original_bgr.json`

#### Overall Validation Comparison (`71` Images, `94` `publisher_mixed` Reference Plates)

| Validation Metric | Baseline `v1` (`enhanced_grayscale`) | Selected `v2` (`original_bgr`) | Change (`v2 - v1`) |
| --- | ---: | ---: | ---: |
| **Images / Ground-truth plates** | `71` / `94` | `71` / `94` | Unchanged |
| **Predicted plates / Matched detections (IoU ≥ 0.5)** | `94` / `91` | `94` / `91` | Unchanged |
| **Detection precision** | `0.9681` (`91 / 94`) | `0.9681` (`91 / 94`) | `0.0000` |
| **Detection recall** | `0.9681` (`91 / 94`) | `0.9681` (`91 / 94`) | `0.0000` |
| **Read coverage (`plates_with_text / GT`)** | `0.9574` (`90 / 94`) | `0.9574` (`90 / 94`) | `0.0000` |
| **End-to-end exact plate match agreement** | `0.0319` (`3 / 94`) | `0.0319` (`3 / 94`) | `0.0000` |
| **Character error rate (CER)** | `0.4584` (`419 / 914`) | `0.4519` (`413 / 914`) | **`-0.0065` (`-6` char edits)** |
| **Accepted reads (`READ`: `conf ≥ 0.75` & `MATCHED`)** | `0` | `1` | **`+1`** |
| **Acceptance coverage (`accepted_matched / GT`)** | `0.0000` (`0 / 94`) | `0.0106` (`1 / 94`) | **`+0.0106`** |
| **Wrong accepted reads** | `0` (`rate: null`) | `0` (`rate: 0.0000`) | **`0` wrong accepted** |
| **Mean per-image processing latency** | `324.61 ms` | `323.28 ms` | `-1.33 ms` |

#### Validation Subgroup Comparison by Lighting Condition and Plate Layout

| Validation Subgroup | GT Plates | `v1` Exact Match | `v2` Exact Match | `v1` CER | `v2` CER | `v1` Accepted (Wrong) | `v2` Accepted (Wrong) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Condition: `other` (Day, `47` images)** | `65` | `3 / 65` (`4.62%`) | `1 / 65` (`1.54%`) | `0.4035` (`257 / 637`) | `0.4223` (`269 / 637`) | `0` (`0`) | `1` (`0`) |
| **Condition: `low_light` (Night, `24` images)** | `29` | `0 / 29` (`0.00%`) | `2 / 29` (`6.90%`) | `0.5848` (`162 / 277`) | `0.5199` (`144 / 277`) | `0` (`0`) | `0` (`0`) |
| **Layout: `single_line`** | `31` | `0 / 31` (`0.00%`) | `2 / 31` (`6.45%`) | `0.4702` (`142 / 302`) | `0.4106` (`124 / 302`) | `0` (`0`) | `0` (`0`) |
| **Layout: `two_line`** | `63` | `3 / 63` (`4.76%`) | `1 / 63` (`1.59%`) | `0.4526` (`277 / 612`) | `0.4722` (`289 / 612`) | `0` (`0`) | `1` (`0`) |

- **Honest assessment of validation outcome**: Switching from `enhanced_grayscale`
  (`v1`) to `original_bgr` (`v2`) modestly improved validation CER on
  `single_line` plates (`0.4702 -> 0.4106`, `0 -> 2` exact matches) and on
  `low_light` plates (`0.5848 -> 0.5199`, `0 -> 2` exact matches), reduced total
  validation character edits from `419` to `413` (`CER 0.4584 -> 0.4519`), and
  produced `1` verified `READ` (`0` wrong accepted reads). However, overall
  validation exact-plate agreement remained `3 / 94` (`3.19%`) because generic
  pre-trained English EasyOCR (`english_g2`) is fundamentally limited on angled,
  weathered, two-line commercial Indian truck plates without plate-character OCR
  fine-tuning or 4-corner perspective rectification.

---

## 5. Real Browser Workflow Verification (`http://localhost:3000` -> `http://127.0.0.1:8000`)

With `uvicorn services.api.app.main:app --host 127.0.0.1 --port 8000` and
`next start -p 3000` running, a live Chrome browser session navigated to
`http://localhost:3000`, selected the **ANPR Inspection** tab, and verified:

1. **Connected service state**: The `"ANPR STILL-IMAGE INSPECTION"` header,
   `"Profile: IN_COMMON_V1"` badge, and enabled `"Run Plate Recognition"` button
   rendered without the disconnected-service warning.
2. **First real validation image upload (`witvix-img_0137.png`, `1200×1600 px`)**:
   - Local preview rendered inside the letterboxed viewport showing
     `3.20 MiB • 1200×1600 px`.
   - Clicking **"Run Plate Recognition"** sent raw PNG bytes to
     `POST /api/v1/anpr/images/recognize` (`HTTP 200 OK`) and displayed:
     - Status banner: `NEEDS REVIEW` (`NEEDS_REVIEW`), Request ID
       `53447551-6ef2-4e82-9d90-2b06aa7434a3 • Schema v1.0`, frame summary
       `Normalized Frame: 1200×1600 px • 1 plate`.
     - Bounding box overlay `#1` aligned accurately over the truck's front
       license plate in the letterboxed viewport (`Box: (130, 475) -> (1035, 725) px`,
       `Size: 905×250 px`).
     - Plate `#1 (plate-1)` card showing raw OCR text, normalized text,
       `Format Status (IN_COMMON_V1): Unsupported format`,
       `OCR Score (Uncalibrated): 0.45`, `Detector score: 0.90`, and review
       reason codes `LOW_OCR_CONFIDENCE` and `FORMAT_NOT_SUPPORTED`.
3. **Second real validation image upload & stale-result clearing (`witvix-img_0003.png`, `963×1280 px`, 2 plates)**:
   - Selecting `witvix-img_0003.png` immediately cleared the previous result and
     reset the right panel to `"No recognition result to display"`.
   - Clicking **"Run Plate Recognition"** returned `NEEDS_REVIEW` in `250.3 ms`
     (`Normalized Frame: 963×1280 px • 2 plates`), drawing both `#1`
     (`Box: (278, 658) -> (353, 706) px`, `Detector score: 0.77`,
     `OCR Score: 0.25`) and `#2` (`Box: (242, 779) -> (351, 809) px`,
     `Detector score: 0.91`, `OCR Score: 0.62`) over the two plates on the
     displayed truck image with honest `NEEDS_REVIEW` badges and reason codes.
4. **Repository hygiene**: No screenshots or plate strings were saved in the
   repository working tree.
