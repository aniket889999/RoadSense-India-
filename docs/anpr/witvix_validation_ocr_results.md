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
