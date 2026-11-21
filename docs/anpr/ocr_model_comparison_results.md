# ANPR Recognition Model Comparison: EasyOCR vs. RapidOCR (PP-OCRv4 ONNX)

## Scope and Provenance Boundary

This document records the bounded recognition-model comparison authorized after
confirming that the fine-tuned YOLOv8n plate detector (`96.81%` precision,
`96.81%` recall on `val`) is not the bottleneck for plate reading.

- **Detector & dataset frozen**: All comparisons use the identical single-class
  plate detector checkpoint (`models/anpr/anpr_witvix_detector_mps_v1_best.pt`,
  SHA-256 `37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db`)
  and the prepared Witvix pilot dataset (`data/processed/witvix_anpr_pilot_v1`,
  fingerprint `2c33c828642ba484c4698e561df47193975f491291718a44c2d8d853e0c94a4a`).
- **Held-out test split untouched**: No inference or inspection was performed on
  the held-out `test` split (`70` images, `87` plates).
- **Metric interpretation**: `agreement_with_supplied_labels`. All `94`
  validation annotations carry `label_provenance: "publisher_mixed"`
  (`all_plate_texts_human_verified: false`) and grouping uses
  `grouping_basis: "publisher_vehicle_proxy"`.
- **Review-only policy for the new engine**: Because different OCR engines
  produce uncalibrated scores on incompatible numerical scales,
  `RapidOcrPlateRecognizer` runs with `review_only=True` (`confidence=None`),
  routing all non-empty candidate reads to `NEEDS_REVIEW`
  (`OCR_CONFIDENCE_MISSING`) rather than blindly reusing EasyOCR's `0.75`
  threshold.

---

## 1. Research of Two Candidate Recognizers and Selection Justification

Two candidate recognition frameworks were evaluated from official documentation
and package metadata before implementation:

| Criterion | Candidate A: `fast-plate-ocr` (`ankandrew/fast-plate-ocr` v1.1.0, `cct-s-v2-global-model`) | Candidate B (Selected): `rapidocr_onnxruntime` (`RapidAI/RapidOCR` v1.4.4, PaddleOCR PP-OCRv4 ONNX) |
| --- | --- | --- |
| **Architecture** | Single-stage fixed-slot Compact Convolutional Transformer (`CCT`) classifying up to `max_plate_slots` parallel character heads from the whole crop | Multi-stage scene-text pipeline: DBNet polygon text-line detector (`ch_PP-OCRv4_det_infer.onnx`) + 0°/180° angle classifier (`ch_ppocr_mobile_v2.0_cls_infer.onnx`) + SVTR-LCNet line recognizer (`ch_PP-OCRv4_rec_infer.onnx`) |
| **Suitability for 9–10 char Indian registrations** | **Incompatible out-of-the-box**: Official `global_plate_config.yaml` sets `max_plate_slots: 9` (9 fixed character heads), truncating standard 10-character Indian registrations (`AA00AA0000`, e.g. `MH12AB1234` or `24BH1234AA`) | **Full variable-length support**: Sequence recognizer (`rec_img_shape: [3, 48, 320]`) decodes arbitrary-length alphanumeric strings without a 9-character slot ceiling |
| **Two-line commercial truck plate layout (`69%` of Witvix)** | Single-line crop classifier; requires external line segmentation or custom multi-line grid training | Built-in DBNet multi-line polygon detection detects upper and lower text lines separately; spans are then ordered top-to-bottom, left-to-right by `order_and_merge_ocr_detections` |
| **Licence** | MIT | Apache-2.0 (RapidOCR & Baidu PaddleOCR PP-OCRv4 weights) |
| **Hardware & `.venv` Compatibility** | Uses `onnxruntime` CPU; no conflicts | Uses `onnxruntime==1.32.0` CPU (`arm64` macOS wheel); installs cleanly alongside `torch==2.13.0`, `torchvision==0.28.0`, `ultralytics==8.4.124`, and `opencv-python-headless==5.0.0.93` without touching any existing package |

**Selection**: **`rapidocr_onnxruntime` (`1.4.4`) with offline PP-OCRv4 ONNX
weights** was selected because it natively handles both 10-character Indian
registration strings and two-line commercial truck plates (`69%` of the dataset)
without a 9-slot truncation limit.

---

## 2. Installed Packages and Staged Offline ONNX Assets

### A. Installed `.venv` Packages (No Existing Packages Replaced)
- `rapidocr_onnxruntime==1.4.4` (Apache-2.0)
- `onnxruntime==1.32.0` (MIT)
- `flatbuffers==25.12.19` (Apache-2.0)
- `tqdm==4.70.1` (MPL-2.0 / MIT)

### B. Offline PP-OCRv4 ONNX Weights Staged Under `models/rapidocr/` (Ignored by Git)

Extracted from the official `rapidocr_onnxruntime==1.4.4` PyPI wheel
(`https://pypi.org/project/rapidocr-onnxruntime/1.4.4/`, upstream
`https://github.com/RapidAI/RapidOCR` / Baidu PaddleOCR PP-OCRv4 Apache-2.0) and
staged under `models/rapidocr/` so `RapidOcrPlateRecognizer` loads explicit
repository-confined paths with zero network calls:

| Role | Staged Path (`models/rapidocr/`) | File Size (Bytes) | Locally Calculated SHA-256 | Locally Calculated MD5 |
| --- | --- | ---: | --- | --- |
| Text-line detector (PP-OCRv4 DBNet) | `models/rapidocr/ch_PP-OCRv4_det_infer.onnx` | `4,745,517` | `d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9` | `70e8929d005459b82f8116451caf9e92` |
| Text-line angle classifier (MobileNetV3) | `models/rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx` | `585,532` | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` | `c99c6bfc2df6544427ad3c7c85feab36` |
| Text-line recognizer (PP-OCRv4 SVTR-LCNet) | `models/rapidocr/ch_PP-OCRv4_rec_infer.onnx` | `10,857,958` | `48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b` | `8e4817a3f31c8be29d3b25393eb1b579` |

---

## 3. Training Development Subset Comparison (`40` Images, `50` Plates)

On the deterministic `40`-image (`50`-plate) `train` development subset (`10`
images from each of `(low_light, single_line)`, `(low_light, two_line)`,
`(other, single_line)`, and `(other, two_line)`), recognition was separated from
detection by evaluating both predicted detector crops and ground-truth reference
crops (diagnostic only):

| Engine & Crop Configuration (`40` Train Images, `50` Plates) | Exact Plate Match (`50` Plates) | Character Error Rate (`485` Chars) | `single_line` Exact / CER (`24` Plates) | `two_line` Exact / CER (`26` Plates) | `other` (Day) Exact / CER (`27` Plates) | `low_light` (Night) Exact / CER (`23` Plates) | Non-Empty Read Coverage | `IN_COMMON_V1` Shape Matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **EasyOCR (`english_g2`, `original_bgr`) on predicted crops** | `5 / 50` (`10.0%`) | `0.4062` (`197 / 485`) | `2 / 24` (`8.3%`) / `0.3896` | `3 / 26` (`11.5%`) / `0.4213` | `5 / 27` (`18.5%`) / `0.3245` | `0 / 23` (`0.0%`) / `0.5045` | `50 / 50` (`100.0%`) | `7` |
| *Diagnostic only*: EasyOCR (`english_g2`) on reference-box crops | `3 / 50` (`6.0%`) | `0.4309` (`209 / 485`) | `1 / 24` (`4.2%`) / `0.4113` | `2 / 26` (`7.7%`) / `0.4488` | `2 / 27` (`7.4%`) / `0.3547` | `1 / 23` (`4.3%`) / `0.5227` | `50 / 50` (`100.0%`) | `7` |
| **RapidOCR (`PP-OCRv4`, `original_bgr`) on predicted crops** | `18 / 50` (`36.0%`) | `0.2969` (`144 / 485`) | `9 / 24` (`37.5%`) / `0.2338` | `9 / 26` (`34.6%`) / `0.3543` | `12 / 27` (`44.4%`) / `0.2604` | `6 / 23` (`26.1%`) / `0.3409` | `49 / 50` (`98.0%`) | `20` |
| *Diagnostic only*: RapidOCR (`PP-OCRv4`, `original_bgr`) on reference-box crops | `16 / 50` (`32.0%`) | `0.2825` (`137 / 485`) | `8 / 24` (`33.3%`) / `0.2165` | `8 / 26` (`30.8%`) / `0.3425` | `9 / 27` (`33.3%`) / `0.2604` | `7 / 23` (`30.4%`) / `0.3091` | `49 / 50` (`98.0%`) | `17` |
| **RapidOCR (`PP-OCRv4`, `resized_bgr`) on predicted crops — Frozen Selection** | **`19 / 50` (`38.0%`)** | **`0.2969` (`144 / 485`)** | **`9 / 24` (`37.5%`) / `0.2381`** | **`10 / 26` (`38.5%`) / `0.3504`** | **`12 / 27` (`44.4%`) / `0.2642`** | **`7 / 23` (`30.4%`) / `0.3364`** | **`49 / 50` (`98.0%`)** | **`19`** |

### Training-Subset Diagnostic & Label Audit Observations
1. **Cross-engine vs. reference-label agreement audit**:
   - Out of `50` training plates: `3` matched the reference label in both
     engines, `15` matched the reference label in RapidOCR only, `2` matched in
     EasyOCR only, and `30` differed from the reference label and between the
     two engines (`0` plates where un-adapted EasyOCR and RapidOCR agreed on an
     identical non-reference string on `resized_bgr`; `1` single-character
     near-miss on `original_bgr` in `witvix-img_0065` where both engines had edit
     distance `1` against the `publisher_mixed` reference transcription).
   - No source annotations were modified or discarded, and AI agreement is not
     treated as human verification.
2. **Two-line layout handling**:
   - RapidOCR's DBNet text detector segments the top and bottom lines of
     two-line commercial truck plates into separate polygon spans, which
     `order_and_merge_ocr_detections` groups top-to-bottom, left-to-right. On the
     `26` two-line training plates, exact plate matches jumped from `3 / 26`
     (`11.5%`, EasyOCR) to `10 / 26` (`38.5%`, RapidOCR `resized_bgr`).
   - When DBNet finds no polygon on a tightly cropped single-line plate,
     `RapidOcrPlateRecognizer` falls back once to direct line recognition
     (`use_det=False, use_cls=False, use_rec=True`), achieving `49 / 50`
     non-empty reads on the training subset.
3. **Training-only configuration freeze**:
   - `RapidOcrPlateRecognizer` with `crop_view="resized_bgr"` (falling back once
     to `original_bgr` if empty) and `review_only=True` was frozen from the
     training subset (`19 / 50` exact matches, `0.2969` CER).

---

## 4. Full Validation Split (`71` Images, `94` Plates) Comparison

The frozen RapidOCR configuration was evaluated once across all `71` validation
images (`94` `publisher_mixed` reference plates), preserving all prior prediction
files and denominators:

- **New predictions file (ignored by Git)**:
  `outputs/anpr/witvix_val_predictions_rapidocr_v1.jsonl`
  (SHA-256: `cd851c83ed2c596f6754502b18c8edf127c57479e67d296da324eebbeb5a87a7`)
- **New provenance file (ignored by Git)**:
  `outputs/anpr/witvix_val_predictions_rapidocr_v1.provenance.json`
- **New evaluation report (ignored by Git)**:
  `outputs/anpr/witvix_val_metrics_rapidocr_v1.json`

### A. Overall Validation Metrics (`71` Images, `94` Reference Plates)

| Validation Metric | Un-adapted EasyOCR `v1` (`enhanced_grayscale`) | Un-adapted EasyOCR `v2` (`original_bgr`) | Adapted EasyOCR `v3` (`english_g2_witvix_v1.pth`) | **Selected Alternative: Off-the-Shelf RapidOCR (`PP-OCRv4 ONNX`, `resized_bgr`, Review-Only)** |
| --- | ---: | ---: | ---: | ---: |
| **Detector device / OCR device** | `mps` / `mps` | `mps` / `mps` | `mps` / `mps` | `mps` / `cpu` (ONNX Runtime) |
| **Images / Ground-truth plates** | `71` / `94` | `71` / `94` | `71` / `94` | `71` / `94` |
| **Predicted plates / Matched detections (IoU ≥ 0.5)** | `94` / `91` | `94` / `91` | `94` / `91` | `94` / `91` |
| **False positives / Missed plates** | `3` / `3` | `3` / `3` | `3` / `3` | `3` / `3` |
| **Detection precision / recall** | `0.9681` / `0.9681` | `0.9681` / `0.9681` | `0.9681` / `0.9681` | `0.9681` / `0.9681` |
| **Plates with non-empty OCR text** | `90 / 94` (`0.9574`) | `90 / 94` (`0.9574`) | `90 / 94` (`0.9574`) | **`91 / 94` (`0.9681`, 100% of matched detections)** |
| **Exact full-plate agreement (`correct_plate_reads`)** | `3 / 94` (`3.19%`) | `3 / 94` (`3.19%`) | `22 / 94` (`23.40%`) | **`37 / 94` (`39.36%`)** |
| **Character edits / Expected characters** | `419 / 914` | `413 / 914` | `275 / 914` | **`220 / 914`** |
| **Character Error Rate (CER)** | `0.4584` (`45.84%`) | `0.4519` (`45.19%`) | `0.3009` (`30.09%`) | **`0.2407` (`24.07%`)** |
| **`IN_COMMON_V1` format shape matches** | `5 / 94` | `5 / 94` | `28 / 94` | **`41 / 94`** |
| **Accepted `READ` plates** | `0` | `1` | `10` | `0` *(review-only: `confidence=None`)* |
| **Wrong accepted reads** | `0` | `0` | `0` | `0` (`rate: null`) |
| **Mean per-image processing latency** | `324.61 ms` | `323.28 ms` | `298.92 ms` | `527.52 ms` *(first run: `227.64 ms`)* |

### B. Validation Subgroup Breakdown (Single-Line vs. Two-Line and Day vs. Low-Light)

| Validation Subgroup | GT Plates | Un-adapted EasyOCR `v2` Exact (`%`) | Un-adapted EasyOCR `v2` CER | Adapted EasyOCR `v3` Exact (`%`) | **RapidOCR (`PP-OCRv4`) Exact (`%`)** | **RapidOCR (`PP-OCRv4`) CER** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Layout: `single_line`** | `31` | `2 / 31` (`6.45%`) | `0.4106` (`124 / 302`) | `4 / 31` (`12.90%`) | **`15 / 31` (`48.39%`)** | **`0.1589` (`48 / 302`)** |
| **Layout: `two_line`** | `63` | `1 / 63` (`1.59%`) | `0.4722` (`289 / 612`) | `18 / 63` (`28.57%`) | **`22 / 63` (`34.92%`)** | **`0.2810` (`172 / 612`)** |
| **Condition: `other` (Day, `47` images)** | `65` | `1 / 65` (`1.54%`) | `0.4223` (`269 / 637`) | `17 / 65` (`26.15%`) | **`30 / 65` (`46.15%`)** | **`0.1790` (`114 / 637`)** |
| **Condition: `low_light` (Night, `24` images)** | `29` | `2 / 29` (`6.90%`) | `0.5199` (`144 / 277`) | `5 / 29` (`17.24%`) | **`7 / 29` (`24.14%`)** | **`0.3827` (`106 / 277`)** |

---

## 5. Honest Decision, Tradeoffs, and Reproducible Commands

### Key Findings and Tradeoffs
1. **Substantial recognition improvement**:
   - Compared to the un-adapted EasyOCR baseline (`v2`: `3 / 94` = `3.19%` exact
     match, `45.19%` CER), off-the-shelf **RapidOCR (`PP-OCRv4` ONNX)**
     increases validation exact plate agreement more than **12×** to **`37 / 94`
     (`39.36%`)** and cuts Character Error Rate nearly in half to **`0.2407`
     (`220 / 914` character edits)**.
   - It also outperforms the 15-epoch CTC-adapted EasyOCR checkpoint (`v3`:
     `22 / 94` = `23.40%` exact match, `30.09%` CER) by **`+15` exact validation
     plate reads (`+15.96` percentage points)** and **`-6.02` percentage points
     of CER**, with particularly strong gains on `single_line` plates (`15 / 31`
     = `48.39%` exact match, `15.89%` CER vs. `4 / 31` for `v3`) and daylight
     plates (`30 / 65` = `46.15%` exact match, `17.90%` CER).
   - Every one of the `91` matched detector boxes produced non-empty OCR text
     (`0` `UNREADABLE` / `NO_TEXT` outcomes on matched detections; the only `3`
     missed plates are the `3` detector false negatives).
2. **Runtime and calibration tradeoffs**:
   - **Review-only confidence boundary**: RapidOCR's SVTR-LCNet character score
     distribution is not calibrated to EasyOCR's `0.75` acceptance threshold.
     Keeping `RapidOcrPlateRecognizer(review_only=True)` sets
     `ocr_confidence=None`, so all `94` detected plates remain in `NEEDS_REVIEW`
     with `OCR_CONFIDENCE_MISSING` (`41 / 94` matching `IN_COMMON_V1` shape and
     `53 / 94` additionally carrying `FORMAT_NOT_SUPPORTED`).
   - **CPU ONNX latency**: RapidOCR runs on CPU via `onnxruntime==1.32.0` (`228–528 ms/image`
     depending on CPU load) while the YOLOv8n detector runs on Apple M4 MPS.
   - **Application runtime preserved**: `build_local_runtime()` continues to
     default to `ROADSENSE_ANPR_OCR_ENGINE=easyocr` (and `scripts/serve_anpr_local.py`
     continues to serve the adapted EasyOCR runtime by default) until a dedicated
     score-calibration policy is chosen for RapidOCR on validation data.

### Reproducible Commands

```bash
# Run frozen RapidOCR (PP-OCRv4 ONNX, review-only) across the 71 validation images:
ROADSENSE_ANPR_PLATE_WEIGHTS=models/anpr/anpr_witvix_detector_mps_v1_best.pt \
ROADSENSE_ANPR_PLATE_SHA256=37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db \
ROADSENSE_ANPR_OCR_ENGINE=rapidocr \
ROADSENSE_ANPR_OCR_MODEL_DIR=models/rapidocr \
ROADSENSE_ANPR_OCR_CROP_VIEW=resized_bgr \
ROADSENSE_ANPR_DEVICE=mps \
.venv/bin/python scripts/generate_anpr_predictions.py \
  --dataset data/processed/witvix_anpr_pilot_v1 \
  --split val \
  --output outputs/anpr/witvix_val_predictions_rapidocr_v1.jsonl

# Score frozen RapidOCR predictions:
.venv/bin/python scripts/evaluate_anpr_predictions.py \
  --dataset data/processed/witvix_anpr_pilot_v1 \
  --predictions outputs/anpr/witvix_val_predictions_rapidocr_v1.jsonl \
  --split val \
  --output outputs/anpr/witvix_val_metrics_rapidocr_v1.json
```

### Recommended Next Step
1. If automatic `READ` acceptance (rather than `review_only=True`) is desired
   for `RapidOcrPlateRecognizer`, select a dedicated validation confidence
   threshold on `val` (analyzing the coverage vs. wrong-accepted-read curve on
   the `41` `IN_COMMON_V1`-matching plates) before freezing the pipeline for any
   future held-out `test` evaluation.
2. Address the remaining `24.07%` CER (`28.10%` on two-line plates, `38.27%` on
   low-light plates) by fine-tuning the PP-OCRv4 SVTR recognizer or adding
   4-corner plate rectification where corner annotations are available.
