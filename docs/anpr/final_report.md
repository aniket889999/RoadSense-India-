# RoadSense India — ANPR: final project report

**Status:** experimental local review tool. Not a production, toll, barrier or registration
service. All results below are on the **validation** split unless stated; the one-time
held-out test evaluation is **pending** (§9).

## 1. Problem statement and how each requirement is met

| Requirement | What was built | Evidence |
| --- | --- | --- |
| Detect the plate region | YOLOv8n single-class `license_plate` detector, trained here (50 epochs, Apple MPS) | Val precision 0.998, recall 0.947, mAP50 0.992, mAP50-95 0.847 |
| Segment / read characters | RapidOCR PP-OCRv4 text-line detection + recognition (pretrained, selected after comparison); locally adapted EasyOCR kept for comparison | 37/94 exact as read; 58/94 after explicit cleanup (§4) |
| Validate against a plate format | `IN_COMMON_V1` shape + `IN_PLATE_V2` (state/UT codes, BH series, look-alike hints) | Pattern-fitting reads 58/61 exact; non-fitting 0/29 (§5) |
| Test multiple images, report results | 71 validation images / 94 plates, every figure reproducible from frozen predictions | Scripts and reports listed per section |
| Handle blur and angled-image failures | Image-quality cues (sharpness, brightness, glare, line tilt), failure analysis by condition, car protocol with blur/angle tags | §6 and car protocol |
| Discuss parking / toll applications | Review-first workflow, opt-in automatic acceptance with measured risk, issuing-RTO decoding | §8 |

## 2. Data

Witvix ANPR dataset (CC BY-NC 4.0): 470 images, 589 plates; grouped split 329 / 71 / 70
images (train / val / test) with identity and hash overlap checks. Labels are
`publisher_mixed` (none flagged human-verified); images are mostly commercial trucks;
69% of plates are two-line; 200 images are at night. The held-out test split has never
been used for inference or tuning.

## 3. Pipeline

JPEG/PNG upload (bounded: 10 MiB, 12 MP) → EXIF-normalized decode → plate detector (MPS)
→ copied crops → RapidOCR (CPU) → raw text preserved → normalization (case, spaces,
hyphens) → explicit candidate cleanup (§4) → `IN_COMMON_V1` + `IN_PLATE_V2` checks →
review decision with reasons → browser review with enlarged crop, quality cues, annotated
PNG and JSON export. Inference is local; uploads and OCR text are not stored by the API.

## 4. Reading accuracy and what improved it

| Configuration (same 94 validation plates) | Exact | CER |
| --- | ---: | ---: |
| EasyOCR, pretrained | 3 (3.2%) | 45.2% |
| EasyOCR, adapted on our training crops | 22 (23.4%) | 30.1% |
| RapidOCR PP-OCRv4 (selected), as read | 37 (39.4%) | 24.1% |
| RapidOCR + explicit candidate cleanup | **58 (61.7%)** | **18.4%** |

The [failure analysis](ocr_failure_analysis.md) found that 28 of 57 failures were the
correct plate plus extra characters: the `IND` country code on high-security plates and
dot-like marks. The [cleanup policy](text_cleanup_policy_results.md) removes only those
(dot-like marks; a leading `IND` only when the remainder is a valid plate), never
substitutes characters, keeps the raw text, and marks every cleaned reading for review:
21 plates fixed, 0 broken. The rule came from validation failures, so 58/94 is a
validation estimate until the held-out check.

## 5. Plate-format validation and issuing office

[`IN_PLATE_V2`](format_check_v2_results.md) checks the state series and BH series, the
state/UT code (including legacy `TS`, `DN`, `OR`, `UA`), offers look-alike hints that are
never applied, and decodes the issuing state and RTO office from a public directory built
reproducibly from a pinned Wikipedia revision (35 states/UTs, 1,302 codes; CC BY-SA 4.0).
For example a `HR55…` reading decodes to Haryana · HR-55 · Gurugram (commercial
vehicles). This is a directory of where codes are issued, not a registry lookup, and it
contains no owner or vehicle data. On validation, readings that fit a pattern with a known
code were exact 58/61 times; non-fitting readings were wrong 29/29 times.

## 6. Failure analysis — where OCR fails

From the [failure analysis](ocr_failure_analysis.md) (57 non-exact plates):

1. **Extra non-registration text** (IND emblem, marks): 28 — now handled by cleanup.
2. **Low light**: 7/29 exact as read (12/29 after cleanup); genuine misreads, CER 33% after
   cleanup. Needs better capture or recognizer adaptation, not text rules.
3. **Mixed character errors**: 20 — dropped digits and substitutions.
4. **Missed plates** (detector): 3 of 94.
5. **Look-alike characters only** (e.g. `0` read as `O`): 3 — surfaced as reviewer hints.

Large crops read worse than small ones because close-ups resolve the emblem and fasteners.
Two-line ordering errors were rare (1 reordered, 1 fragment), so row splitting was
deprioritized.

**Blur and angle.** Synthetic controls verify the quality cues: a sharp rendered plate
scores 5593.6 detail, the same plate blurred 19.0; a +15° rotation is measured as −15.1°
line tilt. These are diagnostics that tell the reviewer *why* a reading may be unreliable;
they do not repair the image and are not real-world blur/skew accuracy. Real blur and
angle performance on cars is what the [car protocol](car_evaluation_protocol.md) measures.

## 7. Automatic acceptance and speed

Default: every reading is reviewed. The opt-in [`rapidocr_gated_v1`](acceptance_policy_results.md)
policy accepts a reading only if RapidOCR's score is ≥ 0.90, the format and state code
check pass, and no cleanup was applied. End to end on validation: **33 of 94 plates (35%)
accepted, 0 wrong** (95% upper bound on the wrong-acceptance rate ≈ 8.7%); 61 routed to
review with reasons. OCR score alone could not separate right from wrong readings.

Warm service latency on the M4 host, 71 validation images ([latency results](latency_results.md)):
median ≈ 0.61 s per image with the detector on MPS or CPU (p95 1.78 s MPS, 1.44 s CPU);
excludes model loading and HTTP/UI time. No real-time claim is made.

## 8. Parking and toll applications (what is and is not claimed)

- **Parking entry/exit logging:** read plates at a gate camera; auto-log high-confidence
  readings (policy above), queue the rest for an attendant. Decoded RTO helps spot
  impossible codes.
- **Toll / access control:** the same review-first design suits exception handling;
  automatic barrier control or charging would need a calibrated, independently tested
  error rate, registry integration (e.g. VAHAN) and legal review, none of which is claimed.
- The prototype opens no barrier, charges nobody, and verifies no registration.

## 9. What is pending and why

| Item | Why it matters | Status |
| --- | --- | --- |
| One-time held-out test evaluation of the frozen pipeline | First independent confirmation of 58/94 and of the acceptance policy | Pending your explicit approval |
| Passenger-car set (≥ 100 double-labelled photos) | Cars, real blur and angles were never measured | Protocol and tools ready; needs photos |
| Plate rectification for strong angles | Would address skewed plates rather than only flag them | Not implemented |

## 10. Jury questions

**Where does OCR fail?** Mostly by reading extra text (emblem, marks), then in low light,
then mixed character errors; three plates were never detected. Numbers in §6.

**How would you handle skewed plates?** Today: detect tilt and flag it with retake
guidance; the reader keeps the raw image. Next: find the plate's four corners (keypoint
head on the detector or contour quadrilateral) and warp it to a frontal rectangle before
OCR, then measure before/after on the angle-tagged car set. We do not claim this yet.

**Which components did you train yourself?** The plate detector (YOLOv8n fine-tuned from
COCO weights on the grouped Witvix training split) and an adapted EasyOCR recognizer
(sequence head adapted on our training crops). The selected RapidOCR recognizer is
pretrained; our contributions around it are the engine comparison, the measured cleanup
policy, the format and RTO checks, and the calibrated acceptance policy.

**How do you know the numbers are honest?** Every figure is reproduced by a script from
frozen predictions, cross-checked against the evaluator, and committed with tests that
fail if a documented number changes. The test split is untouched.
