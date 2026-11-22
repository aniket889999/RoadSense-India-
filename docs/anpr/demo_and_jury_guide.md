# ANPR demonstration and submission guide

RoadSense's ANPR module detects a vehicle's plate region, reads candidate text,
checks a limited Indian plate format, and presents the source crop for human
review. Parking and toll workflows are possible applications; the prototype
does not open barriers, charge vehicles or verify registration records.

## Match to the problem statement

| Requirement | Implemented evidence | Remaining limit |
| --- | --- | --- |
| Detect the plate region | Trained single-class YOLOv8n, 50 epochs on Apple MPS | Training source mainly represents commercial trucks |
| Segment/read characters | RapidOCR text-line detection and sequence recognition; preserved EasyOCR comparison | Inaccurate reads remain reviewable |
| Validate output format | Explicit `IN_COMMON_V1` shape check | A match does not establish registration validity; special formats can be unsupported |
| Test multiple images | 71 validation images / 94 supplied plate annotations | Held-out evaluation is pending; labels have mixed publisher provenance |
| Handle blur/angle failures | Bounded OpenCV detail, resolution, illumination and observed line-tilt measurements; retake guidance and crop preview | Heuristics are advisory; insufficient edge evidence produces no angle estimate |
| Discuss parking/toll applications | Operator-assisted gate-entry review and vehicle-observation examples below | No identity-to-bay, payment or barrier-control claims |

The trained detector and fine-tuned EasyOCR comparison are the model-making
work. The selected RapidOCR recognizer is pretrained; describe it accurately
rather than claiming it was trained from scratch in this project.

## Reproduce the local demo

Before a demonstration, run the read-only preflight from the repository root:

```bash
.venv/bin/python scripts/anpr_preflight.py          # add --engine easyocr / --device cpu to match the launch
```

It loads no model and opens no dataset image. It hashes every pinned model
file, compares installed packages with `requirements-anpr.txt` (the recorded
working environment, not an install instruction), confirms the requested
detector device is available, reads the prepared dataset's metadata, and
reports existing listeners on ports 8000/3000 and the frontend build state.
Any `FAIL` means the launcher is expected to return `UNAVAILABLE` or not start;
`WARN` marks drift from the recorded state. If MPS reports unavailable inside a
sandboxed tool, rerun in a normal Terminal before changing device settings.

Then, with the existing local assets and dependencies:

```bash
.venv/bin/python scripts/serve_anpr_local.py
```

From `apps/web`, run `npm run start` after `npm run build`. Open
`http://localhost:3000`, then **ANPR Inspection**.

1. Choose a permitted JPEG/PNG vehicle image with a visible plate.
2. Select **Run Plate Recognition**. Inspect the detected box and enlarged crop.
3. Compare raw OCR with normalized text. Explain the supported-format result.
4. Inspect capture-quality cues. These recommend checking focus, lighting,
   distance or a straighter view; they do not repair missing character detail.
5. Download **annotated PNG** and **review JSON**. Both identify the result as
   unverified. The PNG includes box IDs and candidate readings; JSON preserves
   complete text, measurements, review reasons and request identity.

The API does not retain uploaded images or OCR text. Downloads are explicit
browser actions. Keep real vehicle images and exported readings outside Git.
`--engine easyocr` selects the preserved adapted EasyOCR comparison.

## Results to present honestly

The frozen RapidOCR validation comparison has:

- 37/94 exact full-plate agreements: **39.36%**.
- 220/914 character edits: **24.07% character error rate**.
- 91/94 matched detections, 3 false-positive boxes and 3 missed plates.
- Detection precision / recall: **96.81% / 96.81%**.
- Single-line exact agreement: 15/31; two-line: 22/63.
- Day/other exact agreement: 30/65; low-light: 7/29.

RapidOCR is review-only. Zero automatically accepted readings is intentional,
not evidence of zero recognition errors. Its score scale is not substituted
for the EasyOCR threshold. New results explain this with `OCR_REVIEW_ONLY`.

Quality assessment and export do not change the frozen OCR benchmark. The
model, thresholds and plate text remain unchanged. These results are agreement
against `publisher_mixed` labels and are not a production accuracy guarantee.

## Blur and tilt evidence

Controlled synthetic tests separate diagnostic behavior from OCR accuracy:

| Synthetic control | Observed detail score | Observed 2D line tilt | Cue |
| --- | ---: | ---: | --- |
| Sharp rendered plate | 5593.55 | 0.0° | No configured cue |
| Same pixels with Gaussian blur (sigma 7) | 19.04 | 0.0° | Low sharpness |
| Same plate rotated +15° | 5649.84 | −15.1° | Tilt observed |

The tilt sign follows image coordinates. It is an edge-line measurement, not
the vehicle's 3D pose or depth. No angle is reported when there is insufficient
coherent line evidence. These are synthetic controls, not real blur/skew
classification accuracy results. Reproduce them with
`pytest -q tests/test_anpr_quality.py`; local measurements are saved in
`outputs/anpr/quality_controls_v1.json`.

The heuristics use an aspect-preserving bounded grayscale crop. Detail uses
Laplacian variance, illumination uses native-crop mean and bright-pixel fraction,
and tilt uses a length-weighted consensus of near-horizontal Hough segments.
Review cues use source width <80 or height <24, detail <35, mean brightness <45,
bright-clipped fraction >0.65, or supported absolute tilt ≥8°. These thresholds
are not calibrated against a human-labelled quality benchmark, and no cue is
proof that text is correct or incorrect.

## Jury answers

**Where does OCR fail?** Small characters, motion blur, glare, low illumination,
two-line layouts, extra markings and unfamiliar fonts cause missed or substituted
characters. Normalization preserves raw evidence and does not guess O/0 or I/1.

**How do you handle skewed angles?** Inspect line geometry when evidence exists,
show capture guidance and keep uncertain text for review. Strong perspective
correction would require a reliable quadrilateral from visible boundaries or a
corner model. The prototype does not invent corners or claim recovered detail.

**What did you train?** A dedicated plate-region detector and an adapted EasyOCR
recognizer baseline. RapidOCR performed best in the supplied-label comparison
and is the local launcher's selected recognizer.

**How could parking/toll systems use it?** As an operator-assisted entry/exit
observation linked to a verified plate reading. Payment, barrier actuation and
association with a particular parking bay require separate validated components.

## Remaining submission work

Collect permitted ordinary-car examples with independently checked text, plus
real blurred and angled examples. Current truck-heavy validation does not
establish passenger-car performance. Freeze the final model/settings before
the independent held-out evaluation. Do not claim these steps are complete.

## Upgrade verification

- Full repository suite: 452 passed, with six existing Alembic deprecation warnings.
- Frontend TypeScript checking and production build passed.
- Controlled quality tests and generated OpenAPI contract checks passed.
- A real local browser upload returned schema 1.1, preserved the detected crop
  and text, showed quality measurements and the explicit `OCR_REVIEW_ONLY` reason.
- Annotated PNG and review JSON were downloaded through the real browser.
  The PNG decoded at 1200×1691 for a 1200×1600 source image, including its
  unverified header and reading footer. JSON retained `human_verified: false`,
  source dimensions, full model output and crop quality measurements.
- The ANPR header now describes plate recognition rather than displaying the
  unrelated pothole-model badge. Existing operations views retain their context.
- No model retraining, dependency download or held-out inference was performed
  for this review/export upgrade.
