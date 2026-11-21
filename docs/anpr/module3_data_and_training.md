# Module 3: ANPR data, training and evaluation

This module supplies tools to prepare transcribed images, fine-tune a dedicated
plate detector, and score frozen predictions. The existing local
`models/yolov8n.pt` base checkpoint matches the configured SHA-256. A limited
Witvix truck-plate pilot is prepared at `data/processed/witvix_anpr_pilot_v1`;
its labels have mixed publisher provenance and capture-session independence is
unverified. A 50-epoch Apple M4 MPS detector baseline has been trained on the
`train`/`val` splits (`outputs/training/anpr/anpr_witvix_detector_mps_v1`; see
[Witvix detector baseline results](witvix_detector_baseline_results.md)),
establishing validation plate-region detection metrics only. Held-out `test`
evaluation and real-image OCR recognition have not yet been performed.

## Work split

Codex owns `src/anpr/dataset.py`, `src/anpr/evaluation.py`,
`scripts/prepare_anpr_dataset.py`, `scripts/evaluate_anpr_predictions.py`, their
tests and shared documentation. Gemini owns `src/anpr/training.py`,
`scripts/train_anpr_detector.py`, `configs/training/anpr_plate_detector.yaml`,
`tests/test_anpr_training.py` and its handoff. Codex reviews and commits the pair,
then pushes completed module code to `origin/feat/anpr-foundation`.

## Required source data

Use permitted JPEG/PNG vehicle images with plate bounding boxes and human-verified
text. Detection-only YOLO labels are insufficient for OCR scoring: text
transcriptions must also be supplied. Do not represent an unreadable but visible
plate as an image with no plates. The current pothole dataset cannot substitute
for a plate dataset.

Supply one JSON object per image in an annotation `.jsonl` file. This illustrative
row is synthetic and is not a real sample or a benchmark result:

```json
{"image_id":"img-001","image_path":"images/img-001.jpg","group_id":"capture-session-01","source_id":"original-image-001","plates":[{"bbox":{"x1":100,"y1":200,"x2":300,"y2":260},"text":"MH12AB1234"}],"conditions":["clear"]}
```

- `image_path` is relative to the declared source root.
- `bbox` uses pixels in the image after EXIF orientation is applied, with the
  same half-open x/y convention as the ANPR API.
- `group_id` identifies a shared capture session/sequence. Do not invent a unique
  group per adjacent video frame.
- `grouping_basis` defaults to `declared_capture_session`. Use
  `publisher_vehicle_proxy` only where the source's one-image-per-vehicle claim
  is the grouping evidence, and retain its weaker assurance in reported results.
- `source_id` identifies the original image before augmentation. Every derivative
  shares its original's ID; this is not a camera ID.
- `plates: []` is allowed for genuinely negative images.
- Each plate has `label_provenance`: `human_verified`, `publisher_mixed` or
  `unspecified` (default). Do not mark model-assisted publisher labels as human
  verified without a corresponding review.
- `conditions` accepts `clear`, `blur`, `angled`, `low_light`, `occluded`, `other`.

The splitter transitively joins shared capture groups, source IDs, normalized
plate identities, raw file hashes and normalized-image hashes. A connected group
always belongs to one split. At least three independent groups containing plates
are required, so train/val/test each contain labelled plate evidence. Target
ratios are 70/15/15, but preserving groups takes precedence over exact counts.
Dataset schema 1.1 records grouping assumptions and annotation trust separately
from file/label integrity. A valid fingerprint does not certify either label
correctness or camera/session independence.

Sources remain read-only. Preparation makes EXIF-normalized PNG copies under a
new `data/processed/` child directory, with single-class YOLO labels, transcriptions,
image hashes, grouping evidence and a deterministic dataset fingerprint.

## Commands once real source annotations are available

Audit first; this command does not create a dataset or load a model:

```bash
.venv/bin/python scripts/prepare_anpr_dataset.py \
  --manifest data/raw/anpr/annotations.jsonl \
  --source-root data/raw/anpr \
  --output-dir data/processed/anpr_india_v1 \
  --dry-run
```

Repeat with `--execute` instead of `--dry-run` to create the dataset. Existing
outputs cannot be overwritten. Prepared datasets contain:

```text
anpr.yaml
manifest.jsonl
preparation_metadata.json
images/{train,val,test}/
labels/{train,val,test}/
```

Training preflight verifies the prepared data, class mapping, config, local base
checkpoint hash and proposed output path. It validates the device name without
importing PyTorch; actual execution additionally checks accelerator availability.

```bash
.venv/bin/python scripts/train_anpr_detector.py \
  --config configs/training/anpr_plate_detector.yaml \
  --dataset data/processed/anpr_india_v1 \
  --dry-run
```

Real training is a separate, explicit action using `--execute`. The default
configuration uses CPU, 50 epochs, image size 640 and batch 8. Set a supported
device explicitly if changing that plan. Training uses validation data for
validation/early stopping and does not run test inference. Text-mirroring flips,
mosaic, mixup and copy-paste augmentation are disabled.
Automatic mixed precision is disabled because the installed Ultralytics AMP
check can attempt to fetch an additional reference checkpoint.

Runs live under `outputs/training/anpr/<run_name>/`. Required weights, logs and
snapshots are hashed in `model_metadata.json`. The training wrapper rechecks data
and configuration around execution and records the starting Git SHA. Exporting
the resulting checkpoint into inference's `models/` directory is a later,
verified integration step; training does not replace deployed weights.

## Frozen prediction scoring

Each prediction JSONL row contains exactly `image_id` and `result`, where `result`
is the `AnprImageResult` v1 contract. Prediction dimensions must match the prepared
image. The scorer checks that normalized text agrees with original OCR text.

```bash
.venv/bin/python scripts/evaluate_anpr_predictions.py \
  --dataset data/processed/anpr_india_v1 \
  --predictions outputs/anpr/validation_predictions.jsonl \
  --split val \
  --output outputs/anpr/validation_metrics_v1.json
```

This command scores saved predictions; it does not run inference or tune a model.
Its default split is `val`. Select `test` explicitly only after the model and
thresholds are frozen. It rejects predictions from another split and never
replaces an existing report. Dataset fingerprint, prediction-file hash, metric
settings, Git SHA and timestamp are recorded. A saved prediction file alone does
not establish which checkpoint produced it; retain its generation provenance with
the actual model run when collecting real predictions.

Metrics include detection precision/recall at the declared IoU threshold,
end-to-end exact plate match, character error rate, read/acceptance coverage,
wrong accepted reads and observed mean processing time. Box matching is greedy
one-to-one in descending IoU and does not use plate text. This is not a detector
mAP implementation.

Missing predictions, missed detections and unavailable/unreadable outputs remain
in the ground-truth denominator. Unmatched plates contribute full deletions to
CER; CER can exceed 1.0. Accepted false positives are counted as wrong accepted
reads even on negative images. Undefined rates are JSON `null`, not a claimed
zero error rate. Unavailable runtime timings are excluded from inference timing.
Results are also broken down by condition; condition groups can overlap.
Reports explicitly distinguish agreement with mixed/unspecified supplied labels
from accuracy against human-verified references.

## Current limits

Module 3 unit tests use synthetic fixtures and fake training engines. On the
prepared Witvix pilot (`data/processed/witvix_anpr_pilot_v1`), the 50-epoch MPS
plate-detector baseline (`anpr_witvix_detector_mps_v1`) has completed `train`/`val`
optimization and validation logging; see
[Witvix detector baseline results](witvix_detector_baseline_results.md).
Held-out `test` split evaluation has not been run. Real image OCR and end-to-end
plate text accuracy measurement still require the `easyocr` package and local
OCR model files under `models/`. Downloads and installation are not performed by
these tools. The earlier missing frontend SWC compiler is an independent
environment issue and is not repaired by this module.

The downloaded Witvix sample is audited separately in
[Witvix intake results](witvix_intake_results.md); its annotations and source
grouping use the weaker provenance categories described above.
