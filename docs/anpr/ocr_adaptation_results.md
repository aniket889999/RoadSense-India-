# OCR adaptation: local experimental result

The detector remains frozen. Fine-tuning the existing English Gen2 text
recognizer improved full-plate agreement on the 71-image validation split from
3/94 to 22/94. This is still an experimental, human-reviewed reading pipeline.

## What changed

The first training-only experiment compared unrestricted, alphanumeric,
uppercase and beam-search decoding. None improved the complete-read count
above 4/45 on that experiment's 40-image training development subset.
Component-based text-line cropping also failed to improve that count (2/45).
Those diagnostic methods are reproducible in `scripts/benchmark_anpr_decoding.py`
and are not enabled in the application.

The effective change was adapting the existing `english_g2` recognition weights:

- Source: only plates explicitly marked `single_line` in training images.
- 124 eligible crops; 99 optimizer-training crops and 25 internal development
  crops, separated deterministically by image identity with seed 42.
- Base weights: the existing official English Gen2 checkpoint. No downloads.
- Frozen visual feature extractor; trainable sequence model and prediction head.
- AdamW, learning rate 0.0001, weight decay 0.0001, batch 8, 15 epochs.
- Native EasyOCR height 64, aspect-preserving padded width 320.
- Model operations on Apple MPS; CTC loss explicitly on CPU with differentiable
  transfers. No silent device fallback.
- Epoch selection: highest complete-read count on the internal development
  crops, then lowest character error rate. Epoch 14 was selected.

Internal development reads rose from 1/25 to 7/25, and CER fell from 0.4797
to 0.1992. The prepared `val` and `test` splits were not used for gradient
updates or checkpoint selection. Source labels were preserved, not corrected
using OCR output. The new checkpoint is a tensor state dictionary, not a pickled
model object or a replacement for the original official assets.

## Validation comparison

The selected checkpoint was evaluated once on all 71 validation images, including
94 publisher-mixed plate annotations. Missed plates remain in the denominator.
All configuration and acceptance thresholds remained fixed for this comparison.

| Measurement | Original-BGR baseline v2 | Adapted recognizer v3 |
| --- | ---: | ---: |
| Exact full-plate agreement | 3/94 (3.19%) | 22/94 (23.40%) |
| Character edits / reference characters | 413/914 | 275/914 |
| Character error rate | 45.19% | 30.09% |
| Matched detections | 91/94 | 91/94 |
| Plates with nonempty OCR | 90/94 | 90/94 |
| Accepted reads at the existing policy | 1 | 10 |
| Wrong accepted reads observed | 0 | 0 |
| Mean pipeline processing time | 323.28 ms | 298.92 ms |

| Subgroup | v2 exact reads | v3 exact reads |
| --- | ---: | ---: |
| Single-line plates | 2/31 | 4/31 |
| Two-line plates | 1/63 | 18/63 |
| Low-light plates | 2/29 | 5/29 |
| Other/day plates | 1/65 | 17/65 |

Confidence scores remain uncalibrated. Ten observed accepted reads with no wrong
reads do not establish a reliable error rate. Reference labels have
`publisher_mixed` provenance, capture-session independence is unverified and
there are no negative images. This is validation agreement, not an independent
held-out accuracy estimate. The held-out test split remains untouched.

## Local artifacts and identity

Dataset fingerprint:
`2c33c828642ba484c4698e561df47193975f491291718a44c2d8d853e0c94a4a`.

- Training experiment: `outputs/training/anpr_ocr/codex_ctc_v1/`.
  `experiment.json`, `history.json` and `best_epoch.json` preserve the split and
  checkpoint-selection evidence. The initial experiment preceded addition of
  the CLI's explicit `--execute` guard and extra metadata fields; the optimizer
  settings and selection rule above describe the actual run.
- Candidate/installed checkpoint SHA-256:
  `c72eab9cc00d2aff5a2eb006be5251d346d3be02de225db92aacbf08ad615fd3`.
- Installed copy: `models/anpr_ocr/english_g2_witvix_v1.pth`.
- Predictions: `outputs/anpr/witvix_val_predictions_mps_v3_ctc.jsonl`;
  SHA-256 `31bc6a725461be68e9a0474bb33707b4e857775bfeaba340a670403aa07cff8e`.
- Report: `outputs/anpr/witvix_val_metrics_mps_v3_ctc.json`.
- The original English Gen2 SHA remains
  `e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8`.

The original detector, official OCR weights, v1/v2 predictions and datasets are
unchanged. Models, predictions, crops and training outputs remain excluded from
Git. No real plate strings or images are included in this document.

## Run the improved local API

From the repository root, with the staged local assets present:

```bash
.venv/bin/python scripts/serve_anpr_local.py --engine easyocr
```

This explicitly selects the adapted EasyOCR engine and starts the existing API on `127.0.0.1:8000`, with the pinned detector,
original-BGR crops and adapted recognizer. Use `--device cpu` for explicit CPU
inference, or `--port` to select another API port. Start the dashboard separately
with `npm run start` from `apps/web` after its production build.
The launcher's newer default is review-only RapidOCR following the
[model comparison](ocr_model_comparison_results.md); `--engine easyocr` retains
the adapted model described in this report.

The adapter verifies both the adapted model path under `models/` and its SHA-256
before loading tensor-only weights with strict parameter matching. Official
EasyOCR assets still initialize with automatic downloads disabled. CPU loading
disables dynamic quantization before restoring the float checkpoint.

For manual API configuration, the two additional settings are:

```text
ROADSENSE_ANPR_OCR_RECOGNIZER_WEIGHTS=models/anpr_ocr/english_g2_witvix_v1.pth
ROADSENSE_ANPR_OCR_RECOGNIZER_SHA256=c72eab9cc00d2aff5a2eb006be5251d346d3be02de225db92aacbf08ad615fd3
```

Omitting both retains the original recognizer. Supplying only one, a mismatched
hash or an incompatible checkpoint returns runtime unavailable.

## Reproduce training into a new output directory

```bash
.venv/bin/python scripts/finetune_anpr_ocr.py \
  --dataset data/processed/witvix_anpr_pilot_v1 \
  --source /Users/aniket/Downloads/archive \
  --output outputs/training/anpr_ocr/reproduction_v1 --execute
```

Without `--execute`, this performs a data preflight and prints crop counts.
Existing output directories cannot be overwritten. The trainer uses MPS model
operations explicitly and does not run held-out inference. Seeded settings do
not promise bit-identical MPS checkpoints; verify a new candidate before deployment.

Focused ANPR/intake/API/OpenAPI tests: 112 passed. Python compilation and
whitespace checks passed. A real production-adapter recognition call reproduced
the evaluated candidate's text on the existing demonstration image.
The running API also returned HTTP 200 for that real image upload, and the
dashboard on `http://localhost:3000` returned HTTP 200 with the new API running.
