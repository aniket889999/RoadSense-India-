# RoadSense ANPR Module 3 — Gemini Handoff (Dedicated Plate-Detector Training Tools)

## Git State

- **Branch**: `feat/anpr-foundation`
- **Starting Checkpoint (HEAD SHA)**: `bb376d080809d176543144b8db3fe112b56c02d6`
- **Patch / Whitespace Check (`git diff --check`)**: Clean (exit code `0`)

## Gemini-Owned Module 3 Files

| File | Purpose |
| --- | --- |
| `src/anpr/training.py` | Reproducible training plan builder (`build_training_plan`) and execution wrapper (`execute_training_plan`, `run_anpr_detector_training`) with lazy Ultralytics/Torch loading, `validate_prepared_dataset` integration, SHA-256 base-weight verification, plate-safe augmentation bounds, and `model_metadata.json` provenance recording |
| `scripts/train_anpr_detector.py` | CLI entry point supporting `--dry-run` (default) and explicit `--execute` flags |
| `configs/training/anpr_plate_detector.yaml` | Default training configuration for `license_plate` single-class YOLOv8n fine-tuning with local `models/yolov8n.pt` SHA-256 pin and disabled text-mirroring/aggressive augmentations |
| `tests/test_anpr_training.py` | 11 focused unit tests covering eager-import absence, dry-run side-effect isolation, pothole weight / SHA-256 / existing run rejection, augmentation guards, `val`-only execution parameters, required artifact hashing, and CLI exit codes |
| `docs/anpr/gemini-module3-handoff.md` | This Module 3 handoff document |

## CLI Usage

### 1. Dry-Run Preflight Validation (No ML Imports or Training Side Effects)
```bash
.venv/bin/python scripts/train_anpr_detector.py \
  --config configs/training/anpr_plate_detector.yaml \
  --dataset data/processed/<prepared_anpr_dataset> \
  --dry-run
```
- Calls `src.anpr.dataset.validate_prepared_dataset(dataset_dir)` to verify `anpr.yaml` (`nc: 1`, `names: {0: "license_plate"}`), `manifest.jsonl`, `preparation_metadata.json`, split membership, and `dataset_fingerprint`.
- Verifies local base weights under `models/`, checks for symlinks and expected 64-character lowercase SHA-256 digest, rejects pothole checkpoints, confirms `outputs/training/anpr/<run_name>` does not already exist, and validates plate-safe augmentation limits without importing `torch` or `ultralytics`.

### 2. Explicit Training Execution (`val`-Only Validation)
```bash
.venv/bin/python scripts/train_anpr_detector.py \
  --config configs/training/anpr_plate_detector.yaml \
  --dataset data/processed/<prepared_anpr_dataset> \
  --execute
```
- Trains from the hash-verified local base weights (`models/yolov8n.pt`, SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`).
- Passes `val=True`, `split="val"`, `exist_ok=False`, `deterministic=True`, and the validated plate-safe augmentation dictionary (`fliplr=0.0`, `flipud=0.0`, `shear=0.0`, `perspective=0.0`, `mosaic=0.0`, `mixup=0.0`, `copy_paste=0.0`, `erasing=0.0`).
- Never invokes `test` split evaluation.
- Writes `outputs/training/anpr/<run_name>/model_metadata.json` containing `git_sha`, `dataset_fingerprint`, `split_counts`, `config_sha256`, `base_weights_sha256`, `device`, `elapsed_seconds`, `environment`, `augmentation`, and SHA-256 hashes of all required artifacts (`base_weights`, `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`, `train_config.yaml`, `dataset_preparation_metadata.json`).

## Focused Test Results

- `.venv/bin/pytest -q tests/test_anpr_training.py`: **11 passed in 2.37s**.

## Real-Training Blockers & Asset Status

1. **Prepared ANPR Dataset**: No real annotated license-plate dataset has been registered and prepared under `data/processed/` yet.
2. **Trained Single-Class Plate Checkpoint**: Real detector training (`--execute`) has not been run yet because the real plate dataset is not yet staged; therefore no trained `license_plate` checkpoint exists under `models/` or `outputs/training/anpr/`.
3. **End-to-End Real-Image OCR**: EasyOCR package and offline EasyOCR weights under `models/` remain absent for end-to-end plate recognition evaluation.

## Codex integration review

The original handoff above records Gemini's implementation and checks. Integration
now uses the shared dataset validator and prepared image/text manifest. Review
corrections require the runtime's strict class mapping, recheck data/config around
execution, capture Git provenance before training, and inspect package versions
without importing ML engines just for metadata. AMP is explicitly disabled to
avoid the installed library's auxiliary reference-checkpoint fetch.

Independent combined verification: **93 focused tests passed** across the ANPR
foundation, adapters, API, dataset preparation, metrics, training and OpenAPI
contracts. Python compilation and all three Module 3 CLI help checks passed.
These checks use synthetic data and fake ML engines, not a trained detector.

The actual local base checkpoint hash matches the configured digest. An actual
project dry-run targeting `data/processed/anpr_india_v1` stopped as expected because
that prepared dataset does not exist. No real ANPR training, held-out inference,
dependency installation or model/data download was performed.
