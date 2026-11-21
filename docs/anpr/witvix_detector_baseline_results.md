# Witvix ANPR Plate-Detector MPS Baseline Results

## Scope and Boundary

This document records the measured baseline training run for a dedicated
single-class (`{0: "license_plate"}`) YOLOv8n license-plate region detector on
the prepared Witvix Indian truck license-plate pilot dataset
(`data/processed/witvix_anpr_pilot_v1`).

- **What this run establishes**: Plate-region bounding-box detection metrics on
  the 71-image (`94`-plate) validation split (`val`).
- **What this run does NOT establish**:
  - **OCR character recognition**: EasyOCR and offline OCR model weights are not
    installed or staged in this environment, so no plate character strings were
    read by an OCR engine.
  - **Complete end-to-end ANPR accuracy**: End-to-end plate reading (detection +
    OCR + `IN_COMMON_V1` format assessment) remains unmeasured.
  - **Held-out test evaluation**: Held-out `test` split (`70` images, `87`
    plates) inference and evaluation were **not** performed during or after this
    training run.
  - **Production or general Indian ANPR readiness**: The Witvix pilot is a
    470-image commercial-truck sample with mixed publisher labels, unverified
    capture-session independence, and zero negative images.

## Training Command and Configuration

- **Training Git SHA**: `ffec2ec5bb2e08e44706448d6d80229770e27352` (commit
  `chore: configure Witvix ANPR detector baseline`)
- **Configuration file**:
  `configs/training/anpr_witvix_mps_baseline_v1.yaml`
  (SHA-256: `82cde863d5ce81a069eee16d6662c928e857adbe73b0c8163e1a00cdcb3df65a`)
- **Executed command**:

```bash
caffeinate -dims .venv/bin/python scripts/train_anpr_detector.py \
  --config configs/training/anpr_witvix_mps_baseline_v1.yaml \
  --dataset data/processed/witvix_anpr_pilot_v1 \
  --execute
```

- **Key hyperparameters and safeguards**:
  - `experiment.name`: `anpr_witvix_detector_mps_v1`
  - `experiment.seed`: `42` (`deterministic: true`)
  - `model.base_weights`: `models/yolov8n.pt`
    (SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`)
  - `model.image_size`: `640`
  - `model.class_name`: `license_plate` (`nc: 1`, `names: {0: "license_plate"}`)
  - `training.device`: `mps` (Apple M4 Metal Performance Shaders)
  - `training.epochs`: `50`
  - `training.batch`: `8`
  - `training.patience`: `15`
  - `training.workers`: `2`
  - `amp`: `false` (disables Ultralytics automatic mixed-precision reference
    checkpoint downloads)
  - `val`: `true`, `split`: `"val"` (`test` split never invoked)
  - **Plate-safe augmentation**: `fliplr: 0.0`, `flipud: 0.0`, `degrees: 3.0`,
    `translate: 0.05`, `scale: 0.15`, `shear: 0.0`, `perspective: 0.0`,
    `mosaic: 0.0`, `mixup: 0.0`, `copy_paste: 0.0`, `erasing: 0.0`,
    `hsv_h: 0.01`, `hsv_s: 0.25`, `hsv_v: 0.25`

## Prepared Dataset and Evidence Limitations

- **Prepared dataset path**: `data/processed/witvix_anpr_pilot_v1`
- **Dataset schema version**: `1.1`
- **Dataset fingerprint (SHA-256)**:
  `2c33c828642ba484c4698e561df47193975f491291718a44c2d8d853e0c94a4a`
- **Group policy**:
  `declared_group+source+normalized_plate+raw_and_prepared_sha256_v2`
- **Independent proxy groups**: `470` (`seed: 42`)

| Split | Images | Plate Annotations | Negative Images |
| --- | ---: | ---: | ---: |
| `train` | 329 | 408 | 0 |
| `val` | 71 | 94 | 0 |
| `test` (untouched) | 70 | 87 | 0 |
| **Total** | **470** | **589** | **0** |

### Retained Evidence Limitations
1. **Label provenance (`publisher_mixed`)**: All 589 plate annotations carry
   `label_provenance: "publisher_mixed"` (`all_plate_texts_human_verified: false`).
   The publisher reports that 341 of 589 transcriptions were human-checked, but
   per-annotation review flags are not provided.
2. **Capture-session independence (`not_established_vehicle_proxy`)**: The
   source archive provides no capture-session or camera/site identifiers.
   Grouping uses `grouping_basis: "publisher_vehicle_proxy"` based on one image
   per vehicle stem plus transitive joining on plate text and image hashes.
3. **No negative images**: All 470 images contain at least one plate (`0`
   negative images across `train`, `val`, and `test`), so false-positive rates
   on plate-free background scenes are not measured by this dataset.
4. **Non-`IN_COMMON_V1` transcriptions preserved**: All 6 plate transcriptions
   outside the `IN_COMMON_V1` regular expression remain unmodified in the
   prepared dataset.

## Hardware and Runtime Environment

- **Hardware**: Apple M4 (`macOS-27.0.1-arm64-arm-64bit`), `torch.backends.mps.is_available() == True`
- **Python**: `3.12.7` (`.venv/bin/python`)
- **PyTorch**: `2.13.0`
- **Ultralytics**: `8.4.124`

## Training Duration, Epoch Completion, and Post-Train Cache Fix

- **Epochs completed**: `50 / 50`
- **Early stopping (`patience: 15`)**: Did **not** trigger before epoch 50
   because the highest validation fitness occurred at **epoch 36** (14 epochs
  before epoch 50, within the 15-epoch patience window).
- **Measured runtime**:
  - `2107.23 seconds` (`0.5853 hours`), recorded directly in the cumulative
    `time` column of `outputs/training/anpr/anpr_witvix_detector_mps_v1/results.csv`
    at epoch 50 (and reported by Ultralytics console output as
    `50 epochs completed in 0.586 hours`).
  - Final best-checkpoint validation speed logged by Ultralytics on `val`
    (`71` images): `0.9 ms` preprocess, `26.9 ms` inference, `7.2 ms`
    postprocess per image.

### Post-Training Label-Cache Blocker and Fix
During the `--execute` run, Ultralytics created transient
`data/processed/witvix_anpr_pilot_v1/labels/train.cache` and
`data/processed/witvix_anpr_pilot_v1/labels/val.cache` files while indexing the
training and validation splits (`test.cache` was never created, confirming the
`test` split was untouched). After all 50 training epochs and the `best.pt`
validation pass completed, `execute_training_plan` called
`validate_prepared_dataset(plan.dataset_dir)`, which enforces that `images/` and
`labels/` contain only manifested `.png` and `.txt` files, raising
`DatasetValidationError("Unmanifested or missing image/label files were found.")`
on those two `.cache` files.

- **Fix applied**: Added `_remove_ultralytics_label_caches(dataset_dir)` in
  `src/anpr/training.py` (invoked in a `finally:` block around `model.train` and
  in `finalize_completed_training_run`) to remove regular `labels/train.cache`
  and `labels/val.cache` files before re-running `validate_prepared_dataset`,
  and added a regression assertion in `tests/test_anpr_training.py`.
- **Artifact finalization**: Without re-running or overwriting the completed
  50-epoch training directory, the transient `.cache` files were removed,
  `validate_prepared_dataset` passed with the exact fingerprint, both
  `weights/best.pt` and `weights/last.pt` were verified to expose
  `names == {0: 'license_plate'}`, and `finalize_completed_training_run` wrote
  `model_metadata.json`.

## Validation Split (`val`) Detection Metrics

All metrics below are measured on the `val` split (`71` images, `94` plate
instances). Held-out `test` split evaluation was not performed.

| Measurement Source | Epoch | Box Precision `P` | Box Recall `R` | `mAP50(B)` | `mAP50-95(B)` | `val/box_loss` | `val/cls_loss` | `val/dfl_loss` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `results.csv` peak fitness row (saved as `best.pt`) | 36 | `0.99752` | `0.94681` | `0.99217` | `0.84655` | `0.57901` | `0.43740` | `0.86991` |
| Ultralytics post-training `Validating .../weights/best.pt` console log (fused model) | 36 (`best.pt`) | `0.998` | `0.947` | `0.992` | `0.847` | — | — | — |
| `results.csv` final epoch row (saved as `last.pt`) | 50 | `0.97894` | `0.98893` | `0.99347` | `0.83607` | `0.62533` | `0.40273` | `0.87088` |

- **Class verification**: Loading both `weights/best.pt` and `weights/last.pt`
  with `YOLO(...)` confirms `model.names == {0: 'license_plate'}`.

## Verified Run Artifacts and SHA-256 Hashes

Run directory (ignored by Git under `outputs/`):
`outputs/training/anpr/anpr_witvix_detector_mps_v1`

| Artifact Key / Relative Path | File Size (Bytes) | SHA-256 Digest |
| --- | ---: | --- |
| `base_weights` (`models/yolov8n.pt`) | `6,549,796` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `weights/best.pt` | `6,221,034` | `37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db` |
| `weights/last.pt` | `6,221,034` | `5527bbaece8319874aedb426c52a8c3f49c3aae4fc78d50dee909ec9e03d9b96` |
| `results.csv` | `6,120` | `9fd001204dd9e516bef91782f6831b1b71bbf4a7caef14d84a078ffb9d887529` |
| `args.yaml` | `1,967` | `3e1c12cf5fa95c943aca8ee1f41d48c06a292ffe3c57c0967cfd3b069cb3203d` |
| `train_config.yaml` | `863` | `82cde863d5ce81a069eee16d6662c928e857adbe73b0c8163e1a00cdcb3df65a` |
| `dataset_preparation_metadata.json` | `972` | `676ef66d15aadf1f6a971c3c1f7b2a30a58f20be70e4b793c9988a6a8898422e` |
| `model_metadata.json` | `2,469` | Provenance record containing all artifact hashes above |

The trained `weights/best.pt` checkpoint remains inside
`outputs/training/anpr/anpr_witvix_detector_mps_v1/weights/best.pt` and has
**not** been copied into `models/` or activated in the API runtime during this
module.

## OCR Readiness Audit

An audit of the local virtual environment (`.venv`) and `models/` directory
confirmed:

1. **EasyOCR Python package (`easyocr`)**: **Absent**
   (`importlib.util.find_spec("easyocr") is None`).
2. **Offline EasyOCR detector/recognizer weights under `models/`**: **Absent**
   (`models/` currently contains only `.gitkeep`, `README.md`, and
   `yolov8n.pt`; no EasyOCR directory or `.pth` files such as `craft_mlt_25k.pth`
   or recognition network weights are staged).

Because `EasyOcrPlateRecognizer` enforces `download_enabled=False` and requires
local weight files under `models/` (`ROADSENSE_ANPR_OCR_MODEL_DIR`), OCR and
end-to-end plate text recognition remain blocked until `easyocr` and its offline
weight files are explicitly provisioned.
