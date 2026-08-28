# Custom Pothole Model (Step 3 & Step 4 ML Pipeline)

> **Note:** Custom model training and evaluation are offline workflows. The active Streamlit app remains manual-only.

This document describes the reproducible pipeline for fine-tuning a custom YOLOv8n model using the RDD2022 dataset for RoadSense India.

---

## 1. Environment Setup

Install training dependencies into `.venv`:
```bash
pip install -r requirements-training.txt
```

Place official base weights locally at `models/yolov8n.pt`:
```bash
mkdir -p models
curl -L "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt" -o models/yolov8n.pt
```
*(Note: The base YOLOv8n checkpoint was downloaded once from the official Ultralytics release asset registry, while all fine-tuning, metric validation, and test evaluation occur strictly locally on-device).*

---

## 2. Dataset Preparation

> **Critical Data Rule:** The official unlabeled RDD2022 test folder (`India/test/`) contains no ground-truth annotations and must **never** be used for training, validation, or test evaluation metrics.

### Option A: Raw Pascal-VOC RDD2022 (Step 3)
```bash
python scripts/build_group_manifest.py \
  --rdd-root /path/to/RDD2022 \
  --output data/interim/groups.csv \
  --sequence-block-size 10

python scripts/prepare_rdd2022_potholes.py \
  --rdd-root /path/to/RDD2022 \
  --groups-csv data/interim/groups.csv \
  --output-dir data/processed/rdd2022_india_raw_d40_v1 \
  --audit-near-duplicates
```

### Option B: Roboflow-Exported YOLO Dataset (Step 4)
The Roboflow-exported YOLO dataset contains annotations across its `train/`, `valid/`, and `test/` folders, but its pre-baked splits suffer from random adjacent-frame sequence leakage.

The isolated importer `scripts/import_roboflow_yolo_potholes.py`:
- Discards pre-baked Roboflow splits.
- Pools all labeled images across `train`, `valid`, and `test`.
- Groups contiguous sequential frames (e.g. frames `4229, 4230, 4231` remain in one atomic group).
- Filters and remaps source class `6` (`D40`) -> `0` (`pothole`) and drops non-pothole damages.
- Uses atomic staging and validation before promoting to `data/processed/`.

```bash
# Dry-run inspection
python scripts/import_roboflow_yolo_potholes.py \
  --source-root "/Users/aniket/Downloads/RDD2022-India" \
  --output-dir "data/processed/rdd2022_india_roboflow_d40_v1" \
  --max-consecutive-gap 1 \
  --seed 42 \
  --dry-run

# Full import
python scripts/import_roboflow_yolo_potholes.py \
  --source-root "/Users/aniket/Downloads/RDD2022-India" \
  --output-dir "data/processed/rdd2022_india_roboflow_d40_v1" \
  --max-consecutive-gap 1 \
  --seed 42
```

---

## 3. Dataset Validation

Always validate the prepared dataset before training:
```bash
python scripts/validate_yolo_dataset.py \
  --dataset data/processed/rdd2022_india_roboflow_d40_v1
```

---

## 4. Model Training

### Smoke Test (1 Epoch Pipeline Check)
The smoke run is strictly a pipeline sanity check to confirm hardware acceleration, data loading, and metadata tracking:
```bash
python scripts/train_pothole.py \
  --config configs/training/pothole_yolov8n_smoke.yaml \
  --dataset data/processed/rdd2022_india_roboflow_d40_v1
```

### Full Baseline Training (50 Epochs, MPS)
The 50-epoch baseline model fine-tunes `models/yolov8n.pt` using Apple Silicon Metal (MPS) acceleration with early stopping patience of 15:
```bash
python scripts/train_pothole.py \
  --config configs/training/pothole_yolov8n_rdd2022_india.yaml \
  --dataset data/processed/rdd2022_india_roboflow_d40_v1
```

> **Note on Output Directories:** Ultralytics may create a suffixed output directory (e.g., `pothole_yolov8n_rdd2022_india_mps_baseline_v12`) if a run directory with the requested experiment name already exists. After training finishes, copy the exact output directory printed by the script and use that exact folder's `weights/best.pt` for evaluation.

### Resuming an Interrupted Training Run
If an ongoing training run is interrupted, you can safely resume in-place from its `weights/last.pt` checkpoint using `--resume-from`:

```bash
caffeinate -dims .venv/bin/python scripts/train_pothole.py \
  --config configs/training/pothole_yolov8n_rdd2022_india.yaml \
  --dataset data/processed/rdd2022_india_roboflow_d40_v1 \
  --resume-from outputs/training/pothole_yolov8n_rdd2022_india_mps_baseline_v1/weights/last.pt
```

> **Important Rules for Resume:**
> - Passing the exact same prepared dataset and baseline training configuration is mandatory; preflight validation verifies that all hyperparameters (`epochs`, `batch`, `patience`, `device`, `imgsz`, `seed`, `name`) and dataset paths match the checkpoint's saved `train_args`.
> - Resume mode requires the checkpoint path to resolve strictly to `<training_run_dir>/weights/last.pt` within `outputs/training/`.
> - It resumes the existing training run in-place without creating a new output folder or downloading base weights.
> - Resume is strictly for continuing training optimization; it **must not** be used for the held-out test split evaluation.

---

## 5. Model Evaluation (Held-Out Test Split)

> **Strict One-Time Evaluation Rule:** The held-out test split must be evaluated **exactly once** after training is complete and the baseline checkpoint is frozen. To prevent test-set data snooping, repeat evaluations on the same run ID are blocked by an atomic attempt ledger.

```bash
# Replace with the exact printed run output path from training
python scripts/evaluate_pothole.py \
  --weights outputs/training/<exact_run_dir>/weights/best.pt \
  --dataset data/processed/rdd2022_india_roboflow_d40_v1 \
  --device mps \
  --batch 4 \
  --imgsz 640
```

---

## 6. Limitations & Provenance Warning

- **Group Proxy Limitation**: Grouping is a filename-based contiguous-sequence proxy, not verified ground-truth road-route or capture-session grouping.
- **Internal vs Real-World Metrics**: Test split evaluation reflects performance on contiguous held-out sequence runs from the camera rig, not guaranteed real-world generalization across new geographies, weather conditions, or vehicle types.
- **Provenance Tracking**: Every training and evaluation run records full Git SHAs, environment versions, dataset fingerprints, and artifact hashes under `outputs/`.
