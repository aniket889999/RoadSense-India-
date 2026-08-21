# Custom Pothole Model (Step 3 Offline Scaffold)

> **Note:** Step 3 is an offline custom-model scaffold. The active Streamlit app remains manual-only.

This document describes the pipeline for fine-tuning a custom YOLOv8 model using the RDD2022 dataset for RoadSense.

First, install training dependencies:
```bash
pip install -r requirements-training.txt
```

## Data Preparation

You must download the RDD2022 dataset manually. Ensure the dataset matches the following layout exactly before proceeding:
```text
/path/to/RDD2022/
    India/
        train/
            images/
            annotations/xmls/
```

Before processing data, you must provide a grouping manifest to prevent data leakage across splits (e.g., adjacent video frames from the same route must remain in the same split).

### 1. Build Group Manifest
To use unverified sequential proxies:
```bash
python scripts/build_group_manifest.py --rdd-root /path/to/RDD2022 --output data/interim/groups.csv --sequence-block-size 10
```

### 2. Prepare Dataset
```bash
python scripts/prepare_rdd2022_potholes.py --rdd-root /path/to/RDD2022 --groups-csv data/interim/groups.csv --output-dir data/processed/yolo_potholes --audit-near-duplicates
```

### 3. Validate Dataset
```bash
python scripts/validate_yolo_dataset.py --dataset data/processed/yolo_potholes
```

### 4. Train Model
Place your base weights at `models/yolov8n.pt`. If they are not present, training will fail unless you explicitly opt-in to automatic downloading.

To run a smoke test (1 epoch, minimal config):
```bash
python scripts/train_pothole.py --config configs/training/pothole_yolov8n_smoke.yaml --dataset data/processed/yolo_potholes --allow-weight-download
```

For full training:
```bash
python scripts/train_pothole.py --config configs/training/pothole_yolov8n_rdd2022_india.yaml --dataset data/processed/yolo_potholes
```

> **Note:** Ultralytics may increment the run directory name (e.g., `pothole_smoke2`, `pothole_smoke3`) if you run training multiple times. Always check the exact output directory printed at the end of training.

### 5. Evaluate Model
```bash
python scripts/evaluate_pothole.py --weights outputs/training/pothole_yolov8n_smoke/weights/best.pt --dataset data/processed/yolo_potholes
```

## Roboflow-Exported Dataset Import (Step 4)

If working from a Roboflow-exported RDD2022 dataset (e.g. 10-class YOLO format with `train/`, `valid/`, `test/`), note that the original Roboflow random splits have severe frame-adjacency data leakage.

The isolated importer `scripts/import_roboflow_yolo_potholes.py`:
- Discards the pre-baked Roboflow splits.
- Pools all images across `train`, `valid`, and `test`.
- Groups contiguous sequential frames (e.g. frames `4229, 4230, 4231` remain in one atomic group).
- Filters and remaps source class `6` (`D40`) -> `0` (`pothole`) and drops non-pothole damages.
- Uses atomic staging and validation before promoting to `data/processed/`.

Dry run command:
```bash
python scripts/import_roboflow_yolo_potholes.py \
  --source-root "/path/to/RDD2022-India" \
  --output-dir "data/processed/rdd2022_india_roboflow_d40_v1" \
  --max-consecutive-gap 1 \
  --seed 42 \
  --dry-run
```

> **Limitation Warning**: Grouping is a filename-based contiguous-sequence proxy, not verified road-route or capture-session grouping. No Roboflow cloud training or weights are utilized.

## Important Limitations
- Do not use official unlabeled RDD2022 test images.
- All raw/derived data, weights, and evaluation reports remain local and are ignored by git.
- External evaluation on novel real-world sequences is required before app integration.
