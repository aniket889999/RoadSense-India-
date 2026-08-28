# RoadSense India: Frozen Baseline Results (Step 4.7)

> **Status Notice:** This document records the **frozen, one-time held-out test evaluation** of the custom YOLOv8n pothole detector baseline for RoadSense India. The active Streamlit web application remains a manual annotation baseline only.

---

## 1. Executive Summary & Test Split Metrics

The fine-tuned model (`pothole_yolov8n_rdd2022_india_mps_baseline_v1`) was evaluated **exactly once** on the held-out test split of the leak-aware grouped RDD2022 India dataset.

### Held-Out Test Evaluation Results (Single Evaluation Pass)
- **Evaluated Split**: `test` (1,156 images; 230 positive images with 477 ground-truth D40 pothole instances)
- **Precision**: **35.54%** (`0.3554`)
- **Recall**: **31.45%** (`0.3145`)
- **mAP@50**: **29.26%** (`0.2926`)
- **mAP@50–95**: **10.84%** (`0.1084`)
- **Inference Speed**: 0.7 ms preprocess, 10.8 ms inference, 14.1 ms postprocess per image on Apple Silicon Metal (MPS)

> **Strict Evaluation Rule:** These numbers represent a frozen, one-time evaluation on the held-out test split. To prevent test-set data snooping, no hyperparameter tuning or repeat runs are conducted on this split.

---

## 2. Dataset & Split Provenance

- **Training Source**: Roboflow-exported YOLO derivative of RDD2022 India (`data/processed/rdd2022_india_roboflow_d40_v1`).
- **Underlying Dataset Reference**: Original RDD2022 release (Crowdsensing Road Damage Dataset / Figshare `DOI: 10.6084/m9.figshare.21431547.v1`, Arya et al., 2024). Training did **not** run directly from the original Figshare Pascal-VOC directory.
- **Import & Split Logic**: The dedicated importer (`scripts/import_roboflow_yolo_potholes.py`) discarded Roboflow's pre-baked splits (which suffered from adjacent-frame sequence leakage), pooled all labeled images across `train`, `valid`, and `test`, and created a new leak-aware contiguous-sequence split.
- **Remapping**: Source class `6` (`D40` pothole) remapped to single class `0` (`pothole`). All other damage classes (`D00`, `D10`, `D20`, etc.) were dropped.
- **Dataset Fingerprint**: `18bbaae9402323f96a8161829f1d1a35a14a6e45317abc0330397bb6dddbfe33`
- **Leak-Aware Grouping**: Sequential video frames grouped into contiguous runs using filename numerical continuity (max consecutive gap = 1).
- **Split Distribution**:
  - **Total Discovered Images**: 7,706 (6,176 valid negative background images, 1,530 positive images with 3,187 D40 pothole instances)
  - **Contiguous Sequence Runs**: 1,695 groups
  - **Cross-Split Sequence Leakage**: **0 adjacent numerical pairs cross splits** (0 / 6,011 pairs)
  - **Train**: 5,394 images (1,071 positive, 2,231 D40 instances)
  - **Validation**: 1,156 images (229 positive, 479 D40 instances)
  - **Held-Out Test**: 1,156 images (230 positive, 477 D40 instances)

---

## 3. Training & Evaluation Setup

- **Model Architecture**: Ultralytics YOLOv8n (nano detection model)
- **Base Weights**: `models/yolov8n.pt` (SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`)
- **Hardware Acceleration**: Apple Silicon MPS (`device: mps`)
- **Hyperparameters**:
  - Image size: `640`
  - Batch size: `4`
  - Total Epochs: `50`
  - Seed: `42`
  - Early stopping patience: `15` (trained full 50 epochs)
- **Training Duration**: 13,118.9 seconds (~3.64 hours)
- **Evaluation Duration**: ~59.0 seconds on MPS
- **Git Commit SHA**: `1076e5d`
- **Best Model Checkpoint Hash**:
  - Path: `outputs/training/pothole_yolov8n_rdd2022_india_mps_baseline_v1/weights/best.pt`
  - SHA-256: `bdf07ad81197ee15b795de635671d2ef75243492138d34c37d5352a9a777d430`

---

## 4. Engineering & Safety Limitations

1. **Research & Portfolio Baseline**: This model is an offline benchmark and reproducible research artifact. It is **not** production-ready or safety-critical software.
2. **Grouping Proxy Limitation**: Frame grouping relies on filename-based contiguous-sequence heuristics as a proxy for capture runs, not verified GPS ground-truth road-routes or distinct capture sessions.
3. **Generalization Scope**: These evaluation metrics reflect performance on contiguous held-out sequence runs from the RDD2022 camera rig. They do **not** guarantee generalization to new geographic areas, camera sensor setups, varying weather conditions, vehicle mounting vibration, or nighttime lighting.
4. **Local Artifact Isolation**: Checkpoint weights (`best.pt`), processed datasets, and raw training runs remain strictly local/ignored by `.gitignore` and are not committed to Git.

---

## 5. Next Steps & Future Work

1. **Experimental In-App Inference**: Connect `best.pt` to the Streamlit application as an *explicitly experimental* automated inference option alongside the manual annotation baseline.
2. **External Route Validation**: Evaluate detection and false-alarm performance on a separately collected, geographically distinct road-route dataset with independent ground-truth labeling.
