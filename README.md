# RoadSense India

## Project Roadmap

1. **Step 1:** Establish basic repository layout, CI configs, and foundational structures.
2. **Step 2:** Implement manual video analysis using Streamlit (Active Application).
3. **Step 3 & 4:** Custom Pothole Model. Fine-tune a custom YOLOv8n pothole detector on leak-aware RDD2022 India dataset and evaluate on a held-out test split. *(Status: Offline baseline training and evaluation complete; see [Frozen Baseline Results](docs/frozen_baseline_results.md))*.

## Project Problem Statement
RoadSense provides a video-based road condition analytics platform. The active application currently provides a Manual Annotation Baseline: it samples video frames and turns human-provided pothole boxes into evidence-backed incident reports.

## Target User
Road inspector, campus facilities team, or municipal field team.

## MVP User Flow
```text
Road video
→ real metadata extraction
→ sampled frames
→ downloadable annotation kit
→ human-provided pothole CSV
→ manually grouped incidents
→ evidence-backed report
```

## Current Development Status
- **Active Web Application**: Manual Annotation Baseline (Streamlit app samples frames and processes human-provided CSV annotations).
- **Offline ML Pipeline**: Custom YOLOv8n pothole detector baseline trained for 50 epochs on Apple Silicon (MPS) and evaluated once on a held-out test split (Precision: 35.54%, Recall: 31.45%, mAP@50: 29.26%, mAP@50–95: 10.84%). Detailed report: [Frozen Baseline Results](docs/frozen_baseline_results.md).

## Planned Modules
- Video Input processing *(Active in Streamlit)*
- Detectors (Potholes) - *Active Application is Manual Only; Custom YOLOv8n model trained offline*
- Analytics (Traffic context) - *Not Active*
- Fusion (Duplicate removal) - *Currently Manual Only*
- Scoring (Inspection-priority) - *Not Active*
- Exporters (Reports) *(Active in Streamlit)*

## Local Setup Instructions
1. Create a Python 3.12 virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app:
```bash
streamlit run app.py
```

## Important Scope Limits
- In the active Streamlit app, bounding boxes come from a human-provided CSV. The app has not yet connected the offline-trained custom pothole model.
- Strict column parsing (exact columns only): `incident_id,frame_index,x_min,y_min,x_max,y_max,label,note`
- label must be exactly `pothole`
