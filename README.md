# RoadSense India

## Project Roadmap

1. **Step 1:** Establish basic repository layout, CI configs, and foundational structures.
2. **Step 2:** Implement manual video analysis using Streamlit (Active Application).
3. **Step 3 & 4:** Custom Pothole Model. Fine-tune a custom YOLOv8n pothole detector on leak-aware RDD2022 India dataset and evaluate on a held-out test split. *(Status: Offline baseline training and evaluation complete; see [Frozen Baseline Results](docs/frozen_baseline_results.md))*.
4. **Step 5 (Experimental):** Offer local, hash-pinned model suggestions without changing the manual reporting workflow.
5. **Step 6 (Drive Review):** Replay a bounded window of uploaded dashcam footage with green circles around raw model suggestions. *(Recorded-video review only; not a live driver-alert system.)*
6. **Step 7 (Local Curation):** Export explicitly human-confirmed sampled frames into a separate, local-only curation pool for a future experiment. *(It does not train or change the frozen baseline.)*

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
- **Primary Web Application**: A Streamlit dashboard with upload-backed Dashcam Drive Review and a separate Manual Evidence workspace.
- **Dashcam Drive Review**: Selects a bounded source-video window, samples it locally, and renders raw model suggestions as green circles in a non-real-time replay. It does not confirm or report potholes automatically.
- **Offline ML Pipeline**: Custom YOLOv8n pothole detector baseline trained for 50 epochs on Apple Silicon (MPS) and evaluated once on a held-out test split (Precision: 35.54%, Recall: 31.45%, mAP@50: 29.26%, mAP@50–95: 10.84%). Detailed report: [Frozen Baseline Results](docs/frozen_baseline_results.md).
- **Local Curation Pool**: A CLI can export only explicitly human-confirmed kit frames and boxes for a possible future experiment. See [Manual Curation](docs/manual_curation.md).

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

To enable Drive Review, use the project's local training environment. It includes the normal app requirements plus the local Ultralytics runtime:

```bash
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/streamlit run app.py
```

## Important Scope Limits
- Manual incident reports are built only from a human-provided CSV. Drive Review never creates, groups, or verifies incidents, and it never writes directly into a manual report.
- Drive Review reads an uploaded recording in a bounded local session. It is not a continuous live camera, navigation aid, driver warning, or safety system.
- Experimental suggestions use only the configured local frozen checkpoint after SHA-256 and provenance verification. They never download weights, train, call an external inference API, or re-run the held-out test split.
- The manual curation command does not treat unreviewed frames as negative labels, does not copy the original MP4, and does not modify the frozen RDD2022 dataset or its held-out test split.
- The experimental model is a research/portfolio baseline, not production-ready or safety-critical software. Review every suggested box manually before using it.
- Strict column parsing (exact columns only): `incident_id,frame_index,x_min,y_min,x_max,y_max,label,note`
- label must be exactly `pothole`
