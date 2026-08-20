# RoadSense India

## Project Roadmap

1. **Step 1:** Establish basic repository layout, CI configs, and foundational structures.
2. **Step 2:** Implement manual video analysis using Streamlit (Active Application).
3. **Step 3:** Custom Pothole Model. Note: Step 3 is an offline custom-model scaffold; the active Streamlit app remains manual-only.

## Project Problem Statement
RoadSense currently provides a Manual Annotation Baseline. It samples video frames and turns human-provided pothole boxes into evidence-backed incident reports.

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
Manual Annotation Baseline

## Planned Modules
- Video Input processing
- Detectors (Potholes) - *Currently Manual Only*
- Analytics (Traffic context) - *Not Active*
- Fusion (Duplicate removal) - *Currently Manual Only*
- Scoring (Inspection-priority) - *Not Active*
- Exporters (Reports)

## Local Setup Instructions
1. Create a Python 3.12 virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app:
```bash
streamlit run app.py
```

## Important Scope Limits
- These boxes come from a human-provided CSV. RoadSense has not yet run a trained pothole model, calculated traffic volume, or created a repair priority.
- Strict column parsing (exact columns only): `incident_id,frame_index,x_min,y_min,x_max,y_max,label,note`
- label must be exactly `pothole`
