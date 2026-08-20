# RoadSense India

## Project Problem Statement
Automate the analysis of recorded road videos to detect pothole candidates, add visible traffic context, remove duplicate sightings, and produce a repair/inspection-priority report.

## Target User
Road inspector, campus facilities team, or municipal field team.

## MVP User Flow
```text
Road video + optional GPS
→ pothole candidate detection
→ visible traffic context
→ duplicate removal
→ inspection-priority report
```

## Current Development Status
Foundation setup

## Planned Modules
- Video Input processing
- Detectors (Potholes)
- Analytics (Traffic context)
- Fusion (Duplicate removal)
- Scoring (Inspection-priority)
- Exporters (Reports)

## Local Setup Instructions
1. Create a Python 3.12 virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app:
```bash
streamlit run app.py
```

## Important Scope Limits
- potholes only in MVP
- traffic context is not true city traffic volume
- priority is for human inspection, not accident prediction
- GPS is approximate camera position, not exact pothole position