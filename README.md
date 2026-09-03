# RoadSense India

## Project Roadmap

1. **Step 1:** Establish basic repository layout, CI configs, and foundational structures.
2. **Step 2:** Implement manual video analysis using Streamlit (Active Application).
3. **Step 3 & 4:** Custom Pothole Model. Fine-tune a custom YOLOv8n pothole detector on leak-aware RDD2022 India dataset and evaluate on a held-out test split. *(Status: Offline baseline training and evaluation complete; see [Frozen Baseline Results](docs/frozen_baseline_results.md))*.
4. **Step 5 (Experimental):** Offer local, hash-pinned model suggestions without changing the manual reporting workflow.
5. **Step 6 (Drive Review):** Replay a bounded window of uploaded dashcam footage with green circles around raw model suggestions. *(Recorded-video review only; not a live driver-alert system.)*
6. **Step 7 (Local Curation):** Export explicitly human-confirmed sampled frames into a separate, local-only curation pool for a future experiment. *(It does not train or change the frozen baseline.)*
7. **Step 8 (Operations Command Center):** Industrial-grade local dashcam operations platform with Next.js 14 frontend, FastAPI async backend, PostgreSQL/PostGIS persistence, and immutable audit logs.
8. **Step 9 (Media Intelligence Pipeline):** OpenCV bounded streaming, FFprobe container verification, ByteTrack multi-object tracking, and FFmpeg H.264 web encoding.

## Project Problem Statement
RoadSense provides an industrial-grade, local-first road condition analytics platform. It ingests dashcam video, tracks pothole candidate detections across frames with ByteTrack, consolidates tracks into reviewable Road Events, and equips municipal road inspectors with an auditable verification workflow.

## Target User
Road inspector, municipal public works department, campus facilities team, or fleet safety auditor.

## Production Architecture & Media Pipeline
```text
Dashcam Video (.mp4)
  → FFprobe Secure Media Intake (codec, rotation, duration, fps validation)
  → Memory-Bounded OpenCV Frame Streaming
  → Provenance-Verified Frozen YOLOv8n (Apple MPS / CUDA)
  → ByteTrack Multi-Object Tracking (Session-local Track IDs & Kalman Gating)
  → Road Event Fusion Engine (Deduplicated candidate events)
  → FFmpeg H.264/yuv420p Faststart Browser-Ready Video Encoding
  → Human Inspection Workbench (Confirm / Reject / Revisit decisions)
  → Auditable Field Inspection Dossier (.zip / .csv / .json)
```

## Current Development Status
- **Operations Command Center (`apps/web`)**: Next.js 14 dashboard with synchronized video/canvas viewport, ByteTrack labels, spatial route maps, and audit-logged review drawer.
- **FastAPI Backend (`services/api`)**: Async API with PostgreSQL/PostGIS & SQLite persistence, WebSocket live progress, and media artifact generation.
- **Media Intelligence Pipeline (`src/media/`, `src/tracking/`)**: FFprobe validation, OpenCV streaming memory bounding, ByteTrack adapter, and FFmpeg libx264 encoding.
- **Offline ML Baseline**: Custom single-class YOLOv8n detector trained on Apple Silicon (MPS) and evaluated once on a held-out test split (Precision: 35.54%, Recall: 31.45%, mAP@50: 29.26%, mAP@50–95: 10.84%). Detailed report: [Frozen Baseline Results](docs/frozen_baseline_results.md).

## Local Setup & Quickstart

### 1. Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r services/api/requirements.txt
```

### 2. Start FastAPI Backend Service
```bash
.venv/bin/uvicorn services.api.app.main:app --host 127.0.0.1 --port 8000
```

### 3. Start Next.js Operations Dashboard
```bash
cd apps/web
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to access the Command Center.

## Important Scope & Operational Limits
- **AI Suggestions Are Not Confirmed Potholes**: Every detection is labeled `AI SUGGESTION — NOT HUMAN VERIFIED` until signed off by a human inspector.
- **Zero Coordinate Fabrication**: GPS telemetry is extracted strictly from embedded NMEA/sensor metadata. When absent, the system displays "No GPS Supplied" rather than fabricating artificial coordinates.
- **No Volumetric Depth or Hazard Risk**: Monocular dashcam video cannot measure 3D pothole depth. RoadSense never claims pothole depth, axle risk, or automated repair priority.
- **Not an ADAS / Collision Alert System**: The application is an offline/local review workbench, not a certified real-time driver alert or collision avoidance tool.
- **100% Local Execution**: Zero cloud API calls, telemetry, or external STUN/TURN services.
