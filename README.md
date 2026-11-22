# RoadSense India — ANPR and Visual Inspection Platform

## Current major project: operator-assisted ANPR

The current problem statement is to locate number plates in permitted vehicle
images, read candidate text, check a supported plate format, test multiple
images, and explain blur/angle failures and parking/toll applications. RoadSense
implements vehicle plate detection, OCR, supported Indian-format checks and
human review. It includes enlarged plate crops,
advisory blur/illumination/tilt cues, annotated PNG export and a JSON review report.
The ANPR screen is available in the existing local dashboard.

The trained YOLOv8n plate detector is paired with pretrained RapidOCR PP-OCRv4;
a locally adapted EasyOCR recognizer is preserved as a comparison. Schema 1.2
now exposes explicit cleanup as separate reviewed `candidate_text`, while raw
and normalized OCR evidence remain unchanged.

On the same **71 validation images / 94 plates**, the applied candidate policy
has **58/94 (61.7%) exact agreement, 18.38% CER, and 21 fixed / 0 broken reads**.
The original normalized-text baseline remains 37/94 (39.36%) and 24.07% CER.
The policy was derived from these validation failures, so this is an optimistic
validation agreement with `publisher_mixed` labels, mainly trucks, **not an
independent accuracy estimate**. Cleaned candidates are never auto-accepted;
all RapidOCR readings require human review. The ANPR held-out test is pending.
Source: [candidate cleanup policy results](docs/anpr/text_cleanup_policy_results.md).

With the existing ANPR dependencies and staged, hash-pinned local model assets:

```bash
.venv/bin/python scripts/anpr_preflight.py   # read-only readiness check; exit 1 on any FAIL
.venv/bin/python scripts/serve_anpr_local.py
```

`requirements-anpr.txt` records the package versions of the working environment
used for the reported results; it is compared by the preflight, not installed over
the existing `.venv`.

Start the dashboard from `apps/web` with `npm run start` after `npm run build`,
then select **ANPR Inspection**. Use `--engine easyocr` for the adapted EasyOCR
comparison. Models and sample images are not included in Git.

See the [demonstration and jury guide](docs/anpr/demo_and_jury_guide.md),
[model card](docs/anpr/model_card.md), [data card](docs/anpr/data_card.md),
[privacy and retention card](docs/anpr/privacy_and_retention.md),
[model comparison and asset details](docs/anpr/ocr_model_comparison_results.md),
and [detector training results](docs/anpr/witvix_detector_baseline_results.md).

## Current status and stack

**Implemented** means a local prototype code path exists; it does not establish
field accuracy, deployment security or operational certification. **Partial**
identifies a boundary/schema/configuration without its full proposed workflow.

| Capability | Status | Current boundary |
| --- | --- | --- |
| Still-image ANPR and reviewed cleanup | Implemented | Trained plate detector, pretrained RapidOCR, preserved raw text, candidate/format review and explicit PNG/JSON exports |
| Earlier dashcam and parking workflows | Implemented | Recorded-video review, session-local tracking, verified ROI layouts, camera-stability gate and operator-facing capacity policy; local research evidence |
| Browser camera preview | Implemented | `getUserMedia` preview; server WebRTC inference is **Not implemented** |
| Interactive MapLibre map / GPS extraction | Not implemented | Current map component shows telemetry presence/absence; a dependency and coordinate fields are not a rendered map or an NMEA parser |
| Persistence | Partial | Async SQLAlchemy/Alembic; SQLite is the default, PostgreSQL can be configured; Compose supplies a PostGIS-capable DB image |
| PostGIS spatial columns/queries | Not implemented | Current geometry is JSON/scalar fields, without spatial query use |
| ANPR-to-parking-bay association, registration lookup, toll/barrier control | Not implemented | Separate inspection modules; no identity linkage or automatic actuation |

The stack is Python 3.12, OpenCV/NumPy, PyTorch/Ultralytics, RapidOCR/ONNX Runtime
(CPU OCR), EasyOCR, FastAPI/Pydantic, SQLAlchemy/Alembic, and Next.js 14.2.18 with
React, TypeScript and Tailwind. Earlier video paths use FFmpeg/ffprobe and
ByteTrack. The local ANPR launcher selects the detector device explicitly;
the recorded detector run uses Apple MPS. See the source-linked
[architecture status matrix](docs/architecture/system_overview.md#current-status)
for the difference between implemented paths and research targets.

## Earlier platform milestones

The following sequence preserves the project's road/dashcam/parking history.
ANPR is now the main academic deliverable, using the existing inspection platform.

1. **Step 1:** Establish basic repository layout, CI configs, and foundational structures.
2. **Step 2:** Implement manual video analysis using Streamlit (preserved research interface).
3. **Step 3 & 4:** Custom Pothole Model. Fine-tune a custom YOLOv8n pothole detector on leak-aware RDD2022 India dataset and evaluate on a held-out test split. *(Status: Offline baseline training and evaluation complete; see [Frozen Baseline Results](docs/frozen_baseline_results.md))*.
4. **Step 5 (Experimental):** Offer local, hash-pinned model suggestions without changing the manual reporting workflow.
5. **Step 6 (Drive Review):** Replay a bounded window of uploaded dashcam footage with green circles around raw model suggestions. *(Recorded-video review only; not a live driver-alert system.)*
6. **Step 7 (Local Curation):** Export explicitly human-confirmed sampled frames into a separate, local-only curation pool for a future experiment. *(It does not train or change the frozen baseline.)*
7. **Step 8 (Operations Command Center):** Next.js frontend, FastAPI async backend, relational persistence and application audit records. PostGIS spatial use remains a target, not delivered functionality.
8. **Step 9 (Media Intelligence Pipeline):** OpenCV bounded streaming, FFprobe container verification, ByteTrack multi-object tracking, and FFmpeg H.264 web encoding.

## Earlier road and site-operations problem statement

The earlier RoadSense prototype ingests recorded dashcam video, tracks pothole
candidates across frames, groups them into reviewable Road Events and supports
human review. Parking phases add verified layouts, guarded occupancy jobs and
human-reviewed hazard/capacity evidence. These form the surrounding platform;
they do not automatically associate ANPR readings with road events or bays.

## Target User

The ANPR deliverable serves a local reviewer and academic demonstration. Earlier
platform personas include road inspectors, campus facilities teams and fleet
auditors; deployment benefits for those organizations are not established results.

## Earlier recorded-video media pipeline

```text
Dashcam Video (.mp4)
  → FFprobe Secure Media Intake (codec, rotation, duration, fps validation)
  → Memory-Bounded OpenCV Frame Streaming
  → Provenance-Verified Frozen YOLOv8n (explicit configured device)
  → ByteTrack Multi-Object Tracking (Session-local Track IDs & Kalman Gating)
  → Road Event Fusion Engine (Deduplicated candidate events)
  → FFmpeg H.264/yuv420p Faststart Browser-Ready Video Encoding
  → Human Inspection Workbench (Confirm / Reject / Revisit decisions)
  → Auditable Field Inspection Dossier (.zip / .csv / .json)
```

## Current Development Status

- **Operations Dashboard (`apps/web`)**: ANPR inspection, recorded-video/canvas review, ByteTrack labels and parking views; the spatial view is a placeholder, not an interactive map.
- **FastAPI Backend (`services/api`)**: Async API, configurable relational persistence, WebSocket job progress and media artifacts. A progress WebSocket is not a live camera inference transport.
- **Media Intelligence Pipeline (`src/media/`, `src/tracking/`)**: FFprobe validation, OpenCV streaming memory bounding, ByteTrack adapter, and FFmpeg libx264 encoding.
- **Earlier pothole ML baseline — limited performance**: Single-class YOLOv8n trained on MPS and evaluated once on its held-out split: Precision 35.54%, Recall 31.45%, mAP@50 29.26%, mAP@50–95 10.84%. This is a weak research baseline with contiguous-sequence proxy and generalization limits; it does not support reliable driver alerts. This completed pothole evaluation is separate from the still-pending ANPR test evaluation. [Frozen Baseline Results](docs/frozen_baseline_results.md).

## Historical clean-environment setup

Use the existing-environment ANPR preflight/launcher above for the current demo.
The commands below preserve the original platform setup sequence. They are not
a verified fresh ANPR installation recipe and must not be run over the working
`.venv` to repair it. Models/OCR assets require separate permitted provisioning;
no automatic weight download or environment replacement is part of the demo.

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
- **Geolocation boundary**: Optional telemetry fields and an absence/presence view exist. Automated NMEA extraction, route rendering and verified pothole geolocation are not implemented; no coordinates should be inferred from video alone.
- **No Volumetric Depth or Hazard Risk**: Monocular dashcam video cannot measure 3D pothole depth. RoadSense never claims pothole depth, axle risk, or automated repair priority.
- **Not an ADAS / Collision Alert System**: The application is an offline/local review workbench, not a certified real-time driver alert or collision avoidance tool.
- **Local inference boundary**: Inspected ANPR/media paths use local assets, and the provided ANPR launcher binds to loopback. This is not an end-to-end audit of browser, dependency, IDE or host network behavior. See the [privacy card](docs/anpr/privacy_and_retention.md) for storage/export limits.
