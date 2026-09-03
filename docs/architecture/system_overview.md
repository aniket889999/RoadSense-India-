# RoadSense India: System Architecture & Data Flow

> **Architecture Style:** Monorepo Multi-Layer System (Next.js Dashboard + FastAPI Async Engine + PostgreSQL/PostGIS)

---

## 1. High-Level Architectural Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION LAYER                            │
│                 apps/web (Next.js App Router, TypeScript)              │
│                                                                        │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │   Command Center     │  │   Review Queue   │  │  System Health   │  │
│  │ (Video Player/Canvas)│  │ (Decision Panel) │  │ (Provenance/MPS) │  │
│  └──────────┬───────────┘  └─────────┬────────┘  └─────────┬────────┘  │
└─────────────┼────────────────────────┼─────────────────────┼───────────┘
              │ REST / JSON            │ PATCH Actions       │ WebSocket
              ▼                        ▼                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           API & INFERENCE LAYER                        │
│                   services/api (FastAPI, WebSockets, Uvicorn)          │
│                                                                        │
│  ┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │   Session Service    │  │  RoadEvent Fusion│  │ Live Stream Bndry│  │
│  │   (Upload / Storage) │  │ (Temporal Clust.)│  │ (WebRTC Ready)   │  │
│  └──────────┬───────────┘  └─────────┬────────┘  └─────────┬────────┘  │
│             │                        │                     │           │
│             ▼                        ▼                     ▼           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │     Local ML Inference Adapter (src/ml/local_model_runtime.py)   │  │
│  │  - Fail-Closed Provenance (src/ml/model_provenance.py)           │  │
│  │  - Bounded Drive Review Planner (src/ml/drive_review.py)         │  │
│  │  - Frozen Baseline Checkpoint (outputs/training/.../best.pt)    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │ Async SQLAlchemy / Alembic
                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           PERSISTENCE LAYER                            │
│                     PostgreSQL 16 + PostGIS / SQLite                   │
│                                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ DriveSession │  │ RawDetection │  │  RoadEvent   │  │ReviewAction│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack & Component Responsibilities

### Presentation Layer (`apps/web`)
- **Framework**: Next.js 14/15 App Router with TypeScript.
- **Styling**: Tailored Dark Graphite/Green Command Center palette (`#0B0F12` background, `#10B981` / `#22C55E` detection green, high-contrast typography via Inter/Geist).
- **Video Canvas**: Synchronized overlay canvas rendering mathematically exact bounding circles over sampled video frames.
- **Map View**: MapLibre GL instance activated strictly when real GPS points exist in session telemetry.

### Service Layer (`services/api`)
- **Framework**: FastAPI with Pydantic v2 schemas.
- **Inference Lifecycle**: Reads `configs/inference/frozen_baseline.yaml`, verifies all 7 artifact hashes, creates private non-symlinked memory descriptor, and predicts via Ultralytics without network calls.
- **Event Bus / WebSocket**: Dispatches real-time processing milestones (`queued`, `validating`, `sampling`, `model_loading`, `processing [N/M]`, `rendering`, `complete`).

### Persistence Layer (`services/api/app/db/`)
- **Engine**: SQLAlchemy 2.0 (Async) + Alembic migrations.
- **Database**: PostgreSQL with PostGIS extension (with automatic SQLite fallback for lightweight unit tests).
- **Entities**:
  - `Device`: Registered capture hardware.
  - `DriveSession`: Recorded video processing run with checksums and execution timing.
  - `RawDetection`: Per-frame candidate boxes from YOLOv8n.
  - `RoadEvent`: Temporal cluster of detections representing one candidate pothole location.
  - `ReviewAction`: Immutable audit log of human reviewer confirmations, rejections, and notes.
  - `Artifact`: Output videos, CSV manifests, and JSON summaries.

---

## 3. Upload Processing Data Flow

1. **Client Submission**: User selects local video (`.mp4`, `.mov`, `.avi`). Frontend validates mime type and file size.
2. **Session Creation**: API streams file to temporary secure spool (`outputs/sessions/<uuid>/raw_video.mp4`), computes SHA-256 digest, and writes `DriveSession` with state `QUEUED`.
3. **Background Job Execution**:
   - `validating`: Checks video container, dimensions, framerate, and duration.
   - `sampling`: Computes bounded sampling plan (e.g. 5 FPS, capped at 150 frames per window).
   - `model_loading`: Runs fail-closed provenance verification; loads checkpoint into memory.
   - `processing`: Iterates sampled frames, yields raw predictions, broadcasts frame-by-frame progress via WebSocket.
   - `rendering`: Generates annotated MP4 replay with green circle overlays and transparent watermarks.
   - `fusion`: Clusters raw detections into `RoadEvent` records based on timestamp proximity ($\Delta t < 1.5s$) and spatial box overlap (IoU > 0.15 or centroid distance < 60px).
   - `complete`: Finalizes CSV/JSON exports, records processing duration and artifact hashes.
4. **Interactive Review**: Inspector reviews suggested Road Events, confirms/rejects each, and triggers export download.

---

## 4. Live Camera & WebRTC Boundary

- **Local-Only Policy**: Camera preview and streaming interact solely with `localhost` / local network.
- **No External STUN/TURN**: By default, no third-party STUN/TURN servers are configured.
- **Controlled Ingest**: Streams or frames are processed at a bounded cadence (e.g. 2 FPS) to avoid thermal throttling on edge devices.
- **Explicit Operator Notice**: The UI displays `"Experimental operator review — not a driver alert"` continuously.
