# RoadSense India: System Architecture & Data Flow

> **Architecture Style:** Next.js dashboard + FastAPI + shared Python modules;
> relational storage is SQLite by default, with configurable PostgreSQL.

## Current status

ANPR is the current major academic deliverable; road/dashcam and parking are
earlier platform workflows. This inventory describes code at `21586e9`.
**Implemented** means a local prototype path exists, **Partial** means only part
of the proposed integration exists, and **Not implemented** identifies a target.
Neither code presence nor a synthetic test proves field accuracy or readiness.

| Capability | Status | Source and practical boundary |
| --- | --- | --- |
| Still-image ANPR | Implemented | [API](../../services/api/app/routers/anpr.py), [pipeline](../../src/anpr/pipeline.py); bounded image input, local detector/OCR, operator review |
| Schema 1.2 candidate cleanup | Implemented | [Policy results](../anpr/text_cleanup_policy_results.md); separate candidate fields, raw evidence preserved, never auto-accepted; validation-derived |
| Annotated PNG/review JSON | Implemented | [Browser exports](../../apps/web/src/lib/anprExports.ts); explicit downloads, not a server-side ANPR evidence database |
| Recorded-video tracking/replay | Implemented | [Media worker](../../services/api/app/services/video_processor.py), [fusion](../../services/api/app/services/event_fusion.py); candidate review, not driver alerts |
| Parking layout, stability gate and recorded occupancy | Implemented | [Parking routes](../../services/api/app/routers/parking.py); verified layouts and gated local jobs, not continuous RTSP service |
| Hazard-aware capacity policy | Implemented | [Policy](../../src/parking/capacity_policy.py); conservative operator-facing evidence fusion, not structural safety certification |
| Browser camera preview | Implemented | [Camera hook](../../apps/web/src/hooks/useCamera.ts); MediaDevices preview only |
| Server-side WebRTC inference | Not implemented | [Offer endpoint](../../services/api/app/routers/live.py) returns `supported=false`, `not_connected` |
| Interactive MapLibre map | Not implemented | [Map view](../../apps/web/src/components/MapView.tsx) displays presence/absence of telemetry; it does not instantiate a map |
| GPS/NMEA pipeline | Partial | Optional [JSON/scalar fields](../../services/api/app/models/entities.py) exist; automatic NMEA extraction and geolocation of detections are **Not implemented** |
| PostgreSQL persistence | Partial | [Config](../../services/api/app/core/config.py), [engine](../../services/api/app/db/session.py), [DB Compose service](../../infra/docker-compose.yml); configurable, not an automatically selected fallback or a deployment verified in this task |
| PostGIS spatial columns/queries | Not implemented | Compose offers a PostGIS-capable image; current ORM/migrations use JSON/scalar coordinates, without geometry types or `ST_*` query integration |
| ANPR-to-parking-bay association | Not implemented | Still-image ANPR returns a reading response; no registration lookup, identity linkage, payment or barrier control |
| Surveyed world-coordinate/cross-camera fusion | Not implemented | Current hazard associations use explicit operator/layout links, not an automatic mobile-to-CCTV projection |
| Maintenance forecasting, dispatch and motorist routing | Not implemented | Earlier product diagrams are research targets, not delivered work-order or navigation integrations |
| API container deployment | Not implemented | Current Compose contains the database service only; no API container build/run is established |
| Deployment RBAC and automatic ANPR export retention | Not implemented | See [ANPR privacy boundaries](../anpr/privacy_and_retention.md); CORS/reviewer labels are not authenticated roles |

Current ANPR data flow:

```text
Permitted still image → bounded decode → local plate detector → OCR
  → preserved raw/normalized text + explicit reviewed candidate cleanup
  → operator crop/format/quality review → optional browser PNG/JSON download
```

See the [model card](../anpr/model_card.md) for measured results and the
[data card](../anpr/data_card.md) for mixed labels and held-out-test limits.
The numbered sections below retain the earlier road-operations design and must
be interpreted with this status matrix; they are not a claim that every drawn
integration has been delivered.

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
│  │   (Upload / Storage) │  │ (Temporal Clust.)│  │ (Preview Only)  │  │
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
- **Framework — Implemented**: Next.js 14.2.18 App Router with TypeScript.
- **Styling**: Tailored Dark Graphite/Green Command Center palette (`#0B0F12` background, `#10B981` / `#22C55E` detection green, high-contrast typography via Inter/Geist).
- **Video Canvas — Implemented**: Synchronized suggestion overlays over sampled frames; this does not establish localization accuracy.
- **Map View — Not implemented as a map**: Current UI reports telemetry presence/absence. Installing `maplibre-gl` does not instantiate a MapLibre map.

### Service Layer (`services/api`)
- **Framework**: FastAPI with Pydantic v2 schemas.
- **Inference Lifecycle**: Reads `configs/inference/frozen_baseline.yaml`, verifies all 7 artifact hashes, creates private non-symlinked memory descriptor, and predicts via Ultralytics without network calls.
- **Event Bus / WebSocket — Implemented**: Dispatches recorded-job progress; this is not a live WebRTC inference service or a measured throughput guarantee.

### Persistence Layer (`services/api/app/db/`)
- **Engine**: SQLAlchemy 2.0 (Async) + Alembic migrations.
- **Database — Partial integration**: SQLite is selected by the default `DATABASE_URL`; PostgreSQL is an explicit configuration. There is no automatic database fallback. PostGIS spatial use is **Not implemented**.
- **Entities**:
  - `Device`: Registered capture hardware.
  - `DriveSession`: Recorded video processing run with checksums and execution timing.
  - `RawDetection`: Per-frame candidate boxes from YOLOv8n.
  - `RoadEvent`: Temporal cluster of detections representing one candidate pothole location.
  - `ReviewAction`: Application audit records of human decisions; not a cryptographically immutable ledger or proof of authenticated identity.
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

- **Implemented:** User-initiated browser `getUserMedia` preview and a continuous operator-review notice.
- **Not implemented:** Server peer negotiation, stream decoding/inference, and any live sampling/performance contract. The API reports an unsupported/disconnected boundary.
- **Network qualification:** The inspected preview does not create a peer connection or configure STUN/TURN. This is not an audit of all browser/dependency/host network traffic.
