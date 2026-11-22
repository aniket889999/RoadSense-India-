# RoadSense India: Industry Readiness & Operating Boundaries

> **Status:** Local research/operations prototype; deployment readiness is not established.
> **Current academic deliverable:** Operator-assisted still-image ANPR.
> **Earlier platform:** Recorded dashcam review and gated parking/hazard workflows.

## Current status

The major project now addresses plate detection, OCR, limited format assessment,
capture-quality review and explicit annotated-image/JSON exports. Applied schema
1.2 cleanup preserves original OCR evidence and creates a reviewed candidate.
Its **58/94 (61.7%) agreement, 18.38% CER and 21 fixed / 0 broken** are derived
from the same validation reads used to design the rule; they are not independent
accuracy. No cleaned reading is auto-accepted, and ANPR held-out testing is
pending. Source: [cleanup policy results](../anpr/text_cleanup_policy_results.md).

| Capability | Status | Operating boundary |
| --- | --- | --- |
| ANPR inspection and candidate export | Implemented | Local operator-assisted prototype, not registration verification |
| Recorded dashcam review, parking gates and capacity snapshots | Implemented | Earlier research workflows; field accuracy and facility acceptance remain unestablished |
| Browser camera preview | Implemented | Server-side WebRTC inference is **Not implemented** |
| MapLibre interaction and automatic GPS/NMEA extraction | Not implemented | Telemetry fields and an absence/presence placeholder are **Partial** |
| PostgreSQL deployment | Partial | Configurable backend and DB Compose service; SQLite is the code default |
| PostGIS spatial query integration | Not implemented | Available DB image does not establish spatial column/query use |
| ANPR-to-bay association, registry lookup, toll/barrier control | Not implemented | Separate inspection workflows only |
| API Docker deployment, deployment RBAC, automated repair dispatch | Not implemented | Product/deployment targets, not current deliverables |

Evidence and source links are centralized in the
[architecture status matrix](../architecture/system_overview.md#current-status).
The numbered sections below preserve earlier road-operations requirements and
persona examples. They do not establish buyer outcomes, security certification,
live throughput, repair-system integration or a complete deployment.

---

## 1. Problem Statement & Operational Context

Municipal road authorities, highway concessionaires, campus facilities teams, and infrastructure inspection contractors capture hundreds of hours of dashcam video from patrol vehicles, buses, and survey trucks. Today, identifying road degradation and potholes from this imagery requires labor-intensive manual inspection, leading to inspection backlogs, inconsistent reporting, and uncoordinated repair dispatches.

**RoadSense India** provides a local-first, auditable operations platform that:
1. Accepts recorded dashcam files through the configured API for local-model processing; this is not an audit of all host/network traffic.
2. Validates video containers, orientation, and streams with secure FFprobe intake.
3. Streams frames boundedly through an OpenCV pipeline (preventing memory exhaustion).
4. Runs local, hardware-accelerated computer vision suggestions (using a frozen, provenance-verified YOLOv8n detector on Apple MPS / CUDA).
5. Tracks candidate detections across consecutive frames with ByteTrack to assign session-local track IDs and maintain continuity through temporary occlusions.
6. Consolidates stable ByteTrack tracks into deduplicated, reviewable **Road Events**.
7. Encodes H.264 / yuv420p / faststart browser-compatible annotated MP4 videos using FFmpeg.
8. Provides human inspectors with an evidence-backed review workbench to confirm, reject, or mark suggestions for revisit.
9. Exports auditable, path-free field inspection dossiers and CSV/JSON packages for enterprise GIS and maintenance tracking.

---

## 2. Target Personas & Core Workflows

### Persona A: Municipal Road Inspection Officer
- **Goal**: Review dashcam video captured during morning survey routes and generate verified pothole repair work-orders.
- **Pain Point**: Cannot review 4 hours of raw 30 FPS video manually; needs bounding boxes grouped into distinct road locations so the same pothole appearing across 15 frames is treated as one review item.
- **Workflow**:
  1. Uploads `.mp4` file from dashcam SD card.
  2. Watches sampled playback with highlighted suggestion markers and ByteTrack IDs.
  3. Reviews each grouped Road Event with thumbnail evidence and confidence score.
  4. Confirms valid potholes, rejects false alarms (e.g., shadows, manhole covers).
  5. Exports CSV report for the public works dispatch system.

### Persona B: Fleet Operations & Safety Manager
- **Goal**: Audit road quality along recurring logistics routes or campus roads without leaking proprietary camera feeds to third-party cloud APIs.
- **Pain Point**: Strict corporate data sovereignty rules prevent uploading dashcam video to external cloud providers.
- **Workflow**:
  1. Target: a local/on-premise deployment on Apple Silicon or Linux. A database Compose service exists; an API container deployment is not implemented.
  2. Processes recorded video with local assets; continuous stream ingestion and an end-to-end offline guarantee are not established.
  3. Audits system health, model provenance hashes, and processing latency.

---

## 3. Strict Safety & Liability Boundaries

To prevent misuse, misleading claims, and safety hazards, RoadSense India enforces non-negotiable operational boundaries:

| Dimension | System Capability | Strict Non-Claim / Boundary |
| :--- | :--- | :--- |
| **Pothole Confirmation** | Generates raw candidate suggestions with confidence scores. | **Never claims confirmed potholes without human reviewer sign-off.** Raw model detections are labeled `UNVERIFIED MODEL SUGGESTION`. |
| **Depth & Severity** | Identifies 2D surface bounding boxes (`x_min, y_min, x_max, y_max`). | **Never claims pothole depth, volumetric severity, or structural axle hazard.** Single monocular dashcam video cannot measure depth without calibrated stereoscopy or LiDAR. |
| **Accident / Hazard Risk** | Highlights visual surface irregularities. | **Never generates driving risk scores, driver warnings, or collision predictions.** RoadSense is an offline review tool, not an Advanced Driver Assistance System (ADAS). |
| **Repair Priority** | Groups detections temporally via ByteTrack. | **Never calculates automated repair priority, municipal budgeting, or engineering urgency.** |
| **GPS Geolocation — Partial** | Optional metadata fields and a telemetry presence/absence view. Automatic NMEA extraction and interactive mapping are **Not implemented**. | Do not invent coordinates or claim verified geolocation from video alone. |
| **Real-Time Performance** | Displays actual measured frame processing latency. | **Never claims real-time performance unless directly measured on host hardware.** |
| **Coverage Guarantee** | Evaluates on trained distribution (RDD2022 India). | **Does not claim complete detection coverage.** Reviewers must be aware that weather, glare, camera blur, and novel asphalt textures will produce false negatives and false positives. |

---

## 4. Lifecycle & Incident State Machine

```
   [ Raw Frame Detections (YOLOv8n) ]
                  │
                  ▼
   [ ByteTrack Multi-Object Tracking ] ──> (Assigns session-local Track IDs, Kalman motion gating)
                  │
                  ▼
   [ Road Event Fusion Engine ] ──> (Consolidates stable tracks into RoadEvents, Status: PENDING_REVIEW)
                  │
          ┌───────┴───────────────────┐
          │ Reviewer Action           │
          ▼                           ▼                           ▼
    [ CONFIRMED ]               [ REJECTED ]              [ NEEDS_REVISIT ]
      • Validated by human        • False alarm (shadow/manhole) • Ambiguous/blurred
      • Included in export        • Excluded from export         • Flagged for secondary check
      • Logged in audit trail     • Logged in audit trail        • Logged in audit trail
```

---

## 5. Model Provenance & Verifiable Engineering

The earlier frozen pothole baseline records the following provenance. Its weak
held-out results do not demonstrate reliable road coverage or driver-alert
performance, and they must not be presented as ANPR results. Source:
[frozen pothole baseline](../frozen_baseline_results.md).
- **Baseline Model ID**: `pothole_yolov8n_rdd2022_india_mps_baseline_v1`
- **Frozen Checkpoint SHA-256**: `bdf07ad81197ee15b795de635671d2ef75243492138d34c37d5352a9a777d430`
- **ByteTrack Config SHA-256**: Pinned in `configs/tracking/bytetrack_default.yaml`
- **Training/Eval Git Commit**: `1076e5d`
- **Evaluation Benchmark (Test Split)**: Precision 35.54%, Recall 31.45%, mAP@50 29.26%, mAP@50-95 10.84% (single held-out evaluation pass; no snooping).
