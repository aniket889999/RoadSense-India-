# RoadSense SiteOps — Hazard-Aware Parking & Pavement Intelligence
## Product Definition, Novelty Hypotheses & Operational Specification

> **Working Name:** RoadSense SiteOps — Hazard-Aware Parking and Pavement Intelligence
> **Classification:** Applied Research & Product Architecture Specification
> **Status:** Research Foundation Baseline (Pre-Implementation)
> **Integrity Notice:** This specification documents a defensible research and product architecture. It makes no claims of unverified patentability, academic novelty, or production readiness without rigorous empirical validation.

---

## 1. Executive Summary & Core Product Statement

Traditional smart parking systems monitor space availability by answering a simple binary question: *"Is a vehicle physically occupying this geometric slot?"* Concurrently, pavement condition assessment systems monitor road quality by asking: *"Where are the surface defects located?"*

In commercial, hospital, university, and industrial facilities, these two operational questions are deeply intertwined. A physically empty parking bay that contains severe structural potholes, pooling water over subsurface degradation, or a blocked approach lane is **not usable capacity**. Directing motorists to degraded or obstructed spaces causes vehicle damage, congestion, driver frustration, and facility liability.

### Core Product Statement
> **"RoadSense SiteOps estimates safe usable parking capacity by combining fixed-camera occupancy observations with human-reviewed pavement hazards collected through mobile inspection."**

RoadSense SiteOps bridges the gap between static occupancy monitoring and mobile asset inspection. It transforms raw geometric occupancy into an actionable, hazard-aware operational metric (**Safe Usable Capacity**) while providing facility managers with an evidence-based maintenance dispatch workflow.

---

## 2. Problem Definition & Operational Friction

### 2.1 The Operational Problem
Facility managers at large closed or semi-controlled campuses (hospitals, university campuses, corporate parks, transit hubs, logistics centers) manage thousands of parking spaces. They face two persistent operational failures:
1. **Phantom Capacity / False Availability:** Parking guidance indicators report "12 spaces free on Level 2 / North Lot", but 3 of those spaces have severe pavement subsidence or potholes that risk chassis damage, and 1 space is blocked by illegal staging or debris. Drivers enter, discover unusable stalls, circle aimlessly, and create gridlock.
2. **Disconnected Maintenance Cycles:** Pavement inspection is performed infrequently (annual or bi-annual audits) using disconnected spreadsheet logs or third-party survey reports. Potholes in high-turnover bays go unaddressed until customer claims or facility damage occur, while maintenance crews lack telemetry on *when* lots are empty enough to perform asphalt repairs without disrupting traffic.

### 2.2 Why Existing Isolated Approaches Fail
- **Why Ordinary Parking Occupancy Is Insufficient:** Standard parking vision systems (e.g., YOLO detection over static camera polygons) treat every unobstructed coordinate as "available." They possess zero awareness of surface degradation, road hazards, standing water depth, or blocked approach paths.
- **Why Ordinary Pothole Detection Is Insufficient:** Standalone dashcam pothole detectors (such as the RoadSense offline YOLOv8n baseline) identify bounding boxes on road surfaces from mobile cameras. However, they lack fixed spatial permanence, lot-geometry context, space-association logic, and occupancy scheduling awareness.
- **Why Simply Placing Both in One Dashboard Is NOT Novel:** Displaying a list of detected potholes next to an occupancy counter on the same screen is merely UI juxtaposition, not system intelligence. It does not perform geometric space association, multi-modal evidence fusion, approach-zone reachability checks, temporal state arbitration, or maintenance opportunity scheduling.

---

## 3. Target Users, Buyers, and Operational Workflow

### 3.1 Target Customers & Buyers
| Role | Organization Type | Primary Value Realized |
| :--- | :--- | :--- |
| **Director of Facility Operations** | Major Hospital / Medical Campus | Eliminates emergency patient/visitor parking frustration and reduces slip/trip/chassis claims on hospital grounds. |
| **Head of Campus Infrastructure** | University / Higher Education | Prioritizes parking lot maintenance during scheduled low-occupancy windows without closing active academic lots. |
| **Logistics & Fleet Yard Manager** | Industrial / Freight Terminal | Prevents heavy trailer and fleet vehicle tire/axle damage in designated staging and parking bays. |
| **Municipal Parking Authority** | Controlled Off-Street Public Parking | Increases parking asset revenue efficiency by maintaining true safe usable capacity. |

### 3.2 Real Operational Workflow

```
[Mobile Inspection Vehicle / Cart]
     │ (Scheduled / opportunistic drive with dashcam)
     ▼
[RoadSense Mobile Pipeline (Frozen Pothole Detector)]
     │ (Extracts candidate surface defects & track IDs)
     ▼
[Human Review Queue (Facility Inspector)]
     │ (Confirms/rejects defect, tags severity & space context)
     ▼
[Verified Surface Hazard Layer (Spatial DB)] ───┐
                                                │ (Spatial & Geometric
[Fixed Facility CCTV (Overhead / Pole)]        │  Association Engine)
     │ (Continuous 24/7 RTSP video stream)      │
     ▼                                          │
[Fixed Vision Pipeline (Vehicle Det + Tracker)] │
     │ (ByteTrack + Temporal Hysteresis)        │
     ▼                                          │
[Space Occupancy State Engine] ─────────────────┘
     │
     ▼
[RoadSense SiteOps Command Center]
     ├── Safe Usable Capacity Counter (Real-time motorist guidance feed)
     ├── 6-State Space Map (Color-coded live status)
     └── Maintenance Opportunity Workflow (Recommended repair dispatch during low-occupancy windows)
```

1. **Baseline Configuration:** Facility administrator configures parking zone boundaries, individual parking-space polygons ($P_i$), and directional approach zones ($A_i$) on fixed camera feeds with human verification.
2. **Continuous Fixed Monitoring:** Fixed CCTV cameras continuously monitor vehicle ingress, dwell, and egress. The vision engine applies vehicle detection, ByteTrack temporal tracking, polygon intersection scoring, and temporal hysteresis to maintain robust occupancy states.
3. **Mobile Pavement Inspection:** Security carts, maintenance utility vehicles, or dashcam-equipped inspection vehicles drive through lot aisles periodically (e.g., daily or weekly).
4. **Human-in-the-Loop Verification:** Mobile detections are flagged in the Review Queue. An authorized inspector verifies candidate potholes, confirming defect existence and bounding geometry.
5. **Multi-Stream Spatial Association:** Verified hazards are projected into the site coordinate system and mapped to specific parking bays ($P_i$) or access lanes ($A_i$).
6. **Dynamic Safe Capacity Calculation:** The system computes the true *Safe Usable Capacity*, dynamically downgrading slots with verified hazards from `FREE_SAFE` to `FREE_UNSAFE` or `BLOCKED`.
7. **Maintenance Dispatch Planning:** The system computes a *Maintenance Opportunity Score* ($MOS$), alerting facility teams to repair hazards during predicted low-occupancy periods (e.g., night shifts, weekend intervals) with minimal operational disruption.

---

## 4. Proposed System Contribution & Novelty Hypotheses

### 4.1 Proposed System Contribution
RoadSense SiteOps introduces a unified, multi-modal spatio-temporal framework that combines:
1. **Geometric Space-Hazard Topology:** Explicit mapping between parking-space polygons, ingress approach zones, and localized surface defects within a shared facility coordinate framework.
2. **Evidence-Grounded 6-State Occupancy Model:** A formal state machine distinguishing safe availability from physical emptiness and structural hazard.
3. **Temporal Multi-Source Fusion:** Combining high-cadence fixed camera temporal tracking (ByteTrack) with low-cadence, high-spatial-resolution mobile inspection observations.
4. **Human-Audited Maintenance Scheduling:** A mathematically formulated Maintenance Opportunity Score that couples verified defect severity and persistence with historical occupancy patterns to schedule non-disruptive asphalt repairs.

### 4.2 Explicit Novelty Hypotheses
Before asserting academic or commercial novelty, RoadSense will experimentally validate the following explicit hypotheses:
- **Hypothesis H1 (False-Safe Reduction):** Fusing verified mobile pavement hazard events with fixed-camera polygon occupancy reduces the operational False-Safe Rate ($FSR$) by at least $85\%$ compared to standard vehicle-only occupancy baselines, with zero increase in occupancy false-positive alarm rates.
- **Hypothesis H2 (Temporal Stability & Flicker Reduction):** Incorporating ByteTrack trajectory association with dual-threshold temporal hysteresis reduces per-space state transitions (flicker) by at least $70\%$ under occlusion, glare, and sensor vibration compared to frame-by-frame polygon IoU thresholding.
- **Hypothesis H3 (Maintenance Optimization):** Scheduling pavement repairs using the multi-factor Maintenance Opportunity Score ($MOS$) reduces parking stall downtime during peak operational hours by over $40\%$ compared to static FIFO maintenance ticketing.

---

## 5. Prior-Art Boundary & Product Non-Claims

### 5.1 Prior-Art Boundary
- **Reference Parking Detector Baseline (e.g., `ParkingSpaceDetector`):** YOLOv8 vehicle detection applied to manually drawn parking polygons with an OpenCV text counter.
  > *"YOLOv8 + OpenCV + parking polygons + a free-space counter is a baseline, not RoadSense’s novelty."*
- **Reference Road Damage Baseline (e.g., RDD2022 YOLOv8n):** Pothole bounding-box prediction on forward-facing dashcam video.
  > *"YOLOv8 bounding boxes on dashcam frames are an inspection baseline, not RoadSense's novelty."*
- **Academic & Patent Landscape:** Commercial systems (e.g., smart parking guidance, automated curb management, PWD road maintenance databases) exist in isolation. Integrating fixed video occupancy with mobile road damage data under human audit for safe capacity estimation represents a distinct applied system architecture whose novelty will be validated through controlled benchmark experiments.

### 5.2 Product Non-Claims (What We Do NOT Claim)
1. We do **not** claim production readiness or safety-critical autonomy.
2. We do **not** claim unverified patentability or unreviewed academic precedence.
3. We do **not** claim automatic parking slot discovery from uncalibrated raw video without human verification.
4. We do **not** claim YOLO detection confidence represents physical pothole depth or structural asphalt severity.
5. We do **not** claim the media pipeline defects are fixed; the media pipeline currently has documented correctness findings under active engineering remediation.

---

## 6. Parking-Space State Machine Specification

To replace naive binary (`0/1`, `empty/occupied`) classification, RoadSense SiteOps enforces a deterministic 6-state finite state machine per configured parking space $S_i$.

```
                  ┌──────────────────────────────────────────────────┐
                  │                     UNKNOWN                      │
                  └────────┬─────────────────────────────────────────┘
                           │ Sensor feed active & calibrated
                           ▼
                  ┌──────────────────────────────────────────────────┐
                  │                  NEEDS_REVIEW                    │
                  └────────┬──────────────────────────┬──────────────┘
                           │ Polygons & baseline verified
                           ▼                          ▼
               ┌───────────────────────┐  ┌───────────────────────────┐
               │       FREE_SAFE       │  │        FREE_UNSAFE        │
               └───────┬───────▲───────┘  └───────┬───────────▲───────┘
                       │       │                  │           │
        Vehicle enters │       │ Vehicle leaves   │ Vehicle   │ Vehicle leaves
        & dwells       │       │ & dwells         │ enters    │ & dwells
                       ▼       │                  ▼           │
               ┌───────────────┴──────────────────────────────┴───────┐
               │                       OCCUPIED                       │
               └───────────────────────┬──────────────────────────────┘
                                       │ Approach/space blocked by
                                       │ non-vehicle obstacle
                                       ▼
                               ┌───────────────┐
                               │    BLOCKED    │
                               └───────────────┘
```

### 6.1 State Definitions
1. **`FREE_SAFE`**: The parking space is physically empty (no vehicle detected), its approach zone is clear, and no active verified surface hazards exist within its boundary or approach path. Safely usable for parking.
2. **`FREE_UNSAFE`**: The parking space is physically empty, but an active, human-verified surface hazard (e.g., severe pothole, deep rutting, exposed rebar) is associated with the space or its immediate approach path. Not safely usable.
3. **`OCCUPIED`**: A vehicle track is stably located within the parking-space polygon with spatial overlap exceeding the occupancy threshold for longer than the entry hysteresis time.
4. **`BLOCKED`**: The parking space is unoccupied by a vehicle but rendered physically inaccessible by an obstacle (e.g., construction barricade, dumpsters, fallen tree branch, staging pallet) detected in the space or approach zone.
5. **`UNKNOWN`**: Sensor telemetry is degraded, camera stream is disconnected, severe occlusion/glare prevents reliable inference, or calibration has been invalidated.
6. **`NEEDS_REVIEW`**: Ambiguous state triggered by conflicting sensor observations, newly reported unreviewed mobile hazard candidates, or significant geometric drift.

### 6.2 State Transition Matrix & Evidence Rules

| From State | To State | Mandatory Minimum Evidence Required | Invalidation / Reversion |
| :--- | :--- | :--- | :--- |
| `UNKNOWN` | `NEEDS_REVIEW` | Video stream active, frame validation passed, camera calibration integrity confirmed. | Stream failure $\to$ `UNKNOWN`. |
| `NEEDS_REVIEW` | `FREE_SAFE` | 1. Parking polygon human-verified.<br>2. Zero vehicle overlap for $\ge T_{empty}$ consecutive frames.<br>3. Zero verified active hazards associated with space/approach. | Any unverified detection $\to$ `NEEDS_REVIEW`. |
| `FREE_SAFE` | `OCCUPIED` | 1. Vehicle track detection with polygon overlap $\text{IoU} \ge \theta_{occ}$ (or area ratio $\ge 0.35$).<br>2. ByteTrack track persistence $\ge N_{enter}$ consecutive frames (duration $\ge T_{enter} = 3.0\text{s}$). | Track loss before $T_{enter} \to$ stays `FREE_SAFE`. |
| `OCCUPIED` | `FREE_SAFE` | 1. Vehicle track exits polygon or is terminated.<br>2. Space overlap $< \theta_{vac}$ for $\ge N_{exit}$ frames ($T_{exit} = 5.0\text{s}$).<br>3. No verified hazards present. | Re-detection $\to$ stays `OCCUPIED`. |
| `FREE_SAFE` | `FREE_UNSAFE` | Human reviewer explicitly confirms a mobile inspection surface hazard mapped to space $S_i$ or approach zone $A_i$. | Repair verified by human $\to$ `FREE_SAFE`. |
| `OCCUPIED` | `FREE_UNSAFE` | Vehicle vacates space ($T_{exit}$ satisfied), but an active verified surface hazard exists on space record. | Repair verified $\to$ `FREE_SAFE`. |
| `FREE_SAFE` / `FREE_UNSAFE` | `BLOCKED` | Non-vehicle obstacle detected in space/approach zone for $\ge T_{block} = 10.0\text{s}$ OR human review tags space as obstructed. | Obstacle removal verified $\to$ `FREE_SAFE` / `FREE_UNSAFE`. |
| `*` | `UNKNOWN` | Camera feed offline, camera shift detected ($\Delta \text{homography} > \epsilon$), optical occlusion $> 50\%$, or network loss $\ge 5.0\text{s}$. | Continuous valid stream $\to$ `NEEDS_REVIEW`. |

> **Strict Transition Rule:** **No state transition may occur on the basis of a single weak or noisy frame.** All vehicle transitions require temporal tracking hysteresis across multiple seconds ($T_{enter}, T_{exit}$), and all hazard-induced transitions require human audit confirmation.

---

## 7. Scope Boundaries (MVP vs. Excluded Scope)

### 7.1 MVP Scope (Phase 1)
1. **Single Fixed CCTV Camera:** Ingest 1 RTSP or recorded fixed CCTV stream covering 10–30 parking bays.
2. **Manual Polygon Configuration:** Human-configured and verified 2D parking-space polygons ($P_i$) and approach zones ($A_i$).
3. **Vehicle Detection & ByteTrack:** Local YOLO vehicle detection coupled with ByteTrack temporal association on fixed frames.
4. **Temporal Occupancy State Machine:** Enforcing the 6-state model with temporal hysteresis ($T_{enter}, T_{exit}$).
5. **Mobile Inspection Ingest:** Ingest forward-facing mobile inspection video from campus carts/dashcams.
6. **Frozen Pothole Detection:** Running the pinned, verified offline YOLOv8n pothole detector on mobile frames.
7. **Human Review Queue:** Dedicated interface for facility reviewers to confirm/reject pothole candidates and link them to parking bays.
8. **Safe Usable Capacity Metrics:** Live calculation and display of total spaces, occupied, free-safe, free-unsafe, and blocked.
9. **Annotated MP4 Replay:** Local FFmpeg rendering of bounding overlays, space polygons, and state colors.
10. **Local Persistence & Audit Log:** PostgreSQL/PostGIS append-only storage of review decisions, sessions, and state intervals.

### 7.2 Excluded Scope (Out of Scope for MVP)
- **Automated License Plate Recognition (ALPR / ANPR):** Strictly excluded to preserve privacy and reduce regulatory risk.
- **Payment Processing & Automatic Parking Fines:** No commercial billing or penalty enforcement.
- **Facial Recognition or Biometric Capture:** Strictly prohibited.
- **City-Wide / Multi-Tenant Scaling:** Limited to single controlled facility deployments.
- **Autonomous In-Vehicle Driver Alerts:** The system is an operator decision tool, not an in-vehicle safety system.
- **Automatic Structural Pothole Depth Estimation:** Visual 2D bounding boxes do not estimate volumetric pavement thickness or subsurface voids.
- **Cross-Camera Re-Identification:** No cross-camera vehicle tracking or trip tracking.
- **Unverified GPS Fusion:** No reliance on noisy raw smartphone GPS without surveyed site control points.
- **Automatic Contractor Work Order Dispatch:** Dispatch remains human-mediated.

---

## 8. Privacy, Data Sovereignty, and Ethical Boundaries

1. **Local-First Data Sovereignty:** All video decoding, ML inference, database persistence, and analytics run 100% locally on facility-owned hardware. No outbound telemetry, cloud processing, or third-party APIs.
2. **Privacy Masking & Minimization:**
   - Raw video is processed in memory; only detected bounding boxes, spatial coordinates, and state intervals are persisted by default.
   - Public sidewalks and non-facility roadways are excluded via configured camera region-of-interest (ROI) masks.
3. **Zero Biometric / Identity Profiling:** No license plate OCR or driver facial analysis is performed. Video feeds are treated as anonymous vehicle geometry and pavement texture.
4. **Role-Based Access & Auditability:** Every status change, polygon modification, and hazard confirmation is stamped with user identity, timestamp, and model hash in an immutable audit ledger.

---

## 9. Commercial Value & Economic Rationale

1. **Reduction in Customer Claims & Friction:** Eliminating phantom spaces prevents motorists from driving into degraded bays, reducing facility liability for tire, rim, and suspension damage.
2. **Optimized Pavement Maintenance Costs:** Enables preventative localized patching of potholes before structural water intrusion causes catastrophic base-course failure requiring full-depth reclamation.
3. **Non-Disruptive Maintenance Scheduling:** By correlating defect severity with historical occupancy curves, facility teams schedule paving repairs when spaces are naturally vacant, avoiding daytime lot closures.
4. **Improved Campus Traffic Throughput:** Drivers are routed only to safely usable spaces, cutting campus circling time, search congestion, and idling emissions.

---

## 10. Operational Risks, Failure Modes, and Go/No-Go Criteria

### 10.1 Key Risks and Mitigations
| Failure Mode | Root Cause | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Camera Tampering / Wind Vibration** | High winds, pole sway, physical bump | Misalignment between fixed polygons and real bays | Continuous homography drift detection; auto-transition to `UNKNOWN` if drift $> \epsilon$. |
| **Severe Lighting Glare / Shadows** | Low sun angle, wet pavement reflection, headlights | False vehicle detections or missed empty bays | Temporal hysteresis window ($>3\text{s}$), shadow-suppression bounding heuristics, multi-frame ByteTrack smoothing. |
| **Mobile Pothole False Positive** | Dark oil stains, manhole covers, shadows | False `FREE_UNSAFE` classification | Mandatory human review queue before any hazard impacts usable capacity. |
| **Heavy Vehicle Occlusion** | Large truck/SUV in foreground blocking adjacent space | False `OCCUPIED` or missed vehicle | Approach-zone visibility checks, height-aware polygon projection, conservative `UNKNOWN` transition when occlusion $> 50\%$. |

### 10.2 Go / No-Go Decision Criteria
The RoadSense SiteOps research initiative will advance to prototype implementation **only if** the following gating criteria are satisfied:
- [ ] **Gating Criterion 1 (Provenance & Reproducibility):** Pinned models, configuration files, and evaluation scripts reproduce documented baselines with zero hash mismatches.
- [ ] **Gating Criterion 2 (Pipeline Defect Remediation):** Media pipeline correctness findings (H.264 streaming, fail-closed inspection, bounded memory queues) are fully resolved and passing automated unit tests.
- [ ] **Gating Criterion 3 (Controlled Dataset Consent):** Facility permission, camera mounting consent, and privacy masking approvals are formally executed for the test campus.
- [ ] **Gating Criterion 4 (Metric Defensibility):** Research evaluation metrics (Safe Usable Capacity, False-Safe Rate, Maintenance Opportunity Score) demonstrate mathematical consistency across simulated edge-case test matrices.
