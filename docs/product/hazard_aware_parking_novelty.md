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
Facility managers at large closed or semi-controlled campuses (hospitals, university campuses, corporate parks, transit hubs, logistics centers) manage hundreds to thousands of parking spaces. They face two persistent operational failures:
1. **Phantom Capacity / False Availability:** Parking guidance indicators report spaces as available, but stalls have severe pavement subsidence or potholes that risk chassis damage, or the approach lane is obstructed. Drivers enter, discover unusable stalls, circle aimlessly, and create gridlock.
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
| **Director of Facility Operations** | Major Hospital / Medical Campus | Reduces visitor parking frustration and limits chassis/slip damage liability on hospital grounds. |
| **Head of Campus Infrastructure** | University / Higher Education | Prioritizes parking lot maintenance during scheduled low-occupancy windows without closing active academic lots. |
| **Logistics & Fleet Yard Manager** | Industrial / Freight Terminal | Mitigates fleet vehicle tire and axle damage in designated staging and parking bays. |
| **Municipal Parking Authority** | Controlled Off-Street Public Parking | Improves parking asset operational reliability by reporting true safe usable capacity. |

### 3.2 Real Operational Workflow

```
[Mobile Inspection Vehicle / Cart]
     │ (Scheduled / opportunistic drive with mobile camera)
     ▼
[RoadSense Mobile Pipeline (Frozen Pothole Detector)]
     │ (Extracts candidate surface defects & stream-local track IDs)
     ▼
[Human Review Queue (Facility Inspector)]
     │ (Audits defect candidate, verifies status, links to site polygon)
     ▼
[Verified Surface Hazard Layer (Spatial DB)] ───┐
                                                │ (Spatial & Geometric
[Fixed Facility CCTV (Overhead / Pole)]        │  Association Engine)
     │ (Continuous RTSP video stream)           │
     ▼                                          │
[Fixed Vision Pipeline (Vehicle Det + Tracker)] │
     │ (Stream-Local ByteTrack + Hysteresis)    │
     ▼                                          │
[Space Occupancy State Engine] ─────────────────┘
     │
     ▼
[RoadSense SiteOps Command Center]
     ├── Safe Usable Capacity Counter (Real-time motorist guidance feed)
     ├── Factored Space Map (Multi-dimensional visual display)
     └── Maintenance Opportunity Workflow (Recommended repair dispatch during low-occupancy windows)
```

1. **Baseline Configuration:** Facility administrator configures parking zone boundaries, individual parking-space polygons ($P_i$), and directional approach zones ($A_i$) on fixed camera feeds with human verification.
2. **Continuous Fixed Monitoring:** Fixed CCTV cameras monitor vehicle ingress, dwell, and egress. The vision engine applies vehicle detection, stream-local ByteTrack temporal tracking, polygon intersection scoring, and temporal hysteresis to maintain robust occupancy states.
3. **Mobile Pavement Inspection:** Security carts, utility vehicles, or dashcam-equipped inspection vehicles drive through lot aisles periodically.
4. **Human-in-the-Loop Verification:** Mobile detections are flagged in the Review Queue. An authorized inspector audits candidate potholes, confirming defect existence and assigning space/approach association.
5. **Spatial Association:** Verified hazards are linked to specific parking bays ($P_i$) or access lanes ($A_i$) using verified site coordinates or manual inspector linkage.
6. **Dynamic Safe Capacity Calculation:** The system computes *Safe Usable Capacity*, dynamically evaluating whether physically empty spaces are safe and accessible.
7. **Maintenance Dispatch Planning:** The system computes an operational *Maintenance Opportunity Score* ($MOS$), highlighting opportunities to repair hazards during predicted low-occupancy periods (e.g., night shifts, weekend intervals) with minimal operational disruption.

---

## 4. Proposed System Contribution & Novelty Hypotheses

### 4.1 Proposed System Contribution
RoadSense SiteOps introduces a unified, multi-modal spatio-temporal framework that combines:
1. **Potential Operational and Systems Contribution:** Combining independently observed parking occupancy, human-reviewed pavement hazards, temporal state estimation, and maintenance planning into a unified safe-usable-capacity workflow.
2. **Factored State Modeling:** Decoupling physical vehicle occupancy, pavement surface condition, physical obstruction, and human-review workflow into independent state dimensions.
3. **Temporal Multi-Source Fusion:** Combining high-cadence fixed camera temporal tracking (ByteTrack) with low-cadence, high-spatial-resolution mobile inspection observations under explicit geometric constraints.
4. **Occupancy-Informed Maintenance Scheduling:** Formulating an operational Maintenance Opportunity Score that couples verified defect persistence and location with historical occupancy patterns to identify low-disruption repair windows.

### 4.2 Directional, Testable Research Hypotheses
In accordance with rigorous scientific methodology, no numerical improvements are claimed prior to empirical pilot data collection. The system will evaluate the following pre-registered directional hypotheses:

- **H1 — Safety-Aware Capacity:** The proposed fused system will produce a lower False-Safe Rate ($FSR$) than vehicle-only polygon-overlap occupancy baselines under the defined evaluation conditions.
- **H2 — Temporal Stability:** ByteTrack persistent tracking combined with temporal hysteresis will produce fewer erroneous state changes (flicker) per space-hour than independent frame-level classification.
- **H3 — Maintenance Planning:** A transparent occupancy-informed maintenance policy will reduce estimated peak-occupancy disruption compared with FIFO scheduling on the same historical workload.
- **H4 — Human Review:** Uncertainty-aware triage will reduce the number of observations requiring manual review while maintaining a pre-registered False-Safe safety bound.

> **Methodology Rule:** Minimum practically important effect sizes (e.g., target percentage reductions) will be established following an initial pilot phase and frozen prior to running the held-out evaluation. Effect thresholds will never be altered after examining held-out test split results.

---

## 5. Prior-Art Boundary & Scoped Novelty Positioning

### 5.1 Scoped Positioning
The contributions of RoadSense SiteOps are categorized across five distinct boundaries:
1. **Product Differentiation:** Fusing road surface hazard intelligence with parking occupancy to guide drivers to truly safe spaces and help facility managers schedule repairs.
2. **Engineering Contribution:** Monorepo architecture, local model runtime with fail-closed SHA-256 verification, stream-local ByteTrack tracking, and PostgreSQL/PostGIS spatial modeling.
3. **Research Hypotheses:** Testable empirical questions (H1–H4) evaluating multi-modal fusion, temporal stability, and maintenance scheduling.
4. **Formal Academic Novelty:** Deferred until systematic literature reviews and peer-reviewed benchmark comparisons are completed.
5. **Legal Patentability:** No claims of patentability or intellectual property exclusivity are made.

### 5.2 Prior-Art Boundary
- **Reference Parking Detector Baseline (e.g., `ParkingSpaceDetector`):** YOLOv8 vehicle detection applied to manually drawn parking polygons with an OpenCV text counter.
  > *"YOLOv8 + OpenCV + parking polygons + a free-space counter is a baseline, not RoadSense’s novelty."*
- **Reference Road Damage Baseline (e.g., RDD2022 YOLOv8n):** Pothole bounding-box prediction on forward-facing dashcam video.
  > *"YOLOv8 bounding boxes on dashcam frames are an inspection baseline, not RoadSense's novelty."*

### 5.3 Product Non-Claims (What We Do NOT Claim)
1. We do **not** claim a proven novel algorithm or the first system of its kind.
2. We do **not** claim patentability, clinical/safety certification, or production readiness.
3. We do **not** claim automatic parking slot discovery from uncalibrated raw video without human verification.
4. We do **not** claim YOLO detection confidence represents physical pothole depth, structural severity, or material volume.
5. We do **not** claim guaranteed downtime reduction prior to experimental measurement.
6. We do **not** claim the media pipeline defects are fixed; the media pipeline currently has documented correctness findings under active engineering remediation.

---

## 6. Factored Parking-Space State Model

To eliminate ambiguities caused by conflating physical occupancy, surface damage, and review workflow into a single scalar state, RoadSense SiteOps enforces a **Factored State Model** consisting of four orthogonal dimensions.

### 6.1 State Dimensions

1. **Occupancy Dimension ($D_{occ} \in \{\text{FREE}, \text{OCCUPIED}, \text{UNKNOWN}\}$):**
   - `FREE`: No vehicle is physically present in the space.
   - `OCCUPIED`: A vehicle is stably present within the space polygon.
   - `UNKNOWN`: Occlusion, sensor dropout, or extreme lighting prevents vehicle occupancy determination.

2. **Surface Dimension ($D_{surf} \in \{\text{SAFE}, \text{UNSAFE}, \text{UNKNOWN}\}$):**
   - `SAFE`: No verified, active surface hazards are associated with the space or its approach zone.
   - `UNSAFE`: An active, human-verified surface hazard (e.g., pothole, severe degradation) exists within the space or its approach path.
   - `UNKNOWN`: Pavement surface has not been inspected, or inspection data is invalid/expired.

3. **Obstruction Dimension ($D_{obs} \in \{\text{CLEAR}, \text{BLOCKED}, \text{UNKNOWN}\}$):**
   - `CLEAR`: The space and its approach zone are free of physical non-vehicle obstacles.
   - `BLOCKED`: An obstacle (debris, barrier, dumpster, staging equipment) blocks entry or parking.
   - `UNKNOWN`: Visibility into the approach zone or stall interior is occluded.

4. **Review Dimension ($D_{rev} \in \{\text{UNREVIEWED}, \text{CONFIRMED}, \text{REJECTED}, \text{NEEDS\_REVIEW}\}$):**
   - `UNREVIEWED`: Automated proposal awaiting initial human inspection.
   - `CONFIRMED`: Human auditor has verified the candidate observation or association.
   - `REJECTED`: Human auditor rejected the automated candidate observation.
   - `NEEDS\_REVIEW`: Operational workflow flag triggered by conflicting evidence or sensor drift.

### 6.2 Key State Model Principles & Precedence Rules
- **Derived-Status Precedence:**
  1. If evidence quality is insufficient or occupancy is `UNKNOWN` ($D_{occ} = \text{UNKNOWN}$): derived status = `UNKNOWN`.
  2. If obstruction is `BLOCKED` ($D_{obs} = \text{BLOCKED}$): derived status = `BLOCKED`.
  3. If occupancy is `OCCUPIED` ($D_{occ} = \text{OCCUPIED}$): derived status = `OCCUPIED`. Underlying surface hazards ($D_{surf} = \text{UNSAFE}$) are preserved as a separate maintenance flag.
  4. If occupancy is `FREE` ($D_{occ} = \text{FREE}$) and a `CONFIRMED` surface hazard affects the bay or approach zone ($D_{surf} = \text{UNSAFE} \land D_{rev} = \text{CONFIRMED}$): derived status = `FREE_UNSAFE`.
  5. If occupancy is `FREE`, obstruction is `CLEAR`, and the reviewed surface state is `SAFE` ($D_{occ} = \text{FREE} \land D_{obs} = \text{CLEAR} \land D_{surf} = \text{SAFE}$): derived status = `FREE_SAFE`.
  6. An `UNREVIEWED` surface observation ($D_{rev} = \text{UNREVIEWED}$) must **not** establish `SAFE`.
  7. `NEEDS_REVIEW` alters workflow presentation but must **not** erase known physical facts.

- **Coexistence of States:** An occupied space may simultaneously have an unsafe surface ($D_{occ}=\text{OCCUPIED}, D_{surf}=\text{UNSAFE}$). A blocked space may also contain a pavement hazard.
- **Workflow vs. Physical Reality:** `NEEDS_REVIEW` is strictly a workflow triage state, not a physical space condition.
- **Asymmetric Evidence:** Human rejection of one candidate observation does not prove that a physical surface is safe; it merely indicates that the specific candidate detection was a false alarm.
- **Temporal Asymmetry:** Parking occupancy changes rapidly (seconds to minutes); pavement degradation persists over weeks or months until explicit repair verification or human closure occurs. They must not share the same hysteresis timing.

### 6.3 Derived Operator-Facing Availability Truth Table

The presentation layer evaluates the factored state tuple $(D_{occ}, D_{surf}, D_{obs}, D_{rev})$ through the strict precedence rules:

| Occupancy ($D_{occ}$) | Surface ($D_{surf}$) | Obstruction ($D_{obs}$) | Review ($D_{rev}$) | Derived Operator Status | Operational Interpretation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `FREE` | `SAFE` | `CLEAR` | `CONFIRMED` | **`FREE_SAFE`** | Safely usable empty space; eligible for driver routing. |
| `FREE` | `SAFE` | `CLEAR` | `UNREVIEWED` | **`UNKNOWN`** | Empty and unobstructed, but surface state unreviewed; cannot confirm safe. |
| `FREE` | `UNSAFE` | `CLEAR` | `CONFIRMED` | **`FREE_UNSAFE`** | Empty stall, but confirmed surface hazard present; not safely usable. |
| `FREE` | `UNSAFE` | `CLEAR` | `REJECTED` | **`FREE_SAFE`** | Candidate defect rejected by human review; stall confirmed safe. |
| `FREE` | Any | `BLOCKED` | Any | **`BLOCKED`** | Empty stall, but blocked by obstacle or staging in approach lane. |
| `OCCUPIED` | `SAFE` | Any | Any | **`OCCUPIED`** | Space physically occupied by vehicle; normal surface. |
| `OCCUPIED` | `UNSAFE` | Any | `CONFIRMED` | **`OCCUPIED`** *(Hazard Flag)* | Space occupied, but underlying surface hazard recorded for maintenance. |
| `UNKNOWN` | Any | Any | Any | **`UNKNOWN`** | Camera offline, calibration invalid, or severe visual occlusion. |
| Any | Any | Any | `NEEDS_REVIEW` | **`UNKNOWN`** *(Review Pending)* | Sensor conflict or unverified state; displayed with triage indicator. |

---

## 7. Temporal Policies & Tracking Boundaries

### 7.1 Configurable Temporal Policies
Fixed durations (e.g., 3.0s entry, 5.0s exit) are initial baseline configuration parameters that must be validated empirically based on camera framerate, camera mounting angle, vehicle approach speed, occlusion patterns, and facility geometry.

The system enforces six distinct temporal policies:
1. **Vehicle Arrival Policy ($T_{arrival}$):** Requires persistent vehicle track overlap with space polygon for $N_{arr} = \lceil T_{arrival} \times \text{FPS} \rceil$ consecutive frames (nominal default: $T_{arrival} = 3.0\text{s}$) to transition $D_{occ}$ from `FREE` to `OCCUPIED`.
2. **Vehicle Departure Policy ($T_{departure}$):** Requires zero vehicle track overlap for $N_{dep} = \lceil T_{departure} \times \text{FPS} \rceil$ consecutive frames (nominal default: $T_{departure} = 5.0\text{s}$) to transition $D_{occ}$ from `OCCUPIED` to `FREE`.
3. **Temporary Occlusion Grace Period ($T_{occlusion}$):** If a vehicle track is momentarily lost due to foreground traffic passing across the camera view, $D_{occ}$ is held in `OCCUPIED` for $T_{occlusion}$ (nominal default: $2.0\text{s}$) before triggering state reassessment.
4. **Camera Interruption Policy ($T_{dropout}$):** If the video feed stalls or frames fail validation for $T_{dropout} \ge 2.0\text{s}$, $D_{occ}$ transitions immediately to `UNKNOWN`.
5. **Surface-Hazard Persistence Policy ($T_{hazard}$):** Verified surface hazards ($D_{surf} = \text{UNSAFE}$) remain active indefinitely in the spatial database until an explicit repair verification or human auditor status override occurs.
6. **Post-Repair Verification Policy:** Following reported asphalt patching, the space surface status transitions to `SAFE` only after human inspector sign-off or confirmation during a post-repair inspection drive.

### 7.2 Strict Tracking Identity Boundaries
- **Stream-Local / Session-Local IDs Only:** ByteTrack tracking IDs are transient numerical identifiers scoped strictly to a single video processing session or continuous camera stream.
- **Non-Persistence Rule:** ByteTrack IDs must **never** be interpreted as permanent vehicle identities (no vehicle re-identification) and must **never** be used as permanent physical hazard identifiers across sessions.

---

## 8. Scope Boundaries (MVP vs. Excluded Scope)

### 8.1 MVP Scope (Phase 1)
1. **Single Fixed CCTV Camera:** Ingest 1 RTSP or recorded fixed CCTV stream covering 10–30 parking bays.
2. **Manual Polygon Configuration:** Human-configured and verified 2D parking-space polygons ($P_i$) and approach zones ($A_i$).
3. **Vehicle Detection & Stream-Local ByteTrack:** Local YOLO vehicle detection coupled with ByteTrack temporal association on fixed frames.
4. **Factored Occupancy State Model:** Enforcing multi-dimensional states with configurable temporal policies.
5. **Mobile Inspection Ingest:** Ingest forward-facing mobile inspection video from campus carts/dashcams.
6. **Frozen Pothole Detection:** Running the pinned, verified offline YOLOv8n pothole detector on mobile frames.
7. **Human Review Queue:** Dedicated interface for facility reviewers to confirm/reject pothole candidates and manually associate them with parking bays.
8. **Safe Usable Capacity Metrics:** Live calculation and display of total spaces, occupied, free-safe, free-unsafe, and blocked.
9. **Annotated MP4 Replay:** Local FFmpeg rendering of bounding overlays, space polygons, and factored state colors.
10. **Local Persistence & Append-Only Audit History:** PostgreSQL/PostGIS storage of review decisions, sessions, and state intervals.

### 8.2 Excluded Scope (Out of Scope for MVP)
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

## 9. Privacy, Data Sovereignty, and Retention Policy

### 9.1 Honest Privacy Engineering
- **Minimise Incidental Capture:** Configure camera angles and fields of view to focus on pavement and vehicle geometry rather than pedestrian walkways.
- **No Intentional Identification:** The system does not attempt to identify individuals or vehicles.
- **Privacy Masking:** Non-facility regions, public thoroughfares, and known pedestrian pathways are masked prior to persistent recording where technically feasible.
- **Documenting Limitations:** Camera mounting height and optical resolution do not provide absolute privacy guarantees. Potential masking failures or edge cases must be documented honestly.
- **Access Control:** Role-based local access controls restrict video access to authorized facility personnel.

### 9.2 Layered Data Retention Policy
1. **Default Operational Raw Video Retention:** Disabled by default, or configured to a short rolling window (nominal pilot default: **24 to 72 hours**) solely for operational review, after which raw frames are deleted unless an operator flags a specific incident.
2. **Research Benchmark Retention:** Video explicitly collected for research benchmarks requires separate site consent, documented study protocols, dedicated access controls, and a fixed deletion schedule upon project completion.
3. **Derived Metadata Retention:** Anonymized bounding box coordinates, spatial polygons, and state interval logs are retained according to facility analytics requirements (e.g., 365 days).
4. **Human-Approved Evidence Retention:** Cropped visual evidence snippets supporting confirmed maintenance work orders are retained for the lifecycle of the repair case and purged upon verified completion.

---

## 10. Audit History & System Governance

- **Audit Ledger Implementation:** The system maintains an **append-only review history at the application level**, logging all human confirmations, polygon edits, and state overrides with timestamps and reviewer IDs.
- **Evidentiary Limitation:** Cryptographically verifiable or immutable ledger storage is deferred until explicitly implemented and audited. The system does not claim formal regulatory evidentiary immutability.

---

## 11. Operational Risks, Failure Modes, and Go/No-Go Criteria

### 11.1 Key Risks and Mitigations
| Failure Mode | Root Cause | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Camera Tampering / Wind Vibration** | High winds, pole sway, physical bump | Misalignment between fixed polygons and real bays | Background feature alignment monitoring; state transitions to `UNKNOWN` if drift exceeds tolerance. |
| **Severe Lighting Glare / Shadows** | Low sun angle, wet pavement reflection, headlights | False vehicle detections or missed empty bays | Configurable temporal hysteresis window, multi-frame track smoothing. |
| **Mobile Pothole False Positive** | Dark oil stains, manhole covers, shadows | False `FREE_UNSAFE` classification | Mandatory human review queue before any hazard impacts usable capacity. |
| **Heavy Vehicle Occlusion** | Large truck/SUV in foreground blocking adjacent space | False `OCCUPIED` or missed vehicle | Approach-zone visibility checks, conservative `UNKNOWN` transition when occlusion $> 50\%$. |

### 11.2 Go / No-Go Decision Criteria
The RoadSense SiteOps research initiative will advance to prototype implementation **only if** the following gating criteria are satisfied:
- [ ] **Gating Criterion 1 (Provenance & Reproducibility):** Pinned models, configuration files, and evaluation scripts reproduce documented baselines with zero hash mismatches.
- [ ] **Gating Criterion 2 (Pipeline Defect Remediation):** Media pipeline correctness findings (H.264 streaming, fail-closed inspection, bounded memory queues) are fully resolved and passing automated unit tests.
- [ ] **Gating Criterion 3 (Controlled Dataset Consent):** Facility permission, camera mounting consent, and privacy masking approvals are formally executed for the test campus.
- [ ] **Gating Criterion 4 (Metric Defensibility):** Research evaluation metrics (Safe Usable Capacity, False-Safe Rate, Maintenance Opportunity Score) demonstrate mathematical consistency across simulated edge-case test matrices.
