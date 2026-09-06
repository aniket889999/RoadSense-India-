# RoadSense SiteOps — Hazard-Aware Parking Evaluation Protocol
## Research Methodology, Evaluation Metrics & Experimental Framework

> **Classification:** Applied Computer Vision & Systems Research Specification
> **Status:** Proposed Experimental Benchmark (Pre-Execution / Future Experiments)
> **Notice:** All metrics, evaluation setups, and benchmark matrices defined herein represent a formal protocol for future empirical execution. No benchmark results are fabricated or claimed prior to experimental execution.

---

## 1. Research Objectives & Evaluation Principles

The goal of this research evaluation is to quantitatively benchmark whether fusing mobile pavement hazard intelligence with fixed-camera temporal parking tracking produces a superior, safer, and more operationally reliable parking capacity estimate than traditional vision-only occupancy baselines.

### Evaluation Principles:
1. **Physical Safety Over Raw Count:** Traditional parking metrics reward binary vehicle presence accuracy. RoadSense SiteOps measures whether a reported "available" space is genuinely safe for vehicle entry and driver egress.
2. **Leak-Free Temporal Splits:** Video sequences from the same recording session, time-of-day interval, or identical vehicle maneuvering event must never cross between validation and test splits.
3. **Transparent Non-Fabricated Metrics:** All metric formulas are mathematically bounded and grounded in verifiable ground-truth annotations without reliance on uncalibrated model confidence heuristics.

---

## 2. Mathematical Metric Definitions

Let $N$ be the total number of configured parking spaces in a facility zone, indexed by $i \in \{1, 2, \dots, N\}$.
Let $T$ be the evaluation time horizon consisting of discrete sampled video frames $t \in \{1, 2, \dots, |T|\}$.

For each space $i$ at time $t$:
- $y_{i,t} \in \{0, 1\}$ is the ground-truth physical vehicle occupancy (0 = empty, 1 = occupied).
- $\hat{y}_{i,t} \in \{0, 1\}$ is the predicted vehicle occupancy.
- $h_{i,t} \in \{0, 1\}$ is the ground-truth pavement hazard status (1 = verified active surface hazard present, 0 = clear surface).
- $b_{i,t} \in \{0, 1\}$ is the ground-truth obstruction status of space $i$ or its approach zone $A_i$ (1 = blocked, 0 = clear).
- $s_{i,t}^* \in \{\text{FREE\_SAFE}, \text{FREE\_UNSAFE}, \text{OCCUPIED}, \text{BLOCKED}, \text{UNKNOWN}, \text{NEEDS\_REVIEW}\}$ is the ground-truth multi-state.
- $\hat{s}_{i,t}$ is the predicted multi-state emitted by the system.

---

### 2.1 Standard Vehicle Occupancy Metrics
Evaluates the binary accuracy of vehicle presence:

$$\text{Precision}_{occ} = \frac{TP_{occ}}{TP_{occ} + FP_{occ}}, \quad \text{Recall}_{occ} = \frac{TP_{occ}}{TP_{occ} + FN_{occ}}, \quad F1_{occ} = \frac{2 \cdot \text{Precision}_{occ} \cdot \text{Recall}_{occ}}{\text{Precision}_{occ} + \text{Recall}_{occ}}$$

*Plain Language Definition:* Measures how accurately the vision pipeline detects whether a car is physically sitting in a parking stall, ignoring surface conditions.

---

### 2.2 Safe Usable Capacity ($SUC$)
A parking space $i$ is defined as **Safely Usable** at time $t$ if and only if it is physically empty, free of active surface hazards, unblocked, and accessible through its approach zone:

$$\text{SafeUsable}_{i,t} = \mathbb{I}(y_{i,t} = 0 \land h_{i,t} = 0 \land b_{i,t} = 0)$$

The aggregate ground-truth Safe Usable Capacity across the zone at time $t$ is:

$$SUC^*(t) = \sum_{i=1}^{N} \text{SafeUsable}_{i,t} = \sum_{i=1}^{N} \mathbb{I}(y_{i,t} = 0 \land h_{i,t} = 0 \land b_{i,t} = 0)$$

The predicted Safe Usable Capacity is:

$$\widehat{SUC}(t) = \sum_{i=1}^{N} \mathbb{I}(\hat{s}_{i,t} = \text{FREE\_SAFE})$$

*Plain Language Definition:* The actual number of empty parking spaces that a motorist can safely drive into without hitting an obstacle, damaging their suspension in a pothole, or encountering a blocked driveway.

---

### 2.3 Safe-Capacity Counting Error
The instantaneous and mean absolute error in safe usable capacity estimation across time horizon $T$:

$$\text{SCCE}(t) = \widehat{SUC}(t) - SUC^*(t)$$

$$\text{MAE}_{safe} = \frac{1}{|T|} \sum_{t=1}^{|T|} \left| \widehat{SUC}(t) - SUC^*(t) \right|$$

$$\text{RMSE}_{safe} = \sqrt{ \frac{1}{|T|} \sum_{t=1}^{|T|} \left( \widehat{SUC}(t) - SUC^*(t) \right)^2 }$$

*Plain Language Definition:* The average numerical difference between the number of safe stalls reported on the dashboard and the true number of safe stalls available in the lot.

---

### 2.4 False-Safe Rate ($FSR$) — Safety-Critical Metric
The proportion of space-time instances where the system incorrectly announces a space as safely available (`FREE_SAFE`), when in reality it is physically occupied, blocked, or contains a surface hazard:

$$FSR = \frac{\sum_{t=1}^{|T|} \sum_{i=1}^{N} \mathbb{I}(\hat{s}_{i,t} = \text{FREE\_SAFE} \land \text{SafeUsable}_{i,t} = 0)}{\sum_{t=1}^{|T|} \sum_{i=1}^{N} \mathbb{I}(\text{SafeUsable}_{i,t} = 0)}$$

*Plain Language Definition:* The percentage of hazardous, blocked, or occupied spaces that the system erroneously tells drivers are safe and open. In facility operations, this is the primary failure metric to minimize.

---

### 2.5 State Flicker Rate ($SFR$)
Measures temporal instability and noise in state prediction per parking space per hour:

$$SFR = \frac{3600 \cdot \sum_{i=1}^N \sum_{t=2}^{|T|} \mathbb{I}(\hat{s}_{i,t} \neq \hat{s}_{i,t-1})}{N \cdot \text{Duration}_{\text{hours}}}$$

*Plain Language Definition:* The average number of times a single parking stall's reported status erratically switches (e.g., green $\to$ red $\to$ green) per hour due to camera noise, shadows, or track dropouts.

---

### 2.6 Time to Stable Occupancy State ($TSS$)
The latency (in seconds) from the moment a vehicle physically comes to a complete rest within polygon $P_i$ (time $t_{event}$) to the moment the system stably transitions $\hat{s}_i$ to `OCCUPIED` for at least $K$ consecutive seconds:

$$TSS = \min \{ \Delta t \mid \hat{s}_{i, t_{event} + \Delta t} = \text{OCCUPIED} \quad \forall \tau \in [0, K] \}$$

*Plain Language Definition:* How many seconds it takes for the system to reliably lock in an "occupied" or "vacated" status after a vehicle parks or departs, filtering out transient drive-throughs.

---

### 2.7 Hazard-to-Space Association Accuracy ($HSA$)
Evaluates the geometric projection and assignment of human-reviewed mobile inspection hazards onto the correct parking space polygon $P_i$ or approach zone $A_i$:

$$HSA = \frac{\sum_{m=1}^{M} \mathbb{I}(\hat{P}_{hazard_m} = P^*_{hazard_m})}{M}$$

where $M$ is the total number of verified mobile pavement hazard events and $P^*_{hazard_m}$ is the true spatial bay coordinate.

*Plain Language Definition:* The percentage of inspected potholes correctly mapped to their actual parking stall number on the facility site map.

---

### 2.8 Human Review Workload ($HRW$)
Quantifies operator audit efficiency:

$$HRW = \frac{\text{Total Candidate Hazard Detections Submitted}}{\text{Confirmed Actionable Potholes}} = \frac{N_{proposals}}{N_{confirmed}}$$

$$\text{Review Time Rate} = \frac{\text{Total Human Review Seconds}}{\text{Kilometers of Inspected Facility Roadway}}$$

*Plain Language Definition:* How many candidate detections an operator must inspect to verify one genuine pothole, and the operational time required to audit a full campus inspection run.

---

### 2.9 Event Deduplication Effectiveness ($EDE$)
Measures the system's ability to cluster multi-frame mobile detections of the same physical pothole into a single spatial entity:

$$EDE = 1 - \frac{|\text{Over-clustered Duplicates}| + |\text{Under-clustered Splits}|}{M_{ground\_truth\_hazards}}$$

*Plain Language Definition:* How effectively the mobile pipeline groups 10 camera frames showing the same pothole into a single repair ticket instead of generating 10 separate duplicate reports.

---

### 2.10 Processing Latency and Memory Footprint
- **Inference Latency ($L_{inf}$):** Milliseconds per frame for vehicle detection and tracking.
- **End-to-End Latency ($L_{e2e}$):** Milliseconds from frame ingestion to WebSocket UI state broadcast.
- **Peak Resident Memory ($RAM_{peak}$):** Megabytes of RSS memory utilized by the background processing job.

---

### 2.11 Maintenance Opportunity Score ($MOS$)
To guide facility maintenance crews without relying on uncalibrated YOLO confidence scores as physical damage severity, the $MOS$ is computed using transparent, auditable operational factors:

$$MOS(i) = w_1 \cdot P_{hazard}(i) + w_2 \cdot N_{affected}(i) + w_3 \cdot U_{historical}(i) + w_4 \cdot Z_{priority}(i) + w_5 \cdot \Delta t_{age}(i) + w_6 \cdot W_{low\_occ}(i)$$

Where:
- $P_{hazard}(i) \in [0, 1]$: **Hazard Persistence** (ratio of inspection runs confirming defect presence over time).
- $N_{affected}(i) \in [1, 5]$: **Number of Affected Stalls** (1 for single stall, higher if located in shared ingress bottleneck $A_i$).
- $U_{historical}(i) \in [0, 1]$: **Historical Space Utilization** (fraction of daylight hours space $i$ is typically occupied).
- $Z_{priority}(i) \in [0, 1]$: **Site Zone Priority** (e.g., Emergency Room entrance = 1.0, General Staff lot = 0.5, Remote Overflow lot = 0.2).
- $\Delta t_{age}(i) \in [0, 1]$: **Normalized Age Since Confirmation** (days since initial verification / 30 days).
- $W_{low\_occ}(i) \in [0, 1]$: **Upcoming Low-Occupancy Window Opportunity** (indicator that scheduled off-peak window is imminent, e.g., overnight interval).
- Weights $\sum_{k=1}^6 w_k = 1.0$ with default configuration: $[w_1=0.25, w_2=0.20, w_3=0.15, w_4=0.15, w_5=0.10, w_6=0.15]$.

> **Explicit Metric Constraint:** Pothole detection bounding box confidence ($\text{conf} \in [0, 1]$) from YOLO is **NOT** used as structural severity. Structural impact is derived strictly from human-reviewed classification, spatial footprint relative to tire contact patches, and historical persistence.

---

## 3. Experimental Evaluation Setup & Baseline Comparisons

### 3.1 Systems Under Comparison

```
┌────────────────────────────────────────────────────────────────────────────┐
│                             SYSTEMS BENCHMARK                              │
├────────────────────────────────────────────────────────────────────────────┤
│ Baseline A:  Standard YOLOv8x + Static Polygon Overlap (No Tracker/FSR)    │
│ Baseline B:  Per-Space Patch ResNet/MobileNet Occupancy Classifier        │
│ System C:    RoadSense SiteOps (YOLO + ByteTrack + Hysteresis + Fused      │
│              Reviewed Pothole Layer + Approach-Zone Geometry)              │
└────────────────────────────────────────────────────────────────────────────┘
```

1. **Baseline A (Geometric Overlap Baseline - e.g., `ParkingSpaceDetector`):**
   - Frame-by-frame YOLOv8 vehicle detection.
   - Bounding box intersection with fixed polygons (IoU $> 0.15$).
   - Binary empty/occupied decision with zero temporal smoothing and zero hazard awareness.
2. **Baseline B (Patch Classification Baseline):**
   - Perspective-warped image crop per parking space polygon.
   - Dedicated binary image classifier (ResNet-18 / MobileNetV3) fine-tuned on parking stall patches.
   - No hazard or approach-zone awareness.
3. **Proposed System C (RoadSense SiteOps Multi-Modal Framework):**
   - YOLO vehicle detection with ByteTrack persistent identity tracking.
   - Dual-threshold temporal hysteresis state machine ($T_{enter} = 3\text{s}, T_{exit} = 5\text{s}$).
   - Spatial integration with human-reviewed mobile pavement hazard records.
   - Ingress approach zone ($A_i$) reachability evaluation.

---

### 3.2 Required Ablation Matrix
To isolate the contribution of each architectural component, System C will be evaluated under 6 systematic ablations:

| Experiment ID | Configuration Description | ByteTrack | Temporal Hysteresis | Pavement Hazard Layer | Approach Zone Model | Human Review Gating |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C-Full** | Full Proposed RoadSense SiteOps | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| **Ablation 1** | Without ByteTrack (Frame-by-frame boxes) | No | Yes | Yes | Yes | Yes |
| **Ablation 2** | Without Temporal Hysteresis (Instantaneous) | Yes | No | Yes | Yes | Yes |
| **Ablation 3** | Without Pavement Hazard Layer (Vehicle-only) | Yes | Yes | No | Yes | N/A |
| **Ablation 4** | Without Approach Zone Geometry (Stall only) | Yes | Yes | Yes | No | Yes |
| **Ablation 5** | Without Human Review (Direct raw detector) | Yes | Yes | Yes (Raw) | Yes | No |
| **Ablation 6** | Without Spatial Calibration (Unregistered) | Yes | Yes | Yes (Uncal) | Yes | Yes |

---

### 3.3 Operating Conditions & Stress-Test Matrix
The benchmark will evaluate performance across 10 challenging real-world operational environments:

| Test Condition ID | Operational Environment | Primary Failure Mechanism Evaluated |
| :--- | :--- | :--- |
| **ENV-01** | Clear Daytime (Direct Sunlight) | High-contrast vehicle drop-shadows causing false stall overlap. |
| **ENV-02** | Low-Light Twilight / Night (Artificial Pole Lighting) | Low SNR, vehicle headlight beam bloom, dark stall occlusion. |
| **ENV-03** | Moderate to Heavy Rainfall | Surface water puddling, lens droplet distortion, pavement reflection. |
| **ENV-04** | Severe Optical Glare (Low Sun Angle) | Lens flare, saturated highlights washing out polygon demarcation lines. |
| **ENV-05** | Complex Dynamic Tree Shadows | High-frequency luminance changes triggering false vehicle patch triggers. |
| **ENV-06** | Severe Perspective & Partial Occlusion | Tall commercial vans in foreground occluding rear compact vehicles. |
| **ENV-07** | High-Density Crowded Lot ($>95\%$ Occupancy) | Boundary spillover across adjacent narrow parking slots. |
| **ENV-08** | Sparsely Occupied Lot ($<10\%$ Occupancy) | Isolated pedestrian / cart traffic traversing empty stalls. |
| **ENV-09** | Camera Pole Sway & Wind Vibration | Geometric polygon misalignment relative to static scene features. |
| **ENV-10** | Temporary Aisle Obstruction | Delivery trucks idling in access lane $A_i$ blocking 4 adjacent stalls. |

---

## 4. Controlled Data-Collection & Annotation Plan

### 4.1 Ethical, Consent & Facility Permission Protocol
- **Site Approval:** Data collection shall occur strictly within consenting private partner facilities (e.g., university research lot, hospital staff deck, industrial campus).
- **Camera Mounting Consent:** Fixed cameras mounted on facility light poles or building facades with formal administrative and safety permits.
- **Privacy Minimization (No ALPR / Faces):**
  - High-angle camera placement ($>35^\circ$ elevation) to prioritize roof/hood vehicle silhouettes over windshields and license plates.
  - Optical resolution and focal length configured such that license plate characters are sub-Nyquist/unreadable or programmatically masked prior to storage.
  - No facial images or pedestrian identification retained.

### 4.2 Annotation Standards & Multi-Class Ground Truth
Trained human annotators will generate frame-level ground-truth using a standardized annotation protocol:
1. **Polygon Layer:** Exact 4-point quadrilateral coordinates for every parking space boundary ($P_i$) and 4-to-6 point polygon for each approach zone ($A_i$).
2. **Per-Frame Vehicle Bounding Boxes:** Tight 2D bounding boxes tagged with class (`sedan`, `suv`, `van`, `truck`, `motorcycle`, `bicycle`) and unique continuous track ID across frames.
3. **Per-Space Multi-State Ground Truth ($s^*_{i,t}$):** Assigned every 1.0 second:
   - `FREE_SAFE`: Empty, hazard-free, unblocked.
   - `FREE_UNSAFE`: Empty, but containing ground-truth verified pothole or rut.
   - `OCCUPIED`: Vehicle present inside boundary.
   - `BLOCKED`: Non-vehicle obstacle present or approach lane blocked.
   - `UNKNOWN`: Heavy occlusion ($>50\%$) or camera failure.

### 4.3 Data Split Protocol & Leakage Prevention
To ensure strict scientific validity:
- **Session & Date Isolation:** The dataset will be partitioned into Train ($60\%$), Validation ($20\%$), and Held-Out Test ($20\%$) partitioned strictly by **distinct calendar dates and physical recording sessions**.
- **Zero Adjacent-Frame Leakage:** Continuous video recordings from the same morning/afternoon session will never be distributed across splits.
- **Geographic Cross-Lot Validation:** Where multi-lot footage is collected, one entire distinct parking lot zone will be reserved exclusively for zero-shot generalization testing.

### 4.4 Data Retention & Disposal Policy
- Raw unmasked video footage will be retained in encrypted local storage (`0700` POSIX permissions) for the duration of the research benchmark (max 180 days) and purged upon benchmark completion.
- Vectorized spatial annotations, bounding box coordinates, and aggregated performance metrics will be permanently archived as open research benchmarks.
