# RoadSense SiteOps — Hazard-Aware Parking Evaluation Protocol
## Research Methodology, Evaluation Metrics & Experimental Framework

> **Classification:** Applied Computer Vision & Systems Research Specification
> **Status:** Proposed Experimental Benchmark (Pre-Execution / Future Experiments)
> **Notice:** All metrics, evaluation setups, and benchmark matrices defined herein represent a formal protocol for future empirical execution. No benchmark results are fabricated or claimed prior to experimental execution.

---

## 1. Research Objectives & Directional Hypotheses

The primary goal of this research evaluation is to quantitatively assess whether fusing mobile pavement hazard intelligence with fixed-camera temporal parking tracking produces a safer, more stable, and operationally reliable parking capacity estimate than traditional vision-only occupancy baselines.

### 1.1 Directional, Testable Hypotheses
In accordance with pre-registration principles, specific numerical target effect sizes will be established following initial pilot runs and frozen before held-out test evaluation. The study evaluates four directional hypotheses:

- **H1 — Safety-Aware Capacity:** The proposed fused system will produce a lower False-Safe Rate ($FSR$) than vehicle-only polygon-overlap occupancy baselines under the defined evaluation conditions.
- **H2 — Temporal Stability:** ByteTrack persistent tracking combined with temporal hysteresis will produce fewer erroneous state changes (flicker) per space-hour than independent frame-level classification.
- **H3 — Maintenance Planning:** A transparent occupancy-informed maintenance policy will reduce estimated peak-occupancy disruption compared with FIFO scheduling on the same historical workload.
- **H4 — Human Review:** Uncertainty-aware triage will reduce the number of observations requiring manual review while maintaining a pre-registered False-Safe safety bound.

> **Evaluation Rule:** Minimum practically important effect sizes will be selected after a pilot and frozen before the held-out evaluation. Effect thresholds will never be altered after examining held-out results.

---

## 2. Mathematical Metric Definitions

Let $N$ be the total number of configured parking spaces in a facility zone, indexed by $i \in \{1, 2, \dots, N\}$.
Let $T$ be the evaluation time horizon consisting of discrete sampled video frames $t \in \{1, 2, \dots, |T|\}$.

For each space $i$ at time $t$, ground truth consists of factored dimensions:
- $y_{i,t} \in \{0, 1\}$: Ground-truth vehicle occupancy ($0 = \text{free}, 1 = \text{occupied}$).
- $h_{i,t} \in \{0, 1\}$: Ground-truth pavement hazard status ($0 = \text{safe}, 1 = \text{verified active hazard present}$).
- $b_{i,t} \in \{0, 1\}$: Ground-truth obstruction status ($0 = \text{clear}, 1 = \text{blocked}$).
- $\hat{a}_{i,t} \in \{\text{FREE\_SAFE}, \text{FREE\_UNSAFE}, \text{OCCUPIED}, \text{BLOCKED}, \text{UNKNOWN}\}$: Predicted operator availability status.

---

### 2.1 Standard Vehicle Occupancy Metrics
Evaluates physical vehicle presence detection accuracy independently of surface condition:

$$\text{Precision}_{occ} = \frac{TP_{occ}}{TP_{occ} + FP_{occ}}, \quad \text{Recall}_{occ} = \frac{TP_{occ}}{TP_{occ} + FN_{occ}}, \quad F1_{occ} = \frac{2 \cdot \text{Precision}_{occ} \cdot \text{Recall}_{occ}}{\text{Precision}_{occ} + \text{Recall}_{occ}}$$

*Plain Language Definition:* Measures how accurately the detector and tracker determine whether a vehicle is physically sitting in a parking stall.

---

### 2.2 Safe Usable Capacity ($SUC$)
A parking space $i$ is ground-truth **Safely Usable** at time $t$ if and only if it is physically empty, free of active surface hazards, unblocked, and accessible through its approach zone:

$$\text{SafeUsable}_{i,t} = \mathbb{I}(y_{i,t} = 0 \land h_{i,t} = 0 \land b_{i,t} = 0)$$

Ground-truth Safe Usable Capacity across the zone at time $t$:

$$SUC^*(t) = \sum_{i=1}^{N} \mathbb{I}(y_{i,t} = 0 \land h_{i,t} = 0 \land b_{i,t} = 0)$$

Predicted Safe Usable Capacity emitted by the system:

$$\widehat{SUC}(t) = \sum_{i=1}^{N} \mathbb{I}(\hat{a}_{i,t} = \text{FREE\_SAFE})$$

---

### 2.3 Safe-Capacity Counting Error ($\text{MAE}_{safe}, \text{RMSE}_{safe}$)
The difference between reported and true safe usable stalls across time horizon $T$:

$$\text{MAE}_{safe} = \frac{1}{|T|} \sum_{t=1}^{|T|} \left| \widehat{SUC}(t) - SUC^*(t) \right|$$

$$\text{RMSE}_{safe} = \sqrt{ \frac{1}{|T|} \sum_{t=1}^{|T|} \left( \widehat{SUC}(t) - SUC^*(t) \right)^2 }$$

---

### 2.4 False-Safe Rate ($FSR$) — Safety-Critical Metric
The proportion of observations where the system predicted a space as safely available (`FREE_SAFE`), but the ground truth indicates the space was physically occupied, blocked, or surface-unsafe:

$$FSR = \frac{\sum_{t=1}^{|T|} \sum_{i=1}^{N} \mathbb{I}(\hat{a}_{i,t} = \text{FREE\_SAFE} \land (y_{i,t} = 1 \lor h_{i,t} = 1 \lor b_{i,t} = 1))}{\sum_{t=1}^{|T|} \sum_{i=1}^{N} \mathbb{I}(\hat{a}_{i,t} = \text{FREE\_SAFE})}$$

**Mandatory Reporting Standards for $FSR$:**
- **Explicit Denominator:** The denominator $\sum_{t=1}^{|T|} \sum_{i=1}^{N} \mathbb{I}(\hat{a}_{i,t} = \text{FREE\_SAFE})$ (total count of predicted-safe observations) must always be reported alongside $FSR$.
- **Disaggregation:** Must report false-safe count, total predicted-safe count, $95\%$ binomial confidence intervals, and breakdowns across day, night, rain, glare, and shadow conditions.

---

### 2.5 State Flicker Rate ($SFR$)
Measures temporal state churn and instability per space per hour:

$$SFR = \frac{3600 \cdot \sum_{i=1}^N \sum_{t=2}^{|T|} \mathbb{I}(\hat{a}_{i,t} \neq \hat{a}_{i,t-1})}{N \cdot \text{Duration}_{\text{hours}}}$$

*Plain Language Definition:* The average number of times a single parking stall's reported availability flips between states per hour due to noise, track dropouts, or shadows.

---

### 2.6 Time to Stable Occupancy State ($TSS$)
The elapsed time (in seconds) from when a vehicle physically comes to a complete halt within polygon $P_i$ ($t_{event}$) to when the system stably emits `OCCUPIED` for at least $K$ consecutive seconds:

$$TSS = \min \{ \Delta t \mid \hat{a}_{i, t_{event} + \Delta t} = \text{OCCUPIED} \quad \forall \tau \in [0, K] \}$$

---

### 2.7 Hazard-to-Space Association Accuracy ($HSA$)
Evaluates the geometric mapping of mobile-inspection hazard events onto facility polygons, evaluated **only** where ground-truth hazard-to-polygon associations were independently surveyed:

$$HSA_{\text{stall}} = \frac{\sum_{m \in M_{\text{stall}}} \mathbb{I}(\hat{P}_{m} = P^*_{m})}{|M_{\text{stall}}|}, \quad HSA_{\text{approach}} = \frac{\sum_{m \in M_{\text{approach}}} \mathbb{I}(\hat{A}_{m} = A^*_{m})}{|M_{\text{approach}}|}$$

- **Requirement:** Stall associations ($HSA_{\text{stall}}$) and approach-zone associations ($HSA_{\text{approach}}$) must be reported separately.
- **Unregistered Handling:** When geometric registration is unavailable or ambiguous, the system must emit `UNKNOWN` rather than guessing.

---

### 2.8 Human Review Workload ($HRW$)
Quantifies operational audit demand on facility inspectors:
1. **Reviews Per Camera-Hour ($RCH$):** Number of candidate observations flagged for human review per hour of active CCTV stream.
2. **Reviews Per Inspection Session ($RSS$):** Number of candidate defect proposals requiring audit per mobile inspection run.
3. **Median Review Time ($\tau_{review}$):** Median seconds required for an operator to audit and act on a candidate proposal.
4. **Automated Temporal Resolution Rate ($ARR$):** Percentage of transient sensor ambiguities resolved automatically by temporal evidence without operator escalation.

*(Note: "Per-kilometer" metrics are restricted to mobile inspection route evaluations and are not used for fixed-camera parking evaluations).*

---

### 2.9 Event Deduplication Effectiveness ($EDE$)
Measures the system's ability to cluster multi-frame observations into discrete physical events:
- **Within-Session Deduplication ($EDE_{\text{within}}$):** Evaluated from track-level ground truth within a single video pass:

$$EDE_{\text{within}} = 1 - \frac{|\text{Over-clustered Duplicates}| + |\text{Under-clustered Splits}|}{M_{\text{ground\_truth\_tracks}}}$$

- **Cross-Session Deduplication Boundary:** Cross-session physical-hazard deduplication must **never** be claimed without verified shared survey coordinates or explicit human verification.

---

### 2.10 Processing Latency and Memory Footprint
- **Inference Latency ($L_{inf}$):** Milliseconds per frame for detection and tracking.
- **End-to-End Latency ($L_{e2e}$):** Milliseconds from frame ingestion to WebSocket UI broadcast.
- **Peak Resident Memory ($RAM_{peak}$):** Megabytes of RSS memory utilized during execution.

---

### 2.11 Maintenance Opportunity Score ($MOS$)
An operational policy metric designed to prioritize asphalt repair windows without disrupting parking operations. Every input has defined units and bounded ranges:

$$MOS(i) = w_1 \cdot P_{hazard}(i) + w_2 \cdot N_{affected}(i) + w_3 \cdot U_{historical}(i) + w_4 \cdot Z_{priority}(i) + w_5 \cdot \Delta t_{age}(i) + w_6 \cdot W_{low\_occ}(i)$$

| Input Factor | Symbol | Unit / Representation | Permitted Range | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Hazard Persistence** | $P_{hazard}(i)$ | Ratio (unitless) | $[0.0, 1.0]$ | Fraction of mobile inspection runs confirming defect presence. |
| **Affected Stalls** | $N_{affected}(i)$ | Scaled count (unitless) | $[0.2, 1.0]$ | $\min(N_{\text{stalls}}, 5) / 5.0$; higher if in shared approach lane $A_i$. |
| **Historical Utilization** | $U_{historical}(i)$ | Fraction (unitless) | $[0.0, 1.0]$ | Average daylight occupancy fraction for stall $i$. |
| **Zone Priority** | $Z_{priority}(i)$ | Policy weight (unitless) | $[0.1, 1.0]$ | Facility zone tier (Emergency=1.0, Visitor=0.7, Remote=0.3). |
| **Defect Age** | $\Delta t_{age}(i)$ | Normalized days | $[0.0, 1.0]$ | $\min(\text{Days since confirmation}, 60) / 60.0$. |
| **Upcoming Low-Occupancy Window** | $W_{low\_occ}(i)$ | Probability / indicator | $[0.0, 1.0]$ | Indicator of scheduled off-peak window in next 24h. |

- **Operational Policy Parameters:** Weights $\mathbf{w} = [w_1, \dots, w_6]$ ($\sum w_k = 1.0$) are defined as **operational policy parameters**, not scientific constants. Initial nominal defaults: $[w_1=0.25, w_2=0.20, w_3=0.15, w_4=0.15, w_5=0.10, w_6=0.15]$. In future iterations, weights may be elicited from facility operations staff or learned from historical work-order decisions.
- **Non-Use of YOLO Confidence:** Bounding-box confidence scores from YOLO are strictly excluded from structural severity calculations.

---

## 3. Experimental Evaluation Setup & Baseline Comparisons

### 3.1 Systems Under Comparison
1. **Baseline A (Geometric Overlap Baseline - e.g., `ParkingSpaceDetector`):**
   - Frame-by-frame YOLO vehicle detection.
   - Polygon IoU thresholding ($> 0.15$).
   - Binary empty/occupied output; zero temporal tracking, zero hazard awareness.
2. **Baseline B (Per-Space Patch Classifier Baseline):**
   - Perspective-warped image crops per parking stall polygon.
   - Dedicated binary image classifier (ResNet-18 / MobileNetV3) fine-tuned on parking patches.
   - No hazard, approach-zone, or tracking awareness.
3. **Proposed System C (RoadSense SiteOps Multi-Modal Framework):**
   - YOLO vehicle detection with stream-local ByteTrack temporal tracking.
   - Configurable temporal hysteresis state engine.
   - Factored multi-dimensional state model.
   - Spatial integration with human-reviewed mobile pavement hazard records.
   - Approach-zone reachability checks.

---

### 3.2 Required Ablation Matrix
To quantify individual component contributions, System C will be evaluated under 6 systematic ablations:

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
The benchmark evaluates 10 real-world operating conditions:
- **ENV-01:** Clear Daytime (Direct Sunlight & Drop Shadows)
- **ENV-02:** Low-Light Twilight / Night (Artificial Pole Lighting & Headlight Glare)
- **ENV-03:** Moderate to Heavy Rainfall (Puddling & Lens Droplets)
- **ENV-04:** Severe Optical Glare (Low Sun Angle)
- **ENV-05:** Dynamic Tree Shadows
- **ENV-06:** Perspective Occlusion (Tall Commercial Vehicles in Foreground)
- **ENV-07:** High-Density Lot ($>95\%$ Occupancy)
- **ENV-08:** Low-Density Lot ($<10\%$ Occupancy)
- **ENV-09:** Camera Vibration / Pole Sway
- **ENV-10:** Temporary Aisle Obstruction (Staged Equipment / Delivery Idling)

---

## 4. Controlled Data Collection & Split Methodology

### 4.1 Single-Camera MVP Split Methodology
For the initial single-camera prototype:
- **Session-Based Grouping:** Frames are grouped strictly by continuous recording sessions.
- **Date/Session Partitioning:** Splits are divided by distinct recording dates and sessions. A standard percentage ratio (e.g., 60/20/20) alone does **not** ensure independence without strict session separation.
- **Zero Adjacent-Frame Leakage:** Frames from the same continuous vehicle parking event or session will never cross between training, validation, and test sets.
- **Held-Out Generalization Boundary:** The held-out test split will contain unseen recording dates, weather conditions, and lighting sessions. Claims of generalization to unseen camera angles or unseen sites will **not** be made from single-camera data.

### 4.2 Multi-Camera & Multi-Site Research Protocol (Future Phase)
When multi-camera footage is collected:
- **Leave-One-Camera-Out (LOCO):** Evaluate model performance on a camera angle never seen during training or validation.
- **Leave-One-Site-Out (LOSO):** Evaluate zero-shot domain transfer on an entirely distinct facility campus.
- **Disaggregated Reporting:** Performance must be reported per camera mount, per sensor resolution, and per site.

### 4.3 Data Retention & Privacy Boundaries
- **Minimising Incidental Capture:** Camera angles focus on vehicle stalls and road surfaces.
- **Short Operational Retention:** Operational raw video is retained for a configurable short window (**24–72 hours** default for pilots) and deleted automatically unless flagged for maintenance review.
- **Research Data Retention:** Dedicated research video data is collected under explicit site agreements, stored in access-controlled storage, and scheduled for deletion upon study completion.
- **Privacy Minimization:** Non-facility areas are masked where technically feasible; the protocol acknowledges that optical resolution and mounting height do not eliminate privacy risks.
