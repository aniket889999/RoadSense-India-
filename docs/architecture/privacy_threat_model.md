# RoadSense India: Privacy & Security Threat Model

> **Security posture:** Local inference design with route-specific file/process
> controls; this is not a completed deployment security audit.

## Current status

- **Implemented:** ANPR local-asset inference and explicit browser exports;
  [ANPR privacy/retention](../anpr/privacy_and_retention.md) explains sensitive
  data boundaries. Model candidates, including applied schema 1.2 cleanup,
  still require review.
- **Implemented:** User-initiated camera preview. **Not implemented:** server
  WebRTC inference; no peer connection is created by the inspected preview.
- **Partial:** Configurable relational storage and application review records.
  **Not implemented:** PostGIS spatial queries, deployment RBAC, authenticated
  reviewer identity, and an automatic ANPR export expiry service.
- **Not implemented:** Interactive MapLibre rendering, ANPR-to-bay identity
  linkage, registration lookup, payment or barrier control.

The threat scenarios below preserve the earlier design review. Mitigation
descriptions are scoped implementation intentions unless verified per route;
they do not prove zero outbound traffic, complete path sanitization, encrypted
storage or a certified immutable ledger. See the source-linked
[current architecture matrix](system_overview.md#current-status).

---

## 1. Threat Profile & Assets to Protect

| Asset | Sensitivity | Threat Vectors | Mitigations |
| :--- | :--- | :--- | :--- |
| **Raw Dashcam Video** | High (Pedestrians, vehicle license plates, geolocation) | Remote exfiltration, unauthorized network upload, path traversal | Local processing/session storage design; route-specific confinement needs verification. No end-to-end zero-traffic guarantee. |
| **Model Weights (`best.pt`)** | High (Proprietary model artifact) | Checkpoint poisoning, symlink injection, malicious pickle substitution | Fail-closed SHA-256 validation; no-follow file descriptor traversal; copy to private `0600` memory snapshot before PyTorch loading. |
| **ByteTrack Configuration** | High (Temporal association parameters) | Parameter tampering, arbitrary config injection | SHA-256 pinned in `configs/tracking/bytetrack_default.yaml`; validated prior to tracker instantiation. |
| **Inspector Review Decisions** | High (Review provenance) | Tampering, unauthenticated modification, data loss | Application review records and transactions; reviewer labels do not establish authenticated identity or immutable storage. |
| **Local Filesystem Paths** | Medium (Information disclosure) | Exposing user laptop paths in JSON/API responses | Some artifact/export paths have confinement/sanitization checks; a complete API disclosure audit is not established here. |

---

## 2. Attack Vectors & Mitigations

### 2.1. Path Traversal & Symlink Attacks
- **Threat**: Attacker supplies a filename or `--resume-from` / `--weights` path like `../../../../etc/passwd` or creates a symlinked `weights/best.pt` pointing to an arbitrary file.
- **Mitigation**:
  - Sensitive artifact/model paths have lexical/resolved confinement and symlink checks; inspect each route/runtime rather than assume one uniform policy for all files.
  - Some protected model-loading paths use no-follow descriptors and private snapshots. This is not a claim that every filesystem operation uses those mechanisms.

### 2.2. Subprocess Command Injection (FFmpeg / FFprobe)
- **Threat**: User-controlled filenames containing shell metacharacters (`;`, `|`, `&&`, `$()`) passed to shell execution.
- **Mitigation**:
  - Inspected media subprocesses use argument arrays; shell execution must not be introduced for user-controlled media arguments.
  - Media filenames are remapped to UUIDs on disk immediately upon intake.

### 2.3. Resource Exhaustion & Memory Bomb Attacks
- **Threat**: Gigantic video files or decompression bombs attempting to crash server RAM.
- **Mitigation**:
  - Streaming frame generation: frames are decoded sequentially one-by-one with bounded queue depth (`max_frames`, `target_fps`).
  - Upload/duration/dimension caps are route-specific. The historical video target was 500 MB / 3600s; it must not be presented as a universal ANPR or parking limit. The ANPR bounds are documented in its [upload contract](../anpr/development_plan.md#module-2-file-ownership-and-upload-contract).
  - Explicit cancellation hooks checked between frame inference cycles.

### 2.4. Network Isolation Goals — Partial
- **Threat**: Third-party JavaScript libraries or CDNs exfiltrating video frames or telemetry.
- **Mitigation**:
  - The inspected ANPR path uses local models; camera preview remains browser-side. No cloud OCR call is part of that path.
  - FastAPI adds CSP/security headers, including an `unsafe-inline` allowance. API response headers do not establish the Next.js document's effective CSP.
  - Browser, dependency, build-tool, IDE and host telemetry/network behavior has not been audited end to end. Do not advertise "100% local" or "zero telemetry" as a verified platform guarantee.

### 2.5. Connected Camera Boundary
- **Threat**: Background camera spying or involuntary streaming.
- **Mitigation**:
  - Explicit user interaction required to start browser `MediaDevices.getUserMedia` stream.
  - The inspected preview attaches a MediaStream to a browser video element, without a server peer connection or inference tracker.
  - The stop/unmount path stops its MediaStream tracks. This does not certify immediate erasure of all browser/OS buffers.

---

## 3. Storage & Retention Policies

- **Default Retention**: Manual operator deletion. Sessions persist locally until explicitly purged via the UI or API.
- **Implemented / Partial**: Explicit session/parking cleanup paths exist; retention and filesystem controls vary by workflow. This does not establish a uniform automatic expiry policy or identical permissions for every report.
- **Not implemented**: Automatic deletion of ANPR browser downloads or an ANPR retention scheduler. Downloads and research prediction/details files can contain plate strings and require operator-managed retention.
