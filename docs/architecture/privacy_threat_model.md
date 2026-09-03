# RoadSense India: Privacy & Security Threat Model

> **Security Posture:** Offline-First, Local Data Sovereignty, Subprocess-Hardened, Path-Traversal Protected

---

## 1. Threat Profile & Assets to Protect

| Asset | Sensitivity | Threat Vectors | Mitigations |
| :--- | :--- | :--- | :--- |
| **Raw Dashcam Video** | High (Pedestrians, vehicle license plates, geolocation) | Remote exfiltration, unauthorized network upload, path traversal | Strict local execution; 0 external outbound network calls; no cloud telemetry; files confined to session spool. |
| **Model Weights (`best.pt`)** | High (Proprietary model artifact) | Checkpoint poisoning, symlink injection, malicious pickle substitution | Fail-closed SHA-256 validation; no-follow file descriptor traversal; copy to private `0600` memory snapshot before PyTorch loading. |
| **ByteTrack Configuration** | High (Temporal association parameters) | Parameter tampering, arbitrary config injection | SHA-256 pinned in `configs/tracking/bytetrack_default.yaml`; validated prior to tracker instantiation. |
| **Inspector Review Decisions** | High (Legal auditability for road infrastructure) | Tampering, unauthenticated modification, data loss | Append-only `ReviewAction` audit history with timestamps and user identification; local database transactions. |
| **Local Filesystem Paths** | Medium (Information disclosure) | Exposing user laptop paths in JSON/API responses | Strict path-sanitization: all API responses and JSON exports emit relative safe tokens, UUIDs, or basenames only. |

---

## 2. Attack Vectors & Mitigations

### 2.1. Path Traversal & Symlink Attacks
- **Threat**: Attacker supplies a filename or `--resume-from` / `--weights` path like `../../../../etc/passwd` or creates a symlinked `weights/best.pt` pointing to an arbitrary file.
- **Mitigation**:
  - All filenames undergo `os.path.basename()` extraction and sanitization with strict alphanumeric/safe-character regex.
  - Path components are checked with `os.lstat()` and `O_NOFOLLOW` before opening.
  - Path confinement asserts `Path(target).resolve().relative_to(REPO_ROOT)` without symlink aliases.

### 2.2. Subprocess Command Injection (FFmpeg / FFprobe)
- **Threat**: User-controlled filenames containing shell metacharacters (`;`, `|`, `&&`, `$()`) passed to shell execution.
- **Mitigation**:
  - Direct `subprocess.run(list_of_args)` and `subprocess.Popen(list_of_args)` argument arrays are used exclusively.
  - Shell execution (`shell=True`) is strictly prohibited throughout the entire codebase.
  - Media filenames are remapped to UUIDs on disk immediately upon intake.

### 2.3. Resource Exhaustion & Memory Bomb Attacks
- **Threat**: Gigantic video files or decompression bombs attempting to crash server RAM.
- **Mitigation**:
  - Streaming frame generation: frames are decoded sequentially one-by-one with bounded queue depth (`max_frames`, `target_fps`).
  - Strict file size cap (500 MB upload limit) and duration cap (3600s).
  - Explicit cancellation hooks checked between frame inference cycles.

### 2.4. Zero-Knowledge Network Isolation
- **Threat**: Third-party JavaScript libraries or CDNs exfiltrating video frames or telemetry.
- **Mitigation**:
  - 100% self-hosted static assets; **no CDNs, Google Fonts, or external scripts**.
  - Restrictive Content Security Policy (CSP): `default-src 'self'; script-src 'self'; connect-src 'self' ws://localhost:* http://localhost:*`.
  - Zero telemetry, crash reporters, or analytics trackers.

### 2.5. Connected Camera Boundary
- **Threat**: Background camera spying or involuntary streaming.
- **Mitigation**:
  - Explicit user interaction required to start browser `MediaDevices.getUserMedia` stream.
  - Direct local canvas loopback only; camera never streams over network or STUN/TURN servers.
  - Trackers and buffers are destroyed immediately when the camera stream stops.

---

## 3. Storage & Retention Policies

- **Default Retention**: Manual operator deletion. Sessions persist locally until explicitly purged via the UI or API.
- **Configurable Purge**: `/api/v1/sessions/{id}` supports selective retention (delete raw video only, delete annotated replay, or delete full database record).
- **Spool Permissions**: Uploaded files and generated reports are created with restricted file permissions (`0600` for files, `0700` for session directories).
- **Session Isolation**: Every session is isolated under `outputs/sessions/<session_id>/` with no shared mutable state.
