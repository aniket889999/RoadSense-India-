# RoadSense India: Privacy & Security Threat Model

> **Security Posture:** Offline-First, Local Data Sovereignty, Path-Traversal Hardened

---

## 1. Threat Profile & Assets to Protect

| Asset | Sensitivity | Threat Vectors | Mitigations |
| :--- | :--- | :--- | :--- |
| **Raw Dashcam Video** | High (May contain pedestrians, license plates, locations) | Remote exfiltration, unauthorized network upload, path traversal | Strict local execution; 0 external outbound network calls; no cloud telemetry; files confined to session spool. |
| **Model Weights (`best.pt`)** | High (Proprietary asset) | Checkpoint poisoning, symlink injection, replacement with malicious pickle | Fail-closed SHA-256 validation; no-follow file descriptor traversal; copy to private `0600` memory snapshot before PyTorch loading. |
| **Inspector Review Decisions** | High (Legal auditability for road works) | Tampering, unauthenticated modification, data loss | Append-only `ReviewAction` audit history with timestamps and user identification; local database transactions. |
| **Local Filesystem Paths** | Medium (Information disclosure) | Exposing user laptop paths in JSON/API responses | Strict path-sanitization: all API responses and JSON exports emit relative safe tokens, UUIDs, or basenames only. |

---

## 2. Attack Vectors & Mitigations

### 2.1. Path Traversal & Symlink Attacks
- **Threat**: Attacker supplies a filename or `--resume-from` / `--weights` path like `../../../../etc/passwd` or creates a symlinked `weights/best.pt` pointing to an arbitrary file.
- **Mitigation**:
  - All filenames undergo `os.path.basename()` extraction and sanitization with strict alphanumeric/safe-character regex.
  - Path components are checked with `os.lstat()` and `O_NOFOLLOW` before opening.
  - Path confinement asserts `Path(target).resolve().relative_to(REPO_ROOT)` without symlink aliases.

### 2.2. Arbitrary Code Execution via Deserialization
- **Threat**: PyTorch `torch.load` unpickling arbitrary Python objects from untrusted `.pt` files.
- **Mitigation**:
  - Model checkpoint is strictly verified against the hardcoded SHA-256 hash in `configs/inference/frozen_baseline.yaml` prior to load.
  - Checkpoint loading occurs only from a validated local snapshot.

### 2.3. Zero-Knowledge Network Isolation
- **Threat**: Third-party JavaScript libraries or CDNs exfiltrating video frames or telemetry.
- **Mitigation**:
  - 100% self-hosted static assets; **no CDNs, Google Fonts, or external scripts**.
  - Restrictive Content Security Policy (CSP): `default-src 'self'; script-src 'self'; connect-src 'self' ws://localhost:* http://localhost:*`.
  - Zero telemetry or analytics trackers.

---

## 3. Storage & Retention Policies

- **Default Retention**: Manual operator deletion. Sessions persist locally until explicitly purged via the UI or API.
- **Spool Permissions**: Uploaded files and generated reports are created with restricted file permissions (`0600` for files, `0700` for session directories).
- **Session Isolation**: Every session is isolated under `outputs/sessions/<session_id>/` with no shared mutable state.
