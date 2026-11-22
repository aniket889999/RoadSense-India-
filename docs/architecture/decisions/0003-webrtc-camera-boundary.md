# ADR 0003: WebRTC & Camera Ingest Boundary

## Status
Accepted

## Current status

**Implemented:** User-initiated browser MediaDevices preview and its stop/unmount
cleanup. **Not implemented:** Server WebRTC negotiation, decoding, model inference
or a live sampling/latency contract. The [offer endpoint](../../../services/api/app/routers/live.py)
returns `supported=false`, `status=not_connected`; the
[camera hook](../../../apps/web/src/hooks/useCamera.ts) creates no peer connection.
The original transport/sampling decisions below are future constraints, not
evidence that a WebRTC pipeline exists. See the broader
[status matrix](../system_overview.md#current-status) for the also-unimplemented
MapLibre, PostGIS spatial and ANPR-to-bay integrations.

## Context
Operators may want to inspect real-time video directly from a connected USB webcam or IP dashcam stream. However, real-time computer vision on high-resolution streams can cause severe thermal throttling, frame drops, or unsafe driver-assistance expectations.

## Decision
1. **Explicit Camera Activation**: Camera access begins only after the operator explicitly clicks "Start Camera" in the UI.
2. **Future transport constraint — Not implemented**: A future WebRTC service would require an explicit local-network policy. The current preview has no peer connection or STUN/TURN configuration.
3. **Future bounded sampling — Not implemented**: A proposed conservative live cadence (e.g. 2 FPS) is a design target, not an implemented limit or measured performance.
4. **Transparent Disconnected State**: If full WebRTC server-side peer negotiation cannot be established in a standalone environment without external dependencies, the system provides a typed WebRTC interface and direct browser MediaDevices preview, displays `"Inference: Not connected"`, and strictly refuses to fabricate synthetic detections.
5. **Continuous Safety Warning**: The UI renders a persistent banner: *"Experimental operator review — not a driver alert"*.

## Consequences
- The inspected preview keeps its stream in the browser path; it does not establish a privacy guarantee for all browser/dependency/host behavior.
- The explicit disconnected state avoids presenting simulated detections as inference.
