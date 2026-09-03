# ADR 0003: WebRTC & Camera Ingest Boundary

## Status
Accepted

## Context
Operators may want to inspect real-time video directly from a connected USB webcam or IP dashcam stream. However, real-time computer vision on high-resolution streams can cause severe thermal throttling, frame drops, or unsafe driver-assistance expectations.

## Decision
1. **Explicit Camera Activation**: Camera access begins only after the operator explicitly clicks "Start Camera" in the UI.
2. **Local Loopback Only**: The camera stream and WebRTC peer connection bind strictly to `localhost` with no external STUN/TURN servers.
3. **Bounded Sampling**: Real-time live inference is capped at a conservative rate (e.g. 2 FPS) to maintain low thermal impact and stable UI responsiveness.
4. **Transparent Disconnected State**: If full WebRTC server-side peer negotiation cannot be established in a standalone environment without external dependencies, the system provides a typed WebRTC interface and direct browser MediaDevices preview, displays `"Inference: Not connected"`, and strictly refuses to fabricate synthetic detections.
5. **Continuous Safety Warning**: The UI renders a persistent banner: *"Experimental operator review — not a driver alert"*.

## Consequences
- Protects user privacy by preventing camera data from leaving the local machine.
- Manages operator expectations with zero deceptive simulation.
