# ADR 0001: Monorepo Architecture for RoadSense India Operations Dashboard

## Status
Accepted

## Context
RoadSense India began as a standalone Python CLI and Streamlit prototype. To support professional municipal and fleet operations, the project requires:
1. A rich, high-performance, responsive operator dashboard with frame-accurate video synchronization.
2. A decoupled asynchronous REST/WebSocket API capable of background video processing without blocking the UI.
3. Continued reuse of verified Python machine learning inference, provenance verification, and bounding algorithms.

## Decision
We adopt a monorepo structure containing:
- `apps/web/`: Next.js App Router (TypeScript) frontend.
- `services/api/`: FastAPI backend with asynchronous background task runners.
- `src/ml/`: Shared core Python machine learning logic and verified provenance checking.
- `infra/`: Local Docker Compose configurations for PostgreSQL/PostGIS.
- `docs/`: Comprehensive architecture, product, and API contracts.

`app.py` is preserved as a legacy/research Streamlit interface and is not deleted.

## Consequences
- Clean separation of concerns between presentation and ML execution.
- No code duplication for model inference: the FastAPI service directly imports and executes `src.ml.local_model_runtime` and `src.ml.drive_review`.
- Independent testing for frontend components, API endpoints, and ML modules.
