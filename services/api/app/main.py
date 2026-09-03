"""FastAPI Application Entrypoint for RoadSense India Operations Dashboard."""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from services.api.app.core.config import settings
from services.api.app.core.logging import logger
from services.api.app.db.base import Base
from services.api.app.db.session import engine
from services.api.app.routers import health, live, road_events, sessions


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces strict Content-Security-Policy and security headers."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "connect-src 'self' ws://localhost:* ws://127.0.0.1:* http://localhost:* http://127.0.0.1:*;"
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables
    logger.info("Initializing database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")
    yield
    # Shutdown: dispose database engine
    logger.info("Shutting down database connection pool...")
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Operations API for RoadSense India Dashcam Pothole Review & Field Inspections",
    lifespan=lifespan,
)

# CORS Middleware (confined to local frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ROADSENSE_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Headers Middleware
app.add_middleware(SecurityHeadersMiddleware)

# Include Routers
app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(road_events.router)
app.include_router(live.router)
