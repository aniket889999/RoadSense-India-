"""Optional access control for the ANPR endpoint: API key and per-client rate limit.

Both are OFF unless configured, so the local demonstration behaves exactly as before.

* ``ROADSENSE_API_KEY`` set (non-empty) -> requests must send ``X-API-Key`` with the same
  value, compared in constant time; otherwise HTTP 401. Use this when the API is reachable
  from anything other than the local machine. Do not ship the key to a browser bundle.
* ``ROADSENSE_ANPR_RATE_LIMIT_PER_MINUTE`` set to a positive integer -> at most that many
  requests per client address per rolling 60 s; otherwise HTTP 429 with ``Retry-After``.
  The limiter is in-process memory: per worker process, reset on restart, not shared
  across instances. It is not a substitute for a gateway in a real deployment.

The key, uploaded bytes and plate text are never logged.
"""

from __future__ import annotations

from collections import deque
import hmac
import math
import os
import threading
import time
from typing import Callable, Deque, Dict

from fastapi import HTTPException, Request

ENV_API_KEY = "ROADSENSE_API_KEY"
ENV_RATE_LIMIT = "ROADSENSE_ANPR_RATE_LIMIT_PER_MINUTE"
API_KEY_HEADER = "X-API-Key"
WINDOW_SECONDS = 60.0
MAX_TRACKED_CLIENTS = 10_000


class SlidingWindowLimiter:
    """Thread-safe per-client rolling-window counter with bounded memory."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str, limit: int) -> float | None:
        """Record a hit; return ``None`` if allowed, else seconds until the next slot."""
        now = self._clock()
        with self._lock:
            hits = self._hits.get(client)
            if hits is None:
                if len(self._hits) >= MAX_TRACKED_CLIENTS:
                    self._evict(now)
                hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] >= WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= limit:
                return max(0.0, WINDOW_SECONDS - (now - hits[0]))
            hits.append(now)
            return None

    def _evict(self, now: float) -> None:
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= WINDOW_SECONDS]:
            del self._hits[key]
        if len(self._hits) >= MAX_TRACKED_CLIENTS:  # still full: drop the oldest-active client
            del self._hits[min(self._hits, key=lambda k: self._hits[k][-1])]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


def _rate_limit() -> int:
    raw = os.environ.get(ENV_RATE_LIMIT, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="ANPR rate-limit configuration is invalid.") from exc
    if value < 0:
        raise HTTPException(status_code=500, detail="ANPR rate-limit configuration is invalid.")
    return value


async def require_anpr_access(request: Request) -> None:
    """FastAPI dependency: enforce the optional API key, then the optional rate limit."""
    expected = os.environ.get(ENV_API_KEY, "")
    if expected:
        supplied = request.headers.get(API_KEY_HEADER, "")
        if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.",
                                headers={"WWW-Authenticate": "ApiKey"})
    limit = _rate_limit()
    if limit:
        client = request.client.host if request.client else "unknown"
        wait = limiter.check(client, limit)
        if wait is not None:
            raise HTTPException(status_code=429, detail="Too many ANPR requests; retry later.",
                                headers={"Retry-After": str(max(1, math.ceil(wait)))})
