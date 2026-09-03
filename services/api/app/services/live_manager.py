"""WebSocket connection and real-time event broadcasting manager."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Set
from fastapi import WebSocket
from services.api.app.schemas.session import SessionProgressEvent


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = set()
            self.active_connections[session_id].add(websocket)

    async def disconnect(self, session_id: str, websocket: WebSocket):
        async with self._lock:
            if session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]

    async def broadcast_progress(self, session_id: str, event: SessionProgressEvent):
        async with self._lock:
            sockets = list(self.active_connections.get(session_id, []))

        message = event.model_dump_json()
        for ws in sockets:
            try:
                await ws.send_text(message)
            except Exception:
                pass


ws_manager = ConnectionManager()
