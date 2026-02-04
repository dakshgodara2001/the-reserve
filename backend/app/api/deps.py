"""API dependencies for dependency injection."""

from typing import AsyncGenerator, Optional
from fastapi import Depends

from ..models import get_db, async_session
from sqlalchemy.ext.asyncio import AsyncSession


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async for session in get_db():
        yield session


class ConnectionManager:
    """WebSocket connection manager for real-time updates."""

    def __init__(self):
        self.active_connections: list = []

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        import json

        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                self.disconnect(connection)

    async def send_personal(self, message: dict, websocket):
        """Send message to specific client."""
        import json

        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)


# Global connection manager instance
connection_manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get connection manager dependency."""
    return connection_manager
