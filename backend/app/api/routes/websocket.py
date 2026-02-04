"""WebSocket routes for real-time communication."""

import json
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import Table, Staff, Event, get_db, async_session
from ..deps import get_connection_manager, ConnectionManager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_connection_manager),
):
    """Main WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        # Send initial state on connection
        async with async_session() as db:
            # Get all tables
            result = await db.execute(select(Table))
            tables = result.scalars().all()

            # Get all staff
            result = await db.execute(select(Staff))
            staff_members = result.scalars().all()

            # Get recent events (last 50)
            result = await db.execute(
                select(Event).order_by(Event.timestamp.desc()).limit(50)
            )
            events = result.scalars().all()

            # Send initial state
            await manager.send_personal(
                {
                    "type": "initial_state",
                    "data": {
                        "tables": [t.to_dict() for t in tables],
                        "staff": [s.to_dict() for s in staff_members],
                        "events": [e.to_dict() for e in events],
                    },
                },
                websocket,
            )

        # Listen for messages from client
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            await handle_client_message(message, websocket, manager)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def handle_client_message(
    message: dict,
    websocket: WebSocket,
    manager: ConnectionManager,
):
    """Handle incoming WebSocket messages from clients."""
    msg_type = message.get("type")

    if msg_type == "ping":
        # Heartbeat
        await manager.send_personal({"type": "pong"}, websocket)

    elif msg_type == "subscribe":
        # Subscribe to specific event types or tables
        # For future implementation
        await manager.send_personal(
            {"type": "subscribed", "data": message.get("data", {})}, websocket
        )

    elif msg_type == "request_state":
        # Client requesting current state
        async with async_session() as db:
            result = await db.execute(select(Table))
            tables = result.scalars().all()

            result = await db.execute(select(Staff))
            staff_members = result.scalars().all()

            await manager.send_personal(
                {
                    "type": "state_update",
                    "data": {
                        "tables": [t.to_dict() for t in tables],
                        "staff": [s.to_dict() for s in staff_members],
                    },
                },
                websocket,
            )

    elif msg_type == "simulate_event":
        # Allow clients to simulate events for testing
        event_data = message.get("data", {})
        await handle_simulation(event_data, manager)

    else:
        # Unknown message type
        await manager.send_personal(
            {"type": "error", "message": f"Unknown message type: {msg_type}"}, websocket
        )


async def handle_simulation(event_data: dict, manager: ConnectionManager):
    """Handle simulation events from client."""
    from ...models import EventType, EventPriority

    async with async_session() as db:
        try:
            event_type = EventType(event_data.get("event_type", "gesture_detected"))
            priority = EventPriority(event_data.get("priority", "medium"))

            event = Event(
                event_type=event_type,
                priority=priority,
                table_id=event_data.get("table_id"),
                staff_id=event_data.get("staff_id"),
                zone=event_data.get("zone"),
                confidence=event_data.get("confidence", 1.0),
                source="simulation",
                description=event_data.get("description", "Simulated event"),
                extra_data=event_data.get("metadata", {}),
            )

            db.add(event)
            await db.commit()
            await db.refresh(event)

            # Broadcast the simulated event
            await manager.broadcast({"type": "new_event", "data": event.to_dict()})

        except Exception as e:
            await manager.broadcast(
                {"type": "simulation_error", "message": str(e)}
            )
