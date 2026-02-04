"""Event management API routes."""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import Event, EventType, EventPriority, get_db
from ..deps import get_connection_manager, ConnectionManager

router = APIRouter(prefix="/events", tags=["events"])


class EventCreate(BaseModel):
    """Schema for creating an event."""

    event_type: str
    priority: str = "medium"
    table_id: Optional[int] = None
    staff_id: Optional[int] = None
    zone: Optional[str] = None
    x_position: Optional[float] = None
    y_position: Optional[float] = None
    confidence: float = 1.0
    source: str = "manual"
    description: str = ""
    metadata: dict = {}


class EventResponse(BaseModel):
    """Schema for event response."""

    id: int
    event_type: str
    priority: str
    timestamp: Optional[str]
    table_id: Optional[int]
    staff_id: Optional[int]
    zone: Optional[str]
    x_position: Optional[float]
    y_position: Optional[float]
    confidence: float
    source: str
    description: str
    metadata: dict
    resolved: bool
    resolved_at: Optional[str]
    resolved_by: Optional[int]

    class Config:
        from_attributes = True


@router.get("", response_model=List[EventResponse])
async def get_events(
    event_type: Optional[str] = None,
    priority: Optional[str] = None,
    table_id: Optional[int] = None,
    staff_id: Optional[int] = None,
    resolved: Optional[bool] = None,
    since_minutes: int = Query(default=60, description="Get events from last N minutes"),
    limit: int = Query(default=100, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Get events, optionally filtered."""
    query = select(Event).order_by(desc(Event.timestamp))

    # Time filter
    since = datetime.utcnow() - timedelta(minutes=since_minutes)
    query = query.where(Event.timestamp >= since)

    if event_type:
        try:
            et = EventType(event_type)
            query = query.where(Event.event_type == et)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid event type: {event_type}"
            )

    if priority:
        try:
            p = EventPriority(priority)
            query = query.where(Event.priority == p)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")

    if table_id is not None:
        query = query.where(Event.table_id == table_id)

    if staff_id is not None:
        query = query.where(Event.staff_id == staff_id)

    if resolved is not None:
        query = query.where(Event.resolved == resolved)

    query = query.limit(limit)

    result = await db.execute(query)
    events = result.scalars().all()
    return [EventResponse(**e.to_dict()) for e in events]


@router.get("/unresolved", response_model=List[EventResponse])
async def get_unresolved_events(
    priority: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Get unresolved events requiring attention."""
    query = (
        select(Event)
        .where(Event.resolved == False)
        .order_by(desc(Event.priority), desc(Event.timestamp))
    )

    if priority:
        try:
            p = EventPriority(priority)
            query = query.where(Event.priority == p)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")

    query = query.limit(limit)

    result = await db.execute(query)
    events = result.scalars().all()
    return [EventResponse(**e.to_dict()) for e in events]


@router.get("/stats")
async def get_event_stats(
    since_minutes: int = Query(default=60),
    db: AsyncSession = Depends(get_db),
):
    """Get event statistics."""
    since = datetime.utcnow() - timedelta(minutes=since_minutes)

    query = select(Event).where(Event.timestamp >= since)
    result = await db.execute(query)
    events = result.scalars().all()

    # Calculate stats
    total = len(events)
    by_type = {}
    by_priority = {}
    resolved_count = 0

    for event in events:
        # Count by type
        type_key = event.event_type.value
        by_type[type_key] = by_type.get(type_key, 0) + 1

        # Count by priority
        priority_key = event.priority.value
        by_priority[priority_key] = by_priority.get(priority_key, 0) + 1

        # Count resolved
        if event.resolved:
            resolved_count += 1

    return {
        "total": total,
        "resolved": resolved_count,
        "unresolved": total - resolved_count,
        "by_type": by_type,
        "by_priority": by_priority,
        "time_range_minutes": since_minutes,
    }


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific event by ID."""
    result = await db.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return EventResponse(**event.to_dict())


@router.post("", response_model=EventResponse)
async def create_event(
    event_data: EventCreate,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Create a new event (for manual injection/testing)."""
    try:
        event_type = EventType(event_data.event_type)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid event type: {event_data.event_type}"
        )

    try:
        priority = EventPriority(event_data.priority)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid priority: {event_data.priority}"
        )

    event = Event(
        event_type=event_type,
        priority=priority,
        table_id=event_data.table_id,
        staff_id=event_data.staff_id,
        zone=event_data.zone,
        x_position=event_data.x_position,
        y_position=event_data.y_position,
        confidence=event_data.confidence,
        source=event_data.source,
        description=event_data.description,
        extra_data=event_data.metadata,
    )

    db.add(event)
    await db.commit()
    await db.refresh(event)

    # Broadcast to all clients
    await ws_manager.broadcast({"type": "new_event", "data": event.to_dict()})

    return EventResponse(**event.to_dict())


@router.put("/{event_id}/resolve")
async def resolve_event(
    event_id: int,
    staff_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Mark an event as resolved."""
    result = await db.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    event.resolved = True
    event.resolved_at = datetime.utcnow()
    event.resolved_by = staff_id

    await db.commit()
    await db.refresh(event)

    # Broadcast update
    await ws_manager.broadcast({"type": "event_resolved", "data": event.to_dict()})

    return EventResponse(**event.to_dict())


@router.delete("/{event_id}")
async def delete_event(event_id: int, db: AsyncSession = Depends(get_db)):
    """Delete an event."""
    result = await db.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    await db.delete(event)
    await db.commit()

    return {"message": f"Event {event_id} deleted"}
