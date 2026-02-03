"""Table management API routes."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import Table, TableState, get_db
from ..deps import get_connection_manager, ConnectionManager

router = APIRouter(prefix="/tables", tags=["tables"])


class TableCreate(BaseModel):
    """Schema for creating a table."""

    number: int
    capacity: int = 4
    x_position: float = 0.0
    y_position: float = 0.0
    zone: str = "main"


class TableUpdate(BaseModel):
    """Schema for updating a table."""

    state: Optional[str] = None
    current_guests: Optional[int] = None
    assigned_server_id: Optional[int] = None
    priority_score: Optional[float] = None
    hand_raise_detected: Optional[bool] = None
    notes: Optional[str] = None


class TableResponse(BaseModel):
    """Schema for table response."""

    id: int
    number: int
    capacity: int
    x_position: float
    y_position: float
    state: str
    current_guests: int
    state_changed_at: Optional[str]
    seated_at: Optional[str]
    assigned_server_id: Optional[int]
    priority_score: float
    hand_raise_detected: bool
    last_interaction: Optional[str]
    zone: str
    notes: Optional[str]

    class Config:
        from_attributes = True


@router.get("", response_model=List[TableResponse])
async def get_tables(
    state: Optional[str] = None,
    zone: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get all tables, optionally filtered by state or zone."""
    query = select(Table)

    if state:
        try:
            table_state = TableState(state)
            query = query.where(Table.state == table_state)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid state: {state}")

    if zone:
        query = query.where(Table.zone == zone)

    result = await db.execute(query)
    tables = result.scalars().all()
    return [TableResponse(**t.to_dict()) for t in tables]


@router.get("/{table_id}", response_model=TableResponse)
async def get_table(table_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific table by ID."""
    result = await db.execute(select(Table).where(Table.id == table_id))
    table = result.scalar_one_or_none()

    if not table:
        raise HTTPException(status_code=404, detail="Table not found")

    return TableResponse(**table.to_dict())


@router.post("", response_model=TableResponse)
async def create_table(
    table_data: TableCreate,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Create a new table."""
    # Check if table number already exists
    result = await db.execute(select(Table).where(Table.number == table_data.number))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=400, detail=f"Table {table_data.number} already exists"
        )

    table = Table(
        number=table_data.number,
        capacity=table_data.capacity,
        x_position=table_data.x_position,
        y_position=table_data.y_position,
        zone=table_data.zone,
    )

    db.add(table)
    await db.commit()
    await db.refresh(table)

    # Broadcast update
    await ws_manager.broadcast(
        {"type": "table_created", "data": table.to_dict()}
    )

    return TableResponse(**table.to_dict())


@router.put("/{table_id}", response_model=TableResponse)
async def update_table(
    table_id: int,
    table_data: TableUpdate,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Update a table's state or properties."""
    result = await db.execute(select(Table).where(Table.id == table_id))
    table = result.scalar_one_or_none()

    if not table:
        raise HTTPException(status_code=404, detail="Table not found")

    # Track state change
    old_state = table.state

    # Update fields
    if table_data.state is not None:
        try:
            new_state = TableState(table_data.state)
            if new_state != table.state:
                table.state = new_state
                table.state_changed_at = datetime.utcnow()

                # Track seating time
                if new_state == TableState.SEATED:
                    table.seated_at = datetime.utcnow()
                elif new_state == TableState.EMPTY:
                    table.seated_at = None
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid state: {table_data.state}"
            )

    if table_data.current_guests is not None:
        table.current_guests = table_data.current_guests

    if table_data.assigned_server_id is not None:
        table.assigned_server_id = table_data.assigned_server_id

    if table_data.priority_score is not None:
        table.priority_score = table_data.priority_score

    if table_data.hand_raise_detected is not None:
        table.hand_raise_detected = table_data.hand_raise_detected

    if table_data.notes is not None:
        table.notes = table_data.notes

    table.last_interaction = datetime.utcnow()

    await db.commit()
    await db.refresh(table)

    # Broadcast update
    await ws_manager.broadcast(
        {
            "type": "table_updated",
            "data": table.to_dict(),
            "old_state": old_state.value if old_state else None,
        }
    )

    return TableResponse(**table.to_dict())


@router.delete("/{table_id}")
async def delete_table(
    table_id: int,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Delete a table."""
    result = await db.execute(select(Table).where(Table.id == table_id))
    table = result.scalar_one_or_none()

    if not table:
        raise HTTPException(status_code=404, detail="Table not found")

    table_number = table.number
    await db.delete(table)
    await db.commit()

    # Broadcast update
    await ws_manager.broadcast(
        {"type": "table_deleted", "data": {"id": table_id, "number": table_number}}
    )

    return {"message": f"Table {table_number} deleted"}


@router.post("/{table_id}/seat")
async def seat_customers(
    table_id: int,
    guest_count: int,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Seat customers at a table."""
    result = await db.execute(select(Table).where(Table.id == table_id))
    table = result.scalar_one_or_none()

    if not table:
        raise HTTPException(status_code=404, detail="Table not found")

    if table.state != TableState.EMPTY:
        raise HTTPException(status_code=400, detail="Table is not empty")

    if guest_count > table.capacity:
        raise HTTPException(
            status_code=400,
            detail=f"Guest count {guest_count} exceeds table capacity {table.capacity}",
        )

    table.state = TableState.SEATED
    table.current_guests = guest_count
    table.state_changed_at = datetime.utcnow()
    table.seated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(table)

    # Broadcast update
    await ws_manager.broadcast(
        {"type": "customers_seated", "data": table.to_dict()}
    )

    return TableResponse(**table.to_dict())
