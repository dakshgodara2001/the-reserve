"""Staff management API routes."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import Staff, StaffRole, StaffStatus, get_db
from ..deps import get_connection_manager, ConnectionManager

router = APIRouter(prefix="/staff", tags=["staff"])


class StaffCreate(BaseModel):
    """Schema for creating a staff member."""

    name: str
    role: str
    skills: List[str] = []
    max_tables: int = 4


class StaffUpdate(BaseModel):
    """Schema for updating a staff member."""

    status: Optional[str] = None
    x_position: Optional[float] = None
    y_position: Optional[float] = None
    current_zone: Optional[str] = None
    assigned_tables: Optional[List[int]] = None
    current_task_count: Optional[int] = None


class StaffResponse(BaseModel):
    """Schema for staff response."""

    id: int
    name: str
    role: str
    status: str
    x_position: float
    y_position: float
    current_zone: str
    assigned_tables: List[int]
    current_task_count: int
    max_tables: int
    tasks_completed_today: int
    avg_response_time: float
    skills: List[str]
    shift_start: Optional[str]
    shift_end: Optional[str]
    last_activity: Optional[str]

    class Config:
        from_attributes = True


@router.get("", response_model=List[StaffResponse])
async def get_staff(
    role: Optional[str] = None,
    status: Optional[str] = None,
    zone: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get all staff members, optionally filtered."""
    query = select(Staff)

    if role:
        try:
            staff_role = StaffRole(role)
            query = query.where(Staff.role == staff_role)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid role: {role}")

    if status:
        try:
            staff_status = StaffStatus(status)
            query = query.where(Staff.status == staff_status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    if zone:
        query = query.where(Staff.current_zone == zone)

    result = await db.execute(query)
    staff_members = result.scalars().all()
    return [StaffResponse(**s.to_dict()) for s in staff_members]


@router.get("/available", response_model=List[StaffResponse])
async def get_available_staff(
    role: Optional[str] = None, db: AsyncSession = Depends(get_db)
):
    """Get staff members available for assignment."""
    query = select(Staff).where(Staff.status == StaffStatus.AVAILABLE)

    if role:
        try:
            staff_role = StaffRole(role)
            query = query.where(Staff.role == staff_role)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid role: {role}")

    result = await db.execute(query)
    staff_members = result.scalars().all()

    # Filter to those who can take more tables
    available = [s for s in staff_members if s.is_available_for_assignment()]

    return [StaffResponse(**s.to_dict()) for s in available]


@router.get("/{staff_id}", response_model=StaffResponse)
async def get_staff_member(staff_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific staff member by ID."""
    result = await db.execute(select(Staff).where(Staff.id == staff_id))
    staff = result.scalar_one_or_none()

    if not staff:
        raise HTTPException(status_code=404, detail="Staff member not found")

    return StaffResponse(**staff.to_dict())


@router.post("", response_model=StaffResponse)
async def create_staff(
    staff_data: StaffCreate,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Create a new staff member."""
    try:
        role = StaffRole(staff_data.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {staff_data.role}")

    staff = Staff(
        name=staff_data.name,
        role=role,
        skills=staff_data.skills,
        max_tables=staff_data.max_tables,
        shift_start=datetime.utcnow(),
    )

    db.add(staff)
    await db.commit()
    await db.refresh(staff)

    # Broadcast update
    await ws_manager.broadcast({"type": "staff_created", "data": staff.to_dict()})

    return StaffResponse(**staff.to_dict())


@router.put("/{staff_id}", response_model=StaffResponse)
async def update_staff(
    staff_id: int,
    staff_data: StaffUpdate,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Update a staff member's status or location."""
    result = await db.execute(select(Staff).where(Staff.id == staff_id))
    staff = result.scalar_one_or_none()

    if not staff:
        raise HTTPException(status_code=404, detail="Staff member not found")

    # Update fields
    if staff_data.status is not None:
        try:
            staff.status = StaffStatus(staff_data.status)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid status: {staff_data.status}"
            )

    if staff_data.x_position is not None:
        staff.x_position = staff_data.x_position

    if staff_data.y_position is not None:
        staff.y_position = staff_data.y_position

    if staff_data.current_zone is not None:
        staff.current_zone = staff_data.current_zone

    if staff_data.assigned_tables is not None:
        staff.assigned_tables = staff_data.assigned_tables

    if staff_data.current_task_count is not None:
        staff.current_task_count = staff_data.current_task_count

    staff.last_activity = datetime.utcnow()

    await db.commit()
    await db.refresh(staff)

    # Broadcast update
    await ws_manager.broadcast({"type": "staff_updated", "data": staff.to_dict()})

    return StaffResponse(**staff.to_dict())


@router.post("/{staff_id}/assign/{table_id}")
async def assign_table(
    staff_id: int,
    table_id: int,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Assign a table to a staff member."""
    from ...models import Table

    # Get staff member
    result = await db.execute(select(Staff).where(Staff.id == staff_id))
    staff = result.scalar_one_or_none()
    if not staff:
        raise HTTPException(status_code=404, detail="Staff member not found")

    # Get table
    result = await db.execute(select(Table).where(Table.id == table_id))
    table = result.scalar_one_or_none()
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")

    # Check availability
    if not staff.is_available_for_assignment():
        raise HTTPException(
            status_code=400, detail="Staff member not available for assignment"
        )

    # Assign
    if table_id not in staff.assigned_tables:
        staff.assigned_tables = staff.assigned_tables + [table_id]

    table.assigned_server_id = staff_id
    staff.last_activity = datetime.utcnow()

    await db.commit()
    await db.refresh(staff)
    await db.refresh(table)

    # Broadcast update
    await ws_manager.broadcast(
        {
            "type": "table_assigned",
            "data": {"staff": staff.to_dict(), "table": table.to_dict()},
        }
    )

    return {"message": f"Table {table.number} assigned to {staff.name}"}


@router.post("/{staff_id}/unassign/{table_id}")
async def unassign_table(
    staff_id: int,
    table_id: int,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Unassign a table from a staff member."""
    from ...models import Table

    # Get staff member
    result = await db.execute(select(Staff).where(Staff.id == staff_id))
    staff = result.scalar_one_or_none()
    if not staff:
        raise HTTPException(status_code=404, detail="Staff member not found")

    # Get table
    result = await db.execute(select(Table).where(Table.id == table_id))
    table = result.scalar_one_or_none()
    if not table:
        raise HTTPException(status_code=404, detail="Table not found")

    # Unassign
    if table_id in staff.assigned_tables:
        staff.assigned_tables = [t for t in staff.assigned_tables if t != table_id]

    if table.assigned_server_id == staff_id:
        table.assigned_server_id = None

    staff.last_activity = datetime.utcnow()

    await db.commit()
    await db.refresh(staff)
    await db.refresh(table)

    # Broadcast update
    await ws_manager.broadcast(
        {
            "type": "table_unassigned",
            "data": {"staff": staff.to_dict(), "table": table.to_dict()},
        }
    )

    return {"message": f"Table {table.number} unassigned from {staff.name}"}


@router.delete("/{staff_id}")
async def delete_staff(
    staff_id: int,
    db: AsyncSession = Depends(get_db),
    ws_manager: ConnectionManager = Depends(get_connection_manager),
):
    """Delete a staff member."""
    result = await db.execute(select(Staff).where(Staff.id == staff_id))
    staff = result.scalar_one_or_none()

    if not staff:
        raise HTTPException(status_code=404, detail="Staff member not found")

    staff_name = staff.name
    await db.delete(staff)
    await db.commit()

    # Broadcast update
    await ws_manager.broadcast(
        {"type": "staff_deleted", "data": {"id": staff_id, "name": staff_name}}
    )

    return {"message": f"Staff member {staff_name} deleted"}
