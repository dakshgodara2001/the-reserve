"""Staff model for restaurant employees."""

from datetime import datetime
from enum import Enum
from typing import Optional, List

from sqlalchemy import String, Integer, Float, DateTime, Enum as SQLEnum, JSON
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class StaffRole(str, Enum):
    """Staff role types."""

    SERVER = "server"
    BUSSER = "busser"
    HOST = "host"
    MANAGER = "manager"
    BARTENDER = "bartender"


class StaffStatus(str, Enum):
    """Staff availability status."""

    AVAILABLE = "available"
    BUSY = "busy"
    ON_BREAK = "on_break"
    OFF_DUTY = "off_duty"


class Staff(Base):
    """Restaurant staff member model."""

    __tablename__ = "staff"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[StaffRole] = mapped_column(SQLEnum(StaffRole), nullable=False)

    # Current status
    status: Mapped[StaffStatus] = mapped_column(
        SQLEnum(StaffStatus), default=StaffStatus.AVAILABLE
    )

    # Location (normalized 0-1 for floor plan)
    x_position: Mapped[float] = mapped_column(Float, default=0.5)
    y_position: Mapped[float] = mapped_column(Float, default=0.5)
    current_zone: Mapped[str] = mapped_column(String(50), default="main")

    # Workload tracking
    assigned_tables: Mapped[List[int]] = mapped_column(JSON, default=list)
    current_task_count: Mapped[int] = mapped_column(Integer, default=0)
    max_tables: Mapped[int] = mapped_column(Integer, default=4)

    # Performance metrics
    tasks_completed_today: Mapped[int] = mapped_column(Integer, default=0)
    avg_response_time: Mapped[float] = mapped_column(Float, default=0.0)  # seconds

    # Skills (for smart assignment)
    skills: Mapped[List[str]] = mapped_column(JSON, default=list)

    # Shift info
    shift_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    shift_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Last activity
    last_activity: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "x_position": self.x_position,
            "y_position": self.y_position,
            "current_zone": self.current_zone,
            "assigned_tables": self.assigned_tables,
            "current_task_count": self.current_task_count,
            "max_tables": self.max_tables,
            "tasks_completed_today": self.tasks_completed_today,
            "avg_response_time": self.avg_response_time,
            "skills": self.skills,
            "shift_start": self.shift_start.isoformat() if self.shift_start else None,
            "shift_end": self.shift_end.isoformat() if self.shift_end else None,
            "last_activity": self.last_activity.isoformat()
            if self.last_activity
            else None,
        }

    def get_workload_score(self) -> float:
        """Calculate current workload as a score 0-1."""
        if self.max_tables == 0:
            return 1.0
        return min(1.0, len(self.assigned_tables) / self.max_tables)

    def is_available_for_assignment(self) -> bool:
        """Check if staff member can take new assignments."""
        return (
            self.status == StaffStatus.AVAILABLE
            and len(self.assigned_tables) < self.max_tables
        )
