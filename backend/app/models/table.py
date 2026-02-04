"""Table model for restaurant tables."""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import String, Integer, Float, DateTime, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class TableState(str, Enum):
    """Table lifecycle states."""

    EMPTY = "empty"
    SEATED = "seated"
    ORDERING = "ordering"
    WAITING = "waiting"
    SERVED = "served"
    PAYING = "paying"


class Table(Base):
    """Restaurant table model."""

    __tablename__ = "tables"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    number: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, default=4)

    # Position on floor plan (normalized 0-1)
    x_position: Mapped[float] = mapped_column(Float, default=0.0)
    y_position: Mapped[float] = mapped_column(Float, default=0.0)

    # Current state
    state: Mapped[TableState] = mapped_column(
        SQLEnum(TableState), default=TableState.EMPTY
    )
    current_guests: Mapped[int] = mapped_column(Integer, default=0)

    # Timing
    state_changed_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    seated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Assignment
    assigned_server_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Priority tracking
    priority_score: Mapped[float] = mapped_column(Float, default=0.0)
    hand_raise_detected: Mapped[bool] = mapped_column(default=False)
    last_interaction: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )

    # Metadata
    zone: Mapped[str] = mapped_column(String(50), default="main")
    notes: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "number": self.number,
            "capacity": self.capacity,
            "x_position": self.x_position,
            "y_position": self.y_position,
            "state": self.state.value,
            "current_guests": self.current_guests,
            "state_changed_at": self.state_changed_at.isoformat()
            if self.state_changed_at
            else None,
            "seated_at": self.seated_at.isoformat() if self.seated_at else None,
            "assigned_server_id": self.assigned_server_id,
            "priority_score": self.priority_score,
            "hand_raise_detected": self.hand_raise_detected,
            "last_interaction": self.last_interaction.isoformat()
            if self.last_interaction
            else None,
            "zone": self.zone,
            "notes": self.notes,
        }

    def get_wait_time_seconds(self) -> float:
        """Calculate time since last state change."""
        if self.state_changed_at:
            return (datetime.utcnow() - self.state_changed_at).total_seconds()
        return 0.0
