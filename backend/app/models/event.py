"""Event model for tracking restaurant events."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import String, Integer, Float, DateTime, Enum as SQLEnum, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class EventType(str, Enum):
    """Types of events detected by the system."""

    # Customer actions
    CUSTOMER_SEATED = "customer_seated"
    CUSTOMER_LEFT = "customer_left"
    HAND_RAISE = "hand_raise"
    MENU_REQUEST = "menu_request"
    ORDER_PLACED = "order_placed"
    FOOD_DELIVERED = "food_delivered"
    CHECK_REQUEST = "check_request"
    PAYMENT_COMPLETE = "payment_complete"

    # Staff actions
    STAFF_ASSIGNED = "staff_assigned"
    STAFF_ARRIVED = "staff_arrived"
    TASK_COMPLETED = "task_completed"

    # Audio events
    VERBAL_REQUEST = "verbal_request"
    COMPLAINT = "complaint"

    # System events
    PRIORITY_ALERT = "priority_alert"
    ZONE_CHANGE = "zone_change"

    # Detection events
    GESTURE_DETECTED = "gesture_detected"
    PERSON_DETECTED = "person_detected"


class EventPriority(str, Enum):
    """Event priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Event(Base):
    """Event log model for tracking all system events."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Event classification
    event_type: Mapped[EventType] = mapped_column(SQLEnum(EventType), nullable=False)
    priority: Mapped[EventPriority] = mapped_column(
        SQLEnum(EventPriority), default=EventPriority.MEDIUM
    )

    # Timing
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Context
    table_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    staff_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Location
    zone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    x_position: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y_position: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Detection details
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[str] = mapped_column(
        String(50), default="system"
    )  # vision, audio, manual

    # Content
    description: Mapped[str] = mapped_column(Text, default="")
    extra_data: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Resolution
    resolved: Mapped[bool] = mapped_column(default=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    resolved_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "table_id": self.table_id,
            "staff_id": self.staff_id,
            "zone": self.zone,
            "x_position": self.x_position,
            "y_position": self.y_position,
            "confidence": self.confidence,
            "source": self.source,
            "description": self.description,
            "metadata": self.extra_data,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
        }

    @classmethod
    def create_detection_event(
        cls,
        event_type: EventType,
        table_id: Optional[int] = None,
        confidence: float = 1.0,
        source: str = "vision",
        description: str = "",
        **kwargs,
    ) -> "Event":
        """Factory method for creating detection events."""
        priority = EventPriority.MEDIUM

        # Set priority based on event type
        if event_type in [EventType.COMPLAINT, EventType.PRIORITY_ALERT]:
            priority = EventPriority.URGENT
        elif event_type in [EventType.HAND_RAISE, EventType.VERBAL_REQUEST]:
            priority = EventPriority.HIGH
        elif event_type in [EventType.CHECK_REQUEST, EventType.ORDER_PLACED]:
            priority = EventPriority.MEDIUM
        else:
            priority = EventPriority.LOW

        return cls(
            event_type=event_type,
            priority=priority,
            table_id=table_id,
            confidence=confidence,
            source=source,
            description=description,
            extra_data=kwargs.get("metadata", {}),
        )
