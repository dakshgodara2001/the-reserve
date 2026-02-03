"""Notification service for dispatching alerts to staff and dashboard."""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class NotificationChannel(str, Enum):
    """Notification delivery channels."""

    DASHBOARD = "dashboard"
    EARPIECE = "earpiece"
    MOBILE = "mobile"
    DISPLAY = "display"


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notification data structure."""

    id: str
    title: str
    message: str
    priority: NotificationPriority
    channels: List[NotificationChannel]
    target_staff_ids: List[int] = field(default_factory=list)
    table_id: Optional[int] = None
    event_id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    delivered: bool = False
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "priority": self.priority.value,
            "channels": [c.value for c in self.channels],
            "target_staff_ids": self.target_staff_ids,
            "table_id": self.table_id,
            "event_id": self.event_id,
            "created_at": self.created_at.isoformat(),
            "delivered": self.delivered,
            "acknowledged": self.acknowledged,
            "metadata": self.metadata,
        }


class NotificationService:
    """Service for managing and dispatching notifications."""

    def __init__(self):
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.notification_history: List[Notification] = []
        self.max_history = 1000
        self._running = False
        self._counter = 0

    async def start(self):
        """Start the notification processing loop."""
        self._running = True
        print("Notification service started")

        while self._running:
            try:
                # Wait for notifications with timeout
                try:
                    notification = await asyncio.wait_for(
                        self.notification_queue.get(), timeout=1.0
                    )
                    await self._process_notification(notification)
                except asyncio.TimeoutError:
                    continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Notification processing error: {e}")

        print("Notification service stopped")

    async def stop(self):
        """Stop the notification service."""
        self._running = False

    def _generate_id(self) -> str:
        """Generate unique notification ID."""
        self._counter += 1
        return f"notif_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self._counter}"

    async def send(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        channels: Optional[List[NotificationChannel]] = None,
        target_staff_ids: Optional[List[int]] = None,
        table_id: Optional[int] = None,
        event_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """Queue a notification for delivery."""
        if channels is None:
            channels = [NotificationChannel.DASHBOARD]

        notification = Notification(
            id=self._generate_id(),
            title=title,
            message=message,
            priority=priority,
            channels=channels,
            target_staff_ids=target_staff_ids or [],
            table_id=table_id,
            event_id=event_id,
            metadata=metadata or {},
        )

        await self.notification_queue.put(notification)
        return notification

    async def send_staff_alert(
        self,
        staff_id: int,
        title: str,
        message: str,
        table_id: Optional[int] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> Notification:
        """Send alert to specific staff member."""
        return await self.send(
            title=title,
            message=message,
            priority=priority,
            channels=[NotificationChannel.EARPIECE, NotificationChannel.DASHBOARD],
            target_staff_ids=[staff_id],
            table_id=table_id,
        )

    async def send_table_alert(
        self,
        table_id: int,
        title: str,
        message: str,
        assigned_staff_id: Optional[int] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> Notification:
        """Send alert about a table."""
        target_staff = [assigned_staff_id] if assigned_staff_id else []
        return await self.send(
            title=title,
            message=message,
            priority=priority,
            channels=[NotificationChannel.DASHBOARD, NotificationChannel.EARPIECE],
            target_staff_ids=target_staff,
            table_id=table_id,
        )

    async def send_urgent_alert(
        self,
        title: str,
        message: str,
        table_id: Optional[int] = None,
    ) -> Notification:
        """Send urgent alert to all channels."""
        return await self.send(
            title=title,
            message=message,
            priority=NotificationPriority.URGENT,
            channels=[
                NotificationChannel.DASHBOARD,
                NotificationChannel.EARPIECE,
                NotificationChannel.DISPLAY,
            ],
            table_id=table_id,
        )

    async def _process_notification(self, notification: Notification):
        """Process and dispatch a notification."""
        try:
            # Dispatch to each channel
            for channel in notification.channels:
                await self._dispatch_to_channel(notification, channel)

            notification.delivered = True

            # Store in history
            self.notification_history.append(notification)

            # Trim history if needed
            if len(self.notification_history) > self.max_history:
                self.notification_history = self.notification_history[-self.max_history :]

            # Broadcast to WebSocket clients
            await self._broadcast_notification(notification)

        except Exception as e:
            print(f"Failed to process notification {notification.id}: {e}")

    async def _dispatch_to_channel(
        self, notification: Notification, channel: NotificationChannel
    ):
        """Dispatch notification to specific channel."""
        if channel == NotificationChannel.DASHBOARD:
            # Dashboard notifications handled via WebSocket broadcast
            pass

        elif channel == NotificationChannel.EARPIECE:
            # TTS placeholder for staff earpiece
            await self._send_to_earpiece(notification)

        elif channel == NotificationChannel.MOBILE:
            # Mobile push notification placeholder
            await self._send_to_mobile(notification)

        elif channel == NotificationChannel.DISPLAY:
            # Kitchen/host display placeholder
            await self._send_to_display(notification)

    async def _send_to_earpiece(self, notification: Notification):
        """Send notification to staff earpiece (TTS placeholder)."""
        # In production, this would integrate with TTS service
        print(
            f"[EARPIECE] Staff {notification.target_staff_ids}: "
            f"{notification.title} - {notification.message}"
        )

    async def _send_to_mobile(self, notification: Notification):
        """Send push notification to mobile devices."""
        # In production, integrate with Firebase/APNS
        print(f"[MOBILE] {notification.title}: {notification.message}")

    async def _send_to_display(self, notification: Notification):
        """Send notification to display screens."""
        # In production, integrate with display system
        print(f"[DISPLAY] {notification.title}: {notification.message}")

    async def _broadcast_notification(self, notification: Notification):
        """Broadcast notification to WebSocket clients."""
        from ..api.deps import connection_manager

        await connection_manager.broadcast(
            {"type": "notification", "data": notification.to_dict()}
        )

    def get_recent_notifications(
        self, limit: int = 50, staff_id: Optional[int] = None
    ) -> List[Notification]:
        """Get recent notifications, optionally filtered by staff."""
        notifications = self.notification_history[-limit:]

        if staff_id:
            notifications = [
                n
                for n in notifications
                if not n.target_staff_ids or staff_id in n.target_staff_ids
            ]

        return notifications

    def get_unacknowledged(
        self, staff_id: Optional[int] = None
    ) -> List[Notification]:
        """Get unacknowledged notifications."""
        notifications = [n for n in self.notification_history if not n.acknowledged]

        if staff_id:
            notifications = [
                n
                for n in notifications
                if not n.target_staff_ids or staff_id in n.target_staff_ids
            ]

        return notifications

    async def acknowledge(self, notification_id: str, staff_id: int):
        """Mark notification as acknowledged."""
        for notification in self.notification_history:
            if notification.id == notification_id:
                notification.acknowledged = True
                notification.metadata["acknowledged_by"] = staff_id
                notification.metadata["acknowledged_at"] = datetime.utcnow().isoformat()
                return True
        return False


# Global notification service instance
notification_service = NotificationService()
