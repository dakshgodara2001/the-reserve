"""Priority queue for managing urgent customer needs."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import heapq


class RequestType(str, Enum):
    """Types of customer requests."""

    GREETING = "greeting"
    DRINK_ORDER = "drink_order"
    FOOD_ORDER = "food_order"
    REFILL = "refill"
    QUESTION = "question"
    CHECK_ON = "check_on"
    CHECK_REQUEST = "check_request"
    COMPLAINT = "complaint"
    PAYMENT = "payment"
    CLEANUP = "cleanup"


@dataclass(order=True)
class PriorityItem:
    """Item in the priority queue."""

    # Priority score (negative for max-heap behavior)
    priority: float = field(compare=True)

    # Unique ID
    request_id: str = field(compare=False)

    # Request details
    request_type: RequestType = field(compare=False)
    table_id: int = field(compare=False)
    created_at: datetime = field(compare=False, default_factory=datetime.utcnow)

    # Context
    description: str = field(compare=False, default="")
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)

    # Assignment
    assigned_staff_id: Optional[int] = field(compare=False, default=None)
    assigned_at: Optional[datetime] = field(compare=False, default=None)

    # Resolution
    resolved: bool = field(compare=False, default=False)
    resolved_at: Optional[datetime] = field(compare=False, default=None)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "priority": abs(self.priority),  # Return positive score
            "request_type": self.request_type.value,
            "table_id": self.table_id,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "assigned_staff_id": self.assigned_staff_id,
            "resolved": self.resolved,
            "age_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
        }


class PriorityQueue:
    """
    Priority queue for customer service requests.

    Implements urgency scoring based on:
    - Base wait time score
    - Gesture detection (hand raise)
    - Frustration indicators
    - Verbal requests
    - Payment readiness
    """

    # Base priority by request type
    BASE_PRIORITIES = {
        RequestType.COMPLAINT: 80,
        RequestType.CHECK_REQUEST: 60,
        RequestType.PAYMENT: 55,
        RequestType.FOOD_ORDER: 50,
        RequestType.DRINK_ORDER: 45,
        RequestType.REFILL: 40,
        RequestType.QUESTION: 35,
        RequestType.CHECK_ON: 30,
        RequestType.GREETING: 25,
        RequestType.CLEANUP: 20,
    }

    # Score modifiers
    GESTURE_BONUS = 30
    FRUSTRATION_BONUS = 50
    VERBAL_REQUEST_BONUS = 40
    PAYMENT_READY_BONUS = 20
    WAIT_TIME_FACTOR = 0.5  # Points per minute waiting

    def __init__(self):
        self._heap: List[PriorityItem] = []
        self._items: Dict[str, PriorityItem] = {}  # For O(1) lookup
        self._counter = 0

        # Callbacks
        self._on_add_callbacks: List[Callable] = []
        self._on_resolve_callbacks: List[Callable] = []

    def _generate_id(self) -> str:
        """Generate unique request ID."""
        self._counter += 1
        return f"req_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self._counter}"

    def add(
        self,
        request_type: RequestType,
        table_id: int,
        description: str = "",
        gesture_detected: bool = False,
        frustration_detected: bool = False,
        verbal_request: bool = False,
        payment_ready: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PriorityItem:
        """
        Add a new request to the queue.

        Args:
            request_type: Type of request
            table_id: Table making request
            description: Description of request
            gesture_detected: Customer raised hand
            frustration_detected: Customer shows frustration
            verbal_request: Customer made verbal request
            payment_ready: Customer ready to pay
            metadata: Additional context

        Returns:
            The created priority item
        """
        # Calculate priority score
        base_score = self.BASE_PRIORITIES.get(request_type, 30)

        urgency = base_score
        if gesture_detected:
            urgency += self.GESTURE_BONUS
        if frustration_detected:
            urgency += self.FRUSTRATION_BONUS
        if verbal_request:
            urgency += self.VERBAL_REQUEST_BONUS
        if payment_ready:
            urgency += self.PAYMENT_READY_BONUS

        # Create item (negative priority for max-heap)
        item = PriorityItem(
            priority=-urgency,  # Negative for max-heap
            request_id=self._generate_id(),
            request_type=request_type,
            table_id=table_id,
            description=description,
            metadata=metadata or {},
        )

        # Store metadata about urgency factors
        item.metadata.update(
            {
                "gesture_detected": gesture_detected,
                "frustration_detected": frustration_detected,
                "verbal_request": verbal_request,
                "payment_ready": payment_ready,
                "base_score": base_score,
            }
        )

        # Add to heap and lookup
        heapq.heappush(self._heap, item)
        self._items[item.request_id] = item

        # Emit callbacks
        for callback in self._on_add_callbacks:
            try:
                callback(item)
            except Exception as e:
                print(f"Add callback error: {e}")

        return item

    def pop(self) -> Optional[PriorityItem]:
        """Remove and return highest priority item."""
        while self._heap:
            item = heapq.heappop(self._heap)
            # Skip resolved items
            if not item.resolved and item.request_id in self._items:
                return item
        return None

    def peek(self) -> Optional[PriorityItem]:
        """Return highest priority item without removing."""
        # Update priorities based on wait time
        self._update_wait_priorities()

        while self._heap:
            item = self._heap[0]
            if not item.resolved and item.request_id in self._items:
                return item
            heapq.heappop(self._heap)
        return None

    def _update_wait_priorities(self):
        """Update priorities based on wait time."""
        updated_items = []

        for item in self._items.values():
            if item.resolved:
                continue

            # Calculate wait time bonus
            wait_minutes = (datetime.utcnow() - item.created_at).total_seconds() / 60
            wait_bonus = wait_minutes * self.WAIT_TIME_FACTOR

            # Recalculate total priority
            base_score = item.metadata.get("base_score", 30)
            urgency = base_score + wait_bonus

            if item.metadata.get("gesture_detected"):
                urgency += self.GESTURE_BONUS
            if item.metadata.get("frustration_detected"):
                urgency += self.FRUSTRATION_BONUS
            if item.metadata.get("verbal_request"):
                urgency += self.VERBAL_REQUEST_BONUS
            if item.metadata.get("payment_ready"):
                urgency += self.PAYMENT_READY_BONUS

            # Update if changed significantly
            if abs(item.priority + urgency) > 1:  # More than 1 point change
                item.priority = -urgency
                updated_items.append(item)

        # Re-heapify if items were updated
        if updated_items:
            heapq.heapify(self._heap)

    def get(self, request_id: str) -> Optional[PriorityItem]:
        """Get item by ID."""
        return self._items.get(request_id)

    def resolve(self, request_id: str, staff_id: Optional[int] = None) -> bool:
        """Mark a request as resolved."""
        item = self._items.get(request_id)
        if not item:
            return False

        item.resolved = True
        item.resolved_at = datetime.utcnow()

        if staff_id and not item.assigned_staff_id:
            item.assigned_staff_id = staff_id
            item.assigned_at = datetime.utcnow()

        # Remove from items dict (will be skipped in heap)
        del self._items[request_id]

        # Emit callbacks
        for callback in self._on_resolve_callbacks:
            try:
                callback(item)
            except Exception as e:
                print(f"Resolve callback error: {e}")

        return True

    def assign(self, request_id: str, staff_id: int) -> bool:
        """Assign a request to staff."""
        item = self._items.get(request_id)
        if not item:
            return False

        item.assigned_staff_id = staff_id
        item.assigned_at = datetime.utcnow()
        return True

    def get_pending(self, limit: int = 50) -> List[PriorityItem]:
        """Get pending requests sorted by priority."""
        self._update_wait_priorities()

        pending = [item for item in self._items.values() if not item.resolved]
        pending.sort(key=lambda x: x.priority)  # Sort by priority (negative)
        return pending[:limit]

    def get_for_table(self, table_id: int) -> List[PriorityItem]:
        """Get all pending requests for a table."""
        return [
            item
            for item in self._items.values()
            if item.table_id == table_id and not item.resolved
        ]

    def get_unassigned(self) -> List[PriorityItem]:
        """Get unassigned requests."""
        return [
            item
            for item in self._items.values()
            if not item.resolved and item.assigned_staff_id is None
        ]

    def get_assigned_to(self, staff_id: int) -> List[PriorityItem]:
        """Get requests assigned to specific staff."""
        return [
            item
            for item in self._items.values()
            if item.assigned_staff_id == staff_id and not item.resolved
        ]

    def cancel_for_table(self, table_id: int):
        """Cancel all pending requests for a table."""
        to_remove = [
            rid for rid, item in self._items.items() if item.table_id == table_id
        ]
        for rid in to_remove:
            del self._items[rid]

    def on_add(self, callback: Callable):
        """Register callback for new requests."""
        self._on_add_callbacks.append(callback)

    def on_resolve(self, callback: Callable):
        """Register callback for resolved requests."""
        self._on_resolve_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending = [item for item in self._items.values() if not item.resolved]

        if not pending:
            return {
                "total_pending": 0,
                "total_assigned": 0,
                "avg_wait_seconds": 0,
                "max_wait_seconds": 0,
                "by_type": {},
            }

        wait_times = [(datetime.utcnow() - item.created_at).total_seconds() for item in pending]
        type_counts = {}
        for item in pending:
            type_counts[item.request_type.value] = type_counts.get(item.request_type.value, 0) + 1

        return {
            "total_pending": len(pending),
            "total_assigned": sum(1 for item in pending if item.assigned_staff_id),
            "avg_wait_seconds": sum(wait_times) / len(wait_times),
            "max_wait_seconds": max(wait_times),
            "by_type": type_counts,
        }

    def __len__(self) -> int:
        return len(self._items)
