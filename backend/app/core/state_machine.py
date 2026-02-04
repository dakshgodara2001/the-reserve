"""Table lifecycle state machine."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Any
from enum import Enum


class TableState(str, Enum):
    """Table lifecycle states."""

    EMPTY = "empty"
    SEATED = "seated"
    ORDERING = "ordering"
    WAITING = "waiting"
    SERVED = "served"
    PAYING = "paying"


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: TableState
    to_state: TableState
    timestamp: datetime
    trigger: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class TableStateMachine:
    """
    Manages table lifecycle state transitions.

    State flow:
    EMPTY → SEATED (person detected)
    SEATED → ORDERING (menu interaction / hand raise)
    ORDERING → WAITING (order placed)
    WAITING → SERVED (food delivered)
    SERVED → PAYING (check requested)
    PAYING → EMPTY (payment complete)

    Also supports:
    * → EMPTY (reset)
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        TableState.EMPTY: [TableState.SEATED],
        TableState.SEATED: [TableState.ORDERING, TableState.EMPTY],
        TableState.ORDERING: [TableState.WAITING, TableState.SEATED, TableState.EMPTY],
        TableState.WAITING: [TableState.SERVED, TableState.ORDERING, TableState.EMPTY],
        TableState.SERVED: [TableState.PAYING, TableState.ORDERING, TableState.EMPTY],
        TableState.PAYING: [TableState.EMPTY],
    }

    # Expected duration in each state (seconds) for alerting
    EXPECTED_DURATIONS = {
        TableState.EMPTY: None,
        TableState.SEATED: 120,  # 2 minutes to take drink order
        TableState.ORDERING: 300,  # 5 minutes to place order
        TableState.WAITING: 900,  # 15 minutes for food
        TableState.SERVED: 1800,  # 30 minutes to eat
        TableState.PAYING: 300,  # 5 minutes to pay
    }

    def __init__(self, table_id: int, initial_state: TableState = TableState.EMPTY):
        self.table_id = table_id
        self.current_state = initial_state
        self.state_entered_at = datetime.utcnow()
        self.history: List[StateTransition] = []

        # Callbacks
        self._on_transition_callbacks: List[Callable] = []
        self._on_timeout_callbacks: List[Callable] = []

        # Metrics
        self.total_transitions = 0
        self.guest_count = 0

    def can_transition(self, to_state: TableState) -> bool:
        """Check if transition to state is valid."""
        # Always allow reset to empty
        if to_state == TableState.EMPTY:
            return True

        valid_next = self.VALID_TRANSITIONS.get(self.current_state, [])
        return to_state in valid_next

    def transition(
        self,
        to_state: TableState,
        trigger: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Attempt to transition to a new state.

        Args:
            to_state: Target state
            trigger: What triggered the transition
            metadata: Additional context

        Returns:
            True if transition succeeded
        """
        if not self.can_transition(to_state):
            return False

        # Record transition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=to_state,
            timestamp=datetime.utcnow(),
            trigger=trigger,
            metadata=metadata or {},
        )
        self.history.append(transition)

        # Update state
        old_state = self.current_state
        self.current_state = to_state
        self.state_entered_at = datetime.utcnow()
        self.total_transitions += 1

        # Emit callbacks
        for callback in self._on_transition_callbacks:
            try:
                callback(self.table_id, old_state, to_state, trigger, metadata)
            except Exception as e:
                print(f"Transition callback error: {e}")

        return True

    def get_time_in_state(self) -> float:
        """Get seconds since entering current state."""
        return (datetime.utcnow() - self.state_entered_at).total_seconds()

    def is_overdue(self) -> bool:
        """Check if table has been in current state too long."""
        expected = self.EXPECTED_DURATIONS.get(self.current_state)
        if expected is None:
            return False
        return self.get_time_in_state() > expected

    def get_overdue_seconds(self) -> float:
        """Get seconds overdue, or 0 if not overdue."""
        expected = self.EXPECTED_DURATIONS.get(self.current_state)
        if expected is None:
            return 0.0
        time_in_state = self.get_time_in_state()
        return max(0, time_in_state - expected)

    def seat_customers(self, guest_count: int) -> bool:
        """Transition to seated with guest count."""
        if self.transition(TableState.SEATED, "customer_seated", {"guest_count": guest_count}):
            self.guest_count = guest_count
            return True
        return False

    def request_service(self) -> bool:
        """Customer requesting service (hand raise, etc.)."""
        if self.current_state == TableState.SEATED:
            return self.transition(TableState.ORDERING, "service_requested")
        return False

    def place_order(self) -> bool:
        """Order has been placed."""
        if self.current_state in [TableState.SEATED, TableState.ORDERING]:
            return self.transition(TableState.WAITING, "order_placed")
        return False

    def deliver_food(self) -> bool:
        """Food has been delivered."""
        if self.current_state == TableState.WAITING:
            return self.transition(TableState.SERVED, "food_delivered")
        return False

    def request_check(self) -> bool:
        """Customer requesting check."""
        if self.current_state == TableState.SERVED:
            return self.transition(TableState.PAYING, "check_requested")
        return False

    def complete_payment(self) -> bool:
        """Payment complete, table reset."""
        return self.transition(TableState.EMPTY, "payment_complete")

    def reset(self) -> bool:
        """Force reset to empty."""
        if self.transition(TableState.EMPTY, "reset"):
            self.guest_count = 0
            return True
        return False

    def on_transition(self, callback: Callable):
        """Register callback for state transitions."""
        self._on_transition_callbacks.append(callback)

    def on_timeout(self, callback: Callable):
        """Register callback for state timeouts."""
        self._on_timeout_callbacks.append(callback)

    def check_timeout(self) -> bool:
        """Check and emit timeout if overdue."""
        if self.is_overdue():
            for callback in self._on_timeout_callbacks:
                try:
                    callback(
                        self.table_id,
                        self.current_state,
                        self.get_time_in_state(),
                        self.get_overdue_seconds(),
                    )
                except Exception as e:
                    print(f"Timeout callback error: {e}")
            return True
        return False

    def get_suggested_actions(self) -> List[str]:
        """Get suggested staff actions based on current state."""
        actions = {
            TableState.EMPTY: [],
            TableState.SEATED: ["greet_customers", "offer_drinks", "present_menu"],
            TableState.ORDERING: ["take_order", "answer_questions", "make_recommendations"],
            TableState.WAITING: ["check_order_status", "refill_drinks", "update_customer"],
            TableState.SERVED: ["check_satisfaction", "offer_dessert", "clear_plates"],
            TableState.PAYING: ["process_payment", "thank_customer"],
        }
        return actions.get(self.current_state, [])

    def get_status(self) -> Dict[str, Any]:
        """Get current status summary."""
        return {
            "table_id": self.table_id,
            "state": self.current_state.value,
            "time_in_state": self.get_time_in_state(),
            "is_overdue": self.is_overdue(),
            "overdue_seconds": self.get_overdue_seconds(),
            "guest_count": self.guest_count,
            "total_transitions": self.total_transitions,
            "suggested_actions": self.get_suggested_actions(),
            "expected_duration": self.EXPECTED_DURATIONS.get(self.current_state),
        }


class RestaurantStateMachineManager:
    """Manages state machines for all tables."""

    def __init__(self):
        self.tables: Dict[int, TableStateMachine] = {}
        self._global_transition_callbacks: List[Callable] = []

    def register_table(
        self, table_id: int, initial_state: TableState = TableState.EMPTY
    ) -> TableStateMachine:
        """Register a new table."""
        sm = TableStateMachine(table_id, initial_state)

        # Add global callback forwarding
        sm.on_transition(self._forward_transition)

        self.tables[table_id] = sm
        return sm

    def get_table(self, table_id: int) -> Optional[TableStateMachine]:
        """Get table state machine."""
        return self.tables.get(table_id)

    def get_tables_in_state(self, state: TableState) -> List[TableStateMachine]:
        """Get all tables in a specific state."""
        return [t for t in self.tables.values() if t.current_state == state]

    def get_overdue_tables(self) -> List[TableStateMachine]:
        """Get all tables that are overdue."""
        return [t for t in self.tables.values() if t.is_overdue()]

    def get_active_tables(self) -> List[TableStateMachine]:
        """Get all tables with customers (not empty)."""
        return [t for t in self.tables.values() if t.current_state != TableState.EMPTY]

    def check_all_timeouts(self):
        """Check timeouts for all tables."""
        for table in self.tables.values():
            table.check_timeout()

    def on_any_transition(self, callback: Callable):
        """Register callback for transitions on any table."""
        self._global_transition_callbacks.append(callback)

    def _forward_transition(
        self,
        table_id: int,
        from_state: TableState,
        to_state: TableState,
        trigger: str,
        metadata: Optional[Dict],
    ):
        """Forward transition to global callbacks."""
        for callback in self._global_transition_callbacks:
            try:
                callback(table_id, from_state, to_state, trigger, metadata)
            except Exception as e:
                print(f"Global transition callback error: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tables."""
        state_counts = {state.value: 0 for state in TableState}
        for table in self.tables.values():
            state_counts[table.current_state.value] += 1

        return {
            "total_tables": len(self.tables),
            "active_tables": len(self.get_active_tables()),
            "overdue_tables": len(self.get_overdue_tables()),
            "state_counts": state_counts,
            "total_guests": sum(t.guest_count for t in self.tables.values()),
        }
