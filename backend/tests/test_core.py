"""Tests for core AI engine components."""

import pytest
from datetime import datetime

from app.core.state_machine import TableStateMachine, TableState
from app.core.priority_queue import PriorityQueue, RequestType
from app.core.assignment import StaffAssigner, StaffInfo, TaskInfo


class TestTableStateMachine:
    """Tests for table state machine."""

    def test_initial_state(self):
        """Test initial state is EMPTY."""
        sm = TableStateMachine(table_id=1)
        assert sm.current_state == TableState.EMPTY

    def test_valid_transition(self):
        """Test valid state transition."""
        sm = TableStateMachine(table_id=1)
        result = sm.transition(TableState.SEATED, "test")
        assert result is True
        assert sm.current_state == TableState.SEATED

    def test_invalid_transition(self):
        """Test invalid state transition is rejected."""
        sm = TableStateMachine(table_id=1)
        # Cannot go from EMPTY to ORDERING
        result = sm.transition(TableState.ORDERING, "test")
        assert result is False
        assert sm.current_state == TableState.EMPTY

    def test_reset_from_any_state(self):
        """Test reset to EMPTY works from any state."""
        sm = TableStateMachine(table_id=1)
        sm.transition(TableState.SEATED, "test")
        sm.transition(TableState.ORDERING, "test")

        result = sm.reset()
        assert result is True
        assert sm.current_state == TableState.EMPTY

    def test_seat_customers(self):
        """Test seating customers updates guest count."""
        sm = TableStateMachine(table_id=1)
        result = sm.seat_customers(4)

        assert result is True
        assert sm.current_state == TableState.SEATED
        assert sm.guest_count == 4

    def test_full_lifecycle(self):
        """Test complete table lifecycle."""
        sm = TableStateMachine(table_id=1)

        # Seat customers
        assert sm.seat_customers(2)
        assert sm.current_state == TableState.SEATED

        # Request service
        assert sm.request_service()
        assert sm.current_state == TableState.ORDERING

        # Place order
        assert sm.place_order()
        assert sm.current_state == TableState.WAITING

        # Deliver food
        assert sm.deliver_food()
        assert sm.current_state == TableState.SERVED

        # Request check
        assert sm.request_check()
        assert sm.current_state == TableState.PAYING

        # Complete payment
        assert sm.complete_payment()
        assert sm.current_state == TableState.EMPTY

    def test_time_in_state(self):
        """Test time tracking in state."""
        sm = TableStateMachine(table_id=1)
        time_in_state = sm.get_time_in_state()
        assert time_in_state >= 0

    def test_transition_history(self):
        """Test transition history is recorded."""
        sm = TableStateMachine(table_id=1)
        sm.transition(TableState.SEATED, "test_trigger")

        assert len(sm.history) == 1
        assert sm.history[0].from_state == TableState.EMPTY
        assert sm.history[0].to_state == TableState.SEATED
        assert sm.history[0].trigger == "test_trigger"


class TestPriorityQueue:
    """Tests for priority queue."""

    def test_add_request(self):
        """Test adding request to queue."""
        queue = PriorityQueue()
        item = queue.add(
            request_type=RequestType.GREETING,
            table_id=1,
            description="Test greeting",
        )

        assert item is not None
        assert item.table_id == 1
        assert item.request_type == RequestType.GREETING

    def test_priority_ordering(self):
        """Test requests are ordered by priority."""
        queue = PriorityQueue()

        # Add low priority
        queue.add(request_type=RequestType.CLEANUP, table_id=1)
        # Add high priority
        queue.add(request_type=RequestType.COMPLAINT, table_id=2)
        # Add medium priority
        queue.add(request_type=RequestType.FOOD_ORDER, table_id=3)

        # Highest priority should come first
        item = queue.peek()
        assert item.request_type == RequestType.COMPLAINT

    def test_gesture_bonus(self):
        """Test gesture detection increases priority."""
        queue = PriorityQueue()

        item1 = queue.add(
            request_type=RequestType.QUESTION,
            table_id=1,
            gesture_detected=False,
        )

        item2 = queue.add(
            request_type=RequestType.QUESTION,
            table_id=2,
            gesture_detected=True,
        )

        # Item with gesture should have higher priority (more negative)
        assert item2.priority < item1.priority

    def test_frustration_bonus(self):
        """Test frustration detection increases priority."""
        queue = PriorityQueue()

        item1 = queue.add(
            request_type=RequestType.QUESTION,
            table_id=1,
            frustration_detected=False,
        )

        item2 = queue.add(
            request_type=RequestType.QUESTION,
            table_id=2,
            frustration_detected=True,
        )

        assert item2.priority < item1.priority

    def test_resolve_request(self):
        """Test resolving a request."""
        queue = PriorityQueue()
        item = queue.add(request_type=RequestType.GREETING, table_id=1)

        result = queue.resolve(item.request_id, staff_id=1)
        assert result is True

        # Should not be in pending anymore
        pending = queue.get_pending()
        assert len(pending) == 0

    def test_assign_request(self):
        """Test assigning staff to request."""
        queue = PriorityQueue()
        item = queue.add(request_type=RequestType.GREETING, table_id=1)

        result = queue.assign(item.request_id, staff_id=5)
        assert result is True

        updated_item = queue.get(item.request_id)
        assert updated_item.assigned_staff_id == 5

    def test_get_for_table(self):
        """Test getting requests for specific table."""
        queue = PriorityQueue()
        queue.add(request_type=RequestType.GREETING, table_id=1)
        queue.add(request_type=RequestType.FOOD_ORDER, table_id=1)
        queue.add(request_type=RequestType.GREETING, table_id=2)

        table1_requests = queue.get_for_table(1)
        assert len(table1_requests) == 2

    def test_cancel_for_table(self):
        """Test canceling all requests for a table."""
        queue = PriorityQueue()
        queue.add(request_type=RequestType.GREETING, table_id=1)
        queue.add(request_type=RequestType.FOOD_ORDER, table_id=1)

        queue.cancel_for_table(1)

        table1_requests = queue.get_for_table(1)
        assert len(table1_requests) == 0


class TestStaffAssigner:
    """Tests for staff assignment algorithm."""

    def setup_method(self):
        """Set up test data."""
        self.staff = [
            StaffInfo(
                staff_id=1,
                name="Alice",
                x_position=0.2,
                y_position=0.3,
                current_workload=1,
                max_workload=4,
                skills=["fine_dining"],
                is_available=True,
            ),
            StaffInfo(
                staff_id=2,
                name="Bob",
                x_position=0.6,
                y_position=0.4,
                current_workload=2,
                max_workload=4,
                skills=["speed"],
                is_available=True,
            ),
            StaffInfo(
                staff_id=3,
                name="Carol",
                x_position=0.5,
                y_position=0.5,
                current_workload=4,
                max_workload=4,
                skills=["cocktails"],
                is_available=True,
            ),
        ]

        self.tasks = [
            TaskInfo(
                task_id="task1",
                table_id=1,
                table_x=0.25,
                table_y=0.35,
                priority=50,
                required_skills=[],
                task_type="greeting",
            ),
        ]

    def test_assign_optimal(self):
        """Test optimal assignment returns result."""
        assigner = StaffAssigner()
        assignments = assigner.assign_optimal(self.staff, self.tasks)

        assert len(assignments) == 1
        # Alice is closest and has lowest workload
        assert assignments[0].staff_id == 1

    def test_assign_respects_capacity(self):
        """Test assignment respects max workload."""
        assigner = StaffAssigner()

        # Carol is at max capacity
        task = TaskInfo(
            task_id="task2",
            table_id=2,
            table_x=0.5,
            table_y=0.5,  # Carol is closest
            priority=50,
            required_skills=[],
            task_type="greeting",
        )

        assignments = assigner.assign_optimal(self.staff, [task])

        # Should not assign to Carol despite being closest
        assert len(assignments) == 1
        assert assignments[0].staff_id != 3

    def test_suggest_single(self):
        """Test single task suggestion."""
        assigner = StaffAssigner()
        assignment = assigner.suggest_single(self.staff[:2], self.tasks[0])

        assert assignment is not None
        assert assignment.staff_id == 1  # Alice is closest

    def test_skill_matching(self):
        """Test skill matching affects assignment."""
        assigner = StaffAssigner()

        task_with_skill = TaskInfo(
            task_id="cocktail_task",
            table_id=3,
            table_x=0.5,
            table_y=0.5,
            priority=50,
            required_skills=["cocktails"],
            task_type="drink_order",
        )

        # Only Bob and Carol can take this (Carol at capacity)
        available_staff = [s for s in self.staff if s.current_workload < s.max_workload]
        recommendations = assigner.get_staff_recommendations(
            available_staff, task_with_skill, limit=3
        )

        assert len(recommendations) >= 1

    def test_no_available_staff(self):
        """Test handling when no staff available."""
        assigner = StaffAssigner()

        # All staff unavailable
        unavailable_staff = [
            StaffInfo(
                staff_id=1,
                name="Alice",
                x_position=0.2,
                y_position=0.3,
                current_workload=0,
                max_workload=4,
                skills=[],
                is_available=False,
            ),
        ]

        assignments = assigner.assign_optimal(unavailable_staff, self.tasks)
        assert len(assignments) == 0


class TestIntegration:
    """Integration tests for core components."""

    def test_full_flow(self):
        """Test full flow from seating to assignment."""
        # Create state machine
        sm = TableStateMachine(table_id=1)
        sm.seat_customers(2)

        # Create priority queue and add request
        queue = PriorityQueue()
        item = queue.add(
            request_type=RequestType.GREETING,
            table_id=1,
            description="New customers seated",
        )

        # Create staff
        staff = [
            StaffInfo(
                staff_id=1,
                name="Alice",
                x_position=0.3,
                y_position=0.3,
                current_workload=1,
                max_workload=4,
                skills=[],
                is_available=True,
            ),
        ]

        # Create task from request
        task = TaskInfo(
            task_id=item.request_id,
            table_id=1,
            table_x=0.2,
            table_y=0.3,
            priority=abs(item.priority),
            required_skills=[],
            task_type="greeting",
        )

        # Assign
        assigner = StaffAssigner()
        assignments = assigner.assign_optimal(staff, [task])

        assert len(assignments) == 1
        assert assignments[0].table_id == 1
        assert assignments[0].staff_id == 1

        # Update queue
        queue.assign(item.request_id, assignments[0].staff_id)
        queue.resolve(item.request_id, assignments[0].staff_id)

        # Verify resolved
        assert len(queue.get_pending()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
