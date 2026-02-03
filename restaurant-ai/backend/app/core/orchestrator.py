"""Main AI orchestrator that coordinates all system components."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable

from .state_machine import TableStateMachine, TableState, RestaurantStateMachineManager
from .priority_queue import PriorityQueue, RequestType, PriorityItem
from .assignment import StaffAssigner, StaffInfo, TaskInfo, Assignment
from ..services.fusion import SensorFusion, SensorSignal, SignalSource, SignalType, FusedEvent
from ..services.notification import notification_service, NotificationPriority
from ..services.analytics import analytics_service


@dataclass
class TableContext:
    """Context information for a table."""

    table_id: int
    table_number: int
    x_position: float
    y_position: float
    state_machine: TableStateMachine
    assigned_staff_id: Optional[int] = None
    last_vision_detection: Optional[datetime] = None
    last_audio_detection: Optional[datetime] = None
    person_count: int = 0


class AIOrchestrator:
    """
    Main orchestrator that coordinates:
    - Vision processing results
    - Audio transcription and intent
    - Table state management
    - Priority queue management
    - Staff assignment
    - Notifications

    This is the "brain" of the restaurant AI system.
    """

    def __init__(self):
        # Core components
        self.state_manager = RestaurantStateMachineManager()
        self.priority_queue = PriorityQueue()
        self.staff_assigner = StaffAssigner()
        self.sensor_fusion = SensorFusion()

        # Context tracking
        self.tables: Dict[int, TableContext] = {}
        self.staff: Dict[int, StaffInfo] = {}

        # Processing loop
        self._running = False
        self._process_interval = 1.0  # seconds

        # Event handlers
        self._event_handlers: List[Callable] = []

        # Setup internal callbacks
        self._setup_callbacks()

    def _setup_callbacks(self):
        """Setup internal callback handlers."""
        # State machine transitions
        self.state_manager.on_any_transition(self._on_table_transition)

        # Priority queue events
        self.priority_queue.on_add(self._on_request_added)
        self.priority_queue.on_resolve(self._on_request_resolved)

        # Sensor fusion events
        self.sensor_fusion.register_event_handler(self._on_fused_event)

    async def start(self):
        """Start the orchestrator processing loop."""
        self._running = True
        print("AI Orchestrator started")

        while self._running:
            try:
                await self._process_cycle()
                await asyncio.sleep(self._process_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Orchestrator error: {e}")
                await asyncio.sleep(self._process_interval)

        print("AI Orchestrator stopped")

    async def stop(self):
        """Stop the orchestrator."""
        self._running = False

    async def _process_cycle(self):
        """Single processing cycle."""
        # 1. Check for state timeouts
        self.state_manager.check_all_timeouts()

        # 2. Update priorities based on wait times
        pending_requests = self.priority_queue.get_pending()

        # 3. Process unassigned requests
        unassigned = self.priority_queue.get_unassigned()
        if unassigned:
            await self._assign_pending_requests(unassigned)

        # 4. Check for overdue tables
        overdue_tables = self.state_manager.get_overdue_tables()
        for table_sm in overdue_tables:
            await self._handle_overdue_table(table_sm)

        # 5. Broadcast state updates
        await self._broadcast_state()

    # ==================== Table Management ====================

    def register_table(
        self,
        table_id: int,
        table_number: int,
        x_position: float,
        y_position: float,
    ) -> TableContext:
        """Register a table with the orchestrator."""
        state_machine = self.state_manager.register_table(table_id)

        context = TableContext(
            table_id=table_id,
            table_number=table_number,
            x_position=x_position,
            y_position=y_position,
            state_machine=state_machine,
        )
        self.tables[table_id] = context

        return context

    def get_table_context(self, table_id: int) -> Optional[TableContext]:
        """Get table context."""
        return self.tables.get(table_id)

    # ==================== Staff Management ====================

    def register_staff(
        self,
        staff_id: int,
        name: str,
        x_position: float = 0.5,
        y_position: float = 0.5,
        skills: Optional[List[str]] = None,
        max_workload: int = 4,
    ) -> StaffInfo:
        """Register staff member."""
        info = StaffInfo(
            staff_id=staff_id,
            name=name,
            x_position=x_position,
            y_position=y_position,
            current_workload=0,
            max_workload=max_workload,
            skills=skills or [],
            is_available=True,
        )
        self.staff[staff_id] = info
        return info

    def update_staff_position(self, staff_id: int, x: float, y: float):
        """Update staff position."""
        if staff_id in self.staff:
            self.staff[staff_id].x_position = x
            self.staff[staff_id].y_position = y

    def update_staff_availability(self, staff_id: int, available: bool):
        """Update staff availability."""
        if staff_id in self.staff:
            self.staff[staff_id].is_available = available

    # ==================== Vision Event Handling ====================

    async def handle_person_detection(
        self,
        table_id: int,
        person_count: int,
        confidence: float,
    ):
        """Handle person detection from vision system."""
        context = self.tables.get(table_id)
        if not context:
            return

        context.last_vision_detection = datetime.utcnow()
        context.person_count = person_count

        # Add to sensor fusion
        signal = SensorSignal(
            source=SignalSource.VISION,
            signal_type=SignalType.PERSON_DETECTED,
            confidence=confidence,
            table_id=table_id,
            metadata={"person_count": person_count},
        )
        event = self.sensor_fusion.add_signal(signal)

        # Check if this is a new seating
        if context.state_machine.current_state == TableState.EMPTY and person_count > 0:
            context.state_machine.seat_customers(person_count)

            # Create greeting request
            self.priority_queue.add(
                request_type=RequestType.GREETING,
                table_id=table_id,
                description=f"Greet {person_count} guests at table {context.table_number}",
            )

        if event:
            await self.sensor_fusion.emit_event(event)

    async def handle_gesture_detection(
        self,
        table_id: int,
        gesture_type: str,
        confidence: float,
        person_id: Optional[int] = None,
    ):
        """Handle gesture detection (e.g., hand raise)."""
        context = self.tables.get(table_id)
        if not context:
            return

        # Add to sensor fusion
        signal = SensorSignal(
            source=SignalSource.VISION,
            signal_type=SignalType.HAND_RAISE if gesture_type == "hand_raise" else SignalType.GESTURE,
            confidence=confidence,
            table_id=table_id,
            person_id=str(person_id) if person_id else None,
            metadata={"gesture_type": gesture_type},
        )
        event = self.sensor_fusion.add_signal(signal)

        if gesture_type == "hand_raise":
            # Determine request type based on current state
            state = context.state_machine.current_state

            if state == TableState.SEATED:
                request_type = RequestType.DRINK_ORDER
                context.state_machine.request_service()
            elif state == TableState.SERVED:
                request_type = RequestType.CHECK_ON
            else:
                request_type = RequestType.QUESTION

            # Add to priority queue with gesture bonus
            self.priority_queue.add(
                request_type=request_type,
                table_id=table_id,
                description=f"Hand raise detected at table {context.table_number}",
                gesture_detected=True,
            )

            # Record analytics
            analytics_service.record_hand_raise(table_id)

        if event:
            await self.sensor_fusion.emit_event(event)

    # ==================== Audio Event Handling ====================

    async def handle_verbal_request(
        self,
        table_id: int,
        transcript: str,
        intent: str,
        entities: List[Dict[str, str]],
        confidence: float,
    ):
        """Handle verbal request from audio system."""
        context = self.tables.get(table_id)
        if not context:
            return

        context.last_audio_detection = datetime.utcnow()

        # Add to sensor fusion
        signal = SensorSignal(
            source=SignalSource.AUDIO,
            signal_type=SignalType.VERBAL_REQUEST,
            confidence=confidence,
            table_id=table_id,
            metadata={
                "transcript": transcript,
                "intent": intent,
                "entities": entities,
            },
        )
        event = self.sensor_fusion.add_signal(signal)

        # Map intent to request type
        intent_mapping = {
            "order": RequestType.FOOD_ORDER,
            "drink": RequestType.DRINK_ORDER,
            "check": RequestType.CHECK_REQUEST,
            "bill": RequestType.CHECK_REQUEST,
            "refill": RequestType.REFILL,
            "question": RequestType.QUESTION,
            "complaint": RequestType.COMPLAINT,
        }

        request_type = intent_mapping.get(intent, RequestType.QUESTION)

        # Check for frustration indicators
        frustration_words = ["wait", "slow", "where", "long", "hello", "excuse"]
        frustration_detected = any(word in transcript.lower() for word in frustration_words)

        # Add to priority queue
        self.priority_queue.add(
            request_type=request_type,
            table_id=table_id,
            description=f"'{transcript[:50]}...' at table {context.table_number}",
            verbal_request=True,
            frustration_detected=frustration_detected,
            metadata={"intent": intent, "entities": entities},
        )

        if event:
            await self.sensor_fusion.emit_event(event)

    # ==================== State Transitions ====================

    def _on_table_transition(
        self,
        table_id: int,
        from_state: TableState,
        to_state: TableState,
        trigger: str,
        metadata: Optional[Dict],
    ):
        """Handle table state transitions."""
        context = self.tables.get(table_id)
        if not context:
            return

        # Record analytics
        analytics_service.record_event(f"transition_{from_state.value}_to_{to_state.value}")

        # Create appropriate requests for new state
        if to_state == TableState.ORDERING:
            self.priority_queue.add(
                request_type=RequestType.FOOD_ORDER,
                table_id=table_id,
                description=f"Take order at table {context.table_number}",
            )

        elif to_state == TableState.PAYING:
            self.priority_queue.add(
                request_type=RequestType.PAYMENT,
                table_id=table_id,
                description=f"Process payment at table {context.table_number}",
                payment_ready=True,
            )

        elif to_state == TableState.EMPTY:
            # Clear pending requests for this table
            self.priority_queue.cancel_for_table(table_id)

            # Create cleanup request
            self.priority_queue.add(
                request_type=RequestType.CLEANUP,
                table_id=table_id,
                description=f"Bus table {context.table_number}",
            )

    async def _handle_overdue_table(self, table_sm: TableStateMachine):
        """Handle a table that's been waiting too long."""
        context = self.tables.get(table_sm.table_id)
        if not context:
            return

        overdue_seconds = table_sm.get_overdue_seconds()

        # Check if we already have a high-priority request for this table
        existing = self.priority_queue.get_for_table(table_sm.table_id)
        high_priority_exists = any(
            item.metadata.get("frustration_detected") for item in existing
        )

        if not high_priority_exists:
            # Create urgent request
            self.priority_queue.add(
                request_type=RequestType.CHECK_ON,
                table_id=table_sm.table_id,
                description=f"Table {context.table_number} waiting {int(overdue_seconds)}s overdue",
                frustration_detected=True,
            )

            # Send notification
            await notification_service.send_table_alert(
                table_id=table_sm.table_id,
                title="Table Overdue",
                message=f"Table {context.table_number} has been waiting {int(overdue_seconds / 60)} minutes",
                assigned_staff_id=context.assigned_staff_id,
                priority=NotificationPriority.HIGH,
            )

    # ==================== Request Management ====================

    def _on_request_added(self, item: PriorityItem):
        """Handle new request added to queue."""
        analytics_service.record_event(f"request_{item.request_type.value}")

    def _on_request_resolved(self, item: PriorityItem):
        """Handle request resolved."""
        # Calculate wait time
        wait_seconds = (item.resolved_at - item.created_at).total_seconds()
        analytics_service.record_wait_time(item.table_id, wait_seconds)

        if item.assigned_staff_id:
            analytics_service.record_task_completion(item.assigned_staff_id)

    async def _assign_pending_requests(self, requests: List[PriorityItem]):
        """Assign staff to pending requests."""
        if not requests or not self.staff:
            return

        # Build task list
        tasks = []
        for req in requests:
            context = self.tables.get(req.table_id)
            if not context:
                continue

            task = TaskInfo(
                task_id=req.request_id,
                table_id=req.table_id,
                table_x=context.x_position,
                table_y=context.y_position,
                priority=abs(req.priority),
                required_skills=self._get_required_skills(req.request_type),
                task_type=req.request_type.value,
            )
            tasks.append(task)

        if not tasks:
            return

        # Get available staff
        staff_list = list(self.staff.values())

        # Get optimal assignments
        assignments = self.staff_assigner.assign_optimal(staff_list, tasks)

        # Apply assignments
        for assignment in assignments:
            self.priority_queue.assign(assignment.task_id, assignment.staff_id)

            # Update staff workload
            if assignment.staff_id in self.staff:
                self.staff[assignment.staff_id].current_workload += 1

            # Send notification to staff
            context = self.tables.get(assignment.table_id)
            if context:
                await notification_service.send_staff_alert(
                    staff_id=assignment.staff_id,
                    title="New Task",
                    message=f"Table {context.table_number}: {assignment.reasoning}",
                    table_id=assignment.table_id,
                )

    def _get_required_skills(self, request_type: RequestType) -> List[str]:
        """Get required skills for request type."""
        skill_map = {
            RequestType.FOOD_ORDER: ["order_taking"],
            RequestType.DRINK_ORDER: ["cocktails"],
            RequestType.COMPLAINT: ["conflict_resolution"],
            RequestType.PAYMENT: ["payment_processing"],
            RequestType.CLEANUP: ["bussing"],
        }
        return skill_map.get(request_type, [])

    # ==================== Fused Events ====================

    async def _on_fused_event(self, event: FusedEvent):
        """Handle fused sensor event."""
        # Emit to external handlers
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"Event handler error: {e}")

        # Broadcast via WebSocket
        from ..api.deps import connection_manager

        await connection_manager.broadcast(
            {"type": "fused_event", "data": event.to_dict()}
        )

    def on_event(self, handler: Callable):
        """Register handler for fused events."""
        self._event_handlers.append(handler)

    # ==================== State Broadcasting ====================

    async def _broadcast_state(self):
        """Broadcast current state to connected clients."""
        from ..api.deps import connection_manager

        state = {
            "type": "state_update",
            "data": {
                "tables": {
                    tid: {
                        "state": ctx.state_machine.current_state.value,
                        "time_in_state": ctx.state_machine.get_time_in_state(),
                        "is_overdue": ctx.state_machine.is_overdue(),
                        "person_count": ctx.person_count,
                        "assigned_staff_id": ctx.assigned_staff_id,
                    }
                    for tid, ctx in self.tables.items()
                },
                "queue_stats": self.priority_queue.get_stats(),
                "restaurant_stats": self.state_manager.get_summary(),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        await connection_manager.broadcast(state)

    # ==================== API Methods ====================

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        return {
            "tables": [
                {
                    "table_id": tid,
                    "table_number": ctx.table_number,
                    "state": ctx.state_machine.current_state.value,
                    "time_in_state": ctx.state_machine.get_time_in_state(),
                    "is_overdue": ctx.state_machine.is_overdue(),
                    "person_count": ctx.person_count,
                    "assigned_staff_id": ctx.assigned_staff_id,
                    "x_position": ctx.x_position,
                    "y_position": ctx.y_position,
                    "suggested_actions": ctx.state_machine.get_suggested_actions(),
                }
                for tid, ctx in self.tables.items()
            ],
            "staff": [
                {
                    "staff_id": s.staff_id,
                    "name": s.name,
                    "x_position": s.x_position,
                    "y_position": s.y_position,
                    "is_available": s.is_available,
                    "current_workload": s.current_workload,
                    "max_workload": s.max_workload,
                }
                for s in self.staff.values()
            ],
            "pending_requests": [r.to_dict() for r in self.priority_queue.get_pending(20)],
            "queue_stats": self.priority_queue.get_stats(),
            "restaurant_stats": self.state_manager.get_summary(),
            "analytics": analytics_service.get_summary_stats(),
        }


# Global orchestrator instance
orchestrator = AIOrchestrator()
