"""Location simulation for testing without real BLE/WiFi hardware."""

import asyncio
import random
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum


class DeviceType(str, Enum):
    """Types of trackable devices."""

    STAFF_BADGE = "staff_badge"
    CUSTOMER_TRACKER = "customer_tracker"
    TABLE_BEACON = "table_beacon"


class MovementPattern(str, Enum):
    """Movement patterns for simulation."""

    STATIONARY = "stationary"
    RANDOM_WALK = "random_walk"
    PATROL = "patrol"
    WAYPOINT = "waypoint"
    FOLLOW_PATH = "follow_path"


@dataclass
class SimulatedDevice:
    """Simulated tracking device."""

    device_id: str
    device_type: DeviceType
    entity_id: int  # Staff ID or customer track ID

    # Current position (normalized 0-1)
    x: float = 0.5
    y: float = 0.5

    # Movement
    movement_pattern: MovementPattern = MovementPattern.RANDOM_WALK
    speed: float = 0.02  # Units per update
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    current_waypoint_idx: int = 0

    # Constraints
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    # State
    is_active: bool = True
    last_update: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "entity_id": self.entity_id,
            "position": {"x": self.x, "y": self.y},
            "movement_pattern": self.movement_pattern.value,
            "is_active": self.is_active,
            "last_update": self.last_update.isoformat(),
        }


class LocationSimulator:
    """
    Simulates BLE/WiFi location tracking for prototype testing.

    Features:
    - Simulated staff movement patterns
    - Configurable movement constraints
    - Zone-aware movement
    - Real-time position updates
    """

    def __init__(self, update_interval: float = 1.0):
        """
        Initialize simulator.

        Args:
            update_interval: Seconds between position updates
        """
        self.update_interval = update_interval
        self.devices: Dict[str, SimulatedDevice] = {}
        self._running = False

        # Callbacks
        self._position_callbacks: List[Callable] = []

        # Predefined patrol routes for staff
        self.patrol_routes = {
            "server_route_1": [
                (0.2, 0.3),
                (0.4, 0.3),
                (0.4, 0.5),
                (0.2, 0.5),
            ],
            "server_route_2": [
                (0.5, 0.2),
                (0.7, 0.2),
                (0.7, 0.4),
                (0.5, 0.4),
            ],
            "busser_route": [
                (0.3, 0.3),
                (0.5, 0.3),
                (0.7, 0.3),
                (0.7, 0.5),
                (0.5, 0.5),
                (0.3, 0.5),
            ],
            "host_route": [
                (0.05, 0.1),
                (0.1, 0.1),
                (0.1, 0.15),
                (0.05, 0.15),
            ],
        }

    def add_device(
        self,
        device_id: str,
        device_type: DeviceType,
        entity_id: int,
        initial_x: float = 0.5,
        initial_y: float = 0.5,
        movement_pattern: MovementPattern = MovementPattern.RANDOM_WALK,
        patrol_route: Optional[str] = None,
    ) -> SimulatedDevice:
        """Add a simulated device."""
        device = SimulatedDevice(
            device_id=device_id,
            device_type=device_type,
            entity_id=entity_id,
            x=initial_x,
            y=initial_y,
            movement_pattern=movement_pattern,
        )

        if patrol_route and patrol_route in self.patrol_routes:
            device.waypoints = self.patrol_routes[patrol_route]
            device.movement_pattern = MovementPattern.PATROL

        self.devices[device_id] = device
        return device

    def remove_device(self, device_id: str):
        """Remove a device."""
        if device_id in self.devices:
            del self.devices[device_id]

    def get_device(self, device_id: str) -> Optional[SimulatedDevice]:
        """Get device by ID."""
        return self.devices.get(device_id)

    def get_position(self, device_id: str) -> Optional[Tuple[float, float]]:
        """Get current position of device."""
        device = self.devices.get(device_id)
        if device:
            return (device.x, device.y)
        return None

    def set_position(self, device_id: str, x: float, y: float):
        """Manually set device position."""
        if device_id in self.devices:
            self.devices[device_id].x = x
            self.devices[device_id].y = y
            self.devices[device_id].last_update = datetime.utcnow()

    async def start(self):
        """Start the simulation loop."""
        self._running = True
        print("Location simulator started")

        while self._running:
            try:
                await self._update_positions()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Simulator error: {e}")

        print("Location simulator stopped")

    async def stop(self):
        """Stop the simulation."""
        self._running = False

    async def _update_positions(self):
        """Update all device positions."""
        for device in self.devices.values():
            if not device.is_active:
                continue

            old_x, old_y = device.x, device.y

            # Update based on movement pattern
            if device.movement_pattern == MovementPattern.STATIONARY:
                pass  # No movement

            elif device.movement_pattern == MovementPattern.RANDOM_WALK:
                self._update_random_walk(device)

            elif device.movement_pattern == MovementPattern.PATROL:
                self._update_patrol(device)

            elif device.movement_pattern == MovementPattern.WAYPOINT:
                self._update_waypoint(device)

            # Clamp to bounds
            device.x = max(device.x_min, min(device.x_max, device.x))
            device.y = max(device.y_min, min(device.y_max, device.y))

            device.last_update = datetime.utcnow()

            # Emit position update if moved
            if abs(device.x - old_x) > 0.001 or abs(device.y - old_y) > 0.001:
                await self._emit_position_update(device)

    def _update_random_walk(self, device: SimulatedDevice):
        """Update position with random walk."""
        # Random direction
        angle = random.uniform(0, 2 * math.pi)
        distance = device.speed * random.uniform(0.5, 1.0)

        device.x += math.cos(angle) * distance
        device.y += math.sin(angle) * distance

    def _update_patrol(self, device: SimulatedDevice):
        """Update position along patrol route."""
        if not device.waypoints:
            return

        target = device.waypoints[device.current_waypoint_idx]
        dx = target[0] - device.x
        dy = target[1] - device.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < device.speed:
            # Reached waypoint, move to next
            device.current_waypoint_idx = (device.current_waypoint_idx + 1) % len(
                device.waypoints
            )
        else:
            # Move towards waypoint
            device.x += (dx / distance) * device.speed
            device.y += (dy / distance) * device.speed

    def _update_waypoint(self, device: SimulatedDevice):
        """Move to current waypoint, then stop."""
        if not device.waypoints or device.current_waypoint_idx >= len(device.waypoints):
            device.movement_pattern = MovementPattern.STATIONARY
            return

        target = device.waypoints[device.current_waypoint_idx]
        dx = target[0] - device.x
        dy = target[1] - device.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < device.speed:
            device.x, device.y = target
            device.current_waypoint_idx += 1
        else:
            device.x += (dx / distance) * device.speed
            device.y += (dy / distance) * device.speed

    def send_to_table(self, device_id: str, table_x: float, table_y: float):
        """Send a device to a table position."""
        device = self.devices.get(device_id)
        if device:
            device.waypoints = [(table_x, table_y)]
            device.current_waypoint_idx = 0
            device.movement_pattern = MovementPattern.WAYPOINT

    def set_patrol_route(self, device_id: str, route_name: str):
        """Set device to follow a patrol route."""
        device = self.devices.get(device_id)
        if device and route_name in self.patrol_routes:
            device.waypoints = self.patrol_routes[route_name]
            device.current_waypoint_idx = 0
            device.movement_pattern = MovementPattern.PATROL

    def on_position_update(self, callback: Callable):
        """Register callback for position updates."""
        self._position_callbacks.append(callback)

    async def _emit_position_update(self, device: SimulatedDevice):
        """Emit position update to callbacks."""
        for callback in self._position_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(device)
                else:
                    callback(device)
            except Exception as e:
                print(f"Position callback error: {e}")

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all device positions."""
        return {
            device_id: {
                "x": device.x,
                "y": device.y,
                "entity_id": device.entity_id,
                "type": device.device_type.value,
                "active": device.is_active,
            }
            for device_id, device in self.devices.items()
        }

    def simulate_staff_arrival(self, staff_id: int, name: str) -> SimulatedDevice:
        """Simulate staff member arriving for shift."""
        device_id = f"staff_{staff_id}"

        # Start at entrance
        device = self.add_device(
            device_id=device_id,
            device_type=DeviceType.STAFF_BADGE,
            entity_id=staff_id,
            initial_x=0.05,
            initial_y=0.1,
            movement_pattern=MovementPattern.RANDOM_WALK,
        )

        return device

    def simulate_customer_entry(self, track_id: int) -> SimulatedDevice:
        """Simulate customer entering restaurant."""
        device_id = f"customer_{track_id}"

        device = self.add_device(
            device_id=device_id,
            device_type=DeviceType.CUSTOMER_TRACKER,
            entity_id=track_id,
            initial_x=0.05,
            initial_y=0.1,
            movement_pattern=MovementPattern.STATIONARY,
        )

        return device

    def simulate_customer_seated(
        self, track_id: int, table_x: float, table_y: float
    ):
        """Simulate customer being seated at table."""
        device_id = f"customer_{track_id}"
        device = self.devices.get(device_id)

        if device:
            device.x = table_x
            device.y = table_y
            device.movement_pattern = MovementPattern.STATIONARY

    def simulate_customer_departure(self, track_id: int):
        """Simulate customer leaving."""
        device_id = f"customer_{track_id}"
        device = self.devices.get(device_id)

        if device:
            # Send to exit
            device.waypoints = [(0.05, 0.1)]
            device.current_waypoint_idx = 0
            device.movement_pattern = MovementPattern.WAYPOINT

    def create_sample_simulation(self):
        """Create a sample simulation with staff."""
        # Add servers
        server1 = self.add_device(
            device_id="staff_1",
            device_type=DeviceType.STAFF_BADGE,
            entity_id=1,
            initial_x=0.2,
            initial_y=0.3,
            patrol_route="server_route_1",
        )

        server2 = self.add_device(
            device_id="staff_2",
            device_type=DeviceType.STAFF_BADGE,
            entity_id=2,
            initial_x=0.5,
            initial_y=0.3,
            patrol_route="server_route_2",
        )

        # Add busser
        busser = self.add_device(
            device_id="staff_4",
            device_type=DeviceType.STAFF_BADGE,
            entity_id=4,
            initial_x=0.4,
            initial_y=0.4,
            patrol_route="busser_route",
        )

        # Add host
        host = self.add_device(
            device_id="staff_5",
            device_type=DeviceType.STAFF_BADGE,
            entity_id=5,
            initial_x=0.05,
            initial_y=0.1,
            patrol_route="host_route",
        )

        # Add bartender (stationary)
        bartender = self.add_device(
            device_id="staff_6",
            device_type=DeviceType.STAFF_BADGE,
            entity_id=6,
            initial_x=0.2,
            initial_y=0.85,
            movement_pattern=MovementPattern.STATIONARY,
        )

        return [server1, server2, busser, host, bartender]
