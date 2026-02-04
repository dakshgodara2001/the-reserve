"""Zone management for restaurant areas."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
import math


class ZoneType(str, Enum):
    """Types of restaurant zones."""

    DINING = "dining"
    BAR = "bar"
    KITCHEN = "kitchen"
    ENTRANCE = "entrance"
    RESTROOM = "restroom"
    PRIVATE = "private"
    OUTDOOR = "outdoor"
    STAFF_ONLY = "staff_only"


@dataclass
class Zone:
    """Restaurant zone definition."""

    id: str
    name: str
    zone_type: ZoneType

    # Bounding box (normalized 0-1)
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    # Properties
    capacity: int = 50
    current_occupancy: int = 0
    tables: List[int] = field(default_factory=list)

    # State
    is_active: bool = True
    alert_threshold: float = 0.9  # Alert when occupancy reaches this ratio

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is within zone."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def get_center(self) -> Tuple[float, float]:
        """Get zone center point."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
        )

    def get_area(self) -> float:
        """Get zone area (normalized)."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def get_occupancy_ratio(self) -> float:
        """Get current occupancy as ratio."""
        if self.capacity == 0:
            return 0.0
        return self.current_occupancy / self.capacity

    def is_near_capacity(self) -> bool:
        """Check if zone is near capacity."""
        return self.get_occupancy_ratio() >= self.alert_threshold

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "zone_type": self.zone_type.value,
            "bounds": {
                "x_min": self.x_min,
                "y_min": self.y_min,
                "x_max": self.x_max,
                "y_max": self.y_max,
            },
            "center": self.get_center(),
            "capacity": self.capacity,
            "current_occupancy": self.current_occupancy,
            "occupancy_ratio": self.get_occupancy_ratio(),
            "is_active": self.is_active,
            "tables": self.tables,
        }


@dataclass
class ZoneTransition:
    """Record of entity moving between zones."""

    entity_id: str
    entity_type: str  # "staff" or "customer"
    from_zone: Optional[str]
    to_zone: Optional[str]
    timestamp: float
    position: Tuple[float, float]


class ZoneManager:
    """
    Manages restaurant zones and tracks entity positions.

    Features:
    - Zone definition and lookup
    - Position to zone mapping
    - Zone transition detection
    - Occupancy tracking
    """

    def __init__(self):
        self.zones: Dict[str, Zone] = {}
        self.entity_zones: Dict[str, str] = {}  # entity_id -> zone_id
        self.transition_history: List[ZoneTransition] = []

        # Callbacks
        self._on_enter_callbacks: List[Callable] = []
        self._on_exit_callbacks: List[Callable] = []
        self._on_capacity_callbacks: List[Callable] = []

        # Initialize default zones
        self._create_default_zones()

    def _create_default_zones(self):
        """Create default restaurant zones."""
        default_zones = [
            Zone(
                id="entrance",
                name="Entrance",
                zone_type=ZoneType.ENTRANCE,
                x_min=0.0,
                y_min=0.0,
                x_max=0.15,
                y_max=0.2,
                capacity=10,
            ),
            Zone(
                id="main_dining",
                name="Main Dining",
                zone_type=ZoneType.DINING,
                x_min=0.15,
                y_min=0.0,
                x_max=0.7,
                y_max=0.7,
                capacity=60,
            ),
            Zone(
                id="bar",
                name="Bar Area",
                zone_type=ZoneType.BAR,
                x_min=0.0,
                y_min=0.7,
                x_max=0.5,
                y_max=1.0,
                capacity=20,
            ),
            Zone(
                id="private",
                name="Private Dining",
                zone_type=ZoneType.PRIVATE,
                x_min=0.7,
                y_min=0.0,
                x_max=1.0,
                y_max=0.5,
                capacity=15,
            ),
            Zone(
                id="kitchen",
                name="Kitchen",
                zone_type=ZoneType.KITCHEN,
                x_min=0.7,
                y_min=0.5,
                x_max=1.0,
                y_max=1.0,
                capacity=10,
            ),
            Zone(
                id="outdoor",
                name="Outdoor Patio",
                zone_type=ZoneType.OUTDOOR,
                x_min=0.5,
                y_min=0.7,
                x_max=0.7,
                y_max=1.0,
                capacity=25,
            ),
        ]

        for zone in default_zones:
            self.zones[zone.id] = zone

    def add_zone(self, zone: Zone):
        """Add a zone."""
        self.zones[zone.id] = zone

    def remove_zone(self, zone_id: str):
        """Remove a zone."""
        if zone_id in self.zones:
            del self.zones[zone_id]

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get zone by ID."""
        return self.zones.get(zone_id)

    def get_zone_at_position(self, x: float, y: float) -> Optional[Zone]:
        """Get zone containing a position."""
        for zone in self.zones.values():
            if zone.is_active and zone.contains_point(x, y):
                return zone
        return None

    def get_zones_by_type(self, zone_type: ZoneType) -> List[Zone]:
        """Get all zones of a type."""
        return [z for z in self.zones.values() if z.zone_type == zone_type]

    def update_entity_position(
        self,
        entity_id: str,
        x: float,
        y: float,
        entity_type: str = "staff",
    ) -> Optional[ZoneTransition]:
        """
        Update entity position and detect zone transitions.

        Args:
            entity_id: Unique entity identifier
            x: X position (0-1)
            y: Y position (0-1)
            entity_type: "staff" or "customer"

        Returns:
            ZoneTransition if entity changed zones
        """
        # Find current zone
        new_zone = self.get_zone_at_position(x, y)
        new_zone_id = new_zone.id if new_zone else None

        # Get previous zone
        old_zone_id = self.entity_zones.get(entity_id)

        # Check for transition
        if old_zone_id != new_zone_id:
            import time

            transition = ZoneTransition(
                entity_id=entity_id,
                entity_type=entity_type,
                from_zone=old_zone_id,
                to_zone=new_zone_id,
                timestamp=time.time(),
                position=(x, y),
            )

            # Update tracking
            if new_zone_id:
                self.entity_zones[entity_id] = new_zone_id
            elif entity_id in self.entity_zones:
                del self.entity_zones[entity_id]

            # Update occupancy
            if old_zone_id and old_zone_id in self.zones:
                self.zones[old_zone_id].current_occupancy = max(
                    0, self.zones[old_zone_id].current_occupancy - 1
                )

            if new_zone_id and new_zone_id in self.zones:
                self.zones[new_zone_id].current_occupancy += 1

                # Check capacity
                if self.zones[new_zone_id].is_near_capacity():
                    self._emit_capacity_alert(self.zones[new_zone_id])

            # Record and emit
            self.transition_history.append(transition)
            self._emit_transition(transition)

            return transition

        return None

    def get_entity_zone(self, entity_id: str) -> Optional[str]:
        """Get zone ID for an entity."""
        return self.entity_zones.get(entity_id)

    def get_entities_in_zone(self, zone_id: str) -> List[str]:
        """Get all entities in a zone."""
        return [
            eid for eid, zid in self.entity_zones.items() if zid == zone_id
        ]

    def calculate_distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
    ) -> float:
        """Calculate distance between two positions."""
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

    def find_nearest_zone(
        self,
        x: float,
        y: float,
        zone_type: Optional[ZoneType] = None,
    ) -> Optional[Tuple[Zone, float]]:
        """
        Find nearest zone to a position.

        Args:
            x: X position
            y: Y position
            zone_type: Optional filter by zone type

        Returns:
            Tuple of (zone, distance) or None
        """
        nearest = None
        min_distance = float("inf")

        for zone in self.zones.values():
            if not zone.is_active:
                continue
            if zone_type and zone.zone_type != zone_type:
                continue

            center = zone.get_center()
            distance = self.calculate_distance((x, y), center)

            if distance < min_distance:
                min_distance = distance
                nearest = zone

        if nearest:
            return (nearest, min_distance)
        return None

    def assign_table_to_zone(self, table_id: int, zone_id: str):
        """Assign a table to a zone."""
        if zone_id in self.zones:
            if table_id not in self.zones[zone_id].tables:
                self.zones[zone_id].tables.append(table_id)

    def get_zone_for_table(self, table_id: int) -> Optional[Zone]:
        """Get zone containing a table."""
        for zone in self.zones.values():
            if table_id in zone.tables:
                return zone
        return None

    def on_zone_enter(self, callback: Callable):
        """Register callback for zone entry."""
        self._on_enter_callbacks.append(callback)

    def on_zone_exit(self, callback: Callable):
        """Register callback for zone exit."""
        self._on_exit_callbacks.append(callback)

    def on_capacity_alert(self, callback: Callable):
        """Register callback for capacity alerts."""
        self._on_capacity_callbacks.append(callback)

    def _emit_transition(self, transition: ZoneTransition):
        """Emit zone transition event."""
        if transition.to_zone:
            for callback in self._on_enter_callbacks:
                try:
                    callback(transition)
                except Exception as e:
                    print(f"Zone enter callback error: {e}")

        if transition.from_zone:
            for callback in self._on_exit_callbacks:
                try:
                    callback(transition)
                except Exception as e:
                    print(f"Zone exit callback error: {e}")

    def _emit_capacity_alert(self, zone: Zone):
        """Emit capacity alert."""
        for callback in self._on_capacity_callbacks:
            try:
                callback(zone)
            except Exception as e:
                print(f"Capacity alert callback error: {e}")

    def get_occupancy_summary(self) -> Dict[str, Any]:
        """Get occupancy summary for all zones."""
        return {
            zone_id: {
                "name": zone.name,
                "type": zone.zone_type.value,
                "occupancy": zone.current_occupancy,
                "capacity": zone.capacity,
                "ratio": zone.get_occupancy_ratio(),
                "near_capacity": zone.is_near_capacity(),
            }
            for zone_id, zone in self.zones.items()
        }

    def get_all_zones(self) -> List[Dict]:
        """Get all zones as dictionaries."""
        return [zone.to_dict() for zone in self.zones.values()]
