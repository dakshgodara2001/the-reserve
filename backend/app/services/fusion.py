"""Sensor fusion module for combining vision, audio, and location signals."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio


class SignalSource(str, Enum):
    """Source of sensor signal."""

    VISION = "vision"
    AUDIO = "audio"
    LOCATION = "location"
    MANUAL = "manual"


class SignalType(str, Enum):
    """Type of detected signal."""

    PERSON_DETECTED = "person_detected"
    HAND_RAISE = "hand_raise"
    GESTURE = "gesture"
    VERBAL_REQUEST = "verbal_request"
    FRUSTRATION = "frustration"
    MOVEMENT = "movement"
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"


@dataclass
class SensorSignal:
    """Individual sensor signal."""

    source: SignalSource
    signal_type: SignalType
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    table_id: Optional[int] = None
    person_id: Optional[str] = None
    staff_id: Optional[int] = None
    position: Optional[tuple] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_seconds(self) -> float:
        """Get age of signal in seconds."""
        return (datetime.utcnow() - self.timestamp).total_seconds()

    def is_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Check if signal is too old."""
        return self.age_seconds() > max_age_seconds


@dataclass
class FusedEvent:
    """Event resulting from fused sensor data."""

    event_type: str
    confidence: float
    table_id: Optional[int]
    staff_id: Optional[int]
    signals: List[SensorSignal]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "confidence": self.confidence,
            "table_id": self.table_id,
            "staff_id": self.staff_id,
            "timestamp": self.timestamp.isoformat(),
            "priority_score": self.priority_score,
            "signal_count": len(self.signals),
            "sources": list(set(s.source.value for s in self.signals)),
            "metadata": self.metadata,
        }


class SensorFusion:
    """Combines signals from multiple sensors to create confident events."""

    def __init__(self):
        self.signal_buffer: Dict[str, List[SensorSignal]] = {}
        self.fusion_window_seconds = 5.0  # Combine signals within this window
        self.min_confidence_threshold = 0.6
        self.stale_signal_age = 30.0  # Remove signals older than this

        # Weight factors for different sources
        self.source_weights = {
            SignalSource.VISION: 1.0,
            SignalSource.AUDIO: 0.9,
            SignalSource.LOCATION: 0.8,
            SignalSource.MANUAL: 1.0,
        }

        # Corroboration bonuses
        self.multi_source_bonus = 0.2

        # Event handlers
        self._event_handlers: List = []

    def add_signal(self, signal: SensorSignal) -> Optional[FusedEvent]:
        """Add a sensor signal and attempt fusion."""
        # Create key for grouping signals
        key = self._get_signal_key(signal)

        if key not in self.signal_buffer:
            self.signal_buffer[key] = []

        self.signal_buffer[key].append(signal)

        # Clean stale signals
        self._clean_stale_signals()

        # Attempt fusion
        return self._try_fusion(key)

    def _get_signal_key(self, signal: SensorSignal) -> str:
        """Generate key for signal grouping."""
        parts = [signal.signal_type.value]

        if signal.table_id is not None:
            parts.append(f"table_{signal.table_id}")
        elif signal.person_id is not None:
            parts.append(f"person_{signal.person_id}")
        elif signal.staff_id is not None:
            parts.append(f"staff_{signal.staff_id}")

        return "_".join(parts)

    def _clean_stale_signals(self):
        """Remove stale signals from buffer."""
        for key in list(self.signal_buffer.keys()):
            self.signal_buffer[key] = [
                s for s in self.signal_buffer[key] if not s.is_stale(self.stale_signal_age)
            ]
            if not self.signal_buffer[key]:
                del self.signal_buffer[key]

    def _try_fusion(self, key: str) -> Optional[FusedEvent]:
        """Attempt to fuse signals for a given key."""
        if key not in self.signal_buffer:
            return None

        signals = self.signal_buffer[key]
        if not signals:
            return None

        # Get signals within fusion window
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.fusion_window_seconds)
        recent_signals = [s for s in signals if s.timestamp >= window_start]

        if not recent_signals:
            return None

        # Calculate fused confidence
        fused_confidence = self._calculate_fused_confidence(recent_signals)

        if fused_confidence < self.min_confidence_threshold:
            return None

        # Create fused event
        event = self._create_fused_event(recent_signals, fused_confidence)

        # Clear processed signals
        self.signal_buffer[key] = [s for s in signals if s not in recent_signals]

        return event

    def _calculate_fused_confidence(self, signals: List[SensorSignal]) -> float:
        """Calculate combined confidence from multiple signals."""
        if not signals:
            return 0.0

        # Weighted average of confidences
        total_weight = 0.0
        weighted_sum = 0.0

        for signal in signals:
            weight = self.source_weights.get(signal.source, 0.5)
            weighted_sum += signal.confidence * weight
            total_weight += weight

        base_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Multi-source bonus
        unique_sources = set(s.source for s in signals)
        if len(unique_sources) > 1:
            bonus = self.multi_source_bonus * (len(unique_sources) - 1)
            base_confidence = min(1.0, base_confidence + bonus)

        return base_confidence

    def _create_fused_event(
        self, signals: List[SensorSignal], confidence: float
    ) -> FusedEvent:
        """Create a fused event from signals."""
        # Determine event type from signal types
        signal_types = [s.signal_type for s in signals]

        # Map signal types to event types
        if SignalType.HAND_RAISE in signal_types:
            event_type = "attention_needed"
        elif SignalType.VERBAL_REQUEST in signal_types:
            event_type = "verbal_request"
        elif SignalType.FRUSTRATION in signal_types:
            event_type = "customer_frustration"
        elif SignalType.ZONE_ENTRY in signal_types:
            event_type = "zone_entry"
        elif SignalType.ZONE_EXIT in signal_types:
            event_type = "zone_exit"
        elif SignalType.PERSON_DETECTED in signal_types:
            event_type = "person_detected"
        else:
            event_type = "activity_detected"

        # Extract table/staff IDs
        table_id = None
        staff_id = None
        for signal in signals:
            if signal.table_id is not None:
                table_id = signal.table_id
            if signal.staff_id is not None:
                staff_id = signal.staff_id

        # Calculate priority
        priority_score = self._calculate_priority(signals, event_type)

        # Aggregate metadata
        metadata = {}
        for signal in signals:
            metadata.update(signal.metadata)

        return FusedEvent(
            event_type=event_type,
            confidence=confidence,
            table_id=table_id,
            staff_id=staff_id,
            signals=signals,
            priority_score=priority_score,
            metadata=metadata,
        )

    def _calculate_priority(
        self, signals: List[SensorSignal], event_type: str
    ) -> float:
        """Calculate priority score for event."""
        base_priority = {
            "customer_frustration": 80.0,
            "attention_needed": 60.0,
            "verbal_request": 50.0,
            "zone_entry": 20.0,
            "zone_exit": 10.0,
            "person_detected": 15.0,
            "activity_detected": 10.0,
        }.get(event_type, 10.0)

        # Boost for high confidence
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        confidence_boost = (avg_confidence - 0.5) * 20  # -10 to +10

        # Boost for multi-source corroboration
        unique_sources = len(set(s.source for s in signals))
        source_boost = (unique_sources - 1) * 10

        return max(0, min(100, base_priority + confidence_boost + source_boost))

    def register_event_handler(self, handler):
        """Register callback for fused events."""
        self._event_handlers.append(handler)

    async def emit_event(self, event: FusedEvent):
        """Emit fused event to all handlers."""
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"Event handler error: {e}")

    def get_active_signals(self, table_id: Optional[int] = None) -> List[SensorSignal]:
        """Get all active (non-stale) signals."""
        signals = []
        for signal_list in self.signal_buffer.values():
            for signal in signal_list:
                if not signal.is_stale(self.stale_signal_age):
                    if table_id is None or signal.table_id == table_id:
                        signals.append(signal)
        return signals

    def clear_signals_for_table(self, table_id: int):
        """Clear all signals for a specific table."""
        for key in list(self.signal_buffer.keys()):
            self.signal_buffer[key] = [
                s for s in self.signal_buffer[key] if s.table_id != table_id
            ]


# Global sensor fusion instance
sensor_fusion = SensorFusion()
