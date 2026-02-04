"""Analytics service for collecting and reporting metrics."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import statistics


@dataclass
class MetricPoint:
    """Single metric data point."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TableMetrics:
    """Metrics for a single table."""

    table_id: int
    total_guests_served: int = 0
    total_orders: int = 0
    avg_wait_time_seconds: float = 0.0
    avg_dining_duration_minutes: float = 0.0
    hand_raises_count: int = 0
    complaints_count: int = 0
    turnover_rate: float = 0.0  # Seatings per hour


@dataclass
class StaffMetrics:
    """Metrics for a single staff member."""

    staff_id: int
    tasks_completed: int = 0
    tables_served: int = 0
    avg_response_time_seconds: float = 0.0
    distance_walked_meters: float = 0.0
    customer_interactions: int = 0


class AnalyticsService:
    """Service for collecting and analyzing restaurant metrics."""

    def __init__(self):
        self.metrics_buffer: List[MetricPoint] = []
        self.max_buffer_size = 10000
        self.table_metrics: Dict[int, TableMetrics] = {}
        self.staff_metrics: Dict[int, StaffMetrics] = {}

        # Time series data
        self.wait_times: List[tuple] = []  # (timestamp, table_id, wait_seconds)
        self.response_times: List[tuple] = []  # (timestamp, staff_id, response_seconds)
        self.event_counts: Dict[str, List[datetime]] = defaultdict(list)

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Record a metric point."""
        metric = MetricPoint(name=name, value=value, tags=tags or {})
        self.metrics_buffer.append(metric)

        # Trim buffer if needed
        if len(self.metrics_buffer) > self.max_buffer_size:
            self.metrics_buffer = self.metrics_buffer[-self.max_buffer_size :]

    def record_wait_time(self, table_id: int, wait_seconds: float):
        """Record customer wait time."""
        self.wait_times.append((datetime.utcnow(), table_id, wait_seconds))
        self.record_metric("wait_time", wait_seconds, {"table_id": str(table_id)})

        # Update table metrics
        if table_id not in self.table_metrics:
            self.table_metrics[table_id] = TableMetrics(table_id=table_id)

        metrics = self.table_metrics[table_id]
        # Calculate running average
        table_waits = [w for _, tid, w in self.wait_times if tid == table_id]
        if table_waits:
            metrics.avg_wait_time_seconds = statistics.mean(table_waits)

    def record_response_time(self, staff_id: int, response_seconds: float):
        """Record staff response time."""
        self.response_times.append((datetime.utcnow(), staff_id, response_seconds))
        self.record_metric(
            "response_time", response_seconds, {"staff_id": str(staff_id)}
        )

        # Update staff metrics
        if staff_id not in self.staff_metrics:
            self.staff_metrics[staff_id] = StaffMetrics(staff_id=staff_id)

        metrics = self.staff_metrics[staff_id]
        staff_responses = [r for _, sid, r in self.response_times if sid == staff_id]
        if staff_responses:
            metrics.avg_response_time_seconds = statistics.mean(staff_responses)

    def record_event(self, event_type: str):
        """Record event occurrence for counting."""
        self.event_counts[event_type].append(datetime.utcnow())
        self.record_metric("event_count", 1, {"event_type": event_type})

    def record_table_seating(self, table_id: int, guest_count: int):
        """Record table seating."""
        if table_id not in self.table_metrics:
            self.table_metrics[table_id] = TableMetrics(table_id=table_id)

        self.table_metrics[table_id].total_guests_served += guest_count
        self.record_metric(
            "guests_seated", guest_count, {"table_id": str(table_id)}
        )

    def record_task_completion(self, staff_id: int):
        """Record staff task completion."""
        if staff_id not in self.staff_metrics:
            self.staff_metrics[staff_id] = StaffMetrics(staff_id=staff_id)

        self.staff_metrics[staff_id].tasks_completed += 1
        self.record_metric("task_completed", 1, {"staff_id": str(staff_id)})

    def record_hand_raise(self, table_id: int):
        """Record hand raise detection."""
        if table_id not in self.table_metrics:
            self.table_metrics[table_id] = TableMetrics(table_id=table_id)

        self.table_metrics[table_id].hand_raises_count += 1
        self.record_event("hand_raise")

    def get_summary_stats(
        self, since_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get summary statistics."""
        since = datetime.utcnow() - timedelta(minutes=since_minutes)

        # Filter to recent data
        recent_waits = [w for t, _, w in self.wait_times if t >= since]
        recent_responses = [r for t, _, r in self.response_times if t >= since]

        # Calculate event counts
        event_summary = {}
        for event_type, timestamps in self.event_counts.items():
            event_summary[event_type] = len([t for t in timestamps if t >= since])

        return {
            "time_range_minutes": since_minutes,
            "wait_times": {
                "count": len(recent_waits),
                "avg": statistics.mean(recent_waits) if recent_waits else 0,
                "min": min(recent_waits) if recent_waits else 0,
                "max": max(recent_waits) if recent_waits else 0,
            },
            "response_times": {
                "count": len(recent_responses),
                "avg": statistics.mean(recent_responses) if recent_responses else 0,
                "min": min(recent_responses) if recent_responses else 0,
                "max": max(recent_responses) if recent_responses else 0,
            },
            "events": event_summary,
            "tables_active": len(
                [m for m in self.table_metrics.values() if m.total_guests_served > 0]
            ),
            "staff_active": len(
                [m for m in self.staff_metrics.values() if m.tasks_completed > 0]
            ),
        }

    def get_table_metrics(self, table_id: int) -> Optional[TableMetrics]:
        """Get metrics for specific table."""
        return self.table_metrics.get(table_id)

    def get_staff_metrics(self, staff_id: int) -> Optional[StaffMetrics]:
        """Get metrics for specific staff member."""
        return self.staff_metrics.get(staff_id)

    def get_hourly_breakdown(
        self, metric_name: str, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get hourly breakdown of a metric."""
        now = datetime.utcnow()
        breakdown = []

        for h in range(hours):
            start = now - timedelta(hours=h + 1)
            end = now - timedelta(hours=h)

            # Filter metrics
            values = [
                m.value
                for m in self.metrics_buffer
                if m.name == metric_name and start <= m.timestamp < end
            ]

            breakdown.append(
                {
                    "hour": start.strftime("%H:00"),
                    "count": len(values),
                    "sum": sum(values),
                    "avg": statistics.mean(values) if values else 0,
                }
            )

        return list(reversed(breakdown))

    def get_top_performers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing staff members."""
        staff_scores = []

        for staff_id, metrics in self.staff_metrics.items():
            # Calculate performance score
            score = (
                metrics.tasks_completed * 10
                - metrics.avg_response_time_seconds * 0.1
                + metrics.customer_interactions * 2
            )
            staff_scores.append(
                {
                    "staff_id": staff_id,
                    "score": score,
                    "tasks": metrics.tasks_completed,
                    "avg_response": metrics.avg_response_time_seconds,
                }
            )

        staff_scores.sort(key=lambda x: x["score"], reverse=True)
        return staff_scores[:limit]

    def get_problem_tables(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get tables with potential issues."""
        table_issues = []

        for table_id, metrics in self.table_metrics.items():
            # Calculate issue score (higher = more issues)
            issue_score = (
                metrics.hand_raises_count * 5
                + metrics.complaints_count * 20
                + (metrics.avg_wait_time_seconds / 60) * 2  # Per minute over
            )

            if issue_score > 0:
                table_issues.append(
                    {
                        "table_id": table_id,
                        "issue_score": issue_score,
                        "hand_raises": metrics.hand_raises_count,
                        "complaints": metrics.complaints_count,
                        "avg_wait": metrics.avg_wait_time_seconds,
                    }
                )

        table_issues.sort(key=lambda x: x["issue_score"], reverse=True)
        return table_issues[:limit]

    def reset_daily_metrics(self):
        """Reset daily metrics (call at end of day)."""
        # Keep table/staff metrics but reset counts
        for metrics in self.table_metrics.values():
            metrics.total_guests_served = 0
            metrics.total_orders = 0
            metrics.hand_raises_count = 0
            metrics.complaints_count = 0

        for metrics in self.staff_metrics.values():
            metrics.tasks_completed = 0
            metrics.tables_served = 0
            metrics.customer_interactions = 0
            metrics.distance_walked_meters = 0

        # Clear time series data older than 24 hours
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.wait_times = [(t, tid, w) for t, tid, w in self.wait_times if t >= cutoff]
        self.response_times = [
            (t, sid, r) for t, sid, r in self.response_times if t >= cutoff
        ]

        for event_type in self.event_counts:
            self.event_counts[event_type] = [
                t for t in self.event_counts[event_type] if t >= cutoff
            ]


# Global analytics service instance
analytics_service = AnalyticsService()
