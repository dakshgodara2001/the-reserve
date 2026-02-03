"""Person tracking using DeepSORT-style tracking."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import deque


@dataclass
class TrackedPerson:
    """Tracked person with history."""

    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity: Tuple[float, float] = (0.0, 0.0)
    table_id: Optional[int] = None
    is_staff: bool = False
    metadata: Dict = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.first_seen).total_seconds()

    @property
    def time_since_update(self) -> float:
        return (datetime.utcnow() - self.last_seen).total_seconds()

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "center": self.center,
            "confidence": self.confidence,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "age_seconds": self.age_seconds,
            "velocity": self.velocity,
            "table_id": self.table_id,
            "is_staff": self.is_staff,
        }


class PersonTracker:
    """
    Simple tracking system using IoU matching.
    Inspired by SORT/DeepSORT but simplified for prototype.
    """

    def __init__(
        self,
        max_age: float = 5.0,  # Seconds before track is deleted
        min_hits: int = 3,  # Minimum detections before track is confirmed
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: Dict[int, TrackedPerson] = {}
        self.next_id = 1
        self.frame_count = 0

    def update(
        self,
        detections: List[Tuple[float, float, float, float, float]],
    ) -> List[TrackedPerson]:
        """
        Update tracker with new detections.

        Args:
            detections: List of (x1, y1, x2, y2, confidence)

        Returns:
            List of active tracked persons
        """
        self.frame_count += 1

        if not detections:
            # No detections - age all tracks
            self._age_tracks()
            return self._get_active_tracks()

        # Build detection array
        det_boxes = np.array([d[:4] for d in detections])
        det_confs = np.array([d[4] for d in detections])

        # Get existing track boxes
        track_ids = list(self.tracks.keys())
        if track_ids:
            track_boxes = np.array([self.tracks[tid].bbox for tid in track_ids])

            # Calculate IoU matrix
            iou_matrix = self._calculate_iou_matrix(det_boxes, track_boxes)

            # Hungarian matching (greedy for simplicity)
            matched_det, matched_track, unmatched_det = self._match_detections(
                iou_matrix, track_ids
            )

            # Update matched tracks
            for det_idx, track_id in zip(matched_det, matched_track):
                self._update_track(
                    track_id,
                    tuple(det_boxes[det_idx]),
                    det_confs[det_idx],
                )

            # Create new tracks for unmatched detections
            for det_idx in unmatched_det:
                self._create_track(tuple(det_boxes[det_idx]), det_confs[det_idx])
        else:
            # No existing tracks - create new for all detections
            for i, det in enumerate(detections):
                self._create_track(tuple(det[:4]), det[4])

        # Remove stale tracks
        self._age_tracks()

        return self._get_active_tracks()

    def _calculate_iou_matrix(
        self, det_boxes: np.ndarray, track_boxes: np.ndarray
    ) -> np.ndarray:
        """Calculate IoU between all detection-track pairs."""
        num_det = len(det_boxes)
        num_track = len(track_boxes)
        iou_matrix = np.zeros((num_det, num_track))

        for d in range(num_det):
            for t in range(num_track):
                iou_matrix[d, t] = self._calculate_iou(det_boxes[d], track_boxes[t])

        return iou_matrix

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _match_detections(
        self, iou_matrix: np.ndarray, track_ids: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Match detections to tracks using greedy matching.

        Returns:
            matched_det: List of matched detection indices
            matched_track: List of matched track IDs
            unmatched_det: List of unmatched detection indices
        """
        matched_det = []
        matched_track = []
        unmatched_det = list(range(iou_matrix.shape[0]))

        if iou_matrix.size == 0:
            return matched_det, matched_track, unmatched_det

        # Greedy matching - take best IoU matches iteratively
        while True:
            # Find best remaining match
            max_iou = 0
            best_det = -1
            best_track_idx = -1

            for d in unmatched_det:
                for t_idx in range(len(track_ids)):
                    if track_ids[t_idx] in matched_track:
                        continue
                    if iou_matrix[d, t_idx] > max_iou:
                        max_iou = iou_matrix[d, t_idx]
                        best_det = d
                        best_track_idx = t_idx

            if max_iou < self.iou_threshold:
                break

            matched_det.append(best_det)
            matched_track.append(track_ids[best_track_idx])
            unmatched_det.remove(best_det)

        return matched_det, matched_track, unmatched_det

    def _create_track(
        self, bbox: Tuple[float, float, float, float], confidence: float
    ) -> int:
        """Create new track."""
        track_id = self.next_id
        self.next_id += 1

        track = TrackedPerson(
            track_id=track_id,
            bbox=bbox,
            confidence=confidence,
        )
        track.positions.append(track.center)

        self.tracks[track_id] = track
        return track_id

    def _update_track(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        confidence: float,
    ):
        """Update existing track."""
        track = self.tracks[track_id]

        # Calculate velocity
        old_center = track.center
        track.bbox = bbox
        new_center = track.center

        dt = track.time_since_update
        if dt > 0:
            track.velocity = (
                (new_center[0] - old_center[0]) / dt,
                (new_center[1] - old_center[1]) / dt,
            )

        track.confidence = confidence
        track.last_seen = datetime.utcnow()
        track.positions.append(new_center)

    def _age_tracks(self):
        """Remove tracks that haven't been updated recently."""
        stale_ids = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.tracks[track_id]

    def _get_active_tracks(self) -> List[TrackedPerson]:
        """Get confirmed active tracks."""
        active = []
        for track in self.tracks.values():
            # Track is active if seen recently
            if track.time_since_update < self.max_age:
                active.append(track)
        return active

    def get_track(self, track_id: int) -> Optional[TrackedPerson]:
        """Get specific track by ID."""
        return self.tracks.get(track_id)

    def assign_table(self, track_id: int, table_id: int):
        """Assign a table to a tracked person."""
        if track_id in self.tracks:
            self.tracks[track_id].table_id = table_id

    def mark_as_staff(self, track_id: int, is_staff: bool = True):
        """Mark a tracked person as staff."""
        if track_id in self.tracks:
            self.tracks[track_id].is_staff = is_staff

    def get_persons_at_table(self, table_id: int) -> List[TrackedPerson]:
        """Get all persons assigned to a table."""
        return [t for t in self.tracks.values() if t.table_id == table_id]

    def get_customer_count(self) -> int:
        """Get count of tracked non-staff persons."""
        return sum(1 for t in self.tracks.values() if not t.is_staff)

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
