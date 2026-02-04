"""MediaPipe-based pose estimation for gesture detection."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np


class GestureType(str, Enum):
    """Detected gesture types."""

    HAND_RAISE = "hand_raise"
    WAVING = "waving"
    POINTING = "pointing"
    HEAD_TURN = "head_turn"
    LEANING_FORWARD = "leaning_forward"
    ARMS_CROSSED = "arms_crossed"
    NONE = "none"


@dataclass
class Landmark:
    """Single pose landmark."""

    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    z: float  # Depth estimate
    visibility: float  # Visibility score (0-1)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class PoseResult:
    """Pose estimation result for one person."""

    landmarks: Dict[str, Landmark]
    gestures: List[GestureType]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    person_id: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gestures": [g.value for g in self.gestures],
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "person_id": self.person_id,
            "landmark_count": len(self.landmarks),
            "metadata": self.metadata,
        }


class PoseEstimator:
    """MediaPipe-based pose estimation for gesture detection."""

    # MediaPipe landmark indices
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    LANDMARK_NAMES = {
        0: "nose",
        2: "left_eye",
        5: "right_eye",
        7: "left_ear",
        8: "right_ear",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.pose = None
        self._initialized = False

        # Gesture detection thresholds
        self.hand_raise_threshold = 0.15  # Wrist above shoulder by this margin
        self.wave_motion_threshold = 0.05
        self.forward_lean_threshold = 0.1

    def initialize(self):
        """Initialize MediaPipe pose."""
        if self._initialized:
            return

        try:
            import mediapipe as mp

            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=self.model_complexity,
            )
            self._initialized = True
            print("MediaPipe Pose initialized")
        except ImportError:
            print("Warning: mediapipe not installed, using mock pose estimator")
            self._initialized = True
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            self._initialized = True

    def estimate(
        self,
        frame: np.ndarray,
        person_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[PoseResult]:
        """
        Estimate pose for a single person in frame.

        Args:
            frame: BGR image as numpy array
            person_bbox: Optional (x1, y1, x2, y2) to crop

        Returns:
            PoseResult or None if no pose detected
        """
        if not self._initialized:
            self.initialize()

        if self.pose is None:
            return self._mock_estimate(frame)

        # Crop to person if bbox provided
        if person_bbox:
            x1, y1, x2, y2 = map(int, person_bbox)
            frame = frame[y1:y2, x1:x2]

        # Convert BGR to RGB
        import cv2

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        # Extract landmarks
        landmarks = {}
        for idx, name in self.LANDMARK_NAMES.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
            )

        # Detect gestures
        gestures = self._detect_gestures(landmarks)

        # Calculate overall confidence
        confidence = sum(lm.visibility for lm in landmarks.values()) / len(landmarks)

        return PoseResult(
            landmarks=landmarks,
            gestures=gestures,
            confidence=confidence,
            bbox=person_bbox,
        )

    def estimate_batch(
        self,
        frame: np.ndarray,
        person_bboxes: List[Tuple[float, float, float, float]],
    ) -> List[Optional[PoseResult]]:
        """Estimate pose for multiple persons."""
        results = []
        for bbox in person_bboxes:
            result = self.estimate(frame, bbox)
            if result:
                result.bbox = bbox
            results.append(result)
        return results

    def _detect_gestures(self, landmarks: Dict[str, Landmark]) -> List[GestureType]:
        """Detect gestures from pose landmarks."""
        gestures = []

        # Check hand raise
        if self._is_hand_raised(landmarks):
            gestures.append(GestureType.HAND_RAISE)

        # Check leaning forward
        if self._is_leaning_forward(landmarks):
            gestures.append(GestureType.LEANING_FORWARD)

        # Check arms crossed
        if self._is_arms_crossed(landmarks):
            gestures.append(GestureType.ARMS_CROSSED)

        if not gestures:
            gestures.append(GestureType.NONE)

        return gestures

    def _is_hand_raised(self, landmarks: Dict[str, Landmark]) -> bool:
        """Check if either hand is raised above shoulder."""
        left_wrist = landmarks.get("left_wrist")
        right_wrist = landmarks.get("right_wrist")
        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")

        if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
            return False

        # Check if either wrist is significantly above its shoulder
        # Note: y increases downward, so lower y means higher position
        left_raised = (
            left_wrist.visibility > 0.5
            and left_shoulder.visibility > 0.5
            and (left_shoulder.y - left_wrist.y) > self.hand_raise_threshold
        )

        right_raised = (
            right_wrist.visibility > 0.5
            and right_shoulder.visibility > 0.5
            and (right_shoulder.y - right_wrist.y) > self.hand_raise_threshold
        )

        return left_raised or right_raised

    def _is_leaning_forward(self, landmarks: Dict[str, Landmark]) -> bool:
        """Check if person is leaning forward (looking for attention)."""
        nose = landmarks.get("nose")
        left_hip = landmarks.get("left_hip")
        right_hip = landmarks.get("right_hip")

        if not all([nose, left_hip, right_hip]):
            return False

        if nose.visibility < 0.5 or left_hip.visibility < 0.3 or right_hip.visibility < 0.3:
            return False

        # Calculate hip center
        hip_center_z = (left_hip.z + right_hip.z) / 2

        # If nose is significantly in front of hips (negative z is closer to camera)
        return (hip_center_z - nose.z) > self.forward_lean_threshold

    def _is_arms_crossed(self, landmarks: Dict[str, Landmark]) -> bool:
        """Check if arms are crossed (potential frustration indicator)."""
        left_wrist = landmarks.get("left_wrist")
        right_wrist = landmarks.get("right_wrist")
        left_elbow = landmarks.get("left_elbow")
        right_elbow = landmarks.get("right_elbow")

        if not all([left_wrist, right_wrist, left_elbow, right_elbow]):
            return False

        # Check visibility
        if any(
            lm.visibility < 0.5
            for lm in [left_wrist, right_wrist, left_elbow, right_elbow]
        ):
            return False

        # Arms crossed: wrists are on opposite sides of body
        # Left wrist should be on right side, right wrist on left side
        left_crossed = left_wrist.x > right_elbow.x
        right_crossed = right_wrist.x < left_elbow.x

        return left_crossed and right_crossed

    def _mock_estimate(self, frame: np.ndarray) -> Optional[PoseResult]:
        """Generate mock pose for testing without MediaPipe."""
        import random

        # 70% chance of detecting a pose
        if random.random() > 0.7:
            return None

        # Generate mock landmarks
        landmarks = {}
        for idx, name in self.LANDMARK_NAMES.items():
            landmarks[name] = Landmark(
                x=random.uniform(0.3, 0.7),
                y=random.uniform(0.2, 0.8),
                z=random.uniform(-0.5, 0.5),
                visibility=random.uniform(0.6, 1.0),
            )

        # Random gestures
        gestures = [GestureType.NONE]
        if random.random() > 0.8:
            gestures = [GestureType.HAND_RAISE]
        elif random.random() > 0.9:
            gestures = [GestureType.LEANING_FORWARD]

        return PoseResult(
            landmarks=landmarks,
            gestures=gestures,
            confidence=random.uniform(0.7, 0.95),
            metadata={"mock": True},
        )

    def cleanup(self):
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()
            self.pose = None
            self._initialized = False
