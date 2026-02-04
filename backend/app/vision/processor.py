"""Video stream processor for continuous analysis."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Callable, Dict, Any
from pathlib import Path
import numpy as np

from .detector import ObjectDetector, Detection
from .pose import PoseEstimator, GestureType, PoseResult
from .tracker import PersonTracker, TrackedPerson


@dataclass
class FrameResult:
    """Result of processing a single frame."""

    frame_number: int
    timestamp: datetime
    detections: List[Detection]
    tracked_persons: List[TrackedPerson]
    pose_results: List[PoseResult]
    gestures_detected: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp.isoformat(),
            "detection_count": len(self.detections),
            "tracked_count": len(self.tracked_persons),
            "gestures": self.gestures_detected,
            "metadata": self.metadata,
        }


class VideoProcessor:
    """
    Processes video streams for person detection, tracking, and gesture recognition.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        pose_estimator: Optional[PoseEstimator] = None,
        tracker: Optional[PersonTracker] = None,
    ):
        self.detector = detector or ObjectDetector()
        self.pose_estimator = pose_estimator or PoseEstimator()
        self.tracker = tracker or PersonTracker()

        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.fps = 0.0

        # Callbacks for events
        self._on_gesture_callbacks: List[Callable] = []
        self._on_person_callbacks: List[Callable] = []
        self._on_frame_callbacks: List[Callable] = []

        # Processing settings
        self.process_every_n_frames = 3  # Skip frames for performance
        self.enable_pose = True
        self.enable_tracking = True

    def initialize(self):
        """Initialize all vision components."""
        self.detector.initialize()
        if self.enable_pose:
            self.pose_estimator.initialize()

    async def process_video_file(
        self,
        video_path: str,
        callback: Optional[Callable[[FrameResult], None]] = None,
        max_frames: Optional[int] = None,
    ) -> List[FrameResult]:
        """
        Process a video file asynchronously.

        Args:
            video_path: Path to video file
            callback: Optional callback for each frame result
            max_frames: Optional maximum frames to process

        Returns:
            List of frame results
        """
        try:
            import cv2
        except ImportError:
            print("OpenCV not available, using mock processing")
            return await self._mock_process_video(video_path, max_frames)

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.initialize()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results = []
        self.is_running = True
        self.frame_count = 0

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # Skip frames for performance
                if self.frame_count % self.process_every_n_frames != 0:
                    continue

                # Process frame
                result = await self._process_frame(frame)
                results.append(result)

                # Call callback if provided
                if callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)

                # Check max frames
                if max_frames and self.frame_count >= max_frames:
                    break

                # Yield to event loop
                await asyncio.sleep(0)

        finally:
            cap.release()
            self.is_running = False

        return results

    async def process_webcam(
        self,
        camera_index: int = 0,
        callback: Optional[Callable[[FrameResult], None]] = None,
    ):
        """
        Process webcam stream.

        Args:
            camera_index: Camera device index
            callback: Callback for each frame result
        """
        try:
            import cv2
        except ImportError:
            print("OpenCV not available")
            return

        self.initialize()

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_index}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.is_running = True
        self.frame_count = 0

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                self.frame_count += 1

                # Skip frames for performance
                if self.frame_count % self.process_every_n_frames != 0:
                    await asyncio.sleep(0.01)
                    continue

                # Process frame
                result = await self._process_frame(frame)

                # Call callback
                if callback:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)

                # Small delay to not hog CPU
                await asyncio.sleep(0.01)

        finally:
            cap.release()
            self.is_running = False

    async def _process_frame(self, frame: np.ndarray) -> FrameResult:
        """Process a single frame."""
        timestamp = datetime.utcnow()
        gestures_detected = []

        # 1. Object detection
        detections = self.detector.detect_persons(frame)

        # 2. Update tracker
        tracked_persons = []
        if self.enable_tracking and detections:
            det_list = [
                (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.confidence)
                for d in detections
            ]
            tracked_persons = self.tracker.update(det_list)

            # Link detections to tracks
            for det in detections:
                for track in tracked_persons:
                    if self._boxes_overlap(
                        (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2),
                        track.bbox,
                    ):
                        det.track_id = track.track_id
                        break

        # 3. Pose estimation for detected persons
        pose_results = []
        if self.enable_pose and detections:
            for det in detections:
                bbox = (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)
                pose_result = self.pose_estimator.estimate(frame, bbox)

                if pose_result:
                    pose_result.person_id = det.track_id
                    pose_results.append(pose_result)

                    # Check for significant gestures
                    for gesture in pose_result.gestures:
                        if gesture != GestureType.NONE:
                            gesture_data = {
                                "type": gesture.value,
                                "person_id": det.track_id,
                                "confidence": pose_result.confidence,
                                "timestamp": timestamp.isoformat(),
                                "bbox": bbox,
                            }
                            gestures_detected.append(gesture_data)

                            # Trigger callbacks
                            await self._emit_gesture(gesture_data)

        # Emit frame callbacks
        result = FrameResult(
            frame_number=self.frame_count,
            timestamp=timestamp,
            detections=detections,
            tracked_persons=tracked_persons,
            pose_results=pose_results,
            gestures_detected=gestures_detected,
        )

        await self._emit_frame(result)

        return result

    def _boxes_overlap(
        self,
        box1: tuple,
        box2: tuple,
        threshold: float = 0.5,
    ) -> bool:
        """Check if two boxes overlap significantly."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])

        return (intersection / area1) > threshold if area1 > 0 else False

    def on_gesture(self, callback: Callable):
        """Register callback for gesture detection."""
        self._on_gesture_callbacks.append(callback)

    def on_person(self, callback: Callable):
        """Register callback for person detection."""
        self._on_person_callbacks.append(callback)

    def on_frame(self, callback: Callable):
        """Register callback for each processed frame."""
        self._on_frame_callbacks.append(callback)

    async def _emit_gesture(self, gesture_data: dict):
        """Emit gesture event to callbacks."""
        for callback in self._on_gesture_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(gesture_data)
                else:
                    callback(gesture_data)
            except Exception as e:
                print(f"Gesture callback error: {e}")

    async def _emit_frame(self, result: FrameResult):
        """Emit frame result to callbacks."""
        for callback in self._on_frame_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                print(f"Frame callback error: {e}")

    def stop(self):
        """Stop video processing."""
        self.is_running = False

    async def _mock_process_video(
        self, video_path: str, max_frames: Optional[int]
    ) -> List[FrameResult]:
        """Generate mock results for testing without OpenCV."""
        import random

        results = []
        frames_to_process = max_frames or 100

        for i in range(frames_to_process):
            # Mock frame as random noise
            mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            detections = self.detector.detect(mock_frame)
            pose_results = []
            gestures_detected = []

            # Occasionally add a hand raise
            if random.random() > 0.9:
                gestures_detected.append(
                    {
                        "type": "hand_raise",
                        "person_id": random.randint(1, 5),
                        "confidence": random.uniform(0.7, 0.95),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            result = FrameResult(
                frame_number=i,
                timestamp=datetime.utcnow(),
                detections=detections,
                tracked_persons=[],
                pose_results=pose_results,
                gestures_detected=gestures_detected,
                metadata={"mock": True},
            )
            results.append(result)

            await asyncio.sleep(0.01)

        return results

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        self.pose_estimator.cleanup()
        self.tracker.reset()
