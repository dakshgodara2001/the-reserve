"""YOLO-based object detection for restaurant scene analysis."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box coordinates."""

    x1: float  # Top-left x
    y1: float  # Top-left y
    x2: float  # Bottom-right x
    y2: float  # Bottom-right y

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_normalized(self, frame_width: int, frame_height: int) -> "BoundingBox":
        """Convert to normalized coordinates (0-1)."""
        return BoundingBox(
            x1=self.x1 / frame_width,
            y1=self.y1 / frame_height,
            x2=self.x2 / frame_width,
            y2=self.y2 / frame_height,
        )


@dataclass
class Detection:
    """Single object detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    timestamp: datetime = field(default_factory=datetime.utcnow)
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2,
            },
            "center": self.bbox.center,
            "timestamp": self.timestamp.isoformat(),
            "track_id": self.track_id,
            "metadata": self.metadata,
        }


class ObjectDetector:
    """YOLO-based object detector for restaurant scenes."""

    # COCO class IDs relevant to restaurant
    PERSON_CLASS = 0
    BOTTLE_CLASS = 39
    WINE_GLASS_CLASS = 40
    CUP_CLASS = 41
    FORK_CLASS = 42
    KNIFE_CLASS = 43
    SPOON_CLASS = 44
    BOWL_CLASS = 45
    DINING_TABLE_CLASS = 60
    CHAIR_CLASS = 56
    CELL_PHONE_CLASS = 67

    RELEVANT_CLASSES = {
        PERSON_CLASS: "person",
        BOTTLE_CLASS: "bottle",
        WINE_GLASS_CLASS: "wine_glass",
        CUP_CLASS: "cup",
        FORK_CLASS: "fork",
        KNIFE_CLASS: "knife",
        SPOON_CLASS: "spoon",
        BOWL_CLASS: "bowl",
        DINING_TABLE_CLASS: "dining_table",
        CHAIR_CLASS: "chair",
        CELL_PHONE_CLASS: "cell_phone",
    }

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._initialized = False

    def initialize(self):
        """Load the YOLO model."""
        if self._initialized:
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            self._initialized = True
            print(f"Loaded YOLO model: {self.model_path}")
        except ImportError:
            print("Warning: ultralytics not installed, using mock detector")
            self._initialized = True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self._initialized = True

    def detect(
        self,
        frame: np.ndarray,
        filter_classes: Optional[List[int]] = None,
    ) -> List[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: BGR image as numpy array
            filter_classes: Optional list of class IDs to detect

        Returns:
            List of Detection objects
        """
        if not self._initialized:
            self.initialize()

        if self.model is None:
            # Return mock detections for testing without model
            return self._mock_detect(frame)

        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                # Filter by class if specified
                if filter_classes and class_id not in filter_classes:
                    continue

                # Skip non-relevant classes
                if class_id not in self.RELEVANT_CLASSES:
                    continue

                # Get bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()
                bbox = BoundingBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                )

                detection = Detection(
                    class_id=class_id,
                    class_name=self.RELEVANT_CLASSES.get(class_id, f"class_{class_id}"),
                    confidence=confidence,
                    bbox=bbox,
                )

                detections.append(detection)

        return detections

    def detect_persons(self, frame: np.ndarray) -> List[Detection]:
        """Detect only persons in frame."""
        return self.detect(frame, filter_classes=[self.PERSON_CLASS])

    def detect_tableware(self, frame: np.ndarray) -> List[Detection]:
        """Detect tableware items (glasses, utensils, etc.)."""
        tableware_classes = [
            self.BOTTLE_CLASS,
            self.WINE_GLASS_CLASS,
            self.CUP_CLASS,
            self.FORK_CLASS,
            self.KNIFE_CLASS,
            self.SPOON_CLASS,
            self.BOWL_CLASS,
        ]
        return self.detect(frame, filter_classes=tableware_classes)

    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        """Generate mock detections for testing without model."""
        import random

        detections = []
        h, w = frame.shape[:2]

        # Simulate detecting 1-3 persons
        num_persons = random.randint(1, 3)
        for i in range(num_persons):
            # Random position
            x1 = random.uniform(0.1, 0.7) * w
            y1 = random.uniform(0.1, 0.5) * h
            width = random.uniform(0.1, 0.2) * w
            height = random.uniform(0.3, 0.5) * h

            detection = Detection(
                class_id=self.PERSON_CLASS,
                class_name="person",
                confidence=random.uniform(0.7, 0.95),
                bbox=BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x1 + width,
                    y2=y1 + height,
                ),
                metadata={"mock": True},
            )
            detections.append(detection)

        return detections

    def get_table_regions(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
        table_positions: List[Tuple[float, float]],
        radius: float = 0.15,
    ) -> Dict[int, List[Detection]]:
        """
        Group detections by their proximity to table positions.

        Args:
            detections: List of detections
            frame_width: Width of frame
            frame_height: Height of frame
            table_positions: List of (x, y) normalized table positions
            radius: Normalized radius for grouping

        Returns:
            Dict mapping table index to list of detections
        """
        table_detections: Dict[int, List[Detection]] = {
            i: [] for i in range(len(table_positions))
        }

        for detection in detections:
            # Get normalized center
            norm_bbox = detection.bbox.to_normalized(frame_width, frame_height)
            cx, cy = norm_bbox.center

            # Find closest table
            min_dist = float("inf")
            closest_table = -1

            for i, (tx, ty) in enumerate(table_positions):
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < min_dist and dist < radius:
                    min_dist = dist
                    closest_table = i

            if closest_table >= 0:
                table_detections[closest_table].append(detection)

        return table_detections
