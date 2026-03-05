#!/usr/bin/env python3
"""
Eureka Navigation Library
Library for autonomous rover navigation with visual target recognition.

Developed as part of a research project on open libraries for autonomous
navigation of a six-wheeled rover prototype in moderately rough unknown
terrain with visual recognition of navigation targets.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from calibration_config import (
    get_distance_from_pixels,
    get_angle_from_position
)


# ═══════════════════════════════════════════════════════════════════
# PUBLIC DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    """
    Result of navigation object detection.

    Attributes:
        object_type: Type of object ("arrow" for arrow, "cone" for cone)
        direction: Direction ("left", "right", "none")
        distance_m: Distance to the object in meters
        angle_deg: Angle relative to the camera center in degrees
        confidence: Detection confidence (0.0 - 1.0)
        bbox: Bounding box (x1, y1, x2, y2)
    """
    object_type: str
    direction: str
    distance_m: float
    angle_deg: float
    confidence: float
    bbox: Tuple[int, int, int, int]


# ═══════════════════════════════════════════════════════════════════
# MAIN LIBRARY CLASS
# ═══════════════════════════════════════════════════════════════════

class NavigationDetector:
    """
    Detector of navigation objects for an autonomous rover.

    Provides detection of arrows and traffic cones with calibrated
    measurement of distance and angle for integration with a localization module.
    """

    # Filtering constants
    CONF_THRESHOLD = 0.5
    NMS_IOU_THRESHOLD = 0.4
    MIN_BOX_SIZE = 2
    MAX_BOX_SIZE = 600

    def __init__(self, weights_path: str, device: str = None):
        """
        Initialize the detector.

        Args:
            weights_path: Path to the YOLO model weights file (.pt)
            device: Inference device ('cuda', 'cpu', or None for automatic selection)
        """
        self.model = YOLO(str(weights_path))

        # Automatically determine the device if not explicitly specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Explicitly move the model to the selected device
        self.model.to(self.device)

        self._image_center_x = 320  # Default value for 640x480 images

    def detect_arrow(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect arrows and determine their direction.

        The function detects arrows in the image and determines their direction
        (left/right) for rover navigation.

        Args:
            image: Input image in BGR format (numpy array)

        Returns:
            List of DetectionResult objects containing arrows.
            The direction field contains "left" or "right".
        """
        all_detections = self._detect_objects(image)
        return [d for d in all_detections if d.object_type == "arrow"]

    def detect_cone(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect traffic cones.

        The function detects traffic cones in the image to identify
        navigation targets.

        Args:
            image: Input image in BGR format (numpy array)

        Returns:
            List of DetectionResult objects containing cones.
            The direction field for cones is always "none".
        """
        all_detections = self._detect_objects(image)
        return [d for d in all_detections if d.object_type == "cone"]

    def detect_all(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Detect all navigation objects (arrows and cones).

        Args:
            image: Input image in BGR format (numpy array)

        Returns:
            List of all detected DetectionResult objects
        """
        return self._detect_objects(image)

    # ───────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ───────────────────────────────────────────────────────────────

    def _detect_objects(self, image: np.ndarray) -> List[DetectionResult]:
        """Core object detection logic."""
        h, w = image.shape[:2]
        self._image_center_x = w / 2

        # Run YOLO inference on the selected device
        results = self.model(image, device=self.device)

        # Extract bounding boxes
        boxes = []
        confs = []
        classes = []

        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
            confs.append(float(b.conf.item()))
            classes.append(int(b.cls.item()))

        # Filtering and NMS
        boxes, confs, classes = self._filter_boxes(boxes, confs, classes)

        # Build detection results
        detections = []
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1

            if w_box <= 0 or h_box <= 0:
                continue

            # Determine object type (assumption: class 0 = arrow, class 1 = cone)
            object_type = "arrow" if cls == 0 else "cone"

            # Determine direction for arrows
            direction = "none"
            if object_type == "arrow":
                roi = image[y1:y2, x1:x2]
                direction = self._arrow_direction_pca(roi)
                if direction is None:
                    direction = "none"

            # Calibrated distance measurement (depends on object type)
            distance_m, _ = get_distance_from_pixels(w_box, h_box, use_width=True, object_type=object_type)

            # Angle measurement
            bx = (x1 + x2) / 2
            angle_deg = get_angle_from_position(bx, self._image_center_x)

            detection = DetectionResult(
                object_type=object_type,
                direction=direction,
                distance_m=distance_m,
                angle_deg=angle_deg,
                confidence=conf,
                bbox=box
            )
            detections.append(detection)

        return detections

    def _filter_boxes(self, boxes, confs, classes):
        """Filter bounding boxes by confidence, size, and apply NMS."""
        filtered_boxes = []
        filtered_confs = []
        filtered_classes = []

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            if (conf >= self.CONF_THRESHOLD and
                self.MIN_BOX_SIZE <= w <= self.MAX_BOX_SIZE and
                self.MIN_BOX_SIZE <= h <= self.MAX_BOX_SIZE):
                filtered_boxes.append(box)
                filtered_confs.append(conf)
                filtered_classes.append(cls)

        if filtered_boxes:
            filtered_boxes, filtered_confs, filtered_classes = self._non_maximum_suppression(
                filtered_boxes, filtered_confs, filtered_classes
            )

        return filtered_boxes, filtered_confs, filtered_classes

    @staticmethod
    def _compute_iou(box1: Tuple[int, int, int, int],
                     box2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _non_maximum_suppression(self, boxes, confs, classes):
        """Non-Maximum Suppression to remove overlapping boxes."""
        if not boxes:
            return [], [], []

        sorted_indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
        keep_boxes = []
        keep_confs = []
        keep_classes = []

        while sorted_indices:
            idx = sorted_indices[0]
            keep_boxes.append(boxes[idx])
            keep_confs.append(confs[idx])
            keep_classes.append(classes[idx])

            remaining = []
            for other_idx in sorted_indices[1:]:
                iou = self._compute_iou(boxes[idx], boxes[other_idx])
                if iou < self.NMS_IOU_THRESHOLD:
                    remaining.append(other_idx)

            sorted_indices = remaining

        return keep_boxes, keep_confs, keep_classes

    @staticmethod
    def _arrow_direction_pca(roi: np.ndarray) -> Optional[str]:
        """
        Determine arrow direction using PCA with heuristics.

        Uses majority voting from three heuristics:
        1. Mass distribution (arrowhead contains more pixels)
        2. Width gradient (arrowhead is wider)
        3. Sharpness (the sharpest angle indicates the tip)
        """
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not cnts:
            return None

        pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return None

        # PCA to determine the principal axis
        mean, evecs = cv2.PCACompute(pts, mean=None)
        long_ax, ortho = evecs