#!/usr/bin/env python3
"""
Navigation Library
A library for autonomous Mars rover navigation with visual target recognition.

Developed under the grant: "Development of open-source libraries for autonomous navigation
of a six-wheeled Mars rover prototype in moderately rugged, unfamiliar terrain with visual target recognition."
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
    Navigation object recognition result.

    Attributes:
    object_type: Object type ("arrow" for arrow, "cone" for cone)
    direction: Direction ("left", "right", "none")
    distance_m: Distance to object in meters
    angle_deg: Angle relative to the camera center in degrees
    confidence: Detection confidence (0.0 - 1.0)
    bbox: Bounding rectangle (x1, y1, x2, y2)
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
    Navigation object detector for an autonomous Mars rover.

    Provides recognition of arrows and traffic cones with calibrated
    distance and angle measurements for transmission to the localization module.
    """

    # Filtering constants
    CONF_THRESHOLD = 0.5
    NMS_IOU_THRESHOLD = 0.4
    MIN_BOX_SIZE = 2
    MAX_BOX_SIZE = 600

    def __init__(self, weights_path: str, device: str = None):
       
        """
        Detector initialization.

        Args:
        weights_path: Path to the YOLO model weights file (.pt)
        device: Inference device ('cuda', 'cpu', or None for autodetection)
        """
        self.model = YOLO(str(weights_path))

        # Auto-detect device if not explicitly specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # We explicitly transfer the model to the selected device
        self.model.to(self.device)

        self._image_center_x = 320  # Default for 640x480

    def detect_arrow(self, image: np.ndarray) -> List[DetectionResult]:
                
        """
        Arrow detection with direction determination.

        This function detects arrows in an image and determines their direction (left/right) for rover navigation.

        Args:
        image: Input image in BGR format (numpy array)

        Returns:
        List of DetectionResult objects with arrows.
        The direction field contains "left" or "right".
        """
        all_detections = self._detect_objects(image)
        return [d for d in all_detections if d.object_type == "arrow"]

    def detect_cone(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Traffic cone recognition.

        This function detects traffic cones in an image to determine
        navigation targets.

        Args:
        image: Input image in BGR format (numpy array)

        Returns:
        List of DetectionResult objects with cones.
        The direction field for cones is always "none".
        """
        all_detections = self._detect_objects(image)
        return [d for d in all_detections if d.object_type == "cone"]

    def detect_all(self, image: np.ndarray) -> List[DetectionResult]:
                
        """
        Recognize all navigational objects (arrows and cones).

        Args:
        image: Input image in BGR format (numpy array)

        Returns:
        List of all detected objects DetectionResult
        """
        return self._detect_objects(image)

    # ───────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ───────────────────────────────────────────────────────────────

    def _detect_objects(self, image: np.ndarray) -> List[DetectionResult]:
        """Basic logic of object detection."""
        h, w = image.shape[:2]
        self._image_center_x = w / 2

        # Launch YOLO on a given device
        results = self.model(image, device=self.device)

        # Removing boxes
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

        # Formation of results
        detections = []
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1

            if w_box <= 0 or h_box <= 0:
                continue

            # Determining the object type (assuming: class 0 = arrow, class 1 = cone)
            object_type = "arrow" if cls == 0 else "cone"

            # Determining the direction of arrows
            direction = "none"
            if object_type == "arrow":
                roi = image[y1:y2, x1:x2]
                direction = self._arrow_direction_pca(roi)
                if direction is None:
                    direction = "none"

            # Calibrated distance measurement (taking into account the type of object)
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
        """Filter boxes by confidence, size, and NMS."""
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
        """Calculating IoU between two boxes."""
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
        """Non-Maximum Suppression для удаления перекрывающихся боксов."""
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
        Determines arrow direction using PCA with heuristics.

        Uses majority voting with three heuristics:

        1. Mass distribution (the tip has more pixels)

        2. Width gradient (the tip is wider)

        3. Pointiness (the sharpest angle)
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

        # PCA to find the principal axis
        mean, evecs = cv2.PCACompute(pts, mean=None)
        long_ax, ortho = evecs

        proj = (pts - mean) @ long_ax
        i_min, i_max = np.argmin(proj), np.argmax(proj)
        p_min, p_max = pts[i_min], pts[i_max]

        votes = []

        # Heuristic 1: Mass Distribution
        left_side = pts[pts[:, 0] < mean[0, 0]]
        right_side = pts[pts[:, 0] >= mean[0, 0]]

        if len(left_side) > 0 and len(right_side) > 0:
            if len(left_side) > len(right_side):
                votes.append('left')
            else:
                votes.append('right')

        # Heuristic 2: Width Gradient
        num_samples = 5
        proj_sorted = np.sort(proj)
        widths = []
        for i in range(num_samples):
            idx = min(int(len(proj_sorted) * i / (num_samples - 1)), len(proj_sorted) - 1)
            proj_val = proj_sorted[idx]
            strip = np.abs(proj - proj_val) < 2.0
            if strip.any():
                width = np.ptp((pts[strip] - mean) @ ortho)
                widths.append(width)

        if len(widths) >= 3:
            x = np.arange(len(widths))
            slope = np.polyfit(x, widths, 1)[0]
            if slope > 0:
                votes.append('right')
            else:
                votes.append('left')

        # Heuristic 3: Pointiness
        hull = cv2.convexHull(pts.astype(np.int32), returnPoints=True)
        hull_pts = hull.reshape(-1, 2).astype(np.float32)

        def find_hull_angle(p_extreme):
            dists = np.linalg.norm(hull_pts - p_extreme, axis=1)
            idx = np.argmin(dists)
            n = len(hull_pts)
            p1 = hull_pts[(idx - 1) % n]
            p2 = hull_pts[idx]
            p3 = hull_pts[(idx + 1) % n]
            v1 = p1 - p2
            v2 = p3 - p2
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm > 0:
                angle = np.arccos(np.clip(dot / norm, -1.0, 1.0))
                return np.degrees(angle)
            return 180.0

        angle_min = find_hull_angle(p_min)
        angle_max = find_hull_angle(p_max)

        if angle_min < angle_max:
            votes.append('left' if p_min[0] < p_max[0] else 'right')
        else:
            votes.append('left' if p_max[0] < p_min[0] else 'right')

        # Majority voting
        if not votes:
            return "right" if p_max[0] > p_min[0] else "left"

        left_votes = votes.count('left')
        right_votes = votes.count('right')
        return 'left' if left_votes > right_votes else 'right'
