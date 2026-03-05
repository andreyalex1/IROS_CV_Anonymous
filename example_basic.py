#!/usr/bin/env python3
"""
Basic example of using the computer vision navigation library.
Detection of arrows and cones in an image.

Usage:
    1. Place an image containing an arrow or cone in the root directory
       of the repository and name it test_image.jpg,
       or pass the path to the image as a command line argument.
    2. Run the script from the repository root:
       python3 example_basic.py
       python3 example_basic.py path/to/your_image.jpg
"""

import cv2
import os
import sys
from pathlib import Path

# Add the library path
sys.path.append(str(Path(__file__).parent))

os.environ["YOLO_VERBOSE"] = "False"  # Suppress YOLO debug output
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"  # Suppress OpenCV warnings
from eureka_nav_lib import NavigationDetector


def main():
    # Step 1: Initialize the detector
    print("Инициализация детектора...")
    weights = Path(__file__).parent / "weights" / "best.pt"
    detector = NavigationDetector(
        weights_path=str(weights),
        device=None  # Automatic device selection (GPU if available)
    )

    # Step 2: Load the image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = str(Path(__file__).parent / "test_image.jpg")

    print(f"Загрузка изображения: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Ошибка: не удалось загрузить изображение '{image_path}'")
        print("Place an image containing an arrow or cone in the root")
        print("directory of the repository named test_image.jpg,")
        print("or pass the path as an argument: python3 example_basic.py <path>")
        return

    # Step 3: Detect all objects
    print("Выполнение детекции...")
    detections = detector.detect_all(image)

    # Step 4: Print results
    print(f"\nОбнаружено объектов: {len(detections)}")
    print("-" * 60)

    for i, det in enumerate(detections, 1):
        print(f"Объект #{i}:")
        print(f"  Тип: {det.object_type}")
        print(f"  Направление: {det.direction}")
        print(f"  Расстояние: {det.distance_m:.2f} м")
        print(f"  Угол: {det.angle_deg:.1f}°")
        print(f"  Уверенность: {det.confidence:.2%}")
        print(f"  Координаты: {det.bbox}")
        print()

    # Step 5: Visualize results
    for det in detections:
        x1, y1, x2, y2 = det.bbox

        # Choose color depending on object type
        color = (0, 255, 0) if det.object_type == "arrow" else (0, 165, 255)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = f"{det.object_type}"
        if det.direction != "none":
            label += f" {det.direction}"
        label += f" {det.distance_m:.1f}m"

        # Draw label text
        cv2.putText(image, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save result
    output_path = str(Path(__file__).parent / "result.jpg")
    cv2.imwrite(output_path, image)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()