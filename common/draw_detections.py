"""Shared annotation and timing helpers for the Hailo-8 and RK1820 runners.

Keeping the drawing code in one place guarantees the two result images are visually
identical, so a difference in the picture is a difference in detections rather than
a difference in how the two runners render them.
"""

from __future__ import annotations

import statistics

import cv2
import numpy as np

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def summarise(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
    }


def draw(image: np.ndarray, detections, banner: list[str]) -> np.ndarray:
    """Draw boxes, class labels and a short banner of timing lines onto a copy of the image."""
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection["box_xyxy"])
        colour = PALETTE[detection["class_id"] % len(PALETTE)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3)
        label = f"{detection['class_name']} {detection['confidence']:.3f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(y1 - text_height - baseline - 4, 0)
        cv2.rectangle(canvas, (x1, top), (x1 + text_width + 4, top + text_height + baseline + 4), colour, -1)
        cv2.putText(
            canvas, label, (x1 + 2, top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
    if banner:
        font, scale, thickness, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, 10
        sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text in banner]
        line_height = max(size[1] for size in sizes) + 16
        strip_width = max(size[0] for size in sizes) + 2 * pad
        strip_height = line_height * len(banner) + pad
        cv2.rectangle(canvas, (0, 0), (strip_width, strip_height), (0, 0, 0), -1)
        for row, (text, size) in enumerate(zip(banner, sizes)):
            cv2.putText(
                canvas, text, (pad, line_height * row + size[1] + pad),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
            )
    return canvas