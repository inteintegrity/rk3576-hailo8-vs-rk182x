"""Host-side decode primitives shared by all three accelerator runners.

Every accelerator in this comparison stops at the same six raw detection heads, so
sigmoid, the box mapping and NMS must be the *single* implementation in this module.
No accelerator-specific decode is allowed, or the three results stop being comparable.

Only NumPy is required (letterbox pulls in OpenCV lazily), so the module also runs on
the RK3576 board itself.
"""

from __future__ import annotations

import numpy as np

STRIDES = (8, 16, 32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def nms_class_aware(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thres: float,
    max_wh: float = 7680.0,
) -> np.ndarray:
    """Greedy NMS with the Ultralytics class offset so classes never suppress each other."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    offsets = class_ids.astype(np.float32) * max_wh
    shifted = boxes + offsets[:, None]
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(shifted[current, 0], shifted[rest, 0])
        yy1 = np.maximum(shifted[current, 1], shifted[rest, 1])
        xx2 = np.minimum(shifted[current, 2], shifted[rest, 2])
        yy2 = np.minimum(shifted[current, 3], shifted[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_current = (shifted[current, 2] - shifted[current, 0]) * (
            shifted[current, 3] - shifted[current, 1]
        )
        area_rest = (shifted[rest, 2] - shifted[rest, 0]) * (shifted[rest, 3] - shifted[rest, 1])
        union = area_current + area_rest - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def letterbox(image: np.ndarray, size: int = 640, pad_value: int = 114):
    """Ultralytics-style letterbox; returns (padded, scale, pad_left, pad_top)."""
    import cv2

    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        size - new_height - top,
        left,
        size - new_width - left,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return padded, scale, left, top


def unletterbox_boxes(boxes: np.ndarray, scale: float, left: int, top: int) -> np.ndarray:
    """Map boxes from letterboxed pixels back to original image pixels."""
    converted = boxes.copy().astype(np.float32)
    converted[:, [0, 2]] = (converted[:, [0, 2]] - left) / scale
    converted[:, [1, 3]] = (converted[:, [1, 3]] - top) / scale
    return converted


def to_network_input(padded: np.ndarray) -> np.ndarray:
    """Letterboxed BGR frame -> the (1, 640, 640, 3) RGB uint8 tensor every runner submits."""
    import cv2

    return np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]