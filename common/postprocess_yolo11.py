"""Shared host-side decode for the YOLO11n raw detection heads.

Both accelerators end at the same six raw head tensors (see
model/conversion_manifest.json -> graph_boundary), so sigmoid, DFL decode, anchor
decode and NMS must be the single implementation in this module. The Hailo-8 and
RK1820 runners import it as-is; no accelerator-specific decode is allowed, or the
two results stop being comparable.

Head layout per scale (stride 8/16/32):
    box branch  (H, W, 4 * REG_MAX)  -> DFL distribution over left/top/right/bottom
    cls branch  (H, W, num_classes)  -> raw logits, sigmoid applied here

Only NumPy is required, so the module also runs on the RK3576 board.
"""

from __future__ import annotations

import numpy as np

REG_MAX = 16
STRIDES = (8, 16, 32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=axis, keepdims=True)


def dfl_distances(box_logits: np.ndarray, reg_max: int = REG_MAX) -> np.ndarray:
    """(N, 4 * reg_max) logits -> (N, 4) distances in grid units."""
    reshaped = np.asarray(box_logits, dtype=np.float32).reshape(-1, 4, reg_max)
    probabilities = softmax(reshaped, axis=-1)
    project = np.arange(reg_max, dtype=np.float32)
    return np.sum(probabilities * project, axis=-1)


def decode_detections(
    box_heads: list[np.ndarray],
    cls_heads: list[np.ndarray],
    strides: tuple[int, ...] = STRIDES,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_detections: int = 300,
):
    """Decode all scales, filter by confidence and run class-aware NMS.

    Returns (boxes xyxy in letterboxed 640x640 pixels, scores, class_ids).

    Class scores are argmax-ed on the raw logits and only the surviving anchors go
    through the DFL softmax. Sigmoid is monotonic and DFL is per-anchor, so this is
    numerically identical to decoding every anchor and filtering afterwards, but it
    keeps the host-side decode from dominating the measured latency.
    """
    if len(box_heads) != len(cls_heads) or len(box_heads) != len(strides):
        raise ValueError("box heads, class heads and strides must describe the same scales")
    boxes_all, scores_all, classes_all = [], [], []
    for box_head, cls_head, stride in zip(box_heads, cls_heads, strides):
        width = cls_head.shape[1]
        logits = np.asarray(cls_head, dtype=np.float32).reshape(-1, cls_head.shape[-1])
        class_ids = np.argmax(logits, axis=1)
        scores = sigmoid(logits[np.arange(logits.shape[0]), class_ids])
        keep = np.flatnonzero(scores >= conf_thres)
        if keep.size == 0:
            continue

        rows, columns = np.divmod(keep, width)
        anchor_x = columns.astype(np.float32) + 0.5
        anchor_y = rows.astype(np.float32) + 0.5
        selected = np.asarray(box_head, dtype=np.float32).reshape(-1, box_head.shape[-1])[keep]
        distances = dfl_distances(selected)
        boxes_all.append(
            np.stack(
                [
                    (anchor_x - distances[:, 0]) * stride,
                    (anchor_y - distances[:, 1]) * stride,
                    (anchor_x + distances[:, 2]) * stride,
                    (anchor_y + distances[:, 3]) * stride,
                ],
                axis=-1,
            )
        )
        scores_all.append(scores[keep])
        classes_all.append(class_ids[keep])

    if not boxes_all:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    class_ids = np.concatenate(classes_all, axis=0)
    if scores.size > max_detections:
        top = np.argsort(scores)[::-1][:max_detections]
        boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]

    selected = nms_class_aware(boxes, scores, class_ids, iou_thres)
    return boxes[selected], scores[selected], class_ids[selected]


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
    """Ultralytics-style letterbox, identical to common/prepare_calibration.py."""
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