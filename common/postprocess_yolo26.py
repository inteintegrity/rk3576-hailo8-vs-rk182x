"""Shared host-side decode for YOLO26 raw detection heads.

YOLO26 differs from YOLO11 in one structural way: the box branch emits four direct
values (left, top, right, bottom distances in grid units) instead of a DFL
distribution over 4 * reg_max channels, and the exported graph carries the
one2one (end-to-end) heads. Everything else matches, so this module reuses the
letterbox, NMS and box-mapping helpers from postprocess_yolo11.py and only replaces
the head decode. Both accelerators that serve a YOLO26 graph use this one
implementation, exactly as they share the YOLO11 one.

Head layout per scale (stride 8/16/32), per Rockchip's official YOLO26 example:
    box head    (H, W, 4)   -> distances, no DFL
    score head  (H, W, 80)  -> raw logits, sigmoid applied here

Reference for the decode convention: rknn3-model-zoo examples/yolo26/python/infer.py
    x1 = (-dist_left + col + 0.5) * stride,  y1 = (-dist_top + row + 0.5) * stride
    x2 = ( dist_right + col + 0.5) * stride, y2 = ( dist_bottom + row + 0.5) * stride
"""

from __future__ import annotations

import numpy as np

from postprocess_yolo11 import STRIDES, nms_class_aware, sigmoid

BOX_CHANNELS = 4
SCORE_CHANNELS = 80


def distances_from_box_head(box_head: np.ndarray) -> np.ndarray:
    """(N, 4) direct distances; YOLO26 needs no DFL softmax."""
    return np.asarray(box_head, dtype=np.float32).reshape(-1, BOX_CHANNELS)


def decode_detections_yolo26(
    box_heads: list[np.ndarray],
    score_heads: list[np.ndarray],
    strides: tuple[int, ...] = STRIDES,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_detections: int = 300,
):
    """Decode all scales, filter by confidence and run the shared class-aware NMS."""
    if len(box_heads) != len(score_heads) or len(box_heads) != len(strides):
        raise ValueError("box heads, score heads and strides must describe the same scales")

    boxes_all, scores_all, classes_all = [], [], []
    for box_head, score_head, stride in zip(box_heads, score_heads, strides):
        width = score_head.shape[1]
        logits = np.asarray(score_head, dtype=np.float32).reshape(-1, score_head.shape[-1])
        class_ids = np.argmax(logits, axis=1)
        scores = sigmoid(logits[np.arange(logits.shape[0]), class_ids])
        keep = np.flatnonzero(scores >= conf_thres)
        if keep.size == 0:
            continue

        rows, columns = np.divmod(keep, width)
        anchor_x = columns.astype(np.float32) + 0.5
        anchor_y = rows.astype(np.float32) + 0.5
        distances = distances_from_box_head(
            np.asarray(box_head, dtype=np.float32).reshape(-1, BOX_CHANNELS)[keep]
        )
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
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    class_ids = np.concatenate(classes_all, axis=0)
    if scores.size > max_detections:
        top = np.argsort(scores)[::-1][:max_detections]
        boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]

    selected = nms_class_aware(boxes, scores, class_ids, iou_thres)
    return boxes[selected], scores[selected], class_ids[selected]