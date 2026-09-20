"""Helpers shared by the two Hailo-8 runners (video file and live camera).

Both runners feed the same HEF and must interpret its output tensors the same way, so the
squashing/identification logic lives here rather than in one of them. Import by this module's
own name: the board's script directory also holds older runners whose file names collide.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_hwc(array: np.ndarray, expected_channels: int, name: str) -> np.ndarray:
    squeezed = np.squeeze(array)
    if squeezed.ndim != 3:
        raise RuntimeError(f"{name}: unexpected output rank {array.shape}")
    if squeezed.shape[-1] == expected_channels:
        return squeezed
    if squeezed.shape[0] == expected_channels:
        return np.transpose(squeezed, (1, 2, 0))
    raise RuntimeError(f"{name}: shape {array.shape} does not match {expected_channels} channels")


def collect_heads(raw_outputs: dict):
    """Identify branch and stride from tensor shape, not layer name.

    YOLO11 heads are 64-channel DFL boxes plus 80-channel class logits; YOLO26 heads are
    4-channel direct boxes plus 80-channel score logits. HEF names differ between the two
    (conv51..conv80 vs conv61..conv94), shapes do not.
    """
    strides = {80: 8, 40: 16, 20: 32}
    by_stride: dict[int, dict[str, np.ndarray]] = {}
    for name, array in raw_outputs.items():
        squeezed = np.squeeze(np.asarray(array))
        if squeezed.ndim != 3:
            raise RuntimeError(f"unexpected HEF output rank for {name}: {np.asarray(array).shape}")
        if squeezed.shape[-1] in (4, 64, 80):
            height, width, channels = squeezed.shape
            head = squeezed
        elif squeezed.shape[0] in (4, 64, 80):
            channels, height, width = squeezed.shape
            head = np.transpose(squeezed, (1, 2, 0))
        else:
            raise RuntimeError(f"cannot identify layout of {name}: {np.asarray(array).shape}")
        if height not in strides:
            raise RuntimeError(f"unexpected spatial size {height} for {name}")
        branch = "box" if channels in (4, 64) else "cls"
        by_stride.setdefault(strides[height], {})[branch] = head
    box_heads, cls_heads = [], []
    for stride in sorted(by_stride):
        if set(by_stride[stride]) != {"box", "cls"}:
            raise RuntimeError(f"stride {stride}: incomplete head pair {sorted(by_stride[stride])}")
        box_heads.append(by_stride[stride]["box"])
        cls_heads.append(by_stride[stride]["cls"])
    if len(box_heads) != 3:
        raise RuntimeError(f"expected 3 scales, found {sorted(by_stride)}")
    return box_heads, cls_heads