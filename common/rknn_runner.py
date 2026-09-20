"""Shared RKNN helpers for the RKNN3 (RK1820) and RK3576 NPU runners.

Both Rockchip toolchains end at the same six raw detection heads, so head collection
and dequantization must be identical for the two of them, exactly as they are shared
with the Hailo-8 runner through common/postprocess_yolo11.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

EXPECTED_SCALES = {80: 8, 40: 16, 20: 32}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_heads(outputs):
    """Map the six RKNN outputs onto per-scale box/class heads without trusting order.

    Each output must be a (64, H, W) box head or an (80, H, W) class head for H in
    {80, 40, 20}; the channel count identifies the branch and H identifies the stride.
    """
    by_stride: dict[int, dict[str, np.ndarray]] = {}
    described = []
    for index, array in enumerate(outputs):
        squeezed = np.squeeze(np.asarray(array))
        if squeezed.ndim != 3:
            raise RuntimeError(f"output {index}: unexpected rank {array.shape}")
        if squeezed.shape[0] in (64, 80) and squeezed.shape[1] == squeezed.shape[2]:
            channels, height, width = squeezed.shape
            head = np.transpose(squeezed, (1, 2, 0))
        elif squeezed.shape[-1] in (64, 80) and squeezed.shape[0] == squeezed.shape[1]:
            height, width, channels = squeezed.shape
            head = squeezed
        else:
            raise RuntimeError(f"output {index}: cannot identify layout of {array.shape}")
        if height not in EXPECTED_SCALES:
            raise RuntimeError(f"output {index}: unexpected spatial size {height} in {array.shape}")
        branch = "box" if channels == 64 else "cls"
        stride = EXPECTED_SCALES[height]
        described.append(
            {"index": index, "shape": list(np.asarray(array).shape), "branch": branch,
             "channels": channels, "height": height, "stride": stride}
        )
        by_stride.setdefault(stride, {})[branch] = head

    box_heads, cls_heads = [], []
    for stride in sorted(by_stride):
        if set(by_stride[stride]) != {"box", "cls"}:
            raise RuntimeError(f"stride {stride}: incomplete head pair {sorted(by_stride[stride])}")
        box_heads.append(by_stride[stride]["box"])
        cls_heads.append(by_stride[stride]["cls"])
    if len(box_heads) != 3:
        raise RuntimeError(f"expected 3 scales, found {sorted(by_stride)}")
    return box_heads, cls_heads, described


def dequantize(outputs, output_attrs):
    """Bring integer outputs back to float32 before decoding.

    rknn3_model_test and the runtime report the six heads as INT8 with per-layer
    asymmetric quantization, so a runtime that hands back raw integers must carry
    usable scale/zero-point. This runtime already returns float32, in which case the
    values pass straight through; anything integer without quantization parameters
    raises instead of silently producing meaningless boxes.
    """
    attrs = output_attrs if isinstance(output_attrs, (list, tuple)) else [output_attrs] * len(outputs)
    converted = []
    for index, array in enumerate(outputs):
        array = np.asarray(array)
        if array.dtype not in (np.int8, np.uint8, np.int16, np.int32):
            converted.append(array.astype(np.float32, copy=False))
            continue
        attr = attrs[index] if index < len(attrs) else None
        scale = getattr(attr, "scale", None)
        zero_point = getattr(attr, "zp", None)
        if zero_point is None:
            zero_point = getattr(attr, "zero_point", None)
        if scale is None or zero_point is None:
            info = getattr(attr, "qnt_info", None)
            if info is not None:
                scale = getattr(info, "scale", None)
                zero_point = getattr(info, "zero_point", None)
        if scale is None or zero_point is None:
            raise RuntimeError(
                f"output {index} came back as {array.dtype} but no scale/zero point is "
                f"available on {attr!r}; dequantization is required for a correct decode."
            )
        converted.append((array.astype(np.float32) - float(zero_point)) * float(scale))
    return converted


def run_inference(rknn, network_input, data_format, output_attrs):
    outputs = rknn.inference(inputs=[network_input], data_format=[data_format])
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("rknn.inference returned no output")
    return dequantize(outputs, output_attrs)

def init_runtime_with_fallback(rknn, requested_mask: int, log=print) -> int:
    """Initialise the RKNN3 runtime, honouring the core count the model was built with.

    RKNN3 requires core_mask to match the compile-time core_num and rejects anything else
    ("core_mask 0xff does not match core number 1"). The mask is therefore configurable but
    still verified: if the requested mask is refused, the plausible masks are tried in turn
    and the one the model accepts is used, with a line in the log saying so.
    """
    if requested_mask == 0:
        if rknn.init_runtime(target="rk1820", core_mask=0) == 0:
            log("core_mask 0x0 (auto) accepted")
            return 0
    elif rknn.init_runtime(target="rk1820", core_mask=requested_mask) == 0:
        return requested_mask

    for cores in range(1, 9):
        mask = (1 << cores) - 1
        if mask == requested_mask:
            continue
        if rknn.init_runtime(target="rk1820", core_mask=mask) == 0:
            log(
                f"core_mask {hex(requested_mask)} was rejected by the runtime (it must match the "
                f"model's compile-time core_num); running with {hex(mask)} = {cores} core(s) instead"
            )
            return mask
    raise SystemExit(
        "rknn3lite init_runtime failed for every plausible core mask; check that the RK182x "
        "module is attached and its runtime service reports the device"
    )


def collect_heads_yolo26(outputs):
    """Same contract as collect_heads, but for YOLO26's (H, W, 4) box heads.

    YOLO26's box branch emits four direct distances instead of 4 * reg_max, so the
    channel count is 4 for the box head and 80 for the score head.
    """
    by_stride: dict[int, dict[str, np.ndarray]] = {}
    described = []
    for index, array in enumerate(outputs):
        squeezed = np.squeeze(np.asarray(array))
        if squeezed.ndim != 3:
            raise RuntimeError(f"output {index}: unexpected rank {array.shape}")
        if squeezed.shape[0] in (4, 80) and squeezed.shape[1] == squeezed.shape[2]:
            channels, height, _ = squeezed.shape
            head = np.transpose(squeezed, (1, 2, 0))
        elif squeezed.shape[-1] in (4, 80) and squeezed.shape[0] == squeezed.shape[1]:
            height, _, channels = squeezed.shape
            head = squeezed
        else:
            raise RuntimeError(f"output {index}: cannot identify layout of {array.shape}")
        if height not in EXPECTED_SCALES:
            raise RuntimeError(f"output {index}: unexpected spatial size {height} in {array.shape}")
        branch = "box" if channels == 4 else "score"
        stride = EXPECTED_SCALES[height]
        described.append(
            {"index": index, "shape": list(np.asarray(array).shape), "branch": branch,
             "channels": channels, "height": height, "stride": stride}
        )
        by_stride.setdefault(stride, {})[branch] = head

    box_heads, score_heads = [], []
    for stride in sorted(by_stride):
        if set(by_stride[stride]) != {"box", "score"}:
            raise RuntimeError(f"stride {stride}: incomplete head pair {sorted(by_stride[stride])}")
        box_heads.append(by_stride[stride]["box"])
        score_heads.append(by_stride[stride]["score"])
    if len(box_heads) != 3:
        raise RuntimeError(f"expected 3 scales, found {sorted(by_stride)}")
    return box_heads, score_heads, described
