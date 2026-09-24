"""Shared RKNN helpers for the RK182x (RKNN3) and RK3576 built-in NPU (RKNN2) runners.

Both Rockchip toolchains end at the same six raw detection heads as the Hailo-8 graph, so
head collection and dequantization must be identical for the two of them - exactly as the
decode itself is shared through postprocess_common.py / postprocess_yolo26.py.
"""

from __future__ import annotations

import numpy as np

#: head spatial size -> stride
EXPECTED_SCALES = {80: 8, 40: 16, 20: 32}
BOX_CHANNELS = 4
SCORE_CHANNELS = 80


def collect_heads_yolo26(outputs):
    """Map the six RKNN outputs onto per-scale box/score heads without trusting order.

    Each output must be a (4, H, W) box head or an (80, H, W) score head for H in
    {80, 40, 20}: the channel count identifies the branch and H identifies the stride.
    Returns (box_heads, score_heads, described).
    """
    by_stride: dict[int, dict[str, np.ndarray]] = {}
    described = []
    for index, array in enumerate(outputs):
        squeezed = np.squeeze(np.asarray(array))
        if squeezed.ndim != 3:
            raise RuntimeError(f"output {index}: unexpected rank {array.shape}")
        if squeezed.shape[0] in (BOX_CHANNELS, SCORE_CHANNELS) and squeezed.shape[1] == squeezed.shape[2]:
            channels, height, _ = squeezed.shape
            head = np.transpose(squeezed, (1, 2, 0))
        elif squeezed.shape[-1] in (BOX_CHANNELS, SCORE_CHANNELS) and squeezed.shape[0] == squeezed.shape[1]:
            height, _, channels = squeezed.shape
            head = squeezed
        else:
            raise RuntimeError(f"output {index}: cannot identify layout of {array.shape}")
        if height not in EXPECTED_SCALES:
            raise RuntimeError(f"output {index}: unexpected spatial size {height} in {array.shape}")
        branch = "box" if channels == BOX_CHANNELS else "score"
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


def dequantize(outputs, output_attrs):
    """Bring integer outputs back to float32 before decoding.

    The runtime reports the six heads as INT8 with per-layer asymmetric quantization, so a
    runtime that hands back raw integers must carry usable scale/zero-point. The RKNN3
    runtime returns float32 already, in which case values pass straight through; anything
    integer without quantization parameters raises instead of silently producing
    meaningless boxes.
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


def init_runtime_with_fallback(rknn, requested_mask: int, log=print) -> int:
    """Initialise the RKNN3 runtime, honouring the core count the model was built with.

    RKNN3 requires core_mask to match the compile-time core_num and rejects anything else
    ("core_mask 0xff does not match core number 1"). The mask is therefore configurable but
    verified: if the requested mask is refused, the plausible masks are tried in turn and
    the one the model accepts is used, with a line in the log saying so.
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