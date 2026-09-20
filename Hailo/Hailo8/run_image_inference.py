"""Board-side Hailo-8 YOLO11n image inference.

Runs on the RK3576 host through HailoRT's Python API (InferVStreams) against the
Hailo-8 M.2 module, then decodes the six raw detection heads with the shared
host-side implementation in common/postprocess_yolo11.py.

Timing is reported in three separate layers on purpose, because a single "FPS"
number would be misleading:
    infer_only  pure HailoRT infer() call (hardware streaming, host overhead only)
    pipeline    letterbox + infer + decode + NMS (no drawing, no disk I/O)
    e2e         image read + full pipeline + annotation + JSON write
The HailoRT benchmark figures are passed in for reference and are never mixed
into the measured numbers.

Usage:
    python run_image_inference.py --hef yolo11n_hailo8_int8.hef --image bus.jpg \
        --out-dir out --runs 100 --benchmark-fps 54.2711 --benchmark-latency-ms 14.196
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parents[1] / "common", _HERE.parents[2] / "common"):
    if (_candidate / "postprocess_yolo11.py").is_file():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise SystemExit("cannot locate common/postprocess_yolo11.py")
from draw_detections import COCO_NAMES, draw, summarise  # noqa: E402
from postprocess_yolo11 import decode_detections, letterbox, unletterbox_boxes  # noqa: E402

def collect_heads(raw_outputs: dict) -> tuple[list, list]:
    """Identify branch and stride from the tensor shape, not the layer name.

    YOLO11 heads are 64-channel DFL boxes plus 80-channel class logits; YOLO26 heads are
    4-channel direct boxes plus 80-channel score logits. Names differ between HEFs
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

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_hwc(array: np.ndarray, expected_channels: int, name: str) -> np.ndarray:
    """Accept either (H, W, C) or (C, H, W) layout, plus an optional leading batch dim."""
    squeezed = np.squeeze(array)
    if squeezed.ndim != 3:
        raise RuntimeError(f"{name}: unexpected output rank {array.shape}")
    if squeezed.shape[-1] == expected_channels:
        return squeezed
    if squeezed.shape[0] == expected_channels:
        return np.transpose(squeezed, (1, 2, 0))
    raise RuntimeError(
        f"{name}: shape {array.shape} matches neither (H,W,{expected_channels}) nor "
        f"({expected_channels},H,W)"
    )


def collect_heads(raw_outputs: dict[str, np.ndarray]):
    """Split the six HEF outputs into per-scale box and class heads."""
    by_stride: dict[int, dict[str, np.ndarray]] = {}
    for name, array in raw_outputs.items():
        key = name.split("/")[-1]
        if key not in HEAD_MAP:
            raise RuntimeError(f"unexpected HEF output {name}; expected one of {sorted(HEAD_MAP)}")
        branch, stride = HEAD_MAP[key]
        channels = 64 if branch == "box" else 80
        by_stride.setdefault(stride, {})[branch] = to_hwc(array, channels, key)
    box_heads, cls_heads = [], []
    for stride in sorted(by_stride):
        if set(by_stride[stride]) != {"box", "cls"}:
            raise RuntimeError(f"stride {stride}: incomplete head pair {sorted(by_stride[stride])}")
        box_heads.append(by_stride[stride]["box"])
        cls_heads.append(by_stride[stride]["cls"])
    return box_heads, cls_heads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--batch-repeats", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--benchmark-fps", type=float, default=54.2711)
    parser.add_argument("--benchmark-latency-ms", type=float, default=14.196)
    args = parser.parse_args()

    from hailo_platform import (
        HEF,
        ConfigureParams,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    original_height, original_width = image.shape[:2]

    hef = HEF(str(args.hef))
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    input_info = hef.get_input_vstream_infos()[0]
    output_infos = hef.get_output_vstream_infos()
    log(f"HEF: {args.hef.name} sha256={sha256(args.hef)}")
    log(f"input vstream: {input_info.name} {input_info.format.type}")
    for info in output_infos:
        log(f"output vstream: {info.name} {info.format.type}")

    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    padded, scale, pad_left, pad_top = letterbox(image, 640)
    # HailoRT reads the leading axis of the input tensor as the frame count, so the
    # uint8 NHWC image must be passed with an explicit batch dimension.
    network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]

    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as pipeline:
        with network_group.activate():
            for _ in range(args.warmup):
                pipeline.infer({input_info.name: network_input})

            infer_times, pipeline_times = [], []
            detections: list[dict] = []
            raw_shapes: dict[str, list[int]] = {}
            for _ in range(args.runs):
                started = time.perf_counter()
                raw_outputs = pipeline.infer({input_info.name: network_input})
                infer_ms = (time.perf_counter() - started) * 1000.0
                infer_times.append(infer_ms)
                if not raw_shapes:
                    raw_shapes = {name: list(array.shape) for name, array in raw_outputs.items()}

                started = time.perf_counter()
                box_heads, cls_heads = collect_heads(raw_outputs)
                boxes, scores, classes = decode_detections(
                    box_heads, cls_heads, conf_thres=args.conf, iou_thres=args.iou
                )
                pipeline_times.append((time.perf_counter() - started) * 1000.0 + infer_ms)

            # End-to-end: reread from disk and include annotation, reusing the live pipeline.
            e2e_times = []
            for _ in range(10):
                started = time.perf_counter()
                fresh = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
                padded_e2e, scale_e2e, left_e2e, top_e2e = letterbox(fresh, 640)
                network_input_e2e = np.ascontiguousarray(
                    cv2.cvtColor(padded_e2e, cv2.COLOR_BGR2RGB)
                )[None]
                raw_e2e = pipeline.infer({input_info.name: network_input_e2e})
                box_e2e, cls_e2e = collect_heads(raw_e2e)
                boxes_e2e, scores_e2e, classes_e2e = decode_detections(
                    box_e2e, cls_e2e, conf_thres=args.conf, iou_thres=args.iou
                )
                boxes_e2e = unletterbox_boxes(boxes_e2e, scale_e2e, left_e2e, top_e2e)
                detections_e2e = [
                    {
                        "class_id": int(classes_e2e[i]),
                        "class_name": COCO_NAMES[int(classes_e2e[i])],
                        "confidence": float(scores_e2e[i]),
                        "box_xyxy": [float(v) for v in boxes_e2e[i]],
                    }
                    for i in range(len(scores_e2e))
                ]
                draw(fresh, detections_e2e, [])
                e2e_times.append((time.perf_counter() - started) * 1000.0)

            # Streaming throughput through the Python API: several frames per infer()
            # call amortises the per-call host round trip, which is what separates the
            # single-shot number above from the HailoRT benchmark figure.
            batch = np.repeat(network_input, args.batch_frames, axis=0)
            for _ in range(3):
                pipeline.infer({input_info.name: batch})
            batch_times = []
            for _ in range(args.batch_repeats):
                started = time.perf_counter()
                pipeline.infer({input_info.name: batch})
                batch_times.append((time.perf_counter() - started) * 1000.0 / args.batch_frames)
    e2e_stats = summarise(e2e_times)
    batch_stats = summarise(batch_times)

    boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
    order = np.argsort(-scores)
    for index in order:
        box = boxes[index]
        detections.append(
            {
                "class_id": int(classes[index]),
                "class_name": COCO_NAMES[int(classes[index])],
                "confidence": round(float(scores[index]), 4),
                "box_xyxy": [int(round(float(v))) for v in box],
                "box_xywh": [
                    int(round(float(box[0]))),
                    int(round(float(box[1]))),
                    int(round(float(box[2] - box[0]))),
                    int(round(float(box[3] - box[1]))),
                ],
                "box_xyxy_normalised": [
                    round(float(box[0] / original_width), 4),
                    round(float(box[1] / original_height), 4),
                    round(float(box[2] / original_width), 4),
                    round(float(box[3] / original_height), 4),
                ],
            }
        )
    log(f"detections: {len(detections)} (conf>={args.conf}, iou={args.iou})")
    for detection in detections:
        log(
            f"  {detection['class_name']:<15} {detection['confidence']:.4f} "
            f"xyxy={detection['box_xyxy']}"
        )

    infer_stats = summarise(infer_times)
    pipeline_stats = summarise(pipeline_times)

    banner = [
        "RK3576 + Hailo-8 | YOLO11n INT8",
        f"FPS {args.benchmark_fps:.1f} (Hailo-8) / {1000.0 / pipeline_stats['mean_ms']:.1f} (pipeline)",
    ]
    annotated = draw(image, detections, banner)

    annotated_path = args.out_dir / "hailo8_yolo11n_result.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(args.out_dir / "hailo8_yolo11n_input.jpg"), image)
    cv2.imwrite(str(args.out_dir / "hailo8_yolo11n_letterboxed.png"), padded)

    report = {
        "accelerator": {
            "name": "Hailo-8",
            "runtime": "HailoRT 4.23.0 (Python API, InferVStreams)",
            "interface": "PCIe Gen2 x1 (5.0 GT/s, x1)",
            "device_path": "/dev/hailo0",
        },
        "model": {
            "name": "YOLO11n",
            "task": "detect",
            "classes": 80,
            "precision": "INT8 w8a8",
            "hef": args.hef.name,
            "hef_sha256": sha256(args.hef),
            "graph_boundary": "six raw detection heads, decode + NMS on host",
            "output_vstreams": raw_shapes,
        },
        "input": {
            "image": args.image.name,
            "image_sha256": sha256(args.image),
            "original_size_wh": [original_width, original_height],
            "network_input": "640x640 letterbox, RGB, uint8",
            "letterbox": {"scale": scale, "pad_left": pad_left, "pad_top": pad_top, "pad_value": 114},
        },
        "decode": {
            "implementation": "common/postprocess_yolo11.py (shared with the RK1820 runner)",
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "nms": "class-aware greedy NMS, max_wh=7680",
        },
        "detections": detections,
        "timing": {
            "runs": args.runs,
            "warmup_runs": args.warmup,
            "infer_only": infer_stats,
            "pipeline_without_io": pipeline_stats,
            "end_to_end_with_io_and_draw": e2e_stats,
            "streaming_batched": {
                "frames_per_call": args.batch_frames,
                "repeats": args.batch_repeats,
                "per_frame": batch_stats,
                "note": (
                    "Frames are repeated copies of the same letterboxed input; a CNN's "
                    "throughput does not depend on content, so this isolates the "
                    "per-call host overhead."
                ),
            },
            "python_infer_only_fps": round(1000.0 / infer_stats["mean_ms"], 3),
            "python_pipeline_fps": round(1000.0 / pipeline_stats["mean_ms"], 3),
            "python_streaming_fps": round(1000.0 / batch_stats["mean_ms"], 3),
            "python_end_to_end_fps": round(1000.0 / e2e_stats["mean_ms"], 3),
            "hailort_benchmark_streaming_fps": args.benchmark_fps,
            "hailort_benchmark_hw_latency_ms": args.benchmark_latency_ms,
            "note": (
                "The HailoRT benchmark figure measures hardware streaming throughput with "
                "no host postprocessing. The Python numbers include the host-side "
                "InferVStreams call, letterbox, decode, NMS and drawing, so they are "
                "expected to be lower and must not be presented as the accelerator rate."
            ),
        },
        "environment": {
            "host": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    report_path = args.out_dir / "hailo8_yolo11n_result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"infer only: {json.dumps(infer_stats)} -> {1000.0 / infer_stats['mean_ms']:.2f} FPS")
    log(
        f"streaming ({args.batch_frames} frames/call): {json.dumps(batch_stats)} -> "
        f"{1000.0 / batch_stats['mean_ms']:.2f} FPS per frame"
    )
    log(f"pipeline (no I/O): {json.dumps(pipeline_stats)} -> {1000.0 / pipeline_stats['mean_ms']:.2f} FPS")
    log(f"end to end: {json.dumps(e2e_stats)} -> {1000.0 / e2e_stats['mean_ms']:.2f} FPS")
    log(f"annotated image: {annotated_path}")
    log(f"json report: {report_path}")

    (args.out_dir / "hailo8_image_inference.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()