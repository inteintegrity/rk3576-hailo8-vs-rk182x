"""Board-side RK1820 / RK182x YOLO11n video inference.

Mirrors Hailo/Hailo8/run_video_inference.py frame for frame: same letterbox, same
shared decode, same annotation and the same reported timing layers, so the two
clips differ only by the accelerator.

The accelerator figure on the banner is the single-stream inference-only rate measured
with this same client, so the two figures on screen are the module's best single-stream
rate and the rate this annotated pass actually sustained.

Usage:
    python run_video_inference.py --model yolo11n_rk1820_int8_rt104.rknn \
        --weight yolo11n_rk1820_int8_rt104.weight --video clip.mp4 --out-dir out \
        --core-mask 0x01
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
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
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402
from rknn_runner import (  # noqa: E402
    collect_heads, collect_heads_yolo26, dequantize, init_runtime_with_fallback,
    run_inference, sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--core-mask", type=lambda value: int(value, 0), default=0xff,
                        help="NPU core mask; RKNN3 official examples use 0xff (all 8 cores), but the "
                             "runtime only accepts the model's compile-time core count and falls back "
                             "automatically if the requested mask does not match")
    parser.add_argument("--model-family", choices=("yolo11", "yolo26"), default="yolo11",
                        help="YOLO11 has 64-channel DFL box heads; YOLO26 has 4-channel direct box heads")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means every frame")
    parser.add_argument("--device-fps", type=float, default=22.785,
                        help="single-stream inference-only rate measured with this same client")
    parser.add_argument("--device-latency-ms", type=float, default=43.889)
    args = parser.parse_args()

    from rknn3lite.api.rknn3_lite import RKNN3Lite

    tag = "yolo26n" if getattr(args, "model_family", "yolo11") == "yolo26" else "yolo11n"
    model_label = f"{tag[:-1].upper()}{tag[-1]}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = args.max_frames if args.max_frames > 0 else frame_count

    rknn = RKNN3Lite()
    if rknn.load_rknn(model_path=str(args.model), weight_path=str(args.weight)) != 0:
        raise SystemExit("rknn3lite load_rknn failed")
    active_mask = init_runtime_with_fallback(rknn, args.core_mask, log)
    output_attrs = rknn.get_outputs_tensor_attr()
    log(f"video: {args.video.name} {width}x{height} source_fps={source_fps:.2f} frames={frame_count}")
    log(f"model: {args.model.name} sha256={sha256(args.model)}")
    log(f"weight: {args.weight.name} sha256={sha256(args.weight)}")
    log(f"core_mask requested: {hex(args.core_mask)}, active: {hex(active_mask)}")
    log(f"runtime sdk version:\n{rknn.get_sdk_version()}")

    data_format = "nhwc"

    def infer(frame: np.ndarray):
        padded, scale, pad_left, pad_top = letterbox(frame, 640)
        network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
        raw = rknn.inference(inputs=[network_input], data_format=[data_format])
        if raw is None or len(raw) == 0:
            raise RuntimeError("rknn.inference returned no output")
        return dequantize(raw, output_attrs), scale, pad_left, pad_top

    # Warm up and measure the pipeline rate once so the banner carries a stable,
    # self-measured FPS instead of a per-frame number that would flicker.
    ok, warm_frame = capture.read()
    if not ok:
        raise SystemExit("cannot read the first frame for warm-up")
    for _ in range(3):
        infer(warm_frame)
    warm_pipeline = []
    for _ in range(10):
        warm_started = time.perf_counter()
        outputs, scale, pad_left, pad_top = infer(warm_frame)
        box_heads, cls_heads, _ = (
            collect_heads_yolo26(outputs) if args.model_family == "yolo26" else collect_heads(outputs)
        )
        (decode_detections_yolo26 if args.model_family == "yolo26" else decode_detections)(
            box_heads, cls_heads, conf_thres=args.conf, iou_thres=args.iou
        )
        warm_pipeline.append((time.perf_counter() - warm_started) * 1000.0)
    banner_fps = 1000.0 / summarise(warm_pipeline)["mean_ms"]
    log(f"warm-up pipeline: {summarise(warm_pipeline)['mean_ms']:.2f} ms -> banner FPS {banner_fps:.1f}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    banner = [
        f"RK3576 + RK182x | {model_label} INT8",
        f"FPS {args.device_fps:.1f} infer only / {banner_fps:.1f} this run",
    ]

    infer_times, pipeline_times, read_infer_times, encode_times = [], [], [], []
    per_frame: list[dict] = []
    class_counter: Counter[str] = Counter()
    confidence_by_class: dict[str, list[float]] = {}
    writer = None

    started_run = time.perf_counter()
    for index in range(limit):
        read_started = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            log(f"frame {index}: read failed, stopping")
            break
        padded, scale, pad_left, pad_top = letterbox(frame, 640)
        network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]

        infer_started = time.perf_counter()
        raw = rknn.inference(inputs=[network_input], data_format=[data_format])
        if raw is None or len(raw) == 0:
            raise RuntimeError("rknn.inference returned no output")
        infer_ms = (time.perf_counter() - infer_started) * 1000.0

        decode_started = time.perf_counter()
        outputs = dequantize(raw, output_attrs)
        box_heads, cls_heads, _ = (
            collect_heads_yolo26(outputs) if args.model_family == "yolo26" else collect_heads(outputs)
        )
        boxes, scores, classes = (decode_detections_yolo26 if args.model_family == "yolo26"
                                  else decode_detections)(
            box_heads, cls_heads, conf_thres=args.conf, iou_thres=args.iou
        )
        boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0

        detections = []
        for position in np.argsort(-scores):
            name = COCO_NAMES[int(classes[position])]
            confidence = float(scores[position])
            class_counter[name] += 1
            confidence_by_class.setdefault(name, []).append(confidence)
            detections.append(
                {
                    "class_id": int(classes[position]),
                    "class_name": name,
                    "confidence": round(confidence, 4),
                    "box_xyxy": [int(round(float(v))) for v in boxes[position]],
                }
            )

        infer_times.append(infer_ms)
        pipeline_times.append(infer_ms + decode_ms)
        read_infer_times.append((time.perf_counter() - read_started) * 1000.0)

        encode_started = time.perf_counter()
        annotated = draw(frame, detections, banner)
        if writer is None:
            writer = cv2.VideoWriter(
                str(args.out_dir / f"rk1820_{tag}_result.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                source_fps if source_fps > 0 else 10.0,
                (annotated.shape[1], annotated.shape[0]),
            )
        writer.write(annotated)
        encode_times.append((time.perf_counter() - encode_started) * 1000.0)

        per_frame.append(
            {
                "frame": index,
                "infer_ms": round(infer_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "detections": detections,
            }
        )
    wall_seconds = time.perf_counter() - started_run
    capture.release()
    if writer is not None:
        writer.release()
    rknn.release()

    infer_stats = summarise(infer_times)
    pipeline_stats = summarise(pipeline_times)
    read_stats = summarise(read_infer_times)
    encode_stats = summarise(encode_times)
    frames = len(per_frame)
    total_seconds = wall_seconds

    report = {
        "accelerator": {
            "name": "RK1820 / RK182x",
            "runtime": "RKNN3 runtime (rknn3lite)",
            "interface": "PCIe via rknn3 NTB",
            "core_mask_requested": hex(args.core_mask),
            "core_mask_active": hex(active_mask),
        },
        "model": {
            "name": "YOLO26n" if args.model_family == "yolo26" else "YOLO11n",
            "precision": "INT8 w8a8",
            "rknn": args.model.name,
            "rknn_sha256": sha256(args.model),
            "weight": args.weight.name,
            "weight_sha256": sha256(args.weight),
            "graph_boundary": "six raw detection heads, decode + NMS on host",
        },
        "input": {
            "video": args.video.name,
            "video_sha256": sha256(args.video),
            "source_fps": source_fps,
            "resolution_wh": [width, height],
            "frames_processed": frames,
            "note": "Clip built from COCO128 images at 10 FPS, 640x480.",
        },
        "decode": {
            "implementation": "common/postprocess_yolo11.py (shared with the Hailo-8 runner)",
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
        },
        "timing": {
            "infer_only_per_frame": infer_stats,
            "pipeline_without_io": pipeline_stats,
            "read_letterbox_infer_decode": read_stats,
            "annotate_and_encode_per_frame": encode_stats,
            "python_infer_only_fps": round(1000.0 / infer_stats["mean_ms"], 3),
            "python_pipeline_fps": round(1000.0 / pipeline_stats["mean_ms"], 3),
            "python_end_to_end_fps": round(frames / total_seconds, 3) if total_seconds > 0 else None,
            "total_wall_seconds": round(total_seconds, 3),
            "banner_pipeline_fps": round(banner_fps, 3),
            "device_benchmark_total_fps": args.device_fps,
            "device_benchmark_total_ms": args.device_latency_ms,
            "note": (
                "The device figure is the single-stream inference-only rate measured with this "
                "same client. The Python figures add letterbox, the rknn3lite call, decode, NMS "
                "and drawing on the RK3576 CPU."
            ),
        },
        "detection_summary": {
            "frames": frames,
            "total_detections": int(sum(class_counter.values())),
            "mean_detections_per_frame": round(sum(class_counter.values()) / frames, 3) if frames else 0,
            "per_class_counts": dict(class_counter.most_common()),
            "per_class_mean_confidence": {
                name: round(sum(values) / len(values), 4)
                for name, values in sorted(confidence_by_class.items())
            },
        },
        "per_frame": per_frame,
        "environment": {
            "host": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    report_path = args.out_dir / f"rk1820_{tag}_video_result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"frames: {frames}, wall: {wall_seconds:.2f} s")
    log(f"infer only per frame: {json.dumps(infer_stats)} -> {1000.0 / infer_stats['mean_ms']:.2f} FPS")
    log(f"pipeline per frame: {json.dumps(pipeline_stats)} -> {1000.0 / pipeline_stats['mean_ms']:.2f} FPS")
    log(f"end to end: {frames / total_seconds:.2f} FPS over {frames} frames ({total_seconds:.2f} s total wall)")
    log(f"class counts: {dict(class_counter.most_common())}")
    log(f"annotated video: {args.out_dir / f'rk1820_{tag}_result.mp4'}")
    log(f"json report: {report_path}")
    (args.out_dir / "rk1820_video_inference.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()