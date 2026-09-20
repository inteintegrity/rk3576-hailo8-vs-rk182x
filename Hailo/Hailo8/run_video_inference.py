"""Board-side Hailo-8 YOLO26n video inference, on HailoRT's asynchronous pipeline.

Runs the same pipeline as run_image_inference.py over every frame of a clip:
letterbox -> HailoRT -> shared decode -> annotation -> MP4.

The accelerator side uses the API Hailo recommends for throughput (and what `hailortcli run`
uses internally): an InferModel, configured once, with `--async-depth` inferences kept in flight
through run_async()/AsyncInferJob.wait(). The older submit-one-batch-and-wait loop left the device
idle whenever the host was decoding, which is why it reported 34-35 FPS for a device whose own
benchmark says 52.5 FPS. Measured on this board the device service rate goes 28.6 FPS (depth 1) ->
51.7 FPS (depth 2) -> 52.7 FPS (depth 8).

The clip is swept twice, so each layer can be reported separately:
    pass 1 (no drawing, no encoding)   -> device service rate and the host pipeline rate
    pass 2 (annotated, writes the MP4) -> end-to-end rate, detections and the clip itself

Usage:
    python run_video_inference.py --hef yolo26n_hailo8_official.hef --video clip.mp4 \
        --out-dir out --model-family yolo26 --async-depth 4 \
        --benchmark-fps 52.53 --benchmark-latency-ms 13.89
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
from hailo_async import AsyncPipeline  # noqa: E402
from hailo_runner import sha256  # noqa: E402
from postprocess_yolo11 import decode_detections, letterbox, unletterbox_boxes  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402


def collect_heads(outputs: dict):
    """Identify branch and stride from tensor shape, not layer name."""
    strides = {80: 8, 40: 16, 20: 32}
    by_stride: dict[int, dict[str, np.ndarray]] = {}
    for name, array in outputs.items():
        squeezed = np.squeeze(np.asarray(array))
        if squeezed.ndim != 3:
            raise RuntimeError(f"unexpected HEF output rank for {name}: {np.asarray(array).shape}")
        if squeezed.shape[-1] in (4, 64, 80):
            height, _, channels = squeezed.shape
            head = squeezed
        elif squeezed.shape[0] in (4, 64, 80):
            channels, height, _ = squeezed.shape
            head = np.transpose(squeezed, (1, 2, 0))
        else:
            raise RuntimeError(f"cannot identify layout of {name}: {np.asarray(array).shape}")
        branch = "box" if channels in (4, 64) else "cls"
        by_stride.setdefault(strides[height], {})[branch] = head
    box_heads = [by_stride[stride]["box"] for stride in sorted(by_stride)]
    cls_heads = [by_stride[stride]["cls"] for stride in sorted(by_stride)]
    if len(box_heads) != 3:
        raise RuntimeError(f"expected 3 scales, found {sorted(by_stride)}")
    return box_heads, cls_heads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-family", choices=("yolo11", "yolo26"), default="yolo11")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means every frame")
    parser.add_argument("--async-depth", type=int, default=4,
                        help="inferences kept in flight; 1 is the old synchronous behaviour")
    parser.add_argument("--measure-frames", type=int, default=0,
                        help="frames in the no-drawing pass; 0 means the whole clip")
    parser.add_argument("--benchmark-fps", type=float, default=52.53)
    parser.add_argument("--benchmark-latency-ms", type=float, default=13.89)
    args = parser.parse_args()

    tag = "yolo26n" if args.model_family == "yolo26" else "yolo11n"
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
    measure_limit = args.measure_frames if args.measure_frames > 0 else limit
    log(f"video: {args.video.name} {width}x{height} source_fps={source_fps:.2f} frames={frame_count}")

    pipeline = AsyncPipeline(args.hef, depth=args.async_depth)
    log(f"HEF: {args.hef.name} sha256={sha256(args.hef)}")
    log(f"infer model: async depth {args.async_depth}, input {pipeline.input_name}")

    class_counter: Counter[str] = Counter()
    confidence_by_class: dict[str, list[float]] = {}
    writer = None

    try:
        # ---- warm-up: the device's service rate with the pipeline kept full -------------------
        ok, warm_frame = capture.read()
        if not ok:
            raise SystemExit("cannot read the first frame for warm-up")
        warm_padded, _, _, _ = letterbox(warm_frame, 640)
        warm_input = np.ascontiguousarray(cv2.cvtColor(warm_padded, cv2.COLOR_BGR2RGB))[None]
        for _ in range(args.async_depth):
            pipeline.submit(warm_input)
        for _ in range(args.async_depth):
            pipeline.collect()
        for _ in range(args.async_depth):
            pipeline.submit(warm_input)
        warm_device, previous = [], time.perf_counter()
        for _ in range(10):
            pipeline.collect()
            now = time.perf_counter()
            warm_device.append((now - previous) * 1000.0)
            previous = now
            pipeline.submit(warm_input)
        for _ in range(args.async_depth):
            pipeline.collect()
        device_stats = summarise(warm_device)
        device_fps = 1000.0 / device_stats["mean_ms"]
        log(f"device service: {device_stats['mean_ms']:.2f} ms per frame -> {device_fps:.1f} FPS "
            f"at async depth {args.async_depth}")
        banner = [
            f"RK3576 + Hailo-8 | {model_label} INT8",
            f"FPS {device_fps:.1f} on device / {{}} this run",
        ]

        def sweep(annotate: bool, frames: int, banner_text: list[str] | None = None):
            """Run the clip once; with annotate=False nothing is drawn or encoded."""
            nonlocal writer
            last_decode_end: float | None = None
            cadence: list[float] = []
            latencies: list[float] = []
            encode_times: list[float] = []
            detections_seen: list[dict] = []
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            started = time.perf_counter()

            def process(meta: dict, outputs: dict):
                nonlocal last_decode_end, writer
                completion = time.perf_counter()
                box_heads, cls_heads = collect_heads(outputs)
                decode_head = (decode_detections_yolo26 if box_heads[0].shape[-1] == 4
                               else decode_detections)
                boxes, scores, classes = decode_head(
                    box_heads, cls_heads, conf_thres=args.conf, iou_thres=args.iou
                )
                boxes = unletterbox_boxes(boxes, meta["scale"], meta["pad_left"], meta["pad_top"])
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
                decode_end = time.perf_counter()
                if last_decode_end is not None:
                    cadence.append((decode_end - last_decode_end) * 1000.0)
                last_decode_end = decode_end
                latencies.append((decode_end - meta["read_started"]) * 1000.0)

                if not annotate:
                    return
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
                detections_seen.append(
                    {
                        "frame": meta["index"],
                        "infer_ms": round((completion - meta["submit_time"]) * 1000.0, 3),
                        "decode_ms": round((decode_end - completion) * 1000.0, 3),
                        "detections": detections,
                    }
                )
                encode_started = time.perf_counter()
                line = banner_text if banner_text is not None else banner
                annotated = draw(meta["frame"], detections, line)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(args.out_dir / f"hailo8_{tag}_result.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps if source_fps > 0 else 10.0,
                        (annotated.shape[1], annotated.shape[0]),
                    )
                writer.write(annotated)
                encode_times.append((time.perf_counter() - encode_started) * 1000.0)

            for index in range(frames):
                read_started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    log(f"frame {index}: read failed, stopping")
                    break
                padded, scale, pad_left, pad_top = letterbox(frame, 640)
                network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
                submit_time = time.perf_counter()
                pipeline.submit(network_input, meta={"index": index, "frame": frame, "scale": scale,
                                                     "pad_left": pad_left, "pad_top": pad_top,
                                                     "read_started": read_started,
                                                     "submit_time": submit_time})
                # collect as soon as the pipeline is full, which keeps async_depth inferences in
                # flight: that is what keeps the accelerator busy while the host decodes
                while pipeline.free_slots() == 0:
                    meta, outputs = pipeline.collect()
                    process(meta, outputs)
            while pipeline.in_flight_count() > 0:
                meta, outputs = pipeline.collect()
                process(meta, outputs)
            return {
                "wall_seconds": time.perf_counter() - started,
                "cadence": cadence,
                "latencies": latencies,
                "encode_times": encode_times,
                "detections_seen": detections_seen,
            }

        log("pass 1/2: no drawing, no encoding (host pipeline rate)")
        measure = sweep(annotate=False, frames=measure_limit)
        measure_stats = summarise(measure["cadence"])
        pipeline_fps = 1000.0 / measure_stats["mean_ms"]
        log(f"  pipeline without drawing: {measure_stats['mean_ms']:.2f} ms per frame "
            f"-> {pipeline_fps:.2f} FPS")

        log("pass 2/2: annotated, writing the MP4 (end-to-end rate)")
        # the clip's banner carries the device rate and this pass's own measured rate
        annotated_pass = sweep(annotate=True, frames=limit,
                               banner_text=[banner[0], banner[1].format(f"{pipeline_fps:.1f}")])
    finally:
        pipeline.close()

    capture.release()
    if writer is not None:
        writer.release()

    pipeline_stats = summarise(measure["cadence"]) if measure["cadence"] else summarise([0.0])
    read_stats = summarise(measure["latencies"]) if measure["latencies"] else summarise([0.0])
    encode_stats = (summarise(annotated_pass["encode_times"]) if annotated_pass["encode_times"]
                    else summarise([0.0]))
    per_frame = annotated_pass["detections_seen"]
    frames = len(per_frame)
    total_seconds = annotated_pass["wall_seconds"]
    banner_fps = 1000.0 / pipeline_stats["mean_ms"] if pipeline_stats["mean_ms"] > 0 else 0.0

    report = {
        "accelerator": {
            "name": "Hailo-8",
            "runtime": f"HailoRT 4.23.0 (Python API, InferModel.run_async, depth {args.async_depth})",
            "interface": "PCIe Gen2 x1 (5.0 GT/s, x1)",
        },
        "model": {
            "name": "YOLO26n" if args.model_family == "yolo26" else "YOLO11n",
            "precision": "INT8 w8a8",
            "hef": args.hef.name,
            "hef_sha256": sha256(args.hef),
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
            "implementation": "common/postprocess_yolo11.py (shared with the RK3576 and RK182x runners)",
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
        },
        "timing": {
            "async_depth": args.async_depth,
            "device_service_per_frame": device_stats,
            "device_service_fps": round(device_fps, 3),
            "infer_only_per_frame": pipeline_stats,
            "python_infer_only_fps": round(device_fps, 3),
            "pipeline_without_io": pipeline_stats,
            "python_pipeline_fps": round(1000.0 / pipeline_stats["mean_ms"], 3),
            "read_letterbox_infer_decode": read_stats,
            "python_end_to_end_fps": round(frames / total_seconds, 3) if total_seconds > 0 else None,
            "total_wall_seconds": round(total_seconds, 3),
            "annotate_and_encode_per_frame": encode_stats,
            "banner_pipeline_fps": round(banner_fps, 3),
            "hailort_benchmark_streaming_fps": args.benchmark_fps,
            "hailort_benchmark_hw_latency_ms": args.benchmark_latency_ms,
            "note": (
                "device_service_fps: 10 collect() intervals with the async pipeline kept full, i.e. "
                "the rate the device can serve. infer_only_per_frame / pipeline_without_io: measured "
                "in a pass with no drawing and no encoding, so it is the host pipeline rate "
                "(read + letterbox + submit + decode + NMS). python_end_to_end_fps: the annotated "
                "pass, including drawing and MP4 writing."
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
    report_path = args.out_dir / f"hailo8_{tag}_video_result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"frames: {frames}, wall: {total_seconds:.2f} s")
    log(f"device service: {json.dumps(device_stats)} -> {device_fps:.2f} FPS")
    log(f"pipeline per frame: {json.dumps(pipeline_stats)} -> {pipeline_fps:.2f} FPS")
    log(f"end to end: {frames / total_seconds:.2f} FPS over {frames} frames")
    log(f"class counts: {dict(class_counter.most_common())}")
    log(f"annotated video: {args.out_dir / f'hailo8_{tag}_result.mp4'}")
    log(f"json report: {report_path}")
    (args.out_dir / "hailo8_video_inference.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()