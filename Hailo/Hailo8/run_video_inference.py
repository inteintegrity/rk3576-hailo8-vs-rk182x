"""Hailo-8 video inference for YOLO26n, on HailoRT's pipelined InferModel API.

Every frame of a clip goes through the same pipeline as the other two accelerators:
read -> letterbox -> Hailo-8 -> shared host decode/NMS -> annotation -> MP4.

The device is driven the way Hailo's documentation prescribes for throughput (and what
`hailortcli run` uses internally): one InferModel, configured once, with `--depth`
inferences kept in flight through run_async()/AsyncInferJob.wait(). Submitting one batch
and waiting for it left the device idle whenever the host was decoding, which is why the
same board and HEF measured 34-35 FPS that way and 51-52 FPS this way.

The clip is swept twice, so each layer is reported separately:
    pass 1 (no drawing, no encoding)   -> device service rate and the host pipeline rate
    pass 2 (annotated, writes the MP4) -> end-to-end rate, detections and the clip itself

Usage:
    python Hailo/Hailo8/run_video_inference.py \
        --hef model/Hailo/yolo26n_hailo8_official.hef \
        --video <clip.mp4> --out-dir out/hailo8 --depth 4
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

MODEL_NAME = "YOLO26n"
DEVICE_NAME = "Hailo-8"
DECODE_IMPLEMENTATION = "common/postprocess_common.py + common/postprocess_yolo26.py (shared by all three runners)"


def _common_dir(start: Path) -> Path:
    """Locate common/ by walking up from this script, so the repo runs from any depth."""
    for candidate in (start, *start.parents):
        module_dir = candidate / "common"
        if (module_dir / "postprocess_yolo26.py").is_file():
            return module_dir
    raise SystemExit(
        f"cannot locate common/postprocess_yolo26.py above {start}; run the script from inside "
        f"the repository (the folder that contains common/, model/ and results/)"
    )


def _repo_root(start: Path) -> Path:
    """The folder that holds common/, model/ and video/ - found by walking up from this script."""
    return _common_dir(start).parent

DEFAULT_VIDEO = Path("video") / "test.mp4"


def _resolve_video(requested, start: Path) -> Path:
    """The clip to read: an explicit --video, or the test clip bundled with this repository."""
    if requested is not None:
        video = Path(requested)
        if not video.is_file():
            raise SystemExit(f"video not found: {video}")
        return video
    video = _repo_root(start) / DEFAULT_VIDEO
    if not video.is_file():
        raise SystemExit(
            f"""the bundled test clip is missing: {video}
pass --video <path> to run on another clip, or clone the repository including video/"""
        )
    return video

sys.path.insert(0, str(_common_dir(Path(__file__).resolve().parent)))

from artifacts import sha256  # noqa: E402
from draw_detections import COCO_NAMES, draw, summarise  # noqa: E402
from hailo_pipeline import HailoPipeline, collect_heads  # noqa: E402
from postprocess_common import letterbox, to_network_input, unletterbox_boxes  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{DEVICE_NAME} single-stream video inference ({MODEL_NAME}).")
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None,
                        help="clip to process; default: the test clip bundled with this "
                             "repository (video/test.mp4)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=4,
                        help="inferences kept in flight; 1 means submit-and-wait")
    parser.add_argument("--model-family", choices=("yolo26",), default="yolo26",
                        help="kept for command-line compatibility; this release serves YOLO26n only")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means every frame")
    parser.add_argument("--measure-frames", type=int, default=0,
                        help="frames in the no-drawing pass; 0 means the whole clip")
    parser.add_argument("--benchmark-fps", type=float, default=52.53,
                        help="hailortcli benchmark streaming rate for the same HEF, recorded for reference")
    parser.add_argument("--benchmark-latency-ms", type=float, default=13.89,
                        help="hailortcli benchmark hardware latency for the same HEF, recorded for reference")
    args = parser.parse_args()
    args.video = _resolve_video(args.video, Path(__file__).resolve().parent)

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

    pipeline = HailoPipeline(args.hef, depth=args.depth)
    log(f"HEF: {args.hef.name} sha256={sha256(args.hef)}")
    log(f"infer model: {args.depth} inference(s) in flight, input {pipeline.input_name}")

    class_counter: Counter[str] = Counter()
    confidence_by_class: dict[str, list[float]] = {}
    writer = None

    try:
        # ---- warm-up: the device's service rate with the pipeline kept full -------------------
        ok, warm_frame = capture.read()
        if not ok:
            raise SystemExit("cannot read the first frame for warm-up")
        warm_input = to_network_input(letterbox(warm_frame, 640)[0])
        for _ in range(args.depth):
            pipeline.submit(warm_input)
        for _ in range(args.depth):
            pipeline.collect()
        for _ in range(args.depth):
            pipeline.submit(warm_input)
        warm_device, previous = [], time.perf_counter()
        for _ in range(10):
            pipeline.collect()
            now = time.perf_counter()
            warm_device.append((now - previous) * 1000.0)
            previous = now
            pipeline.submit(warm_input)
        for _ in range(args.depth):
            pipeline.collect()
        device_stats = summarise(warm_device)
        device_fps = 1000.0 / device_stats["mean_ms"]
        log(f"device service: {device_stats['mean_ms']:.2f} ms per frame -> {device_fps:.1f} FPS "
            f"with {args.depth} in flight")
        banner = [
            f"RK3576 + {DEVICE_NAME} | {MODEL_NAME} INT8",
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
                box_heads, score_heads = collect_heads(outputs)
                boxes, scores, classes = decode_detections_yolo26(
                    box_heads, score_heads, conf_thres=args.conf, iou_thres=args.iou
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
                annotated = draw(meta["frame"], detections,
                                 banner_text if banner_text is not None else banner)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(args.out_dir / "annotated.mp4"),
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
                submit_time = time.perf_counter()
                pipeline.submit(
                    to_network_input(padded),
                    meta={"index": index, "frame": frame, "scale": scale,
                          "pad_left": pad_left, "pad_top": pad_top,
                          "read_started": read_started, "submit_time": submit_time},
                )
                # collect as soon as the pipeline is full, which keeps `depth` inferences in
                # flight: that is what keeps the accelerator busy while the host decodes
                while pipeline.free_slots() == 0:
                    process(*pipeline.collect())
            while pipeline.in_flight_count() > 0:
                process(*pipeline.collect())
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
            "name": DEVICE_NAME,
            "runtime": f"HailoRT 4.23.0 (Python API, InferModel.run_async, depth {args.depth})",
            "interface": "PCIe Gen2 x1 (5.0 GT/s, x1)",
        },
        "model": {
            "name": MODEL_NAME,
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
        },
        "decode": {
            "implementation": DECODE_IMPLEMENTATION,
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
        },
        "timing": {
            "depth": args.depth,
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
                "device_service_fps: 10 collect() intervals with the pipeline kept full, i.e. the "
                "rate the device can serve. infer_only_per_frame / pipeline_without_io: measured in "
                "a pass with no drawing and no encoding, so it is the host pipeline rate (read + "
                "letterbox + submit + decode + NMS). python_end_to_end_fps: the annotated pass, "
                "including drawing and MP4 writing. The hailor* fields are what Hailo's own "
                "hailortcli benchmark reports for the same HEF on the same board."
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
    report_path = args.out_dir / "video_result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"frames: {frames}, wall: {total_seconds:.2f} s")
    log(f"device service: {json.dumps(device_stats)} -> {device_fps:.2f} FPS")
    log(f"pipeline per frame: {json.dumps(pipeline_stats)} -> {pipeline_fps:.2f} FPS")
    log(f"end to end: {frames / total_seconds:.2f} FPS over {frames} frames")
    log(f"class counts: {dict(class_counter.most_common())}")
    log(f"annotated video: {args.out_dir / 'annotated.mp4'}")
    log(f"json report: {report_path}")
    (args.out_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
