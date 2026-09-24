"""RK182x (RK1820 module) video inference for YOLO26n, via the RKNN3 runtime.

Mirrors Hailo/Hailo8/run_video_inference.py frame for frame: same letterbox, same shared
decode, same annotation and the same reported timing layers, so the annotated clips differ
only by the accelerator. The module sits on the board's M.2 slot, so every inference is a
PCIe round trip.

Usage:
    python rk182x/rk1820/run_video_inference.py \
        --model model/rk1820/yolo26n_rk1820_int8.rknn \
        --weight model/rk1820/yolo26n_rk1820_int8.weight \
        --video <clip.mp4> --out-dir out/rk1820
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
DEVICE_NAME = "RK1820 / RK182x"
BANNER_NAME = "RK3576 + RK182x"
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
from postprocess_common import letterbox, to_network_input, unletterbox_boxes  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402
from rknn_helpers import collect_heads_yolo26, dequantize, init_runtime_with_fallback  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"{DEVICE_NAME} single-stream video inference ({MODEL_NAME})."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None,
                        help="clip to process; default: the test clip bundled with this "
                             "repository (video/test.mp4)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--core-mask", type=lambda value: int(value, 0), default=0xff,
                        help="NPU core mask; RKNN3 official examples use 0xff (all 8 cores), but the "
                             "runtime only accepts the model's compile-time core count and falls back "
                             "automatically if the requested mask does not match")
    parser.add_argument("--model-family", choices=("yolo26",), default="yolo26",
                        help="kept for command-line compatibility; this release serves YOLO26n only")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means every frame")
    parser.add_argument("--device-fps", type=float, default=22.785,
                        help="single-stream inference-only rate measured with this same client")
    parser.add_argument("--device-latency-ms", type=float, default=43.889)
    args = parser.parse_args()
    args.video = _resolve_video(args.video, Path(__file__).resolve().parent)

    from rknn3lite.api.rknn3_lite import RKNN3Lite

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

    def infer(frame: np.ndarray):
        padded, scale, pad_left, pad_top = letterbox(frame, 640)
        raw = rknn.inference(inputs=[to_network_input(padded)], data_format=["nhwc"])
        if raw is None or len(raw) == 0:
            raise RuntimeError("rknn.inference returned no output")
        return dequantize(raw, output_attrs), scale, pad_left, pad_top

    # Warm up and measure the pipeline rate once, so the banner carries a stable,
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
        box_heads, score_heads, _ = collect_heads_yolo26(outputs)
        decode_detections_yolo26(box_heads, score_heads, conf_thres=args.conf, iou_thres=args.iou)
        warm_pipeline.append((time.perf_counter() - warm_started) * 1000.0)
    banner_fps = 1000.0 / summarise(warm_pipeline)["mean_ms"]
    log(f"warm-up pipeline: {summarise(warm_pipeline)['mean_ms']:.2f} ms -> banner FPS {banner_fps:.1f}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    banner = [
        f"{BANNER_NAME} | {MODEL_NAME} INT8",
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
        network_input = to_network_input(padded)

        infer_started = time.perf_counter()
        raw = rknn.inference(inputs=[network_input], data_format=["nhwc"])
        if raw is None or len(raw) == 0:
            raise RuntimeError("rknn.inference returned no output")
        infer_ms = (time.perf_counter() - infer_started) * 1000.0

        decode_started = time.perf_counter()
        outputs = dequantize(raw, output_attrs)
        box_heads, score_heads, _ = collect_heads_yolo26(outputs)
        boxes, scores, classes = decode_detections_yolo26(
            box_heads, score_heads, conf_thres=args.conf, iou_thres=args.iou
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
        # The pipeline layer here is the RKNN call plus host decode/NMS; the frame read and the
        # letterbox happen before `infer_started` and are covered by read_infer_times instead.
        # The Hailo-8 runner's pipeline figure does include its read and letterbox, so the two
        # columns are not the same measurement - stated in the record's timing note and in
        # results/README.md.
        pipeline_times.append(infer_ms + decode_ms)
        read_infer_times.append((time.perf_counter() - read_started) * 1000.0)

        encode_started = time.perf_counter()
        annotated = draw(frame, detections, banner)
        if writer is None:
            writer = cv2.VideoWriter(
                str(args.out_dir / "annotated.mp4"),
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
            "name": DEVICE_NAME,
            "runtime": "RKNN3 runtime (rknn3lite)",
            "interface": "PCIe via rknn3 NTB",
            "core_mask_requested": hex(args.core_mask),
            "core_mask_active": hex(active_mask),
        },
        "model": {
            "name": MODEL_NAME,
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
        },
        "decode": {
            "implementation": DECODE_IMPLEMENTATION,
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
                "infer_only_per_frame: the rknn3lite call alone. pipeline_without_io: letterbox + "
                "inference + decode + NMS. python_end_to_end_fps: the annotated pass, including "
                "drawing and MP4 writing. device_benchmark_*: the inference-only figures measured "
                "with this same client, shown on the clip's banner."
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

    log(f"frames: {frames}, wall: {wall_seconds:.2f} s")
    log(f"infer only per frame: {json.dumps(infer_stats)} -> {1000.0 / infer_stats['mean_ms']:.2f} FPS")
    log(f"pipeline per frame: {json.dumps(pipeline_stats)} -> {1000.0 / pipeline_stats['mean_ms']:.2f} FPS")
    log(f"end to end: {frames / total_seconds:.2f} FPS over {frames} frames ({total_seconds:.2f} s total wall)")
    log(f"class counts: {dict(class_counter.most_common())}")
    log(f"annotated video: {args.out_dir / 'annotated.mp4'}")
    log(f"json report: {report_path}")
    (args.out_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
