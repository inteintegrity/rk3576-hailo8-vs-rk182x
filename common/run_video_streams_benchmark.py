"""Multi-stream video benchmark for the two Rockchip backends: N processes, one stream each.

Built for the multi-camera question: how many video streams an accelerator can serve at once,
and where the platform stops keeping up. Each worker opens its own VideoCapture on the same
file (offset per stream so the streams are decorrelated), pins itself to one NPU core, runs
the shared pipeline (letterbox -> accelerator -> shared decode/NMS) and counts processed frames.

Deliberately excludes drawing and video encoding: this measures the inference service capacity.
The single-stream runners report the annotated-output cost separately.

Streams may outnumber NPU cores; extra streams then share cores round-robin, which is what a
deployment does when it has more cameras than the accelerator has cores.

Usage (on the board):
    python3 common/run_video_streams_benchmark.py --backend rk1820 \
        --model model/rk1820/yolo26n_rk1820_int8.rknn \
        --weight model/rk1820/yolo26n_rk1820_int8.weight \
        --video <clip.mp4> --instances 1,2,4,8 --frames 200 --json out/rk1820_multi.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np

def _common_dir(start: Path) -> Path:
    """Locate common/ by walking up from this script, so the repo runs from any depth."""
    for candidate in (start, *start.parents):
        if (candidate / "postprocess_yolo26.py").is_file():
            return candidate
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

from postprocess_common import letterbox, to_network_input  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402
from rknn_helpers import collect_heads_yolo26, dequantize  # noqa: E402

#: NPU cores each backend can hand to one stream. RK3576 has two NPU cores (0x1 / 0x2);
#: RK182x has eight. Streams beyond that count share cores round-robin.
CORE_COUNT = {"rk1820": 8, "rk3576": 2}


def open_offset_capture(video: str, stream_index: int):
    """Open the clip and start each stream at a different frame, so the streams differ."""
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        return None, 0
    modulus = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, stream_index * 7 % modulus)
    return capture, modulus


def run_loop(index, capture, frames, infer, to_heads, decode, conf, iou, queue,
             reported: dict | None = None) -> None:
    """Process `frames` frames through the shared pipeline and report the timing window."""
    processed, detections_total, infer_times = 0, 0, []
    loop_start = time.perf_counter()
    while processed < frames:
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        began = time.perf_counter()
        outputs = infer(frame)
        infer_times.append((time.perf_counter() - began) * 1000.0)
        box_heads, score_heads, _ = to_heads(outputs)
        _, scores, _ = decode(box_heads, score_heads, conf_thres=conf, iou_thres=iou)
        detections_total += int(len(scores))
        processed += 1
    payload = {
        "frames": processed,
        "detections": detections_total,
        "mean_infer_ms": float(np.mean(infer_times)) if infer_times else 0.0,
        "loop_start": loop_start,
        "loop_end": time.perf_counter(),
    }
    payload.update(reported or {})
    queue.put((index, payload))


def worker(backend: str, index: int, model: str, weight: str, video: str, frames: int,
           conf: float, iou: float, ready, start, queue, core_count: int = 1) -> None:
    """One stream per process, pinned to one NPU core, driven through the RKNN Python binding."""
    capture, modulus = open_offset_capture(video, index)
    if capture is None:
        queue.put((index, {"error": f"cannot open {video}"}))
        return

    if backend == "rk1820":
        from rknn3lite.api.rknn3_lite import RKNN3Lite

        rknn = RKNN3Lite()
        if rknn.load_rknn(model_path=model, weight_path=weight) != 0:
            queue.put((index, {"error": "load_rknn failed"}))
            return
        # One core per stream: the runtime only accepts a mask that matches the model's
        # compile-time core count, so the requested single-core mask is tried first and the
        # model's own core count is used if the runtime refuses it. The mask that was accepted is
        # reported with the stream's result, so a record shows which core actually served it.
        requested = 1 << (index % 8)
        active = requested if rknn.init_runtime(target="rk1820", core_mask=requested) == 0 else None
        if active is None:
            for cores in range(1, 9):
                if rknn.init_runtime(target="rk1820", core_mask=(1 << cores) - 1) == 0:
                    active = (1 << cores) - 1
                    break
            else:
                queue.put((index, {"error": "init_runtime failed"}))
                return
        output_attrs = rknn.get_outputs_tensor_attr()

        def infer(frame):
            padded = letterbox(frame, 640)[0]
            batch = to_network_input(padded)
            return dequantize(rknn.inference(inputs=[batch], data_format=["nhwc"]), output_attrs)
        reported = {"core_mask_requested": hex(requested), "core_mask_active": hex(active)}
    else:
        from rknnlite.api import RKNNLite

        rknn = RKNNLite()
        if rknn.load_rknn(model) != 0:
            queue.put((index, {"error": "load_rknn failed"}))
            return
        requested = 1 << (index % core_count)
        if rknn.init_runtime(core_mask=requested) != 0:
            queue.put((index, {"error": "init_runtime failed"}))
            return

        def infer(frame):
            padded = letterbox(frame, 640)[0]
            batch = to_network_input(padded)
            return dequantize(rknn.inference(inputs=[batch], data_format="nhwc"), None)
        reported = {"core_mask_requested": hex(requested), "core_mask_active": hex(requested)}

    ok, warm = capture.read()
    if not ok:
        queue.put((index, {"error": "no frames"}))
        return
    collect_heads_yolo26(infer(warm))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index * 7 % modulus)

    ready.put(index)
    start.wait()
    try:
        run_loop(index, capture, frames, infer, collect_heads_yolo26, decode_detections_yolo26,
                 conf, iou, queue, reported)
    finally:
        capture.release()
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate throughput of the RK3576 NPU / RK182x with N concurrent streams."
    )
    parser.add_argument("--backend", choices=("rk1820", "rk3576"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weight")
    parser.add_argument("--video", default=None,
                        help="clip to process; default: the test clip bundled with this "
                             "repository (video/test.mp4)")
    parser.add_argument("--instances", default="1,2,4,8",
                        help="stream counts to test, comma separated")
    parser.add_argument("--frames", type=int, default=200, help="frames per stream")
    parser.add_argument("--model-family", choices=("yolo26",), default="yolo26",
                        help="kept for command-line compatibility; this release serves YOLO26n only")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    args.video = str(_resolve_video(args.video, Path(__file__).resolve().parent))

    counts = [int(value) for value in args.instances.split(",") if value.strip()]
    core_count = CORE_COUNT[args.backend]
    report = {"backend": args.backend, "video": Path(args.video).name,
              "frames_per_stream": args.frames, "model_family": "yolo26",
              "npu_cores": core_count, "results": {}}

    for count in counts:
        if count > core_count:
            print(f"  note: {count} streams share {core_count} NPU core(s) (oversubscribed)")
        ready, start, queue = mp.Queue(), mp.Event(), mp.Queue()
        processes = [
            mp.Process(target=worker,
                       args=(args.backend, index, args.model, args.weight or "", args.video,
                             args.frames, args.conf, args.iou, ready, start, queue, core_count))
            for index in range(count)
        ]
        for process in processes:
            process.start()
        try:
            for _ in range(count):
                ready.get(timeout=180)
        except Exception:
            print(f"instances={count}: warm-up timed out")
        began = time.perf_counter()
        start.set()
        for process in processes:
            process.join(timeout=900)
        wall = time.perf_counter() - began

        per_stream = {}
        for _ in range(count):
            try:
                index, payload = queue.get(timeout=15)
                per_stream[index] = payload
            except Exception:
                break
        errors = [payload["error"] for payload in per_stream.values() if "error" in payload]
        starts = [p["loop_start"] for p in per_stream.values() if "loop_start" in p]
        ends = [p["loop_end"] for p in per_stream.values() if "loop_end" in p]
        window = (max(ends) - min(starts)) if starts and ends else wall
        total_frames = sum(p.get("frames", 0) for p in per_stream.values())
        aggregate = total_frames / window if window > 0 else 0.0
        per_stream_fps = [round(p["frames"] / (p["loop_end"] - p["loop_start"]), 2)
                          for p in per_stream.values() if p.get("frames")]
        entry = {
            "instances": count,
            "streams_completed": len(per_stream),
            "frames_total": total_frames,
            "window_seconds": round(window, 3),
            "aggregate_fps": round(aggregate, 2),
            "per_stream_fps": per_stream_fps,
            "per_stream_mean_infer_ms": [round(p["mean_infer_ms"], 2) for p in per_stream.values()
                                         if "mean_infer_ms" in p],
            "per_stream_core_mask_active": [p.get("core_mask_active") for p in per_stream.values()],
            "detections_total": sum(p.get("detections", 0) for p in per_stream.values()),
            "errors": errors,
        }
        report["results"][str(count)] = entry
        print(f"instances={count}: aggregate {aggregate:.2f} FPS, per stream {per_stream_fps} FPS, "
              f"{total_frames} frames in {window:.2f} s" + (f", errors: {errors}" if errors else ""))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
