"""Hailo-8 aggregate throughput for N concurrent streams on one device.

N reader threads decode and letterbox N videos in parallel and push frames into one queue;
a single InferModel on the device serves all of them with `--depth` inferences kept in
flight, so the device keeps working while the host decodes. This is how Hailo's architecture
serves many streams - one device, internally pipelined - as opposed to the Rockchip way of
one stream per NPU core, and it is what `hailortcli run` does internally.

Reported number: aggregate throughput, i.e. total processed frames per second across all
streams (the same metric the Rockchip multi-stream benchmark reports).

Usage:
    python3 Hailo/Hailo8/run_streams_aggregate.py \
        --hef model/Hailo/yolo26n_hailo8_official.hef \
        --video <clip.mp4> --streams 8 --frames 200 --depth 8 --json out.json

    --no-decode reports the device-only rate: frames are still submitted and collected, but
    the host decode/NMS is skipped, which isolates the device contribution.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

DECODE_IMPLEMENTATION = "common/postprocess_common.py + common/postprocess_yolo26.py (shared with both RK runners)"


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
from hailo_pipeline import HailoPipeline, collect_heads  # noqa: E402
from postprocess_common import letterbox, to_network_input  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402


def reader(stream_id: int, video: str, frames: int, frame_count: int, inbox: queue.Queue) -> None:
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        return
    # decorrelate the streams: every reader starts at a different offset
    offset = (stream_id * 7) % max(frame_count, 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, offset)
    sent = 0
    while sent < frames:
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        inbox.put((stream_id, to_network_input(letterbox(frame, 640)[0])))
        sent += 1
    capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hailo-8 aggregate throughput for N concurrent video streams on one device."
    )
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None,
                        help="clip to process; default: the test clip bundled with this "
                             "repository (video/test.mp4)")
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--frames", type=int, default=200, help="frames per stream")
    parser.add_argument("--depth", type=int, default=8, help="inferences kept in flight on the device")
    parser.add_argument("--model-family", choices=("yolo26",), default="yolo26",
                        help="kept for command-line compatibility; this release serves YOLO26n only")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--no-decode", action="store_true",
                        help="skip the host decode/NMS to isolate the device contribution")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    args.video = _resolve_video(args.video, Path(__file__).resolve().parent)

    pipeline = HailoPipeline(args.hef, depth=args.depth)
    print(f"HEF {args.hef.name} sha256={sha256(args.hef)}", flush=True)
    print(f"{args.depth} inference(s) in flight, streams {args.streams}, frames/stream {args.frames}",
          flush=True)

    per_stream_frames = [0] * args.streams
    per_stream_detections = [0] * args.streams
    inbox: queue.Queue = queue.Queue(maxsize=args.depth * 4)
    total_frames = args.streams * args.frames

    source_frames = 0
    try:
        # warm-up on the first frame of the clip
        warm_capture = cv2.VideoCapture(str(args.video))
        source_frames = int(warm_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, warm_frame = warm_capture.read()
        warm_capture.release()
        if ok:
            warm_input = to_network_input(letterbox(warm_frame, 640)[0])
            for _ in range(args.depth):
                pipeline.submit(warm_input)
            for _ in range(args.depth):
                pipeline.collect()

        threads = [threading.Thread(target=reader,
                                    args=(i, str(args.video), args.frames, source_frames, inbox),
                                    daemon=True)
                   for i in range(args.streams)]
        for thread in threads:
            thread.start()

        processed = 0
        started = time.perf_counter()
        while processed < total_frames:
            # fill the pipeline from the queue, then collect the oldest completion: this keeps
            # `depth` inferences in flight while the host decodes or waits for the readers
            while pipeline.free_slots() > 0:
                try:
                    item = inbox.get(timeout=0.01)
                except queue.Empty:
                    break
                stream_id, frame = item
                pipeline.submit(frame, meta=stream_id)
            if pipeline.in_flight_count() == 0:
                if not any(thread.is_alive() for thread in threads):
                    break
                time.sleep(0.001)
                continue
            stream_id, outputs = pipeline.collect()
            processed += 1
            per_stream_frames[stream_id] += 1
            if not args.no_decode:
                box_heads, score_heads = collect_heads(outputs)
                _, scores, _ = decode_detections_yolo26(
                    box_heads, score_heads, conf_thres=args.conf, iou_thres=args.iou
                )
                per_stream_detections[stream_id] += int(len(scores))
        wall = time.perf_counter() - started
        for thread in threads:
            thread.join(timeout=5)
    finally:
        pipeline.close()

    aggregate = processed / wall if wall else 0.0
    report = {
        "backend": "hailo8",
        "device": "Hailo-8 (M.2, PCIe Gen2 x1)",
        "hef": args.hef.name,
        "hef_sha256": sha256(args.hef),
        "video": args.video.name,
        "video_sha256": sha256(args.video),
        "streams": args.streams,
        "frames_per_stream_target": args.frames,
        "depth": args.depth,
        "decode": "host decode + NMS" if not args.no_decode else "skipped (--no-decode)",
        "decode_implementation": DECODE_IMPLEMENTATION,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "frames_processed": processed,
        "wall_seconds": round(wall, 3),
        "aggregate_fps": round(aggregate, 3),
        "per_stream_fps": round(aggregate / args.streams, 2),
        "per_stream_frames": per_stream_frames,
        "per_stream_detections": per_stream_detections,
    }
    print(json.dumps(report, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
