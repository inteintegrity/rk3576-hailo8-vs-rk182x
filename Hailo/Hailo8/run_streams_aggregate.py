"""Hailo-8 multi-stream aggregate throughput, on HailoRT's asynchronous pipeline.

N reader threads decode and letterbox N videos in parallel and push frames into one queue; a
single InferModel on the device serves all of them with `--async-depth` inferences kept in flight,
so the device keeps working while the host decodes. This is the Hailo way of serving many streams
(one device, internally pipelined) rather than the Rockchip way (one core per stream), and it is
what `hailortcli run` does internally.

    python run_streams_aggregate.py --hef yolo26n_hailo8_official.hef \
        --video videos/derived/test_640.mp4 --streams 8 --frames 200 --json out.json
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

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parents[1] / "common", _HERE.parents[2] / "common"):
    if (_candidate / "postprocess_yolo11.py").is_file():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise SystemExit("cannot locate common/postprocess_yolo11.py")
from hailo_async import AsyncPipeline  # noqa: E402
from postprocess_yolo11 import decode_detections, letterbox  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402


def split_heads(outputs: dict):
    """Per-scale box/score heads from one frame's output dict."""
    strides = {80: 8, 40: 16, 20: 32}
    by_stride: dict[int, dict[str, np.ndarray]] = {}
    for name, array in outputs.items():
        squeezed = np.squeeze(np.asarray(array))
        if squeezed.ndim != 3:
            raise RuntimeError(f"{name}: unexpected rank {array.shape}")
        if squeezed.shape[-1] in (4, 64, 80):
            height, _, channels = squeezed.shape
            head = squeezed
        elif squeezed.shape[0] in (4, 64, 80):
            channels, height, _ = squeezed.shape
            head = np.transpose(squeezed, (1, 2, 0))
        else:
            raise RuntimeError(f"{name}: cannot identify layout of {array.shape}")
        branch = "box" if channels in (4, 64) else "cls"
        by_stride.setdefault(strides[height], {})[branch] = head
    box_heads = [by_stride[stride]["box"] for stride in sorted(by_stride)]
    cls_heads = [by_stride[stride]["cls"] for stride in sorted(by_stride)]
    return box_heads, cls_heads


def reader(stream_id: int, video: str, frames: int, inbox: queue.Queue) -> None:
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        return
    # decorrelate the streams: every reader starts at a different offset
    capture.set(cv2.CAP_PROP_POS_FRAMES, stream_id * 7 % 394)
    sent = 0
    while sent < frames:
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        padded = letterbox(frame, 640)[0]
        inbox.put((stream_id, np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))))
        sent += 1
    capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--frames", type=int, default=200, help="frames per stream")
    parser.add_argument("--async-depth", type=int, default=8,
                        help="inferences kept in flight on the device")
    parser.add_argument("--model-family", choices=("yolo11", "yolo26"), default="yolo26")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--no-decode", action="store_true",
                        help="skip the host decode/NMS to isolate the device contribution")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    decode = decode_detections_yolo26 if args.model_family == "yolo26" else decode_detections
    pipeline = AsyncPipeline(args.hef, depth=args.async_depth)
    print(f"async depth {args.async_depth}, streams {args.streams}, frames/stream {args.frames}")

    per_stream_frames = [0] * args.streams
    per_stream_detections = [0] * args.streams
    inbox: queue.Queue = queue.Queue(maxsize=args.async_depth * 4)
    total_frames = args.streams * args.frames

    try:
        # warm-up on the first frame of the clip
        warm_capture = cv2.VideoCapture(str(args.video))
        ok, warm_frame = warm_capture.read()
        warm_capture.release()
        if ok:
            padded, _, _, _ = letterbox(warm_frame, 640)
            warm_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
            for _ in range(args.async_depth):
                pipeline.submit(warm_input)
            for _ in range(args.async_depth):
                pipeline.collect()

        threads = [threading.Thread(target=reader, args=(i, str(args.video), args.frames, inbox),
                                    daemon=True)
                   for i in range(args.streams)]
        for thread in threads:
            thread.start()

        processed = 0
        started = time.perf_counter()
        while processed < total_frames:
            # fill the pipeline from the queue, then collect the oldest completion: this keeps
            # async_depth inferences in flight while the host decodes or waits for the readers
            while pipeline.free_slots() > 0:
                try:
                    item = inbox.get(timeout=0.01)
                except queue.Empty:
                    break
                stream_id, frame = item
                pipeline.submit(frame[None], meta=stream_id)
            if pipeline.in_flight_count() == 0:
                if not any(thread.is_alive() for thread in threads):
                    break
                time.sleep(0.001)
                continue
            stream_id, outputs = pipeline.collect()
            processed += 1
            per_stream_frames[stream_id] += 1
            if not args.no_decode:
                box_heads, cls_heads = split_heads(outputs)
                _, scores, _ = decode(box_heads, cls_heads, conf_thres=args.conf,
                                      iou_thres=args.iou)
                per_stream_detections[stream_id] += int(len(scores))
        wall = time.perf_counter() - started
        for thread in threads:
            thread.join(timeout=5)
    finally:
        pipeline.close()

    aggregate = processed / wall if wall else 0.0
    report = {
        "backend": "hailo8",
        "hef": args.hef.name,
        "video": args.video.name,
        "streams": args.streams,
        "frames_per_stream_target": args.frames,
        "async_depth": args.async_depth,
        "no_decode": args.no_decode,
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


if __name__ == "__main__":
    main()