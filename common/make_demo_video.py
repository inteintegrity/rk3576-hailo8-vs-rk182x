"""Demo videos whose on-screen FPS actually moves.

The benchmark runners draw one summary rate per run, so the number on those clips never changes.
This script is the opposite: every frame carries the *live* rolling rate of its own stream, and a
multi-stream run is stitched into one mosaic whose header shows the aggregate live rate, so the
difference between "fixed total throughput" (Hailo-8) and "scales with cores" (RK3576) can be
watched instead of read.

    python make_demo_video.py --backend hailo --hef yolo26n.hef --video clip.mp4 \
        --streams 1 --seconds 20 --out demo_hailo_1stream.mp4
    python make_demo_video.py --backend hailo --hef yolo26n.hef --video clip.mp4 \
        --streams 8 --seconds 20 --out demo_hailo_8stream.mp4
    python make_demo_video.py --backend rk3576 --model yolo26n_rk3576_int8.rknn \
        --video clip.mp4 --streams 2 --cores 2 --seconds 20 --out demo_rk3576_2stream.mp4

Concurrency model, so that a shared device is driven safely:
  Hailo-8    reader threads only read and letterbox into a queue; a single inference thread owns
             the async pipeline (submit while a slot is free, collect the oldest completion). Same
             shape as the validated aggregate benchmark.
  RK3576     one thread per stream, each owning its own RKNNLite instance pinned to its own NPU
             core - the one-stream-per-core model the benchmark uses.

The mosaic is composed on a wall-clock timer (--video-fps, default 15) rather than once per
inference, so the clip plays back at roughly real speed and its length is the measured duration.
Numbers on screen include drawing and encoding: they are demo figures, not the benchmark ones.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parents[0] / "common", _HERE.parents[1] / "common"):
    if (_candidate / "postprocess_yolo11.py").is_file():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise SystemExit("cannot locate common/postprocess_yolo11.py")
from draw_detections import COCO_NAMES, PALETTE  # noqa: E402
from postprocess_yolo11 import decode_detections, letterbox, unletterbox_boxes  # noqa: E402
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
    return ([by_stride[stride]["box"] for stride in sorted(by_stride)],
            [by_stride[stride]["cls"] for stride in sorted(by_stride)])


def detections_from(scores, classes, boxes, limit: int = 60) -> list[dict]:
    detections = []
    for position in np.argsort(-scores)[:limit]:
        class_id = int(classes[position])
        detections.append({
            "class_id": class_id,
            "class_name": COCO_NAMES[class_id],
            "confidence": float(scores[position]),
            "box_xyxy": boxes[position].tolist(),
        })
    return detections


def clip_boxes(boxes, scale, pad_left, pad_top, frame) -> np.ndarray:
    boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame.shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame.shape[0])
    return boxes


class StreamState:
    """Newest annotated frame for one stream, plus that stream's own rolling rate.

    The rate comes from completion stamps (intervals between this stream's decoded frames), not
    from 1/inference-time, so waiting behind a shared device shows up - which is the point when
    eight streams share one Hailo-8.
    """

    def __init__(self) -> None:
        self.frame = None
        self.detections: list[dict] = []
        self.stamps: deque[float] = deque(maxlen=31)
        self.count = 0
        self.first_stamp: float | None = None

    @property
    def live_fps(self) -> float:
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        return ((len(self.stamps) - 1) / span) if span > 0 else 0.0

    @property
    def average_fps(self) -> float:
        if self.first_stamp is None or self.count < 2:
            return 0.0
        span = self.stamps[-1] - self.first_stamp
        return ((self.count - 1) / span) if span > 0 else 0.0

    def completed(self) -> None:
        now = time.perf_counter()
        if self.first_stamp is None:
            self.first_stamp = now
        self.stamps.append(now)
        self.count += 1


def draw_tile(canvas: np.ndarray, column: int, row: int, tile_w: int, tile_h: int,
              state: StreamState, stream_id: int) -> None:
    """Scale one stream's frame into its mosaic cell and annotate it there."""
    tile = cv2.resize(state.frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    scale_x = tile_w / state.frame.shape[1]
    scale_y = tile_h / state.frame.shape[0]
    # drawing on a small tile is cheap per box but adds up over many tiles: keep the budget bounded
    for detection in state.detections[:12]:
        x1, y1, x2, y2 = detection["box_xyxy"]
        top_left = (int(x1 * scale_x), int(y1 * scale_y))
        bottom_right = (int(x2 * scale_x), int(y2 * scale_y))
        colour = PALETTE[detection["class_id"] % len(PALETTE)]
        cv2.rectangle(tile, top_left, bottom_right, colour, 2)
        cv2.putText(tile, f"{detection['class_name']} {detection['confidence']:.2f}",
                    (top_left[0], max(top_left[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
    for index, line in enumerate((f"stream {stream_id + 1}",
                                  f"{state.live_fps:5.1f} FPS live",
                                  f"{state.average_fps:5.1f} FPS avg")):
        # same place, same text as before - only a larger font, so the lines are spaced further
        # apart than they used to be (a bigger glyph would otherwise sit on the next line)
        cv2.putText(tile, line, (8, 18 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85 if index == 0 else 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    canvas[row * tile_h:(row + 1) * tile_h, column * tile_w:(column + 1) * tile_w] = tile


def compose(states: list[StreamState], columns: int, tile_w: int, tile_h: int,
            header: str) -> np.ndarray:
    rows = (len(states) + columns - 1) // columns
    header_h = 30
    canvas = np.zeros((header_h + rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    cv2.putText(canvas, header, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                cv2.LINE_AA)
    for index, state in enumerate(states):
        if state.frame is not None:
            draw_tile(canvas, index % columns, index // columns, tile_w, tile_h, state, index)
    return canvas


def start_reader_threads(video: str, streams: int, size: int, inbox: queue.Queue,
                         stop: threading.Event) -> list[threading.Thread]:
    """One thread per stream: read + letterbox, then hand the tensor to the inference stage."""

    def reader(stream_id: int) -> None:
        capture = cv2.VideoCapture(video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, stream_id * 7 % 394)
        while not stop.is_set():
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            padded, scale, pad_left, pad_top = letterbox(frame, size)
            network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
            # block while the queue is full (a demo should not drop frames, and greedy readers
            # would spin on the GIL and starve the inference thread), but stay interruptible so
            # that stopping the run does not leave threads waiting on a queue nobody drains
            while not stop.is_set():
                try:
                    inbox.put((stream_id, network_input, frame, scale, pad_left, pad_top),
                              timeout=0.2)
                    break
                except queue.Full:
                    continue
        capture.release()

    return [threading.Thread(target=reader, args=(i,), daemon=True) for i in range(streams)]


def hailo_worker(hef: Path, depth: int, conf: float, iou: float, states: list[StreamState],
                 inbox: queue.Queue, stop: threading.Event):
    """Single inference thread owning the device pipeline; readers only feed the queue."""
    from hailo_async import AsyncPipeline

    pipeline = AsyncPipeline(hef, depth=depth)

    def process(meta, outputs) -> None:
        stream_id, frame, scale, pad_left, pad_top = meta
        box_heads, cls_heads = split_heads(outputs)
        decode = decode_detections_yolo26 if box_heads[0].shape[-1] == 4 else decode_detections
        boxes, scores, classes = decode(box_heads, cls_heads, conf_thres=conf, iou_thres=iou)
        boxes = clip_boxes(boxes, scale, pad_left, pad_top, frame)
        state = states[stream_id]
        state.frame = frame
        state.detections = detections_from(scores, classes, boxes)
        state.completed()

    def worker() -> None:
        try:
            while not stop.is_set():
                while pipeline.free_slots() > 0:
                    try:
                        stream_id, network_input, frame, scale, pad_left, pad_top = inbox.get(
                            timeout=0.02)
                    except queue.Empty:
                        break
                    pipeline.submit(network_input[None],
                                    meta=(stream_id, frame, scale, pad_left, pad_top))
                if pipeline.in_flight_count() == 0:
                    continue
                meta, outputs = pipeline.collect()
                process(meta, outputs)
        finally:
            # collect whatever is still queued and leave: the orderly teardown can block in this
            # binding, and main() ends with os._exit() so a hung teardown cannot trap the process
            while pipeline.in_flight_count() > 0:
                pipeline.collect()

    return threading.Thread(target=worker, daemon=True), pipeline


def rk1820_workers(model: Path, weight: Path, conf: float, iou: float, size: int,
                   states: list[StreamState], video: str, streams: int,
                   stop: threading.Event):
    """One thread per stream, each with its own RKNN3 runtime on the module.

    The runtime requires core_mask to match the model's compile-time core_num (our YOLO26 model was
    built with core_num = 1), so every instance asks for one core and the device schedules the
    contexts - the same shape as the published multi-stream measurement.
    """
    from rknn3lite.api.rknn3_lite import RKNN3Lite

    from rknn_runner import (collect_heads, collect_heads_yolo26, dequantize,
                             init_runtime_with_fallback, run_inference)

    family = "yolo26" if "yolo26" in model.name else "yolo11"
    runtimes = []
    for _ in range(streams):
        rknn = RKNN3Lite()
        if rknn.load_rknn(model_path=str(model), weight_path=str(weight)) != 0:
            raise SystemExit("rknn3lite load_rknn failed")
        init_runtime_with_fallback(rknn, 0x1, print)
        runtimes.append((rknn, rknn.get_outputs_tensor_attr()))

    def worker(stream_id: int) -> None:
        rknn, attrs = runtimes[stream_id]
        capture = cv2.VideoCapture(video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, stream_id * 7 % 394)
        while not stop.is_set():
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            padded, scale, pad_left, pad_top = letterbox(frame, size)
            # RKNN3 wants the batch dimension kept: (1, H, W, C) uint8
            network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
            raw = run_inference(rknn, network_input, "nhwc", attrs)
            if family == "yolo26":
                box_heads, cls_heads, _ = collect_heads_yolo26(raw)
            else:
                box_heads, cls_heads = collect_heads(raw)
            decode = (decode_detections_yolo26 if box_heads[0].shape[-1] == 4
                      else decode_detections)
            boxes, scores, classes = decode(box_heads, cls_heads, conf_thres=conf, iou_thres=iou)
            boxes = clip_boxes(boxes, scale, pad_left, pad_top, frame)
            state = states[stream_id]
            state.frame = frame
            state.detections = detections_from(scores, classes, boxes)
            state.completed()
        capture.release()

    return [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(streams)], runtimes


def rk3576_workers(model: Path, conf: float, iou: float, size: int, states: list[StreamState],
                   cores: int, video: str, streams: int, stop: threading.Event):
    """One thread per stream, each owning its own runtime pinned to its own NPU core."""
    from rknnlite.api import RKNNLite

    from rknn_runner import collect_heads, collect_heads_yolo26, dequantize

    family = "yolo26" if "yolo26" in model.name else "yolo11"
    runtimes = []
    # One runtime per stream, each pinned round-robin to a core. With more streams than cores this
    # deliberately oversubscribes the NPU (the benchmark only measured one stream per core), and
    # giving every stream its own instance keeps a runtime object out of two threads at once.
    for index in range(streams):
        rknn = RKNNLite()
        if rknn.load_rknn(str(model)) != 0:
            raise SystemExit("rknnlite load_rknn failed")
        core = index % cores
        if rknn.init_runtime(core_mask=1 << core) != 0:
            raise SystemExit(f"rknnlite init_runtime failed for core {core}")
        runtimes.append(rknn)

    def worker(stream_id: int) -> None:
        rknn = runtimes[stream_id]              # one runtime per stream, see above
        capture = cv2.VideoCapture(video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, stream_id * 7 % 394)
        while not stop.is_set():
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            padded, scale, pad_left, pad_top = letterbox(frame, size)
            network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
            raw = dequantize(rknn.inference(inputs=[network_input], data_format="nhwc"), None)
            if family == "yolo26":
                box_heads, cls_heads, _ = collect_heads_yolo26(raw)
            else:
                box_heads, cls_heads = collect_heads(raw)
            decode = (decode_detections_yolo26 if box_heads[0].shape[-1] == 4
                      else decode_detections)
            boxes, scores, classes = decode(box_heads, cls_heads, conf_thres=conf, iou_thres=iou)
            boxes = clip_boxes(boxes, scale, pad_left, pad_top, frame)
            state = states[stream_id]
            state.frame = frame
            state.detections = detections_from(scores, classes, boxes)
            state.completed()
        capture.release()

    return [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(streams)], runtimes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hailo", "rk3576", "rk1820"), required=True)
    parser.add_argument("--hef", type=Path, help="--backend hailo")
    parser.add_argument("--model", type=Path, help="--backend rk3576 / rk1820")
    parser.add_argument("--weight", type=Path, help="--backend rk1820 (.weight next to the .rknn)")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--pipeline-depth", type=int, default=8,
                        help="inferences kept in flight on the device")
    parser.add_argument("--cores", type=int, default=2, help="RK3576 NPU cores to use")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--video-fps", type=float, default=15.0,
                        help="mosaic frames written per wall-clock second")
    parser.add_argument("--tile-width", type=int, default=0,
                        help="0 picks 640 for one stream, 320 for several")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    probe = cv2.VideoCapture(str(args.video))
    ok, sample = probe.read()
    probe.release()
    if not ok:
        raise SystemExit(f"cannot read {args.video}")
    frame_h, frame_w = sample.shape[:2]
    tile_w = args.tile_width or (frame_w if args.streams == 1 else 320)
    tile_h = int(round(tile_w * frame_h / frame_w))
    columns = args.streams if args.streams <= 2 else 4

    states = [StreamState() for _ in range(args.streams)]
    label = {"hailo": "RK3576 + Hailo-8",
             "rk3576": "RK3576 built-in NPU",
             "rk1820": "RK3576 + RK182x (M.2)"}[args.backend]
    stop = threading.Event()
    inbox: queue.Queue = queue.Queue(maxsize=max(8, args.pipeline_depth * 2))
    runtimes: list = []
    infer_thread = None

    if args.backend == "hailo":
        if not args.hef:
            raise SystemExit("--hef is required for --backend hailo")
        infer_thread, _ = hailo_worker(args.hef, args.pipeline_depth, args.conf, args.iou,
                                       states, inbox, stop)
        threads = start_reader_threads(str(args.video), args.streams, args.size, inbox, stop)
    elif args.backend == "rk3576":
        if not args.model:
            raise SystemExit("--model is required for --backend rk3576")
        threads, runtimes = rk3576_workers(args.model, args.conf, args.iou, args.size, states,
                                           args.cores, str(args.video), args.streams, stop)
    else:
        if not args.model or not args.weight:
            raise SystemExit("--model and --weight are required for --backend rk1820")
        threads, runtimes = rk1820_workers(args.model, args.weight, args.conf, args.iou, args.size,
                                           states, str(args.video), args.streams, stop)

    for thread in threads:
        thread.start()
    if infer_thread is not None:
        infer_thread.start()

    mosaic_w = columns * tile_w
    mosaic_h = 30 + ((args.streams + columns - 1) // columns) * tile_h
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps,
                             (mosaic_w, mosaic_h))
    started = time.perf_counter()
    next_frame_at = started
    written = 0
    history: deque[tuple[float, int]] = deque()

    try:
        while time.perf_counter() - started < args.seconds:
            now = time.perf_counter()
            if now < next_frame_at:
                time.sleep(min(0.002, next_frame_at - now))
                continue
            if now - next_frame_at > 2.0 / args.video_fps:
                # composing fell behind (many tiles, slow encode): resync instead of writing a
                # burst of frames back to back, which would just starve the inference threads
                next_frame_at = now
            next_frame_at += 1.0 / args.video_fps
            elapsed = now - started
            total_completed = sum(state.count for state in states)
            history.append((now, total_completed))
            while history and now - history[0][0] > 2.0:
                history.popleft()
            if len(history) > 1:
                (t0, n0), (t1, n1) = history[0], history[-1]
                aggregate_live = (n1 - n0) / (t1 - t0) if t1 > t0 else 0.0
            else:
                aggregate_live = 0.0
            overall = total_completed / elapsed if elapsed > 0 else 0.0
            header = (f"{label} | {args.streams} stream(s) | aggregate "
                      f"{aggregate_live:5.1f} FPS live / {overall:5.1f} avg | {elapsed:4.1f}s")
            writer.write(compose(states, columns, tile_w, tile_h, header))
            written += 1
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=3)
        if infer_thread is not None:
            infer_thread.join(timeout=5)
        writer.release()
        for entry in runtimes:
            runtime = entry[0] if isinstance(entry, tuple) else entry   # rk1820 keeps (runtime, attrs)
            if hasattr(runtime, "release"):
                runtime.release()

    elapsed = time.perf_counter() - started
    total = sum(state.count for state in states)
    print(f"wrote {args.out} ({written} mosaic frames at {args.video_fps:.0f} fps, "
          f"{elapsed:.1f} s of run)")
    print(f"aggregate {total / elapsed:.2f} FPS including drawing and encoding | per stream avg: "
          + ", ".join(f"{state.average_fps:.2f}" for state in states))
    sys.stdout.flush()
    # Leave immediately: the video is written, and skipping HailoRT's teardown avoids both its
    # occasional hang and its abort-at-exit behaviour.
    os._exit(0)


if __name__ == "__main__":
    main()