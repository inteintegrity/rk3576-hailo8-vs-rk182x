"""Process-per-stream version of the live-FPS demo, for the numbers a single Python process cannot reach.

make_demo_video.py runs every stream as a thread in one interpreter, so the host-side decode and
NMS of all streams share one GIL: measured on this board that ceiling is about 57 frames/s in
total, which is what the eight-stream RK182x clip shows (~7.2 FPS per stream) even though the
module itself serves 88.6 FPS when the host work is spread over separate processes - the way the
published multi-stream measurement was taken.

This script is the same demo with one process per stream. Workers own their own runtime, annotate
their own tile and publish it (JPEG) through a queue; the parent only stitches the newest tile of
each stream into the mosaic and writes the MP4, so the numbers on screen are the ones the device
can actually be driven to.

    python make_demo_video_processes.py --backend rk1820 \
        --model yolo26n_rk1820_int8.rknn --weight yolo26n_rk1820_int8.weight \
        --video clip.mp4 --streams 8 --seconds 20 --video-fps 4 --tile-width 256 --out out.mp4

    # the same for the built-in NPU (rk3576) or Hailo-8 (hailo) shares the worker code paths
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys
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


def clip_boxes(boxes, scale, pad_left, pad_top, frame) -> np.ndarray:
    boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame.shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame.shape[0])
    return boxes


def worker(backend: str, stream_id: int, model: str, weight: str, video: str, size: int,
           tile_w: int, tile_h: int, conf: float, iou: float, core: int, seconds: float,
           stop, out_queue, limits: dict) -> None:
    """Own runtime, own core, own GIL: annotate this stream's tile and publish it."""
    if backend == "rk1820":
        from rknn3lite.api.rknn3_lite import RKNN3Lite

        from rknn_runner import (collect_heads, collect_heads_yolo26, dequantize,
                                 init_runtime_with_fallback, run_inference)

        family = "yolo26" if "yolo26" in Path(model).name else "yolo11"
        rknn = RKNN3Lite()
        if rknn.load_rknn(model_path=model, weight_path=weight) != 0:
            raise SystemExit("rknn3lite load_rknn failed")
        init_runtime_with_fallback(rknn, 0x1, lambda message: None)
        attrs = rknn.get_outputs_tensor_attr()

        def infer(frame):
            padded, scale, pad_left, pad_top = letterbox(frame, size)
            network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
            raw = run_inference(rknn, network_input, "nhwc", attrs)
            if family == "yolo26":
                box_heads, cls_heads, _ = collect_heads_yolo26(raw)
            else:
                box_heads, cls_heads = collect_heads(raw)
            return box_heads, cls_heads, scale, pad_left, pad_top
    else:
        from rknnlite.api import RKNNLite

        from rknn_runner import collect_heads, collect_heads_yolo26, dequantize

        family = "yolo26" if "yolo26" in Path(model).name else "yolo11"
        rknn = RKNNLite()
        if rknn.load_rknn(model) != 0:
            raise SystemExit("rknnlite load_rknn failed")
        if rknn.init_runtime(core_mask=1 << core) != 0:
            raise SystemExit(f"rknnlite init_runtime failed for core {core}")

        def infer(frame):
            padded, scale, pad_left, pad_top = letterbox(frame, size)
            network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
            raw = dequantize(rknn.inference(inputs=[network_input], data_format="nhwc"), None)
            if family == "yolo26":
                box_heads, cls_heads, _ = collect_heads_yolo26(raw)
            else:
                box_heads, cls_heads = collect_heads(raw)
            return box_heads, cls_heads, scale, pad_left, pad_top

    parent = os.getppid()
    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, stream_id * 7 % 394)
    stamps: deque[float] = deque(maxlen=31)
    count = 0
    first: float | None = None
    infer_ms: list[float] = []
    host_ms: list[float] = []
    phase_ms: dict[str, list[float]] = {"decode": [], "tile": [], "encode": []}
    started = time.perf_counter()
    while not stop.is_set() and time.perf_counter() - started < seconds:
        if os.getppid() != parent:            # orphaned (parent killed): leave the device alone
            break
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        device_started = time.perf_counter()
        box_heads, cls_heads, scale, pad_left, pad_top = infer(frame)
        device_ms = (time.perf_counter() - device_started) * 1000.0
        host_started = time.perf_counter()
        decode = (decode_detections_yolo26 if box_heads[0].shape[-1] == 4 else decode_detections)
        boxes, scores, classes = decode(box_heads, cls_heads, conf_thres=conf, iou_thres=iou)
        boxes = clip_boxes(boxes, scale, pad_left, pad_top, frame)
        phase_ms["decode"].append((time.perf_counter() - host_started) * 1000.0)

        tile_started = time.perf_counter()
        tile = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        scale_x = tile_w / frame.shape[1]
        scale_y = tile_h / frame.shape[0]
        for position in np.argsort(-scores)[:12]:
            x1, y1, x2, y2 = boxes[position]
            class_id = int(classes[position])
            colour = PALETTE[class_id % len(PALETTE)]
            top_left = (int(x1 * scale_x), int(y1 * scale_y))
            bottom_right = (int(x2 * scale_x), int(y2 * scale_y))
            cv2.rectangle(tile, top_left, bottom_right, colour, 2)
            cv2.putText(tile, f"{COCO_NAMES[class_id]} {scores[position]:.2f}",
                        (top_left[0], max(top_left[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        colour, 1, cv2.LINE_AA)

        now = time.perf_counter()
        if first is None:
            first = now
        stamps.append(now)
        count += 1
        span = stamps[-1] - stamps[0]
        live = ((len(stamps) - 1) / span) if len(stamps) > 1 and span > 0 else 0.0
        avg = ((count - 1) / (stamps[-1] - first)) if count > 1 else 0.0
        phase_ms["tile"].append((time.perf_counter() - tile_started) * 1000.0)
        encode_started = time.perf_counter()
        ok, encoded = cv2.imencode(".jpg", tile, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        phase_ms["encode"].append((time.perf_counter() - encode_started) * 1000.0)
        if not ok:
            continue
        infer_ms.append(device_ms)
        host_ms.append((time.perf_counter() - host_started) * 1000.0)
        message = (stream_id, encoded.tobytes(), live, avg, count)
        try:
            out_queue.put_nowait(message)
        except queue.Full:
            try:                       # replace a stale tile: only the newest one matters
                out_queue.get_nowait()
                out_queue.put_nowait(message)
            except queue.Empty:
                pass
    capture.release()
    def mean(values):
        return sum(values) / len(values) if values else 0.0

    limits[stream_id] = (count, time.perf_counter() - started,            # frames, work window
                         round(mean(infer_ms), 2), round(mean(host_ms), 2),
                         {name: round(mean(values), 2) for name, values in phase_ms.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("rk1820", "rk3576"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weight", type=Path, help="rk1820 only")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--cores", type=int, default=2, help="rk3576: cores to pin streams to")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--video-fps", type=float, default=4.0,
                        help="mosaic frames written per wall-clock second")
    parser.add_argument("--tile-width", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "rk1820" and not args.weight:
        raise SystemExit("--weight is required for --backend rk1820")

    probe = cv2.VideoCapture(str(args.video))
    ok, sample = probe.read()
    probe.release()
    if not ok:
        raise SystemExit(f"cannot read {args.video}")
    frame_h, frame_w = sample.shape[:2]
    tile_w = args.tile_width
    tile_h = int(round(tile_w * frame_h / frame_w))
    columns = args.streams if args.streams <= 2 else 4
    rows = (args.streams + columns - 1) // columns

    stop = mp.Event()
    out_queue = mp.Queue(maxsize=128)
    limits: dict = mp.Manager().dict()
    processes = [
        mp.Process(target=worker,
                   args=(args.backend, index, str(args.model), str(args.weight or ""),
                         str(args.video), args.size, tile_w, tile_h, args.conf, args.iou,
                         index % args.cores, args.seconds, stop, out_queue, limits),
                   daemon=True)
        for index in range(args.streams)
    ]
    for process in processes:
        process.start()

    label = {"rk1820": "RK3576 + RK182x (M.2, one process per stream)",
             "rk3576": "RK3576 built-in NPU (one process per stream)"}[args.backend]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps,
                             (columns * tile_w, 30 + rows * tile_h))
    latest = {index: None for index in range(args.streams)}
    history: deque[tuple[float, int]] = deque()
    started = time.perf_counter()
    next_frame_at = started
    written = 0
    compose_ms: list[float] = []
    try:
        while time.perf_counter() - started < args.seconds:
            now = time.perf_counter()
            if now < next_frame_at:
                time.sleep(min(0.003, next_frame_at - now))
                continue
            if now - next_frame_at > 2.0 / args.video_fps:
                next_frame_at = now
            next_frame_at += 1.0 / args.video_fps
            while True:                      # keep only the newest tile of each stream
                try:
                    message = out_queue.get_nowait()
                except queue.Empty:
                    break
                latest[message[0]] = message
            total = sum(message[4] for message in latest.values() if message is not None)
            elapsed = now - started
            history.append((now, total))
            while history and now - history[0][0] > 2.0:
                history.popleft()
            if len(history) > 1:
                (t0, n0), (t1, n1) = history[0], history[-1]
                aggregate_live = (n1 - n0) / (t1 - t0) if t1 > t0 else 0.0
            else:
                aggregate_live = 0.0
            overall = total / elapsed if elapsed > 0 else 0.0
            header = (f"{label} | {args.streams} streams | aggregate {aggregate_live:5.1f} FPS live "
                      f"/ {overall:5.1f} avg | {elapsed:4.1f}s")
            canvas = np.zeros((30 + rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
            cv2.putText(canvas, header, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 2, cv2.LINE_AA)
            for index, message in enumerate(latest.values()):
                column, row = index % columns, index // columns
                cell = canvas[30 + row * tile_h:30 + (row + 1) * tile_h,
                              column * tile_w:(column + 1) * tile_w]
                if message is None:
                    cv2.putText(cell, f"stream {index + 1}: waiting", (8, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
                    continue
                _, jpeg, live, avg, _ = message
                tile = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if tile is not None and tile.shape[:2] == (tile_h, tile_w):
                    cell[:] = tile
                for line_index, line in enumerate((f"stream {index + 1}",
                                                   f"{live:5.1f} FPS live",
                                                   f"{avg:5.1f} FPS avg")):
                    # same place, same text; only a larger font
                    cv2.putText(cell, line, (8, 18 + line_index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.85 if line_index == 0 else 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            frame_started = time.perf_counter()
            writer.write(canvas)
            compose_ms.append((time.perf_counter() - frame_started) * 1000.0)
            written += 1
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for process in processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
        writer.release()

    wall_after_loop = time.perf_counter() - started
    elapsed = wall_after_loop
    entries = [limits.get(index) for index in range(args.streams)]
    counts = [int(entry[0]) if entry else 0 for entry in entries]
    windows = [float(entry[1]) if entry else 0.0 for entry in entries]
    work_window = max(windows) if windows else 0.0
    total = sum(counts)
    rates = [count / window if window > 0 else 0.0 for count, window in zip(counts, windows)]
    print(f"wrote {args.out} ({written} mosaic frames at {args.video_fps:.1f} fps, "
          f"wall {elapsed:.1f} s, worker window {work_window:.1f} s)")
    if compose_ms:
        ordered = sorted(compose_ms)
        median = ordered[len(ordered) // 2]
        print(f"mosaic composer: {written} frames in {wall_after_loop:.1f} s = "
              f"{written / wall_after_loop if wall_after_loop else 0:.1f} fps achieved, "
              f"{median:.1f} ms median per frame (ceiling {1000.0 / median if median else 0:.0f} fps)")
    entries = [limits.get(index) for index in range(args.streams)]
    device = [entry[2] for entry in entries if entry]
    host = [entry[3] for entry in entries if entry]
    if device and host:
        print(f"per worker per frame: infer (device round trip) {sum(device)/len(device):.1f} ms, "
              f"host {sum(host)/len(host):.1f} ms")
        phases = [entry[4] for entry in entries if entry and len(entry) > 4]
        if phases:
            for name in ("decode", "tile", "encode"):
                values = [phase[name] for phase in phases if name in phase]
                if values:
                    print(f"    host/{name:6s}: {sum(values)/len(values):5.1f} ms")
    print(f"mosaic composer achieved {written / args.seconds:.1f} fps "
          f"(requested {args.video_fps:.1f}) over the {args.seconds:.0f} s run")
    print(f"aggregate {total / work_window:.2f} FPS | per stream: "
          + ", ".join(f"{rate:.2f}" for rate in rates)
          + "  (inference + decode + tile annotation; mosaic encoding happens in the parent)")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()