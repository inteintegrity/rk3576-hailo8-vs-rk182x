"""Board-side Hailo-8 live camera inference: USB camera -> Hailo-8 -> on-screen FPS.

Same pipeline as run_video_inference.py (letterbox -> HailoRT InferVStreams -> shared decode ->
shared annotation), driven by a V4L2 camera instead of a file, with the live frame rate drawn on
every frame.

Three ways to watch it, because this board is headless:

    --serve-port 8080   MJPEG over HTTP; open http://<board-ip>:8080/ in a browser
    --window            cv2.imshow (only where a display exists)
    (always)            one status line on the terminal every --print-interval seconds

The camera is configured with v4l2-ctl before OpenCV opens it. That matters on this board: when
OpenCV negotiates the frame rate itself the camera settles at 10 FPS, while an explicit
`v4l2-ctl --set-parm` reaches 30 FPS. Auto exposure is also worth pinning: in a dim room the
camera stretches the exposure to ~31 ms, which caps capture at 10 FPS no matter what the
pipeline can do (--exposure-ms / --gain).

Usage:
    # 640x480 at the camera's 30 FPS, live FPS in the terminal and in a browser
    python run_camera_inference.py --hef yolo26n_hailo8_official.hef \
        --model-family yolo26 --device /dev/video0 --width 640 --height 480 \
        --camera-fps 30 --exposure-ms 10 --gain 48 --serve-port 8080

    # also record what you see, and keep a snapshot every 10 s
    python run_camera_inference.py ... --record out/camera/live.mp4 --snapshot-every 10
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
sys.path.insert(0, str(_HERE))
from draw_detections import COCO_NAMES, draw, summarise  # noqa: E402
from postprocess_yolo11 import decode_detections, letterbox, unletterbox_boxes  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402
from hailo_runner import collect_heads, sha256  # noqa: E402


def configure_camera(device: str, width: int, height: int, fps: int, pixel_format: str,
                     exposure_ms: float | None, gain: int | None) -> None:
    """Program the V4L2 device before OpenCV opens it; ignore a missing v4l2-ctl."""
    fourcc = pixel_format.upper()
    commands = [["--set-fmt-video", f"width={width},height={height},pixelformat={fourcc}",
                 "--set-parm", str(fps)]]
    if exposure_ms is not None or gain is not None:
        controls = []
        if exposure_ms is not None:
            # units are 100 us; manual mode is required for the value to take effect
            controls += ["--set-ctrl=auto_exposure=1",
                         f"--set-ctrl=exposure_time_absolute={int(exposure_ms * 10)}"]
        if gain is not None:
            controls.append(f"--set-ctrl=gain={gain}")
        commands.append(controls)
    for arguments in commands:
        try:
            result = subprocess.run(["v4l2-ctl", "-d", device, *arguments],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"v4l2-ctl {' '.join(arguments)} failed: {result.stderr.strip()}", flush=True)
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            print(f"v4l2-ctl unavailable ({error}); continuing with OpenCV defaults", flush=True)
            return


class MjpegServer(ThreadingHTTPServer):
    """Serve the newest annotated frame as multipart MJPEG."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state: dict):
        self.state = state
        super().__init__(address, _MjpegHandler)


class _MjpegHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path not in ("/", "/stream.mjpg"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                frame, stamp = self.server.state["frame"], self.server.state["stamp"]
                if frame is None or stamp == self.server.state["sent"]:
                    time.sleep(0.005)
                    continue
                self.server.state["sent"] = stamp
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args) -> None:
        pass


def camera_frame_source(device: str, width: int, height: int):
    """Yield frames in a thread so capture overlaps inference on the main thread."""
    frames: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()
    delivered = {"count": 0}

    def worker() -> None:
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        # one driver buffer halves the capture rate on this board (15 FPS instead of 30), so the
        # default buffer count is kept; the queue below already discards stale frames
        if not capture.isOpened():
            print(f"cannot open camera {device}", flush=True)
            stop.set()
            return
        while not stop.is_set():
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            delivered["count"] += 1
            if frames.full():          # keep only the newest frame, never a backlog
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
            frames.put(frame)
        capture.release()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return frames, stop, delivered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--model-family", choices=("yolo11", "yolo26"), default="yolo26")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--pixel-format", default="MJPG", choices=("MJPG", "YUYV"))
    parser.add_argument("--exposure-ms", type=float, default=10.0,
                        help="pin exposure (0 disables and leaves the camera on auto exposure)")
    parser.add_argument("--gain", type=int, default=48)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--seconds", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--out-dir", type=Path, default=Path("out/camera"))
    parser.add_argument("--record", type=Path, help="write the annotated frames to this MP4")
    parser.add_argument("--snapshot-every", type=float, default=0.0,
                        help="seconds between annotated JPEG snapshots (0 disables)")
    parser.add_argument("--serve-port", type=int, default=0, help="MJPEG HTTP port (0 disables)")
    parser.add_argument("--window", action="store_true", help="cv2.imshow, needs a display")
    parser.add_argument("--print-interval", type=float, default=2.0)
    args = parser.parse_args()

    from hailo_platform import (
        HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
        InputVStreamParams, OutputVStreamParams, VDevice,
    )

    tag = "yolo26n" if args.model_family == "yolo26" else "yolo11n"
    model_label = f"{tag[:-1].upper()}{tag[-1]}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    configure_camera(args.device, args.width, args.height, args.camera_fps, args.pixel_format,
                     args.exposure_ms if args.exposure_ms > 0 else None, args.gain)

    frames, stop_capture, delivered = camera_frame_source(args.device, args.width, args.height)
    first = None
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        try:
            first = frames.get(timeout=0.5)
            break
        except queue.Empty:
            continue
    if first is None:
        raise SystemExit(f"no frame from {args.device} within 5 s")
    height, width = first.shape[:2]
    log(f"camera: {args.device} {width}x{height} requested {args.camera_fps} FPS "
        f"{args.pixel_format}, exposure pin={args.exposure_ms} ms gain={args.gain}")

    hef = HEF(str(args.hef))
    device = VDevice()
    network_group = device.configure(
        hef, ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    )[0]
    input_info = hef.get_input_vstream_infos()[0]
    log(f"HEF: {args.hef.name} sha256={sha256(args.hef)}")
    log(f"input vstream: {input_info.name} {input_info.format.type}")

    in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    server = None
    if args.serve_port:
        server = MjpegServer(("0.0.0.0", args.serve_port), {"frame": None, "stamp": 0, "sent": -1})
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log(f"MJPEG stream: http://<board-ip>:{args.serve_port}/  (Ctrl+C to stop)")

    writer = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(args.camera_fps), (width, height))

    rolling: deque[float] = deque(maxlen=30)   # per-loop milliseconds: averaging instantaneous
    #                                             FPS values would overstate a jittery rate
    recent: deque[tuple[float, float]] = deque(maxlen=1024)   # (time, interval) for a 5 s window
    infer_times: list[float] = []
    pipeline_times: list[float] = []
    class_counter: Counter[str] = Counter()
    stamp = 0
    snapshots: list[Path] = []
    last_snapshot = 0.0
    started = time.perf_counter()
    next_print = started + args.print_interval

    with InferVStreams(network_group, in_params, out_params) as pipeline:
        with network_group.activate():
            for _ in range(5):
                padded, _, _, _ = letterbox(first, 640)
                pipeline.infer({input_info.name: np.ascontiguousarray(
                    cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]})

            try:
                while not stop_capture.is_set():
                    cycle_started = time.perf_counter()
                    try:
                        frame = frames.get(timeout=1.0)
                    except queue.Empty:
                        log("camera stopped delivering frames")
                        break

                    padded, scale, pad_left, pad_top = letterbox(frame, 640)
                    network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]

                    infer_started = time.perf_counter()
                    raw_outputs = pipeline.infer({input_info.name: network_input})
                    infer_ms = (time.perf_counter() - infer_started) * 1000.0

                    box_heads, cls_heads = collect_heads(raw_outputs)
                    decode = (decode_detections_yolo26 if box_heads[0].shape[-1] == 4
                              else decode_detections)
                    boxes, scores, classes = decode(
                        box_heads, cls_heads, conf_thres=args.conf, iou_thres=args.iou
                    )
                    boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top)
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

                    detections = []
                    for position in np.argsort(-scores)[:300]:
                        name = COCO_NAMES[int(classes[position])]
                        class_counter[name] += 1
                        detections.append({
                            "box_xyxy": boxes[position].tolist(),
                            "class_id": int(classes[position]),
                            "class_name": name,
                            "confidence": float(scores[position]),
                        })

                    # measured up to here: capture wait + preprocess + infer + decode + draw
                    pipeline_ms = (time.perf_counter() - cycle_started) * 1000.0
                    rolling.append(pipeline_ms)
                    infer_times.append(infer_ms)
                    pipeline_times.append(pipeline_ms)
                    now_fps = (len(rolling) * 1000.0 / sum(rolling)) if sum(rolling) > 0 else 0.0
                    elapsed = time.perf_counter() - started
                    average_fps = (len(pipeline_times) * 1000.0 / sum(pipeline_times)
                                   if sum(pipeline_times) > 0 else 0.0)
                    recent.append((elapsed, pipeline_ms))
                    while recent and elapsed - recent[0][0] > 5.0:
                        recent.popleft()
                    window_fps = (len(recent) * 1000.0 / sum(ms for _, ms in recent)
                                  if recent else 0.0)
                    banner = [
                        f"RK3576 + Hailo-8 | {model_label} | live {width}x{height}",
                        f"FPS {now_fps:.1f} / 5s {window_fps:.1f} / {pipeline_ms:.0f} ms",
                    ]
                    annotated = draw(frame, detections, banner)

                    if writer is not None:
                        writer.write(annotated)
                    if server is not None:
                        ok, encoded = cv2.imencode(".jpg", annotated,
                                                   [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            stamp += 1
                            server.state["frame"], server.state["stamp"] = encoded.tobytes(), stamp
                    if args.snapshot_every > 0 and elapsed - last_snapshot >= args.snapshot_every:
                        last_snapshot = elapsed
                        path = args.out_dir / f"snapshot_{elapsed:07.1f}s.jpg"
                        cv2.imwrite(str(path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        snapshots.append(path)
                    if args.window:
                        cv2.imshow("Hailo-8 live", annotated)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break



                    if time.perf_counter() >= next_print:
                        next_print = time.perf_counter() + args.print_interval
                        camera_fps = delivered["count"] / elapsed if elapsed > 0 else 0.0
                        log(f"t={elapsed:6.1f}s  {now_fps:5.1f} FPS (30-frame) / {window_fps:5.1f} (5s)"
                            f" / {average_fps:5.1f} avg  |  camera delivered {camera_fps:5.1f} FPS"
                            f"  |  infer {infer_ms:5.1f} ms  pipeline {pipeline_ms:5.1f} ms"
                            f"  dets {len(detections)}")

                    if args.seconds > 0 and elapsed >= args.seconds:
                        break
            except KeyboardInterrupt:
                log("Ctrl+C: stopping")
            finally:
                stop_capture.set()

    if writer is not None:
        writer.release()
    if server is not None:
        server.shutdown()
    if args.window:
        cv2.destroyAllWindows()
    for path in snapshots:
        log(f"snapshot: {path}")

    elapsed = time.perf_counter() - started
    summary = {
        "device": args.device,
        "resolution": [width, height],
        "requested_camera_fps": args.camera_fps,
        "pixel_format": args.pixel_format,
        "exposure_ms": args.exposure_ms,
        "gain": args.gain,
        "hef": args.hef.name,
        "hef_sha256": sha256(args.hef),
        "model": model_label,
        "frames": len(pipeline_times),
        "camera_frames_delivered": delivered["count"],
        "wall_seconds": round(elapsed, 3),
        "average_fps_wall_clock": round(len(pipeline_times) / elapsed, 3) if elapsed > 0 else None,
        "average_fps_measured": (round(len(pipeline_times) * 1000.0 / sum(pipeline_times), 3)
                                 if pipeline_times else None),
        "infer_ms": summarise(infer_times) if infer_times else None,
        "pipeline_ms": summarise(pipeline_times) if pipeline_times else None,
        "per_class_counts": dict(class_counter.most_common()),
        "snapshots": [str(p) for p in snapshots],
        "record": str(args.record) if args.record else None,
    }
    report = args.out_dir / "camera_live_result.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"frames: {len(pipeline_times)}, wall: {elapsed:.2f} s, "
        f"average {summary['average_fps']} FPS")
    if pipeline_times:
        log(f"infer only: {json.dumps(summary['infer_ms'])}")
        log(f"pipeline (capture wait + letterbox + infer + decode + draw): "
            f"{json.dumps(summary['pipeline_ms'])}")
    log(f"json report: {report}")


if __name__ == "__main__":
    main()