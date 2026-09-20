"""Live USB camera -> RK3576 built-in NPU real-time detection, with a live FPS readout.

The Hailo-8 version of this script (run_hailo.py) with the accelerator swapped: same v4l2 camera
setup, same letterbox, same shared decode/NMS and annotation, same timing convention - the clock
stops once the boxes are drawn, so the number on screen means the same thing on both.

Notes that only apply to this board's NPU:

  * rknn-toolkit-lite2 wants a core mask for the RK3576's two NPU cores. NPU_CORE_0_1 (3, both
    cores) is the default here; pass --core-mask 1 or 2 to pin a single core, or 0 for auto.
    A value the platform does not support (e.g. 7) is replaced by the runtime and the accepted
    mask is printed.
  * the model is not an HEF but an .rknn built by this project with the same six raw detection
    heads, so decode and NMS run in the shared host implementation, exactly as in the published
    measurements.

Usage:
    python run_rk3576.py                                   # yolo26n, both NPU cores, MJPEG :8080
    python run_rk3576.py --model /home/seeed/hailo-vs-rk182x/yolo11n_rk3576_int8.rknn
    python run_rk3576.py --core-mask 1 --record out/live.mp4 --snapshot-every 10
"""

import argparse
import json
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw_detections import COCO_NAMES, draw  # noqa: E402
from postprocess_yolo11 import decode_detections, letterbox, unletterbox_boxes  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402
from rknn_runner import collect_heads, collect_heads_yolo26, dequantize, sha256  # noqa: E402

# ================= Configuration =================
MODEL_PATH = '/home/seeed/hailo-vs-rk182x/yolo26n_rk3576_int8.rknn'
DEVICE_ID = "/dev/video0"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CORE_MASK = 3                      # both RK3576 NPU cores; 1 or 2 pin a single core, 0 = auto
#                                    (the published single-stream runs passed 7, which this SoC
#                                    replaces with the same two cores)
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30
EXPOSURE_MS, GAIN = 20.0, 60
# ==================================================


def loop_percentiles(values):
    """min/median/p95/max of the per-loop times, in ms."""
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min_ms": round(ordered[0], 2),
        "median_ms": round(ordered[len(ordered) // 2], 2),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 2),
        "max_ms": round(ordered[-1], 2),
    }


def configure_camera(device, width, height, fps, exposure_ms, gain):
    """Program the V4L2 device before OpenCV opens it; silently skip if v4l2-ctl is missing."""
    calls = [["--set-fmt-video", f"width={width},height={height},pixelformat=MJPG",
              "--set-parm", str(fps)]]
    controls = []
    if exposure_ms and exposure_ms > 0:
        # 100 us units, and manual mode is required for the value to take effect
        controls += ["--set-ctrl=auto_exposure=1",
                     f"--set-ctrl=exposure_time_absolute={int(exposure_ms * 10)}"]
    if gain:
        controls.append(f"--set-ctrl=gain={gain}")
    if controls:
        calls.append(controls)
    for arguments in calls:
        try:
            result = subprocess.run(["v4l2-ctl", "-d", device, *arguments],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print(f"[warn] v4l2-ctl {' '.join(arguments)}: {result.stderr.strip()}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            print(f"[warn] v4l2-ctl unavailable ({error}); using OpenCV defaults")
            return


def init_runtime(rknn, requested_mask: int, log=print) -> int:
    """Start the RK3576 runtime, reporting which core mask it actually accepted.

    RKNNLite takes no target argument, and a mask the platform cannot honour is replaced by the
    runtime itself (on this SoC NPU_CORE_0_1_2, 7, becomes NPU_CORE_0_1, 3). Anything the
    runtime refuses outright falls back to both cores, then a single core, then auto.
    """
    if rknn.init_runtime(core_mask=requested_mask) == 0:
        return requested_mask
    for mask in (3, 1, 2, 0):
        if mask == requested_mask:
            continue
        if rknn.init_runtime(core_mask=mask) == 0:
            log(f"[warn] core_mask {hex(requested_mask)} was rejected; running with {hex(mask)}")
            return mask
    raise SystemExit("rknnlite init_runtime failed for every plausible core mask")

class MjpegServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state):
        self.state = state
        super().__init__(address, MjpegHandler)


class MjpegHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        if self.path not in ("/", "/stream.mjpg"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                frame, stamp = self.server.state["frame"], self.server.state["stamp"]
                if frame is None or stamp == last:
                    time.sleep(0.005)
                    continue
                last = stamp
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--device", default=DEVICE_ID)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--core-mask", type=lambda value: int(value, 0), default=CORE_MASK)
    parser.add_argument("--model-family", choices=("auto", "yolo11", "yolo26"), default="auto",
                        help="auto reads it from the model file name")
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    parser.add_argument("--exposure-ms", type=float, default=EXPOSURE_MS)
    parser.add_argument("--gain", type=int, default=GAIN)
    parser.add_argument("--serve-port", type=int, default=8080, help="MJPEG port, 0 disables")
    parser.add_argument("--record", type=Path, help="write annotated frames to this MP4")
    parser.add_argument("--snapshot-every", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("out/camera"))
    parser.add_argument("--seconds", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--show", action="store_true", help="force a cv2 window")
    parser.add_argument("--print-interval", type=float, default=1.0)
    args = parser.parse_args()

    from rknnlite.api import RKNNLite

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[error] model not found: {model_path}")
        print("        available: /home/seeed/hailo-vs-rk182x/yolo26n_rk3576_int8.rknn")
        print("                   /home/seeed/hailo-vs-rk182x/yolo11n_rk3576_int8.rknn")
        return

    def log(message: str) -> None:
        print(message, flush=True)

    rknn = RKNNLite()
    if rknn.load_rknn(str(model_path)) != 0:
        print("[error] rknnlite load_rknn failed")
        return
    active_mask = init_runtime(rknn, args.core_mask, log)
    family = args.model_family
    if family == "auto":
        family = "yolo26" if "yolo26" in model_path.name else "yolo11"
    model_label = "YOLO26n" if family == "yolo26" else "YOLO11n"
    log(f"[info] model {model_path.name} ({family}, sha256={sha256(model_path)})")
    log(f"[info] core mask requested {hex(args.core_mask)}, active {hex(active_mask)}")
    device_label = "RK3576 NPU"

    configure_camera(args.device, args.width, args.height, args.camera_fps,
                     args.exposure_ms, args.gain)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    # Do NOT set CAP_PROP_BUFFERSIZE=1 here: measured on this board, one driver buffer halves
    # the capture rate to 15 FPS (v4l2-ctl --stream-mmap and OpenCV with the default 4 buffers
    # both reach 29.8 FPS). The default buffer count costs at most ~4 frames of latency.
    if not cap.isOpened():
        print(f"[error] cannot open camera {args.device}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height

    server = None
    if args.serve_port:
        server = MjpegServer(("0.0.0.0", args.serve_port), {"frame": None, "stamp": 0})
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log(f"[info] MJPEG stream ready: http://<board-ip>:{args.serve_port}/")

    writer = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(args.camera_fps), (frame_width, frame_height))

    use_window = args.show or bool(__import__("os").environ.get("DISPLAY"))
    rolling: deque[float] = deque(maxlen=30)   # per-loop milliseconds: averaging instantaneous
    #                                             FPS values would overstate a jittery rate
    loop_times: list[float] = []
    infer_times: list[float] = []
    slow_frames = 0
    loop_ms = 0.0
    frames_done = 0
    detections_total = 0
    snapshots = []
    last_snapshot = 0.0
    stamp = 0

    def infer(frame: np.ndarray):
        padded, scale, pad_left, pad_top = letterbox(frame, 640)
        network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
        raw = rknn.inference(inputs=[network_input], data_format="nhwc")
        if raw is None or len(raw) == 0:
            raise RuntimeError("rknnlite inference returned no output")
        return dequantize(raw, None), scale, pad_left, pad_top

    ok, warm_frame = cap.read()
    if not ok:
        print("[error] cannot read a frame for warm-up")
        return
    for _ in range(5):
        infer(warm_frame)
    started = time.perf_counter()
    next_print = started + args.print_interval
    log("[info] warm-up done, running real-time detection (Ctrl+C to stop)")

    try:
        while True:
            loop_started = time.perf_counter()
            ret, frame = cap.read()
            read_ms = (time.perf_counter() - loop_started) * 1000.0
            if not ret:
                print("[warn] camera read failed")
                break

            infer_started = time.perf_counter()
            outputs, scale, pad_left, pad_top = infer(frame)
            infer_ms = (time.perf_counter() - infer_started) * 1000.0

            # YOLO26 box heads carry 4 direct distances, YOLO11 box heads 64 DFL channels
            if family == "yolo26":
                box_heads, cls_heads, _ = collect_heads_yolo26(outputs)
            else:
                box_heads, cls_heads = collect_heads(outputs)
            decode = (decode_detections_yolo26 if box_heads[0].shape[-1] == 4
                      else decode_detections)
            boxes, scores, classes = decode(box_heads, cls_heads,
                                            conf_thres=args.conf, iou_thres=args.iou)
            boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height)

            detections = []
            for position in np.argsort(-scores):
                class_id = int(classes[position])
                detections.append({
                    "class_id": class_id,
                    "class_name": COCO_NAMES[class_id],
                    "confidence": float(scores[position]),
                    "box_xyxy": boxes[position].tolist(),
                })

            # the banner carries the previous frame's measured time; this frame's own time is
            # only known once it has been annotated
            now_fps = len(rolling) * 1000.0 / sum(rolling) if sum(rolling) > 0 else 0.0
            average_fps = (len(loop_times) * 1000.0 / sum(loop_times)
                           if sum(loop_times) > 0 else 0.0)
            elapsed = time.perf_counter() - started
            banner = [
                f"{device_label} | {model_label} INT8 | live {frame_width}x{frame_height}",
                f"FPS {now_fps:.1f} (30-frame) / {average_fps:.1f} avg / {loop_ms:.0f} ms",
            ]
            annotated = draw(frame, detections, banner)

            # Clock stops here, the same convention as the Hailo script and the original tutorial
            # loop: capture wait, preprocess, inference, decode, NMS and annotation are counted;
            # the MJPEG encoding, recording and window display that follow are not.
            loop_ms = (time.perf_counter() - loop_started) * 1000.0
            rolling.append(loop_ms)
            loop_times.append(loop_ms)
            infer_times.append(infer_ms)
            slow_frames += 1 if loop_ms > 50.0 else 0
            frames_done += 1
            detections_total += len(detections)

            if writer is not None:
                writer.write(annotated)
            if server is not None:
                ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    stamp += 1
                    server.state["frame"], server.state["stamp"] = encoded.tobytes(), stamp
            if args.snapshot_every > 0 and elapsed - last_snapshot >= args.snapshot_every:
                last_snapshot = elapsed
                path = args.out_dir / f"snapshot_{elapsed:07.1f}s.jpg"
                cv2.imwrite(str(path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                snapshots.append(str(path))
            if use_window:
                cv2.imshow("RK3576 NPU live", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if time.perf_counter() >= next_print:
                next_print = time.perf_counter() + args.print_interval
                print(f"[live] {now_fps:5.1f} FPS (30-frame) / {average_fps:5.1f} avg"
                      f" | wait-for-frame {read_ms:5.1f} ms, infer {infer_ms:5.1f} ms,"
                      f" whole loop {loop_ms:5.1f} ms | {len(detections)} detections"
                      f" | slow frames so far {slow_frames}", flush=True)
            if args.seconds > 0 and elapsed >= args.seconds:
                break
    except KeyboardInterrupt:
        print("\n[info] Ctrl+C: stopping")

    cap.release()
    if writer is not None:
        writer.release()
    if server is not None:
        server.shutdown()
    if use_window:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    summary = {
        "accelerator": device_label,
        "model": model_path.name,
        "model_sha256": sha256(model_path),
        "core_mask_requested": hex(args.core_mask),
        "core_mask_active": hex(active_mask),
        "device": args.device,
        "resolution": [frame_width, frame_height],
        "frames": frames_done,
        "wall_seconds": round(elapsed, 3),
        "average_fps_measured": (round(len(loop_times) * 1000.0 / sum(loop_times), 3)
                                 if loop_times else None),
        "average_fps_wall_clock": round(frames_done / elapsed, 3) if elapsed else None,
        "note": ("average_fps_measured stops the clock after the boxes are drawn (same convention "
                 "as the on-screen number); average_fps_wall_clock is frames over wall seconds and "
                 "therefore also covers the MJPEG encode, recording and display"),
        "infer_ms": loop_percentiles(infer_times),
        "loop_ms": loop_percentiles(loop_times),
        "frames_slower_than_50ms": slow_frames,
        "mean_detections_per_frame": (round(detections_total / frames_done, 3)
                                      if frames_done else None),
        "snapshots": snapshots,
        "record": str(args.record) if args.record else None,
    }
    report = args.out_dir / "run_rk3576_live_result.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {frames_done} frames in {elapsed:.1f} s: {summary['average_fps_measured']} FPS "
          f"measured (boxes drawn), {summary['average_fps_wall_clock']} FPS wall clock, "
          f"{summary['mean_detections_per_frame']} detections/frame")
    print(f"[done] json report: {report}")


if __name__ == "__main__":
    main()