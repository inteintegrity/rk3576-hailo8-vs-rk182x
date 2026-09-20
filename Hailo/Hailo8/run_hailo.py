"""Live USB camera -> Hailo-8 real-time detection, with a live FPS readout.

Changed from the original tutorial version for this board:

  * no cv2.imshow by default - this board has no framebuffer or X server, so the annotated
    frames are streamed as MJPEG over HTTP instead: open http://<board-ip>:8080/ in a browser.
    (A window is still opened automatically when DISPLAY is set, or with --show.)
  * the live FPS is printed to the terminal every second, and drawn on every frame.
  * the camera is programmed through v4l2-ctl before OpenCV opens it. Both matter here: if
    OpenCV negotiates the frame rate itself the camera settles at 10 FPS, and in a dim room
    auto exposure stretches the exposure to ~31 ms, which also caps capture at 10 FPS.
  * HEF_PATH pointed at a file that does not exist on this board (yolov11n.hef). Both HEF
    flavours that are on the board are supported and picked automatically:
      - NMS-on-chip HEFs (e.g. yolov8n.hef): output is one block of per-class detections
      - raw-head HEFs (this project's yolo11n / yolo26n): six detection heads, decoded on host
  * detection boxes now carry short class labels instead of a bare rectangle.

Usage:
    python run_hailo.py                                  # defaults below, MJPEG on port 8080
    python run_hailo.py --hef /home/seeed/ugen300/models/yolov8n.hef
    python run_hailo.py --record out/live.mp4 --snapshot-every 10 --seconds 30
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (VDevice, HEF, InferVStreams, ConfigureParams,
                            HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
                            FormatType)

# ================= Configuration =================
HEF_PATH = '/home/seeed/hailo-vs-rk182x/yolo26n_hailo8_official.hef'
DEVICE_ID = "/dev/video0"  # Update based on v4l2-ctl output
CONF_THRESHOLD = 0.45
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30
EXPOSURE_MS, GAIN = 20.0, 60

# COCO Dataset 80 Class Labels
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
# ==================================================


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


def detect_flavour(hef):
    """'nms' for on-chip-NMS HEFs, 'raw' for the six-raw-head HEFs of this project."""
    outputs = hef.get_output_vstream_infos()
    for info in outputs:
        shape = tuple(info.shape)
        if "nms" in info.name.lower() or (len(shape) == 3 and shape[0] == 80 and shape[1] == 5):
            return "nms", info.name
    return "raw", ", ".join(info.name for info in outputs)


def parse_nms_output(raw, confidence_threshold):
    """Yield (class_id, y_min, x_min, y_max, x_max, score) from a per-class detection block.

    Hailo's NMS output on this board is shaped (80 classes, 5 values, N detections), padded
    with zero-score rows; coordinates are normalised to the network input.
    """
    array = np.asarray(raw, dtype=np.float32)
    if array.ndim != 3:
        raise RuntimeError(f"unexpected NMS output shape {array.shape}")
    classes, values, count = array.shape
    for class_id in range(classes):
        block = array[class_id]
        for index in range(count):
            score = float(block[4, index])
            if score < confidence_threshold:
                continue
            yield class_id, float(block[0, index]), float(block[1, index]), \
                float(block[2, index]), float(block[3, index]), score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", default=HEF_PATH)
    parser.add_argument("--device", default=DEVICE_ID)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    parser.add_argument("--exposure-ms", type=float, default=EXPOSURE_MS)
    parser.add_argument("--gain", type=int, default=GAIN)
    parser.add_argument("--iou", type=float, default=0.45, help="raw-head HEFs only")
    parser.add_argument("--serve-port", type=int, default=8080, help="MJPEG port, 0 disables")
    parser.add_argument("--record", type=Path, help="write annotated frames to this MP4")
    parser.add_argument("--snapshot-every", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("out/camera"))
    parser.add_argument("--seconds", type=float, default=0.0, help="0 runs until Ctrl+C")
    parser.add_argument("--show", action="store_true", help="force a cv2 window")
    parser.add_argument("--print-interval", type=float, default=1.0)
    args = parser.parse_args()

    # raw-head HEFs are decoded with this project's shared, validation-checked implementation
    from draw_detections import COCO_NAMES, draw
    from hailo_runner import collect_heads
    from postprocess_yolo11 import decode_detections, letterbox, unletterbox_boxes
    from postprocess_yolo26 import decode_detections_yolo26

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hef_path = Path(args.hef)
    if not hef_path.is_file():
        print(f"[error] HEF not found: {hef_path}")
        print("        available: /home/seeed/hailo-vs-rk182x/yolo26n_hailo8_official.hef")
        print("                   /home/seeed/hailo-vs-rk182x/yolo11n_hailo8_int8.hef")
        print("                   /home/seeed/ugen300/models/yolov8n.hef")
        return
    hef = HEF(str(hef_path))
    input_vstream_info = hef.get_input_vstream_infos()[0]
    input_h, input_w = input_vstream_info.shape[:2]
    flavour, output_name = detect_flavour(hef)
    print(f"[info] HEF {hef_path.name}: input {input_w}x{input_h}, "
          f"{'on-chip NMS' if flavour == 'nms' else 'raw heads'} ({output_name})")

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
        server = MjpegServer(("0.0.0.0", args.serve_port),
                             {"frame": None, "stamp": 0})
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[info] MJPEG stream ready: http://<board-ip>:{args.serve_port}/")

    writer = None
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(args.camera_fps), (frame_width, frame_height))

    use_window = args.show or bool(__import__("os").environ.get("DISPLAY"))
    model_label = Path(args.hef).stem.replace("_", " ")
    device_label = "RK3576 + Hailo-8"
    rolling: deque[float] = deque(maxlen=30)   # per-loop milliseconds, not instantaneous FPS:
    #                                             averaging 1000/ms overstates the rate whenever
    #                                             the frame times jitter
    frames_done = 0
    detections_total = 0
    loop_times: list[float] = []
    slow_frames = 0
    loop_ms = 0.0
    snapshots = []
    last_snapshot = 0.0
    stamp = 0
    started = time.perf_counter()
    next_print = started + args.print_interval
    default_float_output = output_name if flavour == "nms" else None

    with VDevice() as target:
        config_params_dict = ConfigureParams.create_from_hef(hef, HailoStreamInterface.PCIe)
        network_group = target.configure(hef, config_params_dict)[0]
        with network_group.activate():
            in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            with InferVStreams(network_group, in_params, out_params) as vstreams:
                # warm up before the clock starts: the first inferences after activation are
                # several times slower and would otherwise drag the average FPS down
                ok, warm_frame = cap.read()
                if not ok:
                    print("[error] cannot read a frame for warm-up")
                    return
                if flavour == "raw":
                    warm_padded, _, _, _ = letterbox(warm_frame, input_w)
                    warm_input = np.ascontiguousarray(cv2.cvtColor(warm_padded, cv2.COLOR_BGR2RGB))[None]
                else:
                    warm_input = np.expand_dims(cv2.cvtColor(
                        cv2.resize(warm_frame, (input_w, input_h)), cv2.COLOR_BGR2RGB), axis=0)
                for _ in range(5):
                    vstreams.infer({input_vstream_info.name: warm_input})
                started = time.perf_counter()
                next_print = started + args.print_interval
                print("[info] initialization successful, running real-time detection "
                      "(Ctrl+C to stop)")
                try:
                    while True:
                        loop_started = time.perf_counter()
                        ret, frame = cap.read()
                        read_ms = (time.perf_counter() - loop_started) * 1000.0
                        if not ret:
                            print("[warn] camera read failed")
                            break

                        if flavour == "raw":
                            # letterbox keeps the aspect ratio; boxes are mapped back afterwards
                            padded, scale, pad_left, pad_top = letterbox(frame, input_w)
                            network_input = np.ascontiguousarray(
                                cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
                        else:
                            resized = cv2.resize(frame, (input_w, input_h))
                            network_input = np.expand_dims(
                                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), axis=0)

                        infer_started = time.perf_counter()
                        outputs = vstreams.infer({input_vstream_info.name: network_input})
                        infer_ms = (time.perf_counter() - infer_started) * 1000.0

                        detections = []
                        if flavour == "nms":
                            for class_id, y_min, x_min, y_max, x_max, score in parse_nms_output(
                                    list(outputs.values())[0], args.conf):
                                detections.append({
                                    "class_id": class_id,
                                    "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES)
                                    else f"ID {class_id}",
                                    "confidence": score,
                                    "box_xyxy": [x_min * frame_width, y_min * frame_height,
                                                 x_max * frame_width, y_max * frame_height],
                                })
                        else:
                            box_heads, cls_heads = collect_heads(outputs)
                            decode = (decode_detections_yolo26 if box_heads[0].shape[-1] == 4
                                      else decode_detections)
                            boxes, scores, classes = decode(box_heads, cls_heads,
                                                            conf_thres=args.conf, iou_thres=args.iou)
                            boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top)
                            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width)
                            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height)
                            for position in np.argsort(-scores):
                                class_id = int(classes[position])
                                detections.append({
                                    "class_id": class_id,
                                    "class_name": COCO_NAMES[class_id],
                                    "confidence": float(scores[position]),
                                    "box_xyxy": boxes[position].tolist(),
                                })

                        # the banner carries the previous frame's measured time; this frame's own
                        # time is only known once it has been annotated
                        now_fps = len(rolling) * 1000.0 / sum(rolling) if sum(rolling) > 0 else 0.0
                        average_fps = (len(loop_times) * 1000.0 / sum(loop_times)
                                       if sum(loop_times) > 0 else 0.0)
                        elapsed = time.perf_counter() - started
                        banner = [
                            f"{device_label} | {model_label}",
                            f"FPS {now_fps:.1f} (30-frame) / {average_fps:.1f} avg / {loop_ms:.0f} ms",
                        ]
                        annotated = draw(frame, detections, banner)

                        # Clock stops here, the same convention as the original tutorial loop:
                        # capture wait, preprocess, inference, decode, NMS and annotation are
                        # counted; the MJPEG encoding, recording and window display that follow
                        # are not.
                        loop_ms = (time.perf_counter() - loop_started) * 1000.0
                        rolling.append(loop_ms)
                        loop_times.append(loop_ms)
                        slow_frames += 1 if loop_ms > 50.0 else 0

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
                            snapshots.append(str(path))
                        if use_window:
                            cv2.imshow('reComputer RK3576 - Hailo live', annotated)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        frames_done += 1
                        detections_total += len(detections)

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
        "hef": hef_path.name,
        "flavour": flavour,
        "device": args.device,
        "resolution": [frame_width, frame_height],
        "frames": frames_done,
        "wall_seconds": round(elapsed, 3),
        "average_fps_measured": round(len(loop_times) * 1000.0 / sum(loop_times), 3) if loop_times else None,
        "average_fps_wall_clock": round(frames_done / elapsed, 3) if elapsed else None,
        "note": ("average_fps_measured stops the clock after the boxes are drawn (same convention "
                 "as the on-screen number); average_fps_wall_clock is frames over wall seconds and "
                 "therefore also covers the MJPEG encode, recording and display"),
        "mean_detections_per_frame": round(detections_total / frames_done, 3) if frames_done else None,
        "loop_ms": loop_percentiles(loop_times),
        "frames_slower_than_50ms": slow_frames,
        "snapshots": snapshots,
        "record": str(args.record) if args.record else None,
    }
    report = args.out_dir / "run_hailo_live_result.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {frames_done} frames in {elapsed:.1f} s: "
          f"{summary['average_fps_measured']} FPS measured (boxes drawn), "
          f"{summary['average_fps_wall_clock']} FPS wall clock, "
          f"{summary['mean_detections_per_frame']} detections/frame")
    print(f"[done] json report: {report}")


if __name__ == "__main__":
    main()