"""Split the live loop into its parts, so a low FPS number can be attributed.

Measures, on the board, with the camera settings run_hailo.py uses:

    capture only          how fast the camera itself delivers frames
    capture + inference   capture, preprocess, HailoRT infer - no decode, no drawing, no encoding
    inference only        a fixed frame, so the camera is out of the picture

Comparing the three says whether a given FPS is the camera's ceiling, the accelerator's, or the
host-side work added on top (decode, NMS, annotation, MJPEG).

    python measure_live_pipeline.py --hef /home/seeed/hailo-vs-rk182x/yolo26n_hailo8_official.hef
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from hailo_platform import (VDevice, HEF, InferVStreams, ConfigureParams,
                            HailoStreamInterface, InputVStreamParams, OutputVStreamParams,
                            FormatType)


def configure(device: str, width: int, height: int, fps: int, exposure_ms: float, gain: int) -> None:
    arguments = ["-d", device, "--set-fmt-video", f"width={width},height={height},pixelformat=MJPG",
                 "--set-parm", str(fps)]
    if exposure_ms > 0:
        arguments += ["--set-ctrl=auto_exposure=1",
                      f"--set-ctrl=exposure_time_absolute={int(exposure_ms * 10)}"]
    if gain:
        arguments.append(f"--set-ctrl=gain={gain}")
    subprocess.run(["v4l2-ctl", *arguments], capture_output=True, text=True)


def open_camera(device: str) -> cv2.VideoCapture:
    # deliberately no CAP_PROP_BUFFERSIZE here: setting it to 1 caps capture at 15 FPS on this
    # board, while the default 4 buffers reach 29.8 FPS (same as v4l2-ctl --stream-mmap)
    return cv2.VideoCapture(device, cv2.CAP_V4L2)


def capture_only(capture: cv2.VideoCapture, seconds: float) -> float:
    capture.read()
    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        ok, _ = capture.read()
        count += 1 if ok else 0
    return count / (time.perf_counter() - started)


def capture_and_infer(capture: cv2.VideoCapture, pipeline, input_name: str, size: int,
                      seconds: float) -> dict:
    waits, infers, frames = [], [], 0
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        wait_started = time.perf_counter()
        ok, frame = capture.read()
        waits.append((time.perf_counter() - wait_started) * 1000.0)
        if not ok:
            continue
        resized = cv2.resize(frame, (size, size))
        network_input = np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))[None]
        infer_started = time.perf_counter()
        pipeline.infer({input_name: network_input})
        infers.append((time.perf_counter() - infer_started) * 1000.0)
        frames += 1
    elapsed = time.perf_counter() - started
    return {
        "fps": frames / elapsed,
        "wait_ms": float(np.mean(waits)),
        "infer_ms": float(np.mean(infers)),
    }


def infer_only(pipeline, input_name: str, size: int, seconds: float) -> float:
    network_input = np.zeros((1, size, size, 3), dtype=np.uint8)
    for _ in range(5):
        pipeline.infer({input_name: network_input})
    count, started = 0, time.perf_counter()
    while time.perf_counter() - started < seconds:
        pipeline.infer({input_name: network_input})
        count += 1
    return count / (time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--exposure-ms", type=float, default=20.0)
    parser.add_argument("--gain", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()

    hef = HEF(str(args.hef))
    input_info = hef.get_input_vstream_infos()[0]
    input_size = input_info.shape[0]
    print(f"HEF {args.hef.name}: input {input_size}x{input_info.shape[1]}")

    with VDevice() as target:
        network_group = target.configure(
            hef, ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe))[0]
        with network_group.activate():
            in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            with InferVStreams(network_group, in_params, out_params) as pipeline:
                rate = infer_only(pipeline, input_info.name, input_size, args.seconds)
                print(f"  inference only (fixed frame)        : {rate:6.1f} FPS "
                      f"({1000.0 / rate:5.1f} ms per infer)")

                for label, exposure in (("manual exposure", args.exposure_ms), ("auto exposure", 0.0)):
                    configure(args.device, args.width, args.height, args.camera_fps,
                              exposure, args.gain)
                    capture = open_camera(args.device)
                    if not capture.isOpened():
                        print("  cannot open camera"); return
                    if exposure > 0:
                        rate = capture_only(capture, args.seconds)
                        print(f"  capture only ({label:15s})      : {rate:6.1f} FPS")
                    mixed = capture_and_infer(capture, pipeline, input_info.name, input_size,
                                              args.seconds)
                    print(f"  capture + inference ({label:15s}): {mixed['fps']:6.1f} FPS "
                          f"(wait {mixed['wait_ms']:5.1f} ms + infer {mixed['infer_ms']:5.1f} ms)")
                    capture.release()


if __name__ == "__main__":
    main()