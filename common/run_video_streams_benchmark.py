"""Multi-stream video benchmark: N processes each decoding a video and running inference.

Built for the multi-camera question: how many video streams one accelerator can serve at
once, and where the platform stops keeping up. Each worker opens its own VideoCapture on
the same file (offset per stream so the streams are decorrelated), runs the shared pipeline
(letterbox -> accelerator -> shared decode/NMS) and counts processed frames.

Deliberately excludes drawing and video encoding: this measures the inference service
capacity. The single-stream runners report the annotated-output cost separately.

Streams may outnumber NPU cores; extra streams then share cores round-robin, which is what
a deployment does when it has more cameras than the accelerator has cores.

Usage (on the board):
    python run_video_streams_benchmark.py --backend rk1820 --model m.rknn --weight m.weight \
        --video videos/derived/test_640.mp4 --instances 1,2,4,8 --frames 300 --json out.json
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

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parents[1] / "common"):
    if (_candidate / "postprocess_yolo11.py").is_file():
        sys.path.insert(0, str(_candidate))
        break
from postprocess_yolo11 import decode_detections, letterbox  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402
from rknn_runner import collect_heads, collect_heads_yolo26, dequantize  # noqa: E402

HAILO_STRIDES = {80: 8, 40: 16, 20: 32}
CORE_COUNT = {"rk1820": 8, "rk3576": 3, "hailo": 1}


def run_loop(index, capture, frames, infer, to_heads, decode, conf, iou, queue) -> None:
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
        heads = to_heads(outputs)
        _, scores, _ = decode(heads[0], heads[1], conf_thres=conf, iou_thres=iou)
        detections_total += int(len(scores))
        processed += 1
    queue.put((index, {
        "frames": processed,
        "detections": detections_total,
        "mean_infer_ms": float(np.mean(infer_times)) if infer_times else 0.0,
        "loop_start": loop_start,
        "loop_end": time.perf_counter(),
    }))


def worker_hailo(index: int, hef_path: str, video: str, frames: int, family: str,
                 conf: float, iou: float, ready, start, queue, core_count: int) -> None:
    """One InferVStreams pipeline per process, driven with nested `with` blocks.

    HailoRT needs the nested form: entering the context managers manually leaves the network
    group unactivated, and an inline temporary gets collected before it can be used.
    """
    from hailo_platform import (
        HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
        InputVStreamParams, OutputVStreamParams, VDevice,
    )

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        queue.put((index, {"error": f"cannot open {video}"}))
        return
    capture.set(cv2.CAP_PROP_POS_FRAMES, index * 7 % 394)

    hef = HEF(hef_path)
    device = VDevice()
    network_group = device.configure(
        hef, ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    )[0]
    input_info = hef.get_input_vstream_infos()[0]
    in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    decode = decode_detections_yolo26 if family == "yolo26" else decode_detections

    try:
        with InferVStreams(network_group, in_params, out_params) as pipeline:
            with network_group.activate():
                def infer(frame):
                    padded, _, _, _ = letterbox(frame, 640)
                    batch = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
                    raw = pipeline.infer({input_info.name: batch})
                    by_stride = {}
                    for name, array in raw.items():
                        squeezed = np.squeeze(np.asarray(array))
                        if squeezed.shape[-1] in (4, 64, 80):
                            height, channels = squeezed.shape[0], squeezed.shape[-1]
                            head = squeezed
                        else:
                            channels, height = squeezed.shape[0], squeezed.shape[1]
                            head = np.transpose(squeezed, (1, 2, 0))
                        branch = "box" if channels in (4, 64) else "cls"
                        by_stride.setdefault(HAILO_STRIDES[height], {})[branch] = head
                    return ([by_stride[s]["box"] for s in sorted(by_stride)],
                            [by_stride[s]["cls"] for s in sorted(by_stride)])

                ok, warm = capture.read()
                if not ok:
                    queue.put((index, {"error": "no frames"}))
                    return
                infer(warm)
                capture.set(cv2.CAP_PROP_POS_FRAMES, index * 7 % 394)

                ready.put(index)
                start.wait()
                run_loop(index, capture, frames, infer, lambda outputs: outputs, decode,
                         conf, iou, queue)
    except Exception as error:
        queue.put((index, {"error": f"{type(error).__name__}: {error}"}))
    finally:
        capture.release()


def worker_rockchip(backend: str, index: int, model: str, weight: str, video: str, frames: int,
                    family: str, conf: float, iou: float, ready, start, queue,
                    core_count: int = 1) -> None:
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        queue.put((index, {"error": f"cannot open {video}"}))
        return
    capture.set(cv2.CAP_PROP_POS_FRAMES, index * 7 % 394)

    if backend == "rk1820":
        from rknn3lite.api.rknn3_lite import RKNN3Lite

        rknn = RKNN3Lite()
        if rknn.load_rknn(model_path=model, weight_path=weight) != 0:
            queue.put((index, {"error": "load_rknn failed"}))
            return
        if rknn.init_runtime(target="rk1820", core_mask=1 << (index % 8)) != 0:
            for cores in range(1, 9):
                if rknn.init_runtime(target="rk1820", core_mask=(1 << cores) - 1) == 0:
                    break
            else:
                queue.put((index, {"error": "init_runtime failed"}))
                return
        output_attrs = rknn.get_outputs_tensor_attr()

        def infer(frame):
            padded, _, _, _ = letterbox(frame, 640)
            batch = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
            return dequantize(rknn.inference(inputs=[batch], data_format=["nhwc"]), output_attrs)
    else:
        from rknnlite.api import RKNNLite

        rknn = RKNNLite()
        if rknn.load_rknn(model) != 0:
            queue.put((index, {"error": "load_rknn failed"}))
            return
        if rknn.init_runtime(core_mask=1 << (index % core_count)) != 0:
            queue.put((index, {"error": "init_runtime failed"}))
            return

        def infer(frame):
            padded, _, _, _ = letterbox(frame, 640)
            batch = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
            return dequantize(rknn.inference(inputs=[batch], data_format="nhwc"), None)

    decode = decode_detections_yolo26 if family == "yolo26" else decode_detections
    collect = collect_heads_yolo26 if family == "yolo26" else collect_heads

    ok, warm = capture.read()
    if not ok:
        queue.put((index, {"error": "no frames"}))
        return
    collect(infer(warm))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index * 7 % 394)

    ready.put(index)
    start.wait()
    try:
        run_loop(index, capture, frames, infer, collect, decode, conf, iou, queue)
    finally:
        capture.release()
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("rk1820", "rk3576", "hailo"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weight")
    parser.add_argument("--video", required=True)
    parser.add_argument("--instances", default="1,2,3")
    parser.add_argument("--frames", type=int, default=300, help="frames per stream")
    parser.add_argument("--model-family", choices=("yolo11", "yolo26"), default="yolo26")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    counts = [int(value) for value in args.instances.split(",") if value.strip()]
    core_count = CORE_COUNT[args.backend]
    report = {"backend": args.backend, "video": Path(args.video).name,
              "frames_per_stream": args.frames, "model_family": args.model_family,
              "npu_cores": core_count, "results": {}}

    for count in counts:
        if count > core_count:
            print(f"  note: {count} streams share {core_count} NPU core(s) (oversubscribed)")
        ready, start, queue = mp.Queue(), mp.Event(), mp.Queue()
        processes = [
            mp.Process(target=worker_hailo if args.backend == "hailo" else worker_rockchip,
                       args=((index, args.model, args.video, args.frames, args.model_family,
                              args.conf, args.iou, ready, start, queue, core_count)
                             if args.backend == "hailo" else
                             (args.backend, index, args.model, args.weight or "", args.video,
                              args.frames, args.model_family, args.conf, args.iou,
                              ready, start, queue, core_count)))
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