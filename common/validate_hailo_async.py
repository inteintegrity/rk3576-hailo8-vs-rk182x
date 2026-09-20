"""Check the asynchronous pipeline against the published synchronous run, and measure its ceiling.

Two things have to hold before the runners are rewritten on top of common/hailo_async.py:

  1. accuracy: on the same frame the async pipeline must produce the same detections as the
     synchronous run that is already published (results/final_benchmark/hailo8/*.json);
  2. throughput: the device-only rate (submit + collect, no host decode) at increasing pipeline
     depth shows what the depth buys.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "common"))

from hailo_platform import VDevice  # noqa: E402

from hailo_async import AsyncPipeline  # noqa: E402
from postprocess_yolo11 import decode_detections, letterbox, unletterbox_boxes  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402


def to_heads(outputs: dict):
    """Split a batch-1 output dict into the box and score heads of the three scales."""
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
    box_heads = [by_stride[s]["box"] for s in sorted(by_stride)]
    cls_heads = [by_stride[s]["cls"] for s in sorted(by_stride)]
    return box_heads, cls_heads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="published synchronous run to compare frame detections against")
    parser.add_argument("--frame", type=int, default=200)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = capture.read()
    if not ok:
        raise SystemExit("cannot read the test frame")
    padded, scale, pad_left, pad_top = letterbox(frame, 640)
    network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
    height, width = frame.shape[:2]

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    expected = reference["per_frame"][args.frame]["detections"]
    print(f"reference: {args.reference.name} frame {args.frame} -> {len(expected)} detections")

    vdevice = VDevice()          # one device for the whole process: creating and releasing it
    #                              repeatedly inside one process has been observed to segfault
    pipeline = AsyncPipeline(args.hef, depth=1, vdevice=vdevice)
    try:
        pipeline.submit(network_input)
        _, outputs = pipeline.collect()
        box_heads, cls_heads = to_heads(outputs)
        decode = decode_detections_yolo26 if box_heads[0].shape[-1] == 4 else decode_detections
        boxes, scores, classes = decode(box_heads, cls_heads, conf_thres=0.25, iou_thres=0.45)
        boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

        print(f"async pipeline: {len(scores)} detections")
        matched = 0
        for position in np.argsort(-scores):
            for item in expected:
                if (int(classes[position]) == item["class_id"]
                        and abs(float(scores[position]) - item["confidence"]) < 1e-3
                        and max(abs(a - b) for a, b in zip(boxes[position], item["box_xyxy"])) < 1.0):
                    matched += 1
                    break
        print(f"detections matching the published synchronous run: {matched}/{len(expected)}")
        if matched != len(expected):
            raise SystemExit("async pipeline does not reproduce the published detections")
    finally:
        pipeline.close()

    # device-only throughput at increasing depth: submit and collect the same frame repeatedly
    for depth in args.depths:
        pipeline = AsyncPipeline(args.hef, depth=depth, vdevice=vdevice)
        try:
            for _ in range(depth):                 # fill the pipeline
                pipeline.submit(network_input)
            submitted = collected = 0
            started = time.perf_counter()
            while time.perf_counter() - started < args.seconds:
                # collect the oldest completion first, then keep the pipeline full: this keeps
                # exactly `depth` inferences in flight, which is the point of the async API
                pipeline.collect()
                collected += 1
                pipeline.submit(network_input)
                submitted += 1
            wall = time.perf_counter() - started
            print(f"device only, depth {depth:2d}: {collected / wall:6.2f} FPS "
                  f"({collected} completed in {wall:.2f}s)")
        finally:
            pipeline.close()


if __name__ == "__main__":
    main()