"""Board-side RK1820 / RK182x YOLO11n image inference.

Runs on the RK3576 host through the RKNN3 runtime (rknn3lite) against the RK182x
M.2 module, then decodes the six raw detection heads with the shared host-side
implementation, exactly as the Hailo-8 runner does.

The two runners deliberately share:
    common/postprocess_yolo11.py   sigmoid + DFL + anchor decode + class-aware NMS
    common/draw_detections.py      identical boxes, labels, banner and timing stats
so any difference between the two result images comes from the accelerators rather
than from the harness.

Usage:
    python run_image_inference.py --model yolo11n_rk1820_int8.rknn \
        --weight yolo11n_rk1820_int8.weight --image 000000000164.jpg \
        --out-dir out --core-mask 0x01 --runs 100
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
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
from draw_detections import COCO_NAMES, draw, summarise  # noqa: E402
from postprocess_yolo11 import decode_detections, letterbox, unletterbox_boxes  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402
from rknn_runner import (  # noqa: E402
    collect_heads, collect_heads_yolo26, dequantize, init_runtime_with_fallback,
    run_inference, sha256,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--core-mask", type=lambda value: int(value, 0), default=0xff,
                        help="NPU core mask; RKNN3 official examples use 0xff (all 8 cores), but the "
                             "runtime only accepts the model's compile-time core count and falls back "
                             "automatically if the requested mask does not match")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--model-family", choices=("yolo11", "yolo26"), default="yolo11",
                        help="YOLO11 has 64-channel DFL box heads; YOLO26 has 4-channel direct box heads")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    from rknn3lite.api.rknn3_lite import RKNN3Lite

    tag = "yolo26n" if getattr(args, "model_family", "yolo11") == "yolo26" else "yolo11n"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    original_height, original_width = image.shape[:2]

    rknn = RKNN3Lite()
    if rknn.load_rknn(model_path=str(args.model), weight_path=str(args.weight)) != 0:
        raise SystemExit("rknn3lite load_rknn failed")
    active_mask = init_runtime_with_fallback(rknn, args.core_mask, log)
    log(f"model: {args.model.name} sha256={sha256(args.model)}")
    log(f"weight: {args.weight.name} sha256={sha256(args.weight)}")
    log(f"core_mask requested: {hex(args.core_mask)}, active: {hex(active_mask)}")

    input_attrs = rknn.get_inputs_tensor_attr()
    output_attrs = rknn.get_outputs_tensor_attr()
    log(f"runtime sdk version: {rknn.get_sdk_version()}")
    log(f"input tensor attrs: {input_attrs}")
    log(f"output tensor attrs: {output_attrs}")

    padded, scale, pad_left, pad_top = letterbox(image, 640)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # The RKNN was built with input_attrs NHWC uint8, so the tensor is passed as
    # (1, 640, 640, 3); data_format is stated explicitly rather than inferred.
    network_input = np.ascontiguousarray(rgb)[None]
    data_format = "nhwc"

    for _ in range(args.warmup):
        run_inference(rknn, network_input, data_format, output_attrs)

    infer_times, pipeline_times = [], []
    described: list[dict] = []
    raw_shapes: list[list[int]] = []
    boxes = scores = classes = None
    for _ in range(args.runs):
        started = time.perf_counter()
        outputs = run_inference(rknn, network_input, data_format, output_attrs)
        infer_ms = (time.perf_counter() - started) * 1000.0
        infer_times.append(infer_ms)
        if not raw_shapes:
            raw_shapes = [list(np.asarray(array).shape) for array in outputs]

        started = time.perf_counter()
        box_heads, cls_heads, described = (
            collect_heads_yolo26(outputs) if args.model_family == "yolo26" else collect_heads(outputs)
        )
        boxes, scores, classes = (decode_detections_yolo26 if args.model_family == "yolo26" else decode_detections)(
            box_heads, cls_heads, conf_thres=args.conf, iou_thres=args.iou
        )
        pipeline_times.append((time.perf_counter() - started) * 1000.0 + infer_ms)

    # Streaming throughput: repeated single-frame calls, the only batching the
    # rknn3lite inference API exposes (the input list is per model input, not per frame).
    for _ in range(3):
        run_inference(rknn, network_input, data_format, output_attrs)
    stream_times = []
    for _ in range(args.runs):
        started = time.perf_counter()
        run_inference(rknn, network_input, data_format, output_attrs)
        stream_times.append((time.perf_counter() - started) * 1000.0)

    # End to end: reread from disk and include annotation and JSON writing.
    e2e_times = []
    for _ in range(10):
        started = time.perf_counter()
        fresh = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        padded_e2e, scale_e2e, left_e2e, top_e2e = letterbox(fresh, 640)
        inputs_e2e = np.ascontiguousarray(cv2.cvtColor(padded_e2e, cv2.COLOR_BGR2RGB))[None]
        outputs_e2e = run_inference(rknn, inputs_e2e, data_format, output_attrs)
        box_e2e, cls_e2e, _ = (
            collect_heads_yolo26(outputs_e2e) if args.model_family == "yolo26" else collect_heads(outputs_e2e)
        )
        boxes_e2e, scores_e2e, classes_e2e = (decode_detections_yolo26 if args.model_family == "yolo26" else decode_detections)(
            box_e2e, cls_e2e, conf_thres=args.conf, iou_thres=args.iou
        )
        boxes_e2e = unletterbox_boxes(boxes_e2e, scale_e2e, left_e2e, top_e2e)
        detections_e2e = [
            {
                "class_id": int(classes_e2e[i]),
                "class_name": COCO_NAMES[int(classes_e2e[i])],
                "confidence": float(scores_e2e[i]),
                "box_xyxy": [float(v) for v in boxes_e2e[i]],
            }
            for i in range(len(scores_e2e))
        ]
        draw(fresh, detections_e2e, [])
        e2e_times.append((time.perf_counter() - started) * 1000.0)

    infer_stats = summarise(infer_times)
    pipeline_stats = summarise(pipeline_times)
    stream_stats = summarise(stream_times)
    e2e_stats = summarise(e2e_times)

    boxes = unletterbox_boxes(boxes, scale, pad_left, pad_top)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
    detections = []
    for index in np.argsort(-scores):
        box = boxes[index]
        detections.append(
            {
                "class_id": int(classes[index]),
                "class_name": COCO_NAMES[int(classes[index])],
                "confidence": round(float(scores[index]), 4),
                "box_xyxy": [int(round(float(v))) for v in box],
                "box_xywh": [
                    int(round(float(box[0]))),
                    int(round(float(box[1]))),
                    int(round(float(box[2] - box[0]))),
                    int(round(float(box[3] - box[1]))),
                ],
                "box_xyxy_normalised": [
                    round(float(box[0] / original_width), 4),
                    round(float(box[1] / original_height), 4),
                    round(float(box[2] / original_width), 4),
                    round(float(box[3] / original_height), 4),
                ],
            }
        )
    log(f"detections: {len(detections)} (conf>={args.conf}, iou={args.iou})")
    for detection in detections:
        log(
            f"  {detection['class_name']:<15} {detection['confidence']:.4f} "
            f"xyxy={detection['box_xyxy']}"
        )

    banner = [
        "RK3576 + RK1820 | YOLO11n INT8",
        f"FPS {1000.0 / infer_stats['mean_ms']:.1f} (RK1820) / {1000.0 / pipeline_stats['mean_ms']:.1f} (pipeline)",
    ]
    annotated = draw(image, detections, banner)

    annotated_path = args.out_dir / f"rk1820_{tag}_result.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(args.out_dir / f"rk1820_{tag}_input.jpg"), image)
    cv2.imwrite(str(args.out_dir / f"rk1820_{tag}_letterboxed.png"), padded)

    report = {
        "accelerator": {
            "name": "RK1820 / RK182x",
            "runtime": f"RKNN3 runtime (rknn3lite) {rknn.get_sdk_version()}",
            "interface": "PCIe via rknn3 NTB",
            "core_mask_requested": hex(args.core_mask),
            "core_mask_active": hex(active_mask),
        },
        "model": {
            "name": "YOLO26n" if args.model_family == "yolo26" else "YOLO11n",
            "task": "detect",
            "classes": 80,
            "precision": "INT8 w8a8",
            "rknn": args.model.name,
            "rknn_sha256": sha256(args.model),
            "weight": args.weight.name,
            "weight_sha256": sha256(args.weight),
            "graph_boundary": "six raw detection heads, decode + NMS on host",
            "raw_output_shapes": raw_shapes,
            "raw_output_mapping": described,
            "input_tensor_attrs": str(input_attrs),
            "output_tensor_attrs": str(output_attrs),
        },
        "input": {
            "image": args.image.name,
            "image_sha256": sha256(args.image),
            "original_size_wh": [original_width, original_height],
            "network_input": "640x640 letterbox, RGB, uint8, NHWC",
            "letterbox": {"scale": scale, "pad_left": pad_left, "pad_top": pad_top, "pad_value": 114},
        },
        "decode": {
            "implementation": "common/postprocess_yolo11.py (shared with the Hailo-8 runner)",
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "nms": "class-aware greedy NMS, max_wh=7680",
        },
        "detections": detections,
        "timing": {
            "runs": args.runs,
            "warmup_runs": args.warmup,
            "infer_only": infer_stats,
            "pipeline_without_io": pipeline_stats,
            "streaming_repeated_calls": stream_stats,
            "end_to_end_with_io_and_draw": e2e_stats,
            "python_infer_only_fps": round(1000.0 / infer_stats["mean_ms"], 3),
            "python_pipeline_fps": round(1000.0 / pipeline_stats["mean_ms"], 3),
            "python_end_to_end_fps": round(1000.0 / e2e_stats["mean_ms"], 3),
            "note": (
                "No RKNN3 benchmark CLI equivalent of `hailortcli benchmark` was run, so "
                "only host-measured numbers are reported for this accelerator."
            ),
        },
        "environment": {
            "host": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    report_path = args.out_dir / f"rk1820_{tag}_result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"infer only: {json.dumps(infer_stats)} -> {1000.0 / infer_stats['mean_ms']:.2f} FPS")
    log(f"streaming: {json.dumps(stream_stats)} -> {1000.0 / stream_stats['mean_ms']:.2f} FPS")
    log(f"pipeline (no I/O): {json.dumps(pipeline_stats)} -> {1000.0 / pipeline_stats['mean_ms']:.2f} FPS")
    log(f"end to end: {json.dumps(e2e_stats)} -> {1000.0 / e2e_stats['mean_ms']:.2f} FPS")
    log(f"annotated image: {annotated_path}")
    log(f"json report: {report_path}")

    (args.out_dir / "rk1820_image_inference.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    rknn.release()


if __name__ == "__main__":
    main()