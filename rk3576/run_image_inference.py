"""RK3576 built-in NPU YOLO11n image inference.

Third accelerator in the comparison: the SoC's own 6 TOPS NPU, driven through
rknn-toolkit-lite2 (librknnrt 2.3.0). It consumes a model built from the same
canonical ONNX and ending at the same six raw detection heads, so the shared host
decode in common/postprocess_yolo11.py applies unchanged and the result is directly
comparable with the Hailo-8 and RK1820 runs.

Unlike the other two this accelerator is on-die, so there is no PCIe hop and no
vendor benchmark CLI on the image; timings come from the same Python harness used for
the other platforms (rknn-toolkit-lite2 is the vendor runtime API).

Usage:
    python run_image_inference.py --model yolo11n_rk3576_int8.rknn --image 164.jpg \
        --out-dir out --core-mask 7 --runs 100
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
from rknn_runner import collect_heads, collect_heads_yolo26, dequantize, sha256  # noqa: E402

CORE_MASKS = {0: "NPU_CORE_AUTO", 1: "NPU_CORE_0", 3: "NPU_CORE_0_1", 7: "NPU_CORE_0_1_2", 65535: "NPU_CORE_ALL"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--core-mask", type=lambda value: int(value, 0), default=7,
                        help="1=core0, 3=core0+1, 7=all three cores, 0=auto")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--model-family", choices=("yolo11", "yolo26"), default="yolo11",
                        help="YOLO11 has 64-channel DFL box heads; YOLO26 has 4-channel direct box heads")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    from rknnlite.api import RKNNLite

    tag = "yolo26n" if args.model_family == "yolo26" else "yolo11n"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    original_height, original_width = image.shape[:2]

    rknn = RKNNLite()
    if rknn.load_rknn(str(args.model)) != 0:
        raise SystemExit("rknnlite load_rknn failed")
    if rknn.init_runtime(core_mask=args.core_mask) != 0:
        raise SystemExit("rknnlite init_runtime failed")
    log(f"model: {args.model.name} sha256={sha256(args.model)}")
    log(f"core_mask: {hex(args.core_mask)} ({CORE_MASKS.get(args.core_mask, 'custom')})")
    log(f"runtime:\n{rknn.get_sdk_version()}")

    padded, scale, pad_left, pad_top = letterbox(image, 640)
    network_input = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))[None]
    data_format = "nhwc"

    for _ in range(args.warmup):
        raw = rknn.inference(inputs=[network_input], data_format=data_format)
        if raw is None or len(raw) == 0:
            raise RuntimeError("rknnlite inference returned no output")

    infer_times, pipeline_times = [], []
    described: list[dict] = []
    raw_shapes: list[list[int]] = []
    boxes = scores = classes = None
    for _ in range(args.runs):
        started = time.perf_counter()
        raw = rknn.inference(inputs=[network_input], data_format=data_format)
        infer_ms = (time.perf_counter() - started) * 1000.0
        if raw is None or len(raw) == 0:
            raise RuntimeError("rknnlite inference returned no output")
        infer_times.append(infer_ms)
        if not raw_shapes:
            raw_shapes = [list(np.asarray(array).shape) for array in raw]

        started = time.perf_counter()
        box_heads, cls_heads, described = (
            collect_heads_yolo26(dequantize(raw, None)) if args.model_family == "yolo26"
            else collect_heads(dequantize(raw, None))
        )
        boxes, scores, classes = (decode_detections_yolo26 if args.model_family == "yolo26"
                                  else decode_detections)(
            box_heads, cls_heads, conf_thres=args.conf, iou_thres=args.iou
        )
        pipeline_times.append((time.perf_counter() - started) * 1000.0 + infer_ms)

    infer_stats = summarise(infer_times)
    pipeline_stats = summarise(pipeline_times)

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
        log(f"  {detection['class_name']:<15} {detection['confidence']:.4f} xyxy={detection['box_xyxy']}")

    # End to end: reread from disk and include annotation and JSON writing.
    e2e_times = []
    for _ in range(10):
        started = time.perf_counter()
        fresh = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        padded_e2e, scale_e2e, left_e2e, top_e2e = letterbox(fresh, 640)
        inputs_e2e = np.ascontiguousarray(cv2.cvtColor(padded_e2e, cv2.COLOR_BGR2RGB))[None]
        raw_e2e = rknn.inference(inputs=[inputs_e2e], data_format=data_format)
        box_e2e, cls_e2e, _ = (collect_heads_yolo26(dequantize(raw_e2e, None))
                               if args.model_family == "yolo26" else collect_heads(dequantize(raw_e2e, None)))
        boxes_e2e, scores_e2e, classes_e2e = (decode_detections_yolo26 if args.model_family == "yolo26"
                                              else decode_detections)(
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
    e2e_stats = summarise(e2e_times)

    banner = [
        "RK3576 NPU | YOLO11n INT8",
        f"FPS {1000.0 / infer_stats['mean_ms']:.1f} (RK3576 NPU) / "
        f"{1000.0 / pipeline_stats['mean_ms']:.1f} (pipeline)",
    ]
    annotated = draw(image, detections, banner)

    annotated_path = args.out_dir / f"rk3576_npu_{tag}_result.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(args.out_dir / f"rk3576_npu_{tag}_input.jpg"), image)
    cv2.imwrite(str(args.out_dir / f"rk3576_npu_{tag}_letterboxed.png"), padded)

    report = {
        "accelerator": {
            "name": "RK3576 built-in NPU",
            "runtime": f"rknn-toolkit-lite2 (librknnrt {rknn.get_sdk_version().strip().splitlines()[2].strip() if len(rknn.get_sdk_version().splitlines()) > 2 else 'unknown'})",
            "interface": "on-die NPU, no PCIe hop",
            "core_mask": hex(args.core_mask),
            "core_mask_name": CORE_MASKS.get(args.core_mask, "custom"),
        },
        "model": {
            "name": "YOLO26n" if args.model_family == "yolo26" else "YOLO11n",
            "task": "detect",
            "classes": 80,
            "precision": "INT8 w8a8 (rknn-toolkit2 2.3.2, target rk3576)",
            "rknn": args.model.name,
            "rknn_sha256": sha256(args.model),
            "graph_boundary": "six raw detection heads, decode + NMS on host",
            "raw_output_shapes": raw_shapes,
            "raw_output_mapping": described,
        },
        "input": {
            "image": args.image.name,
            "image_sha256": sha256(args.image),
            "original_size_wh": [original_width, original_height],
            "network_input": "640x640 letterbox, RGB, uint8, NHWC",
            "letterbox": {"scale": scale, "pad_left": pad_left, "pad_top": pad_top, "pad_value": 114},
        },
        "decode": {
            "implementation": "common/postprocess_yolo11.py (shared with the Hailo-8 and RK1820 runners)",
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
            "end_to_end_with_io_and_draw": e2e_stats,
            "python_infer_only_fps": round(1000.0 / infer_stats["mean_ms"], 3),
            "python_pipeline_fps": round(1000.0 / pipeline_stats["mean_ms"], 3),
            "python_end_to_end_fps": round(1000.0 / e2e_stats["mean_ms"], 3),
            "note": (
                "No vendor benchmark CLI for this NPU exists on the image; these are the "
                "same host-measured layers used for the other two accelerators. The NPU is "
                "on-die, so there is no PCIe transfer to hide or expose."
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
    report_path = args.out_dir / f"rk3576_npu_{tag}_result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"infer only: {json.dumps(infer_stats)} -> {1000.0 / infer_stats['mean_ms']:.2f} FPS")
    log(f"pipeline (no I/O): {json.dumps(pipeline_stats)} -> {1000.0 / pipeline_stats['mean_ms']:.2f} FPS")
    log(f"end to end: {json.dumps(e2e_stats)} -> {1000.0 / e2e_stats['mean_ms']:.2f} FPS")
    log(f"annotated image: {annotated_path}")
    log(f"json report: {report_path}")
    (args.out_dir / "rk3576_npu_image_inference.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    rknn.release()


if __name__ == "__main__":
    main()