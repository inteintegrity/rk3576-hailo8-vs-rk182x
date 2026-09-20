"""Validate the shared YOLO26 decode against the exported graph's own end-to-end output.

The Ultralytics YOLO26 ONNX carries an NMS-free `output0` of shape [1, 300, 6] that the
graph derives from its one2one heads. Cutting those same six heads out and decoding them
with common/postprocess_yolo26.py must reproduce that output, otherwise the host decode is
wrong and every downstream number would be too.

Usage:
    python common/validate_yolo26_decode.py --onnx model/yolo26n.onnx --image <jpg>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from draw_detections import COCO_NAMES  # noqa: E402
from postprocess_yolo11 import letterbox, unletterbox_boxes  # noqa: E402
from postprocess_yolo26 import decode_detections_yolo26  # noqa: E402

HEADS = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort

    model = onnx.load(str(args.onnx))
    existing = {output.name for output in model.graph.output}
    added = [name for name in HEADS if name not in existing]
    for name in added:
        model.graph.output.extend([onnx.ValueInfoProto(name=name)])
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in session.get_outputs()]
    print("graph outputs:", output_names)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    padded, scale, pad_left, pad_top = letterbox(image, 640)
    rgb = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32)[None] / 255.0

    results = session.run(output_names, {session.get_inputs()[0].name: tensor})
    by_name = dict(zip(output_names, results))

    end_to_end = np.asarray(by_name["output0"])[0]  # (300, 6)
    # the graph's own layout: decide by value ranges, as Rockchip's reference does
    score_first = np.mean((end_to_end[:, 0] >= 0) & (end_to_end[:, 0] <= 1)) > \
                  np.mean((end_to_end[:, 4] >= 0) & (end_to_end[:, 4] <= 1))
    if score_first:
        ref_scores, ref_classes, ref_boxes = end_to_end[:, 0], end_to_end[:, 1].astype(int), end_to_end[:, 2:6]
    else:
        ref_boxes, ref_scores, ref_classes = end_to_end[:, 0:4], end_to_end[:, 4], end_to_end[:, 5].astype(int)
    keep = ref_scores >= args.conf
    ref_boxes, ref_scores, ref_classes = ref_boxes[keep], ref_scores[keep], ref_classes[keep]
    print(f"graph end-to-end output: {len(ref_scores)} detections above conf {args.conf}")

    box_heads = [np.squeeze(np.asarray(by_name[HEADS[i]])) for i in (0, 2, 4)]
    score_heads = [np.squeeze(np.asarray(by_name[HEADS[i]])) for i in (1, 3, 5)]
    for index, head in enumerate(box_heads):
        if head.shape[0] == 4:
            box_heads[index] = np.transpose(head, (1, 2, 0))
    for index, head in enumerate(score_heads):
        if head.shape[0] == 80:
            score_heads[index] = np.transpose(head, (1, 2, 0))

    # compare in the same space the graph works in: the 640x640 letterboxed canvas.
    # Unletterboxing only our boxes would compare different coordinate systems.
    boxes, scores, classes = decode_detections_yolo26(
        box_heads, score_heads, conf_thres=args.conf, iou_thres=args.iou
    )
    display = unletterbox_boxes(boxes, scale, pad_left, pad_top)
    order = np.argsort(-scores)
    print(f"our host decode:         {len(scores)} detections")
    for index in order:
        print(f"   {COCO_NAMES[int(classes[index])]:<14} {scores[index]:.4f} "
              f"letterbox={[int(round(v)) for v in boxes[index]]} "
              f"original={[int(round(v)) for v in display[index]]}")

    print()
    matched = 0
    for index in order:
        best, best_iou = None, 0.0
        for other, (ref_box, ref_score, ref_cls) in enumerate(zip(ref_boxes, ref_scores, ref_classes)):
            if ref_cls != classes[index]:
                continue
            xx1, yy1 = max(boxes[index][0], ref_box[0]), max(boxes[index][1], ref_box[1])
            xx2, yy2 = min(boxes[index][2], ref_box[2]), min(boxes[index][3], ref_box[3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            union = ((boxes[index][2] - boxes[index][0]) * (boxes[index][3] - boxes[index][1])
                     + (ref_box[2] - ref_box[0]) * (ref_box[3] - ref_box[1]) - inter)
            value = inter / union if union > 0 else 0.0
            if value > best_iou:
                best, best_iou = other, value
        if best is not None and best_iou >= 0.5:
            matched += 1
            print(f"   match {COCO_NAMES[int(classes[index])]:<14} ours {scores[index]:.4f} vs graph "
                  f"{ref_scores[best]:.4f}  iou {best_iou:.3f}")
    print(f"\nmatched {matched}/{len(scores)} of ours against the graph's own output "
          f"({len(ref_scores)} detections)")
    if matched < len(scores) - 1:
        raise SystemExit("MISMATCH: host decode does not reproduce the graph's end-to-end output")


if __name__ == "__main__":
    main()