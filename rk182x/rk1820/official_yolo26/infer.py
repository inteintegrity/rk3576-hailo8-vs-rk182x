#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import time

import cv2
import numpy as np


SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
DEFAULT_IMG_FOLDER = osp.normpath(osp.join(SCRIPT_DIR, "../../../datasets/COCO/val2017"))
DEFAULT_ANNO_JSON = osp.normpath(
    osp.join(SCRIPT_DIR, "../../../datasets/COCO/annotations/instances_val2017.json")
)
DEFAULT_MODEL = osp.normpath(osp.join(SCRIPT_DIR, "../model/yolo26s.rknn"))

OBJ_CLASS_NUM = 80
OBJ_NUMB_MAX_SIZE = 300
COCO_ID_LIST = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

def parse_args(default_runtime=None):
    parser = argparse.ArgumentParser(description="YOLO26 COCO eval for ONNX/RKNN")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--runtime", choices=["sim", "board"], default=default_runtime or "sim")
    parser.add_argument("--target", default="rk1820")
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--core-mask", type=lambda x: int(x, 0), default=0x1)
    parser.add_argument("--img-folder", default=DEFAULT_IMG_FOLDER)
    parser.add_argument("--dataset-list", default=None)
    parser.add_argument("--anno-json", default=DEFAULT_ANNO_JSON)
    parser.add_argument("--output-json", default=osp.join(SCRIPT_DIR, "results_yolo26_python.json"))
    parser.add_argument("--model-size", type=int, nargs=2, default=[640, 640], metavar=("H", "W"))
    parser.add_argument("--conf-thresh", type=float, default=0.001)
    parser.add_argument("--nms-thresh", type=float, default=0.70)
    parser.add_argument("--pad-color", type=int, default=114)
    parser.add_argument("--scaleup", action="store_true")
    parser.add_argument("--score-mode", choices=["auto", "prob", "logits"], default="auto")
    parser.add_argument("--coco-map-test", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-img-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox_bgr(img, new_shape=(640, 640), pad_color=114, scaleup=False):
    src_h, src_w = img.shape[:2]
    dst_h, dst_w = new_shape
    scale = min(dst_w / src_w, dst_h / src_h)
    if not scaleup:
        scale = min(scale, 1.0)

    resize_w = max(int(round(src_w * scale)), 1)
    resize_h = max(int(round(src_h * scale)), 1)
    pad_x = (dst_w - resize_w) // 2
    pad_y = (dst_h - resize_h) // 2
    right = dst_w - resize_w - pad_x
    bottom = dst_h - resize_h - pad_y

    if (resize_w, resize_h) != (src_w, src_h):
        img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(
        img, pad_y, bottom, pad_x, right, cv2.BORDER_CONSTANT,
        value=(pad_color, pad_color, pad_color),
    )
    return img, {"scale": scale, "pad_x": pad_x, "pad_y": pad_y, "src_w": src_w, "src_h": src_h}


def unletterbox_xyxy(boxes, info, model_w, model_h):
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= info["pad_x"]
    boxes[:, [1, 3]] -= info["pad_y"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, model_w) / info["scale"]
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, model_h) / info["scale"]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, info["src_w"])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, info["src_h"])
    return boxes


def normalize_head(x, expected_channels=None):
    x = np.asarray(x)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError("Expected head tensor [1,C,H,W] or [1,H,W,C], got {}".format(x.shape))
    if expected_channels is not None:
        if x.shape[0] == expected_channels:
            return x.astype(np.float32)
        if x.shape[-1] == expected_channels:
            return x.transpose(2, 0, 1).astype(np.float32)
    if x.shape[0] in (4, OBJ_CLASS_NUM) or x.shape[0] % 4 == 0:
        return x.astype(np.float32)
    if x.shape[-1] in (4, OBJ_CLASS_NUM) or x.shape[-1] % 4 == 0:
        return x.transpose(2, 0, 1).astype(np.float32)
    return x.astype(np.float32)


def dfl_decode(position):
    c, h, w = position.shape
    if c == 4:
        return position
    if c % 4 != 0:
        raise ValueError("Box channels must be 4 or divisible by 4, got {}".format(c))
    dfl_len = c // 4
    y = position.reshape(4, dfl_len, h, w)
    y = y - np.max(y, axis=1, keepdims=True)
    y = np.exp(y)
    y = y / np.sum(y, axis=1, keepdims=True)
    bins = np.arange(dfl_len, dtype=np.float32).reshape(1, dfl_len, 1, 1)
    return np.sum(y * bins, axis=1)


def decode_box_head(box_head, model_size):
    box_head = normalize_head(box_head)
    box = dfl_decode(box_head)
    _, grid_h, grid_w = box.shape
    model_h, model_w = model_size
    stride = model_h // grid_h

    row, col = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    x1 = (-box[0] + col + 0.5) * stride
    y1 = (-box[1] + row + 0.5) * stride
    x2 = (box[2] + col + 0.5) * stride
    y2 = (box[3] + row + 0.5) * stride
    return np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)


def calculate_iou_cpp_style(box, boxes):
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, xx2 - xx1 + 1.0)
    inter_h = np.maximum(0.0, yy2 - yy1 + 1.0)
    inter = inter_w * inter_h
    area0 = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
    area1 = (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
    union = area0 + area1 - inter
    return np.where(union > 0.0, inter / union, 0.0)


def nms_per_class(boxes, scores, classes, nms_thresh):
    keep_all = []
    for cls_id in np.unique(classes):
        cls_indices = np.where(classes == cls_id)[0]
        order = cls_indices[np.argsort(scores[cls_indices])[::-1]]
        while order.size > 0:
            current = order[0]
            keep_all.append(current)
            if order.size == 1:
                break
            iou = calculate_iou_cpp_style(boxes[current], boxes[order[1:]])
            order = order[1:][iou <= nms_thresh]
    keep_all = np.array(keep_all, dtype=np.int64)
    if keep_all.size == 0:
        return keep_all
    return keep_all[np.argsort(scores[keep_all])[::-1]]


def score_tensor_is_logits(score_head, output_name, score_mode):
    if score_mode == "logits":
        return True
    if score_mode == "prob":
        return False
    if output_name and "sigmoid" in output_name:
        return False
    return bool(np.nanmin(score_head) < 0.0 or np.nanmax(score_head) > 1.0)


def post_process_detection_output(output, conf_thresh):
    output = np.asarray(output)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim == 2 and output.shape[0] == 6 and output.shape[1] != 6:
        output = output.transpose(1, 0)
    if output.ndim != 2 or output.shape[1] < 6:
        raise ValueError("Unexpected detection output shape {}".format(output.shape))

    det = output[:, :6].astype(np.float32)
    score_first = (det[:, 0] >= 0.0) & (det[:, 0] <= 1.0) & (det[:, 1] >= -1.0) & (det[:, 1] <= 81.0)
    score_last = (det[:, 4] >= 0.0) & (det[:, 4] <= 1.0) & (det[:, 5] >= -1.0) & (det[:, 5] <= 81.0)
    if np.mean(score_first) > np.mean(score_last):
        scores = det[:, 0]
        classes = det[:, 1].astype(np.int32)
        boxes = det[:, 2:6]
    else:
        boxes = det[:, 0:4]
        scores = det[:, 4]
        classes = det[:, 5].astype(np.int32)

    valid = (scores >= conf_thresh) & (classes >= 0) & (classes < OBJ_CLASS_NUM)
    if not np.any(valid):
        return None, None, None
    return boxes[valid], classes[valid], scores[valid]


def post_process_yolo26(outputs, output_names, conf_thresh, nms_thresh, model_size, score_mode):
    for idx, output in enumerate(outputs):
        arr = np.asarray(output)
        name = output_names[idx] if output_names and idx < len(output_names) else ""
        looks_like_det = arr.ndim in (2, 3) and (
            arr.shape[-1] >= 6 or (arr.ndim == 2 and arr.shape[0] == 6)
        )
        if name == "output0" or (len(outputs) == 1 and looks_like_det):
            boxes, classes, scores = post_process_detection_output(arr, conf_thresh)
            if boxes is None:
                return None, None, None
            order = np.argsort(scores)[::-1][:OBJ_NUMB_MAX_SIZE]
            return boxes[order], classes[order], scores[order]

    if len(outputs) % 3 != 0:
        raise ValueError("Expected 6 or 9 YOLO26 head outputs, got {}".format(len(outputs)))

    output_per_branch = len(outputs) // 3
    all_boxes, all_scores, all_classes = [], [], []
    for branch in range(3):
        box_idx = branch * output_per_branch
        score_idx = box_idx + 1
        sum_idx = box_idx + 2

        boxes = decode_box_head(outputs[box_idx], model_size)
        score_head = normalize_head(outputs[score_idx], expected_channels=OBJ_CLASS_NUM)
        score_name = output_names[score_idx] if output_names and score_idx < len(output_names) else ""
        if score_tensor_is_logits(score_head, score_name, score_mode):
            score_head = sigmoid(score_head)

        score_flat = score_head.transpose(1, 2, 0).reshape(-1, OBJ_CLASS_NUM)
        classes = np.argmax(score_flat, axis=1).astype(np.int32)
        scores = np.max(score_flat, axis=1).astype(np.float32)
        valid = scores >= conf_thresh

        if output_per_branch == 3:
            score_sum = normalize_head(outputs[sum_idx])
            score_sum = score_sum.reshape(-1)
            valid &= score_sum >= conf_thresh

        if np.any(valid):
            all_boxes.append(boxes[valid])
            all_scores.append(scores[valid])
            all_classes.append(classes[valid])

    if not all_boxes:
        return None, None, None

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    keep = nms_per_class(boxes, scores, classes, nms_thresh)
    keep = keep[:OBJ_NUMB_MAX_SIZE]
    if keep.size == 0:
        return None, None, None
    return boxes[keep], classes[keep], scores[keep]


class ONNXRunner:
    def __init__(self, model_path):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [x.name for x in self.session.get_outputs()]

    def run(self, rgb_img):
        inp = rgb_img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return self.session.run(self.output_names, {self.input_name: inp})

    def release(self):
        return


class RKNNRunner:
    def __init__(self, model_path, runtime, target=None, device_id=None, core_mask=0x1):
        from rknn.api import RKNN

        self.rknn = RKNN()
        weight_path = model_path[:-5] + ".weight" if model_path.endswith(".rknn") else None
        if weight_path and osp.exists(weight_path):
            ret = self.rknn.load_rknn(model_path, weight_path)
        else:
            ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError("load_rknn failed, ret={}".format(ret))

        if runtime == "board":
            ret = self.rknn.init_runtime(target=target, device_id=device_id, core_mask=core_mask)
        else:
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("init_runtime failed, ret={}".format(ret))
        self.output_names = None

    def run(self, rgb_img):
        try:
            return self.rknn.inference(inputs=[rgb_img], data_format="nhwc")
        except TypeError:
            return self.rknn.inference(inputs=[rgb_img])

    def release(self):
        self.rknn.release()


def build_runner(args):
    model_path = osp.abspath(args.model_path)
    if model_path.endswith(".onnx"):
        return ONNXRunner(model_path), "onnx"
    if model_path.endswith(".rknn"):
        return RKNNRunner(model_path, args.runtime, args.target, args.device_id, args.core_mask), "rknn"
    raise ValueError("Unsupported model path: {}".format(model_path))


def collect_images(img_folder, dataset_list):
    if dataset_list:
        with open(dataset_list, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
        return paths
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [
        osp.join(img_folder, name)
        for name in sorted(os.listdir(img_folder))
        if name.lower().endswith(exts)
    ]


def image_id_from_path(path):
    return int(osp.splitext(osp.basename(path))[0])


def coco_eval(anno_json, pred_json):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    anno = COCO(anno_json)
    pred = anno.loadRes(pred_json)
    evaluator = COCOeval(anno, pred, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats





def main(default_runtime=None):
    args = parse_args(default_runtime)
    runner, platform = build_runner(args)
    image_paths = collect_images(args.img_folder, args.dataset_list)
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    print("YOLO26 python eval")
    print("  platform     : {}".format(platform))
    print("  runtime      : {}".format(args.runtime))
    print("  model        : {}".format(osp.abspath(args.model_path)))
    print("  images       : {}".format(len(image_paths)))
    print("  conf/nms     : {:.4f}/{:.2f}".format(args.conf_thresh, args.nms_thresh))
    print("  pad/scaleup  : {}/{}".format(args.pad_color, args.scaleup))
    print("  output json  : {}".format(osp.abspath(args.output_json)))

    records = []
    t0 = time.time()
    try:
        for idx, img_path in enumerate(image_paths, 1):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print("skip unreadable image: {}".format(img_path))
                continue

            lb_img, lb_info = letterbox_bgr(
                img_bgr, tuple(args.model_size), pad_color=args.pad_color, scaleup=args.scaleup
            )
            rgb_img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
            outputs = runner.run(rgb_img)
            boxes, classes, scores = post_process_yolo26(
                outputs, runner.output_names, args.conf_thresh, args.nms_thresh,
                tuple(args.model_size), args.score_mode,
            )
            if boxes is not None:
                boxes = unletterbox_xyxy(boxes, lb_info, args.model_size[1], args.model_size[0])
                img_id = image_id_from_path(img_path)
                for box, cls_id, score in zip(boxes, classes, scores):
                    x1, y1, x2, y2 = box.tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if w <= 0.0 or h <= 0.0:
                        continue
                    records.append({
                        "image_id": img_id,
                        "category_id": COCO_ID_LIST[int(cls_id)],
                        "bbox": [round(x1, 3), round(y1, 3), round(w, 3), round(h, 3)],
                        "score": round(float(score), 5),
                    })

                if args.save_img_dir:
                    os.makedirs(args.save_img_dir, exist_ok=True)
                    draw_img = img_bgr.copy()
                    for box, cls_id, score in zip(boxes, classes, scores):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            draw_img, "{} {:.3f}".format(int(cls_id), float(score)),
                            (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1,
                        )
                    cv2.imwrite(osp.join(args.save_img_dir, osp.basename(img_path)), draw_img)

            if args.verbose or idx == 1 or idx % 50 == 0 or idx == len(image_paths):
                print("  processed {}/{} images, detections={}".format(idx, len(image_paths), len(records)))
    finally:
        runner.release()

    os.makedirs(osp.dirname(osp.abspath(args.output_json)) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(records, f)

    elapsed = time.time() - t0
    print("Total detections: {}".format(len(records)))
    print("Elapsed: {:.1f}s".format(elapsed))
    print("Results saved to {}".format(osp.abspath(args.output_json)))

    if args.coco_map_test:
        stats = coco_eval(args.anno_json, args.output_json)
        print("mAP@[0.50:0.95]={:.4f}, AP50={:.4f}, AP75={:.4f}".format(
            stats[0], stats[1], stats[2]
        ))


if __name__ == "__main__":
    main()
