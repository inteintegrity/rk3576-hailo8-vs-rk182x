"""Convert the canonical YOLO26n ONNX to RKNN for the RK3576 built-in NPU.

Keeps the same graph boundary as the other two accelerators: the graph is cut at the six
one2one detection heads, so decode and NMS stay in the shared host module and the three
accelerators remain comparable.

Runs on the board (rknn-toolkit2 2.3.2 lives in the RK3576 benchmark venv).

Usage:
    python convert_yolo26.py --onnx yolo26n.onnx --dataset calib.txt --output yolo26n_rk3576_int8.rknn
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

HEADS = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv_output_0",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv_output_0",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv_output_0",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked(ret, action: str) -> None:
    if ret not in (None, 0):
        raise RuntimeError(f"{action} failed with return code {ret}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="rk3576")
    parser.add_argument("--precision", choices=("int8", "fp16"), default="int8")
    parser.add_argument("--algorithm", choices=("normal", "mmse"), default="normal")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    from rknn.api import RKNN

    do_quantization = args.precision == "int8"
    config = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": args.target,
        "float_dtype": "float16",
        "optimization_level": args.optimization_level,
    }
    if do_quantization:
        config.update(
            quantized_dtype="w8a8",
            quantized_algorithm=args.algorithm,
            quantized_method="channel",
        )

    started = time.time()
    rknn = RKNN(verbose=False)
    try:
        checked(rknn.config(**config), "config")
        # cut the graph at the six raw detection heads, exactly like the RK182x recipe
        checked(rknn.load_onnx(model=str(args.onnx), outputs=HEADS), "load_onnx")
        checked(
            rknn.build(do_quantization=do_quantization, dataset=str(args.dataset) if do_quantization else None),
            "build",
        )
        checked(rknn.export_rknn(str(args.output)), "export_rknn")
    finally:
        rknn.release()

    metadata = {
        "target": args.target,
        "precision": args.precision,
        "quantized_dtype": "w8a8" if do_quantization else None,
        "optimization_level": args.optimization_level,
        "graph_boundary": "six one2one raw detection heads",
        "onnx": args.onnx.name,
        "onnx_sha256": sha256(args.onnx),
        "dataset_entries": len(args.dataset.read_text(encoding="utf-8").split()),
        "output": args.output.name,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output),
        "conversion_seconds": round(time.time() - started, 1),
        "rknn_toolkit2": importlib.metadata.version("rknn-toolkit2"),
        "python": platform.python_version(),
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()