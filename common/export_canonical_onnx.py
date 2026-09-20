"""Export the single canonical YOLO11n ONNX used by both accelerators.

This script does nothing unless --run-conversion is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--run-conversion", action="store_true")
    args = parser.parse_args()

    if not args.run_conversion:
        print("DRY RUN: no model was loaded or exported")
        print(f"source={args.pt}")
        print(f"target={args.onnx}")
        print(f"imgsz={args.imgsz}, batch=1, opset={args.opset}, dynamic=False, nms=False")
        return

    from ultralytics import YOLO, __version__ as ultralytics_version

    source = args.pt.resolve()
    target = args.onnx.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing canonical ONNX: {target}")

    model = YOLO(str(source))
    if model.task != "detect":
        raise RuntimeError(f"expected detect model, got {model.task!r}")
    if len(model.names) != 80:
        raise RuntimeError(f"expected COCO 80 classes, got {len(model.names)}")

    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            batch=1,
            opset=args.opset,
            simplify=True,
            dynamic=False,
            nms=False,
        )
    ).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if exported != target:
        shutil.copy2(exported, target)

    manifest = {
        "source_pt": str(source),
        "source_pt_sha256": sha256(source),
        "canonical_onnx": str(target),
        "canonical_onnx_sha256": sha256(target),
        "ultralytics_version": ultralytics_version,
        "task": model.task,
        "classes": len(model.names),
        "names": model.names,
        "export": {
            "imgsz": args.imgsz,
            "batch": 1,
            "opset": args.opset,
            "simplify": True,
            "dynamic": False,
            "nms": False,
        },
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

