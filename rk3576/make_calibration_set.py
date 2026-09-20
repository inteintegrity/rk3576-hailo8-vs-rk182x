"""Build a letterboxed 640x640 calibration set on the board (RKNN2-style dataset list).

The RKNN2 toolkit on the board quantises from a file listing image paths. The images must
be preprocessed exactly like inference input, so this letterboxes them with the same
shared implementation used by the runners (bilinear, padding 114).

Usage (on the board):
    python make_calibration_set.py --src calib_src --out calib --list calib.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parents[1] / "common", _HERE.parents[2] / "common"):
    if (_candidate / "postprocess_yolo11.py").is_file():
        sys.path.insert(0, str(_candidate))
        break
from postprocess_yolo11 import letterbox  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    sources = sorted(p for p in args.src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not sources:
        raise SystemExit(f"no images in {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for index, source in enumerate(sources):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read {source}")
        padded, _, _, _ = letterbox(image, args.size)
        target = args.out / f"{index:05d}.png"
        if not cv2.imwrite(str(target), padded):
            raise SystemExit(f"cannot write {target}")
        written.append(str(target))
    args.list.write_text("\n".join(written) + "\n", encoding="utf-8")
    print(f"wrote {len(written)} letterboxed images to {args.out}")
    print(f"dataset list: {args.list}")


if __name__ == "__main__":
    main()