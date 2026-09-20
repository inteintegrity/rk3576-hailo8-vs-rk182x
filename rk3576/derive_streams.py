"""Derive benchmark stream sources from the 4K reference video.

This environment has no hardware video decoder (no Rockchip MPP GStreamer plugin and no
V4L2 codec node), so a software 4K H.264 decode costs ~183 ms per frame on the RK3576 CPU
and caps any pipeline at ~5.5 FPS. That is a host limitation, not an accelerator limit, so
the accelerator comparison runs on a derived source with the same content, while the 4K
source is measured separately to document the decode ceiling.

Outputs (default under --out-dir):
    test_640.mp4   640x640, already letterboxed -> the runners' letterbox becomes identity
    test_1080.mp4  1920x1080                     -> mid-cost source, decode still measurable

Usage (on the board):
    python derive_streams.py --video videos/test.mp4 --out-dir videos/derived
"""

from __future__ import annotations

import argparse
import json
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
from postprocess_yolo11 import letterbox  # noqa: E402

TARGETS = {"640": (640, 640), "1080": (1920, 1080)}


def measure_decode(path: Path, frames: int = 40) -> float:
    capture = cv2.VideoCapture(str(path))
    times = []
    for _ in range(frames):
        began = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        times.append((time.perf_counter() - began) * 1000.0)
    capture.release()
    return float(np.mean(times)) if times else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"source": args.video.name, "derived": {}}

    source_decode_ms = measure_decode(args.video, frames=15)
    print(f"source {args.video.name}: software decode {source_decode_ms:.1f} ms/frame "
          f"-> {1000 / source_decode_ms:.2f} FPS ceiling")
    report["source_decode_ms"] = round(source_decode_ms, 1)

    for label, (width, height) in TARGETS.items():
        target = args.out_dir / f"{args.video.stem}_{label}.mp4"
        capture = cv2.VideoCapture(str(args.video))
        writer = None
        written = 0
        began = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (width, height) == (640, 640):
                out = letterbox(frame, 640)[0]
            else:
                out = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (out.shape[1], out.shape[0]))
            writer.write(out)
            written += 1
        capture.release()
        if writer is not None:
            writer.release()
        decode_ms = measure_decode(target)
        entry = {
            "file": target.name,
            "resolution": [width, height],
            "frames": written,
            "bytes": target.stat().st_size,
            "decode_ms": round(decode_ms, 2),
            "decode_fps_ceiling": round(1000 / decode_ms, 2),
            "derive_seconds": round(time.perf_counter() - began, 1),
        }
        report["derived"][label] = entry
        print(f"  wrote {target.name}: {written} frames, decode {decode_ms:.2f} ms/frame "
              f"-> {1000 / decode_ms:.1f} FPS ceiling")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()