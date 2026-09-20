"""Rebuild an annotated MP4 from a recorded run instead of re-running the accelerator.

The detections come from the run's own per-frame record (the same class ids, confidences
and boxes the runner logged), so the picture is the recorded run replayed; only the
on-screen timing strip is re-drawn. This exists because the RK182x module was swapped out
of the M.2 slot before the screenshots were taken, and the original clip carries a banner
with a hard-coded model name.

The banner text is built the same way the live runner builds it, so a frame taken from the
rebuilt clip is directly comparable to a frame from a live run:

    line 1: "<host> + <accelerator> | <model> INT8"
    line 2: "FPS <accelerator rate> <how it was measured> / <rate of this pass> this run"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from draw_detections import draw  # noqa: E402


def banner_for(record: dict, device: str, device_fps: float, how: str) -> list[str]:
    model = record["model"]["name"]
    label = f"{model[:-1].upper()}{model[-1]}" if model[-1].islower() else model
    per_frame_fps = record["timing"]["banner_pipeline_fps"]
    return [
        f"{device} | {label} INT8",
        f"FPS {device_fps:.1f} {how} / {per_frame_fps:.1f} this run",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True, help="the clip the run processed")
    parser.add_argument("--record", type=Path, required=True, help="the run's result JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", required=True, help='on-screen name, e.g. "RK3576 + RK182x"')
    parser.add_argument("--device-fps", type=float, required=True)
    parser.add_argument("--device-fps-label", default="infer only")
    parser.add_argument("--png-frame", type=int, default=-1, help="also write this frame as PNG")
    parser.add_argument("--png-out", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.out} (pass --force)")
    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    record = json.loads(args.record.read_text(encoding="utf-8"))
    frames = record["per_frame"]
    banner = banner_for(record, args.device, args.device_fps, args.device_fps_label)
    print(f"banner line 1: {banner[0]}")
    print(f"banner line 2: {banner[1]}")
    print(f"recorded frames: {len(frames)}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    writer = None
    written = 0
    png_frame = None

    for index in range(len(frames)):
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"source clip ended after {written} frames, record has {len(frames)}")
        if frame.shape[0] != 640 or frame.shape[1] != 640:
            raise SystemExit(f"unexpected frame size {frame.shape}")
        annotated = draw(frame, frames[index]["detections"], banner)
        if writer is None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), source_fps,
                (annotated.shape[1], annotated.shape[0]),
            )
        writer.write(annotated)
        if index == args.png_frame:
            png_frame = annotated
        written += 1

    capture.release()
    writer.release()
    if written != len(frames):
        raise SystemExit(f"wrote {written} frames, record has {len(frames)}")
    print(f"wrote {args.out} ({written} frames at {source_fps:.2f} fps)")
    if args.png_out and png_frame is not None:
        args.png_out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.png_out), png_frame):
            raise SystemExit(f"failed to write {args.png_out}")
        print(f"wrote {args.png_out}")


if __name__ == "__main__":
    main()