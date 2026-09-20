"""Check that an annotated clip's on-screen strip carries the expected text.

The strip ("RK3576 + Hailo-8 | YOLO26n INT8" / "FPS ...") is drawn by common/draw_detections.py
with fixed font metrics, so the expected strip can be rendered again and compared with the frame
pixel by pixel: a near-zero difference proves the picture carries exactly the device name and
the two timing figures we intend to publish, without anyone having to read them off the image.

Run it with the interpreter and OpenCV build that drew the clip - the two OpenCV builds in this
project (host and board) report different Hershey font metrics, so a cross-machine comparison is
meaningless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def render_strip(banner: list[str]) -> tuple[np.ndarray, int, int]:
    """Re-draw the banner exactly as draw() does, on a canvas the size of the strip."""
    font, scale, thickness, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, 10
    sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text in banner]
    line_height = max(size[1] for size in sizes) + 16
    strip_width = max(size[0] for size in sizes) + 2 * pad
    strip_height = line_height * len(banner) + pad
    canvas = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)
    for row, (text, size) in enumerate(zip(banner, sizes)):
        cv2.putText(
            canvas, text, (pad, line_height * row + size[1] + pad),
            font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
    return canvas, strip_width, strip_height


def ink_score(actual: np.ndarray, expected: np.ndarray) -> float:
    """Jaccard distance between the two ink masks, tolerant to one pixel of antialiasing.

    A variant string can be a few pixels wider or narrower than the frame's strip, so the two
    masks are cropped to their common size first.
    """
    height = min(actual.shape[0], expected.shape[0])
    width = min(actual.shape[1], expected.shape[1])
    ink_actual = (actual[:height, :width].max(axis=2) > 160).astype(np.uint8)
    ink_expected = (expected[:height, :width].max(axis=2) > 160).astype(np.uint8)
    union = np.logical_or(ink_actual, ink_expected).sum()
    if union == 0:
        return 1.0
    return float(np.logical_xor(ink_actual, ink_expected).sum() / union)


# Names that a copy-paste slip or a stale hard-coded string would plausibly produce.
NAME_VARIANTS = {
    "YOLO26n": ["YOLO11n", "YOLO26N", "YOLO8n"],
    "YOLO11n": ["YOLO26n", "YOLO11N"],
    "RK182x": ["RK1820", "RK1821", "RK1808"],
    "Hailo-8": ["Hailo8", "Hailo-7", "Hailo-8L"],
    "Hailo8": ["Hailo-8"],
    "RK3576": ["RK3578", "RK3588"],
}


def decoys(banner: list[str]) -> list[list[str]]:
    """Wrong-digit and wrong-name variants of the expected strip."""
    variants = []
    for line_index, line in enumerate(banner):
        for position, character in enumerate(line):
            if not character.isdigit():
                continue
            for delta in (1, 9):
                altered = list(banner)
                altered[line_index] = line[:position] + str((int(character) + delta) % 10) + line[position + 1:]
                variants.append(altered)
    for name, alternatives in NAME_VARIANTS.items():
        for alternative in alternatives:
            if any(name in line for line in banner):
                variants.append([line.replace(name, alternative) for line in banner])
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True,
                        help="the run's result JSON; the expected strip is derived from it, "
                             "never typed by hand")
    parser.add_argument("--device", required=True, help='on-screen name, e.g. "RK3576 + Hailo-8"')
    parser.add_argument("--device-fps", type=float, required=True)
    parser.add_argument("--device-fps-label", default="infer only")
    parser.add_argument("--frame", type=int, default=200)
    parser.add_argument("--png-out", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.clip))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.clip}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"frame {args.frame} not in {args.clip}")

    record = json.loads(args.record.read_text(encoding="utf-8"))
    model = record["model"]["name"]
    label = f"{model[:-1].upper()}{model[-1]}" if model[-1].islower() else model
    banner = [
        f"{args.device} | {label} INT8",
        f"FPS {args.device_fps:.1f} {args.device_fps_label} / "
        f"{record['timing']['banner_pipeline_fps']:.1f} this run",
    ]
    expected, strip_w, strip_h = render_strip(banner)
    actual = frame[:strip_h, :strip_w]
    changed = float((np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max(axis=2) > 8).mean())
    ink = float((actual.max(axis=2) > 200).mean())
    score = ink_score(actual, expected)

    # A single wrong character moves only a thousandth of the strip's pixels, so instead of an
    # absolute threshold, require the expected text to beat every one-character variant.
    rivals = []
    for variant in decoys(banner):
        rival, _, _ = render_strip(variant)
        rivals.append((ink_score(actual, rival), variant))
    rivals.sort()
    best_rival_score, best_rival = rivals[0]

    result = {
        "clip": str(args.clip),
        "frame": args.frame,
        "banner_line_1": banner[0],
        "banner_line_2": banner[1],
        "strip_wh": [strip_w, strip_h],
        "banner_pixels_changed_fraction": round(changed, 6),
        "white_text_pixel_fraction": round(ink, 5),
        "ink_jaccard_to_expected": round(score, 6),
        "best_rival_ink_jaccard": round(best_rival_score, 6),
        "best_rival_text": best_rival,
        "frame_wh": [frame.shape[1], frame.shape[0]],
        "cv2_version": cv2.__version__,
        "verdict": "expected text wins" if score < best_rival_score * 0.6 else "ambiguous",
    }
    print(json.dumps(result, ensure_ascii=False))
    if args.png_out:
        args.png_out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.png_out), frame):
            raise SystemExit(f"failed to write {args.png_out}")
        print(f"wrote {args.png_out}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not score < best_rival_score * 0.6:
        raise SystemExit(
            f"the strip on {args.clip} matches {best_rival!r} at least as well as the expected "
            f"{banner!r} (ink distance {best_rival_score:.4f} vs {score:.4f})"
        )
    print(f"expected strip wins: ink distance {score:.4f} vs best rival {best_rival_score:.4f}")


if __name__ == "__main__":
    main()