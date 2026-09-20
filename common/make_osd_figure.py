"""Stitch the three on-screen frames into one image: nothing but the three frames.

Each PNG comes from frame 200 of one annotated clip and has already been checked against its
expected strip by common/check_osd_strip.py. The frames are pasted side by side with no scaling,
no labels and no titles - the labels and the explanation belong in the article text.

    python common/make_osd_figure.py

Output: results/final_benchmark/figures/runtime_osd.png
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "results" / "final_benchmark"
SHOTS = BENCH / "figures" / "screenshots"
OUT = BENCH / "figures" / "runtime_osd.png"

FRAMES = [
    "hailo8_yolo26n_frame200.png",
    "rk3576_npu_yolo26n_frame200.png",
    "rk182x_yolo26n_frame200.png",
]


def main() -> None:
    images = []
    for name in FRAMES:
        image = cv2.imread(str(SHOTS / name))
        if image is None:
            raise SystemExit(f"cannot read {SHOTS / name}")
        images.append(image)
    heights = {image.shape[0] for image in images}
    if len(heights) != 1:
        raise SystemExit(f"frames have different heights: {sorted(heights)}")

    mosaic = np.hstack(images)
    if not cv2.imwrite(str(OUT), mosaic):
        raise SystemExit(f"failed to write {OUT}")
    print(f"wrote {OUT} ({mosaic.shape[1]}x{mosaic.shape[0]}, {len(images)} frames side by side)")


if __name__ == "__main__":
    main()