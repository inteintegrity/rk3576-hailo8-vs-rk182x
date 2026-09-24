"""Stitch the three device frames into one image: nothing but the three frames.

Each PNG is frame 200 of one annotated clip and has already been checked against its expected
on-screen text by common/check_osd_figure.py (which also verifies that the copies under
results/figures/ are these same files). The frames are pasted side by side with no scaling, no
labels and no titles - the labels and the explanation belong in the article text.

    python common/make_osd_figure.py

Output: results/figures/runtime_osd.png
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SINGLE = ROOT / "results" / "single_stream"
OUT = ROOT / "results" / "figures" / "runtime_osd.png"

# the order the article reads them in: Hailo-8, RK3576 NPU, RK182x
FRAMES = [
    SINGLE / "hailo8" / "frame200.png",
    SINGLE / "rk3576_npu" / "frame200.png",
    SINGLE / "rk1820" / "frame200.png",
]


def main() -> None:
    images = []
    for path in FRAMES:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"cannot read {path}")
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