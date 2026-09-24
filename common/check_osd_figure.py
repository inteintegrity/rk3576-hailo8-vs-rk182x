"""Check the three published screenshots against the runs that produced them.

Three checks, all against data rather than against a human reading the picture:

1. each screenshot carries the expected on-screen text - device name, model name and the two
   timing figures - rebuilt from that device's own result JSON. The strip is rendered again with
   the same font metrics and compared pixel by pixel, and it must beat every one-character
   variant of itself (a wrong digit or a stale device name moves only ~0.1% of the strip, so
   "close enough" is not a test; beating every decoy is);
2. each screenshot equals frame 200 of that device's annotated clip, so the published still is
   provably from the published video (needs --clips; the annotated clips are not shipped);
3. the three-up mosaic in figures/ is exactly the three screenshots pasted side by side, and the
   convenience copies `figures/<device>_frame200.png` are the per-device stills byte for byte;
   plus: the caption figures on each screenshot are fields of that device's result JSON - the
   mapping is spelled out in BANNER_FPS_FIELD below - and the banner rectangle measured on screen
   is recorded, so a clipped banner (a rectangle too short for its two lines) fails the check.
   The runner paints a black rectangle exactly the size its two lines need, and this check covers
   that rectangle; ink drawn outside it is video content and cannot be told apart from the scene.

OpenCV renders the Hershey font with different metrics in 4.x and 5.x (the same banner comes out
482x78 px under 4.x and 402x86 px under 5.x), so check 1 is only meaningful with the same OpenCV
major version that drew the frame - which each result JSON records. For a device whose frame was
drawn with another major version the script reports "skipped" with the version to use instead of
failing; run it once per build to cover all three.

    python3 common/check_osd_figure.py --clips <dir>       # dir holds annotated_<device>.mp4
    PYTHONPATH=<opencv-4.x> python3 common/check_osd_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SINGLE = ROOT / "results" / "single_stream"
FIGURES = ROOT / "results" / "figures"
FRAME = 200

#: on-screen device name, the label after the first FPS figure, and which timing field of that
#: device's JSON supplies it.
#:   hailo8     - the runner prints the device service rate it measures in its own warm-up;
#:   rk3576_npu - the runner prints the single-stream inference-only rate passed on the command
#:                line (34.88), which is device_benchmark_total_fps in the record; the 394-frame
#:                mean of that same run is python_infer_only_fps (32.9) and is what the tables use;
#:   rk1820     - this clip was re-rendered from the run record after the module was swapped out,
#:                so its banner carries that record's own inference-only mean, 22.785.
BANNER_FPS_FIELD = {
    "hailo8": ("RK3576 + Hailo-8", "on device", "device_service_fps"),
    "rk3576_npu": ("RK3576 NPU", "infer only", "device_benchmark_total_fps"),
    "rk1820": ("RK3576 + RK182x", "infer only", "python_infer_only_fps"),
}

#: names a stale hard-coded string or a copy-paste slip would plausibly produce
NAME_VARIANTS = {
    "YOLO26n": ["YOLO26N", "YOLO8n", "YOLO26s"],
    "RK182x": ["RK1820", "RK1821", "RK1808"],
    "Hailo-8": ["Hailo8", "Hailo-7", "Hailo-8L"],
    "RK3576": ["RK3578", "RK3588"],
}


def render_strip(banner: list[str]) -> tuple[np.ndarray, int, int]:
    """Re-draw the banner exactly as draw_detections.draw() does, on a canvas the strip's size."""
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


def banner_box_wh(image: np.ndarray) -> tuple[int, int]:
    """Height and width of the black banner rectangle in the top-left corner, from the image.

    The runner paints it with cv2.rectangle(..., -1) before the text, so its edge is a run of
    near-black pixels: down column x=1 for the height, along row y=1 for the width. The runs are
    bounded by the image itself, so a dark scene simply ends them at the frame edge.
    """
    dark = image.max(axis=2) < 40
    height = 0
    while height < image.shape[0] and dark[height, 1]:
        height += 1
    width = 0
    while width < image.shape[1] and dark[1, width]:
        width += 1
    return height, width


def check_device(name: str, image: np.ndarray) -> dict:
    device, fps_label, fps_field = BANNER_FPS_FIELD[name]
    record = json.loads((SINGLE / name / "video_result.json").read_text(encoding="utf-8"))
    timing = record["timing"]
    banner = [
        f"{device} | {record['model']['name']} INT8",
        f"FPS {timing[fps_field]:.1f} {fps_label} / {timing['banner_pipeline_fps']:.1f} this run",
    ]
    entry = {
        "device_folder": name,
        "banner_fps_field": fps_field,
        "banner_line_1": banner[0],
        "banner_line_2": banner[1],
        "frame_wh": [image.shape[1], image.shape[0]],
        "cv2_version": cv2.__version__,
    }

    drawn_with = record["environment"]["opencv"]
    if cv2.__version__.split(".")[0] != drawn_with.split(".")[0]:
        entry.update({
            "pixel_check": "skipped",
            "reason": (f"the frame was drawn with OpenCV {drawn_with} and Hershey font metrics "
                       f"differ between OpenCV 4.x and 5.x; re-run this check with an OpenCV "
                       f"{drawn_with.split('.')[0]}.x build to verify the strip"),
            "passed": None,
        })
        return entry

    expected, strip_w, strip_h = render_strip(banner)
    actual = image[:strip_h, :strip_w]
    changed = float((np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max(axis=2) > 8).mean())
    score = ink_score(actual, expected)
    rivals = sorted((ink_score(actual, render_strip(variant)[0]), variant) for variant in decoys(banner))
    best_score, best_text = rivals[0]
    verdict = "expected text wins" if score < best_score * 0.6 else "ambiguous"
    box_h, box_w = banner_box_wh(image)
    entry.update({
        "drawn_with_opencv": drawn_with,
        "pixel_check": "run",
        "strip_wh": [strip_w, strip_h],
        "banner_box_wh_on_screen": [box_w, box_h],
        # the box is exactly what the two lines need, so an extra line would make it taller
        "banner_box_matches_expected": abs(box_h - strip_h) <= 2 and box_w >= strip_w - 2,
        "banner_pixels_changed_fraction": round(changed, 6),
        "white_text_pixel_fraction": round(float((actual.max(axis=2) > 200).mean()), 5),
        "ink_jaccard_to_expected": round(score, 6),
        "best_rival_ink_jaccard": round(best_score, 6),
        "best_rival_text": best_text,
        "verdict": verdict,
        "passed": (verdict == "expected text wins"
                   and abs(box_h - strip_h) <= 2 and box_w >= strip_w - 2),
    })
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path,
                        help="directory holding annotated_<device>.mp4; enables the "
                             "still-versus-clip-frame check (the clips are not shipped)")
    parser.add_argument("--report", type=Path, default=SINGLE / "frame200_check.json")
    args = parser.parse_args()

    checks = []
    for name in BANNER_FPS_FIELD:
        image = cv2.imread(str(SINGLE / name / "frame200.png"))
        if image is None:
            raise SystemExit(f"cannot read {SINGLE / name / 'frame200.png'}")
        checks.append(check_device(name, image))

    # 2. every published still must be the same frame as that device's annotated clip. The clip
    # is lossy, so the comparison allows codec noise; a different frame would differ by orders of
    # magnitude more, and both figures are recorded for inspection.
    if args.clips:
        for entry in checks:
            clip = args.clips / f"annotated_{entry['device_folder']}.mp4"
            capture = cv2.VideoCapture(str(clip))
            if not capture.isOpened():
                raise SystemExit(f"cannot open {clip}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise SystemExit(f"frame {FRAME} not in {clip}")
            still = cv2.imread(str(SINGLE / entry["device_folder"] / "frame200.png"))
            difference = np.abs(frame.astype(np.int16) - still.astype(np.int16))
            entry.update({
                "clip": clip.name,
                "clip_frame_mean_abs_difference": round(float(difference.mean()), 4),
                "clip_frame_pixels_above_2": round(float((difference.max(axis=2) > 2).mean()), 6),
                "still_matches_clip_frame": bool(difference.mean() < 1.0
                                                 and (difference.max(axis=2) > 8).mean() < 1e-4),
            })

    # 3. the mosaic must be exactly the three screenshots side by side, and the convenience
    #    copies under figures/ must be the same files as the per-device stills
    images = [cv2.imread(str(SINGLE / name / "frame200.png")) for name in BANNER_FPS_FIELD]
    copies_ok = {}
    for name, expected in zip(BANNER_FPS_FIELD, images):
        copy = FIGURES / f"{name}_frame200.png"
        loaded = cv2.imread(str(copy))
        copies_ok[name] = loaded is not None and loaded.shape == expected.shape and float(
            np.abs(loaded.astype(np.int16) - expected.astype(np.int16)).max()) == 0
    mosaic = cv2.imread(str(FIGURES / "runtime_osd.png"))
    if mosaic is None:
        raise SystemExit(f"cannot read {FIGURES / 'runtime_osd.png'}")
    expected_mosaic = np.hstack(images)
    mosaic_ok = mosaic.shape == expected_mosaic.shape and float(
        np.abs(mosaic.astype(np.int16) - expected_mosaic.astype(np.int16)).max()) <= 2

    # merge into the shipped report: entries verified under another build are preserved, so the
    # report accumulates the results of one run per OpenCV major version
    report = {"frame": FRAME, "devices": {}}
    if args.report.is_file():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        for entry in previous.get("devices", []):
            if entry.get("pixel_check") == "run" and entry.get("cv2_version") != cv2.__version__:
                report["devices"][entry["device_folder"]] = entry
    for entry in checks:
        if entry.get("pixel_check") == "run" or entry["device_folder"] not in report["devices"]:
            report["devices"][entry["device_folder"]] = entry
    report["mosaic_matches_three_screenshots"] = mosaic_ok
    report["figure_copies_match_device_stills"] = copies_ok
    report["devices"] = list(report["devices"].values())

    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = [entry["device_folder"] for entry in report["devices"]
              if entry.get("pixel_check") == "run" and not entry.get("passed")]
    if failed:
        raise SystemExit(f"on-screen text check failed for: {failed}")
    if not mosaic_ok:
        raise SystemExit("figures/runtime_osd.png is not the three screenshots side by side")
    if not all(copies_ok.values()):
        raise SystemExit(f"figures/ copies differ from the device stills: "
                         f"{[name for name, ok in copies_ok.items() if not ok]}")
    if args.clips:
        mismatched = [entry["device_folder"] for entry in report["devices"]
                      if not entry.get("still_matches_clip_frame")]
        if mismatched:
            raise SystemExit(f"screenshot does not match the clip's frame {FRAME}: {mismatched}")
    verified = [entry["device_folder"] for entry in report["devices"]
                if entry.get("pixel_check") == "run"]
    skipped = [entry["device_folder"] for entry in report["devices"]
               if entry.get("pixel_check") == "skipped"]
    if not verified:
        raise SystemExit(
            f"no device could be verified with OpenCV {cv2.__version__}: the frames were drawn "
            f"with {[entry.get('drawn_with_opencv') for entry in report['devices']]}; run this "
            f"script once per OpenCV major version"
        )
    print(f"strip checks recorded in the report (this run used OpenCV {cv2.__version__}): {verified}")
    if skipped:
        print(f"still needs the build that drew them: {skipped} - run this script once per OpenCV "
              f"major version, the report keeps the entries")
    print("checks passed: on-screen text, mosaic composition, figure copies"
          + (", still-versus-clip frame" if args.clips else ""))


if __name__ == "__main__":
    main()