"""Environment check for the RK3576 built-in NPU backend.

Run it through the venv the install script creates - it is the same interpreter the runner uses:

    bash scripts/install-rk3576.sh          # creates .venvs/rk3576 and calls this check
    .venvs/rk3576/bin/python scripts/check-rk3576.py

It verifies the architecture and Python version, the RKNN Python binding, the runtime library and
the NPU driver, the shipped model and sample clip (against checksums.sha256), and then runs one
real inference on one frame of the sample clip. Exit code 0 means ready.
"""

from __future__ import annotations

import argparse
import ctypes.util
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _probe import (  # noqa: E402
    Report, load_checksums, read_first_frame, repo_root, run, verify_artefact,
)

MODEL = "model/rk3576/yolo26n_rk3576_int8.rknn"
CORE_MASKS = (0x7, 0x3, 0x1)   # the runner's default first, then both cores, then one


def probe_runtime(report: Report) -> bool:
    """librknnrt is the runtime library the binding calls into; the driver exposes the NPU."""
    library = ctypes.util.find_library("rknnrt")
    if not library:
        candidates = glob.glob("/usr/lib/*/librknnrt.so") + glob.glob("/usr/lib/librknnrt.so") + \
            glob.glob("/usr/local/lib/librknnrt.so")
        library = candidates[0] if candidates else None
    ok = report.check("librknnrt.so is installed", bool(library), library or
                      "not found; it ships with the board's firmware (RKNPU2 runtime)")

    driver = []
    for path in ("/dev/rknpu", "/sys/module/rknpu", "/sys/kernel/debug/rknpu"):
        if Path(path).exists():
            driver.append(path)
    modules = run(["lsmod"]) or ""
    if "rknpu" in modules:
        driver.append("rknpu kernel module")
    report.info(f"NPU driver evidence: {', '.join(driver) if driver else 'none found'}")
    return ok


def probe_binding(report: Report) -> bool:
    try:
        from rknnlite.api import RKNNLite
    except ImportError as error:
        report.check("rknnlite binding imports", False, f"{error}")
        return False
    report.check("rknnlite binding imports", True, str(Path(RKNNLite.__module__.replace(".", "/"))))
    try:
        version = (RKNNLite().get_sdk_version() or "").strip().splitlines()
        report.info(f"runtime sdk version: {' | '.join(version[:2])}")
    except Exception as error:  # some runtimes only answer this once a model is loaded
        report.info(f"runtime sdk version not available yet ({type(error).__name__})")
    return True


def probe_dependencies(report: Report) -> bool:
    ok = True
    for name in ("numpy", "cv2"):
        try:
            module = __import__(name)
            report.check(f"{name} imports", True, getattr(module, "__version__", "present"))
        except ImportError as error:
            ok = report.check(f"{name} imports", False, f"{error}; try: sudo apt install -y "
                                                      f"python3-{'numpy' if name == 'numpy' else 'opencv'}")
    return ok


def one_frame_inference(report: Report, root: Path) -> bool:
    """The decisive check: load the model on the NPU and decode one frame of the sample clip."""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        return report.check("one-frame inference", False, "the binding is not importable")

    from postprocess_common import letterbox, to_network_input
    from postprocess_yolo26 import decode_detections_yolo26
    from rknn_helpers import collect_heads_yolo26, dequantize

    frame, frames, fps = read_first_frame(root)
    if frame is None:
        return report.check("one-frame inference", False, "cannot read the sample clip")

    rknn = RKNNLite()
    mask = None
    try:
        if rknn.load_rknn(str(root / MODEL)) != 0:
            return report.check("one-frame inference", False, "load_rknn failed")
        for candidate in CORE_MASKS:
            if rknn.init_runtime(core_mask=candidate) == 0:
                mask = candidate
                break
        if mask is None:
            return report.check("one-frame inference", False,
                                f"init_runtime failed for masks {[hex(m) for m in CORE_MASKS]}")
        network_input = to_network_input(letterbox(frame, 640)[0])
        raw = rknn.inference(inputs=[network_input], data_format="nhwc")
        if raw is None or not len(raw):
            return report.check("one-frame inference", False, "inference returned no output")
        box_heads, score_heads, described = collect_heads_yolo26(dequantize(raw, None))
        boxes, scores, classes = decode_detections_yolo26(box_heads, score_heads,
                                                          conf_thres=0.25, iou_thres=0.45)
    except Exception as error:
        return report.check("one-frame inference", False, f"{type(error).__name__}: {error}")
    finally:
        try:
            rknn.release()
        except Exception:
            pass

    scales = ",".join(str(d["stride"]) for d in described)
    return report.check("one-frame inference", True,
                        f"core_mask {hex(mask)}, 6 heads (stride {scales}), {len(scores)} detection(s) "
                        f"from frame 1 of {frames} @ {fps:.0f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the RK3576 built-in NPU environment.")
    parser.parse_args()

    root = repo_root()
    report = Report("rk3576", "RK3576 built-in NPU")
    if not report.platform_gate():
        raise SystemExit(report.finish())

    entries = load_checksums(root)
    for label, relative in (("model file", MODEL), ("sample clip", "video/test.mp4")):
        ok, detail = verify_artefact(root, relative, entries)
        report.check(label, ok, detail)
    probe_dependencies(report)
    probe_runtime(report)
    if probe_binding(report):
        one_frame_inference(report, root)
    raise SystemExit(report.finish())


if __name__ == "__main__":
    main()