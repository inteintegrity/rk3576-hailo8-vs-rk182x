"""Environment check for the Hailo-8 backend.

Run it through the venv the install script creates:

    bash scripts/install-hailo8.sh
    .venvs/hailo8/bin/python scripts/check-hailo8.py

Hailo-8 does not ship a wheel in this repository (the HailoRT Python bindings come from Hailo's
Developer Zone), so this check prefers whatever HailoRT is already installed on the board: the
driver, /dev/hailo0, the `hailortcli` tool and the `hailo_platform` binding. If HailoRT is not
installed, pass a wheel to the install script with HAILORT_WHEEL=/path/to/hailort-*.whl.
It checks the model and the sample clip against checksums.sha256 and runs one real inference.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _probe import (  # noqa: E402
    Report, load_checksums, read_first_frame, repo_root, run, verify_artefact, which,
)

MODEL = "model/Hailo/yolo26n_hailo8_official.hef"


def probe_driver(report: Report) -> bool:
    """The PCIe card needs its kernel driver; /dev/hailo0 or a loaded module both prove it."""
    evidence = []
    if Path("/dev/hailo0").exists():
        evidence.append("/dev/hailo0")
    modules = run(["lsmod"]) or ""
    if "hailo_pci" in modules:
        evidence.append("hailo_pci kernel module")
    pci = run(["lspci"]) or ""
    for line in pci.splitlines():
        if "hailo" in line.lower():
            evidence.append(line.strip())
    ok = report.check("Hailo-8 device is present", bool(evidence),
                      "; ".join(evidence) if evidence else
                      "no /dev/hailo0, no hailo_pci module and no Hailo PCIe device")
    return ok


def probe_tool(report: Report) -> bool:
    tool = which("hailortcli")
    if not tool:
        return report.check("hailortcli is installed", False,
                            "not found; it comes with the HailoRT package for this board")
    report.check("hailortcli is installed", True, tool)
    version = run(["hailortcli", "--version"])
    if version:
        report.info(f"hailortcli: {version.splitlines()[0]}")
    identify = run(["hailortcli", "fw-control", "identify"], timeout=30)
    if identify:
        for line in identify.splitlines()[:6]:
            report.info(f"fw-control: {line.strip()}")
    else:
        report.info("fw-control identify returned nothing (needs the module in the M.2 slot)")
    return True


def probe_binding(report: Report) -> bool:
    try:
        module = importlib.import_module("hailo_platform")
    except ImportError as error:
        return report.check("hailo_platform binding imports", False,
                            f"{error}; HailoRT is not installed - pass HAILORT_WHEEL=/path/to/"
                            f"hailort-4.23.0-cp311-cp311-linux_aarch64.whl to the install script")
    version = getattr(module, "__version__", "present")
    return report.check("hailo_platform binding imports", True, f"{version} ({module.__file__})")


def probe_dependencies(report: Report) -> bool:
    ok = True
    for name in ("numpy", "cv2"):
        try:
            module = __import__(name)
            report.check(f"{name} imports", True, getattr(module, "__version__", "present"))
        except ImportError as error:
            ok = report.check(f"{name} imports", False, f"{error}")
    return ok


def one_frame_inference(report: Report, root: Path) -> bool:
    """Verify the HEF end to end on the device: one frame in, decoded boxes out."""
    try:
        from hailo_pipeline import HailoPipeline, collect_heads
    except ImportError as error:
        return report.check("one-frame inference", False, f"{error}")

    from postprocess_common import letterbox, to_network_input
    from postprocess_yolo26 import decode_detections_yolo26

    frame, frames, fps = read_first_frame(root)
    if frame is None:
        return report.check("one-frame inference", False, "cannot read the sample clip")

    pipeline = None
    try:
        pipeline = HailoPipeline(root / MODEL, depth=1)
        pipeline.submit(to_network_input(letterbox(frame, 640)[0]))
        _, outputs = pipeline.collect()
        box_heads, score_heads = collect_heads(outputs)
        boxes, scores, classes = decode_detections_yolo26(box_heads, score_heads,
                                                          conf_thres=0.25, iou_thres=0.45)
    except Exception as error:
        return report.check("one-frame inference", False, f"{type(error).__name__}: {error}")
    finally:
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                pass

    return report.check("one-frame inference", True,
                        f"{len(scores)} detection(s) from frame 1 of {frames} @ {fps:.0f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Hailo-8 environment.")
    parser.parse_args()

    root = repo_root()
    report = Report("hailo8", "Hailo-8 (HailoRT)")
    if not report.platform_gate():
        raise SystemExit(report.finish())

    entries = load_checksums(root)
    for label, relative in (("model file", MODEL), ("sample clip", "video/test.mp4")):
        ok, detail = verify_artefact(root, relative, entries)
        report.check(label, ok, detail)
    probe_dependencies(report)
    probe_driver(report)
    probe_tool(report)
    if probe_binding(report):
        one_frame_inference(report, root)
    raise SystemExit(report.finish())


if __name__ == "__main__":
    main()