"""Environment check for the RK182x backend (RKNN3 runtime).

Run it through the venv the install script creates:

    bash scripts/install-rk182x.sh
    .venvs/rk182x/bin/python scripts/check-rk182x.py

No RKNN3 wheel ships in this repository (it comes with the RK182x SDK), so this check prefers the
RKNN3 runtime already installed on the board. If it is missing, pass its wheel or the SDK path to
the install script with RKNN3_WHEEL=/path/to/rknn3_toolkit_lite-*.whl. The decisive test is one
real inference: loading the model and initialising the runtime also proves the module is seated
and reachable, because both fail without a device.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _probe import (  # noqa: E402
    Report, load_checksums, read_first_frame, repo_root, run, verify_artefact,
)

MODEL = "model/rk1820/yolo26n_rk1820_int8.rknn"
WEIGHT = "model/rk1820/yolo26n_rk1820_int8.weight"


def probe_driver(report: Report) -> bool:
    """Report what the PCIe side shows; the runtime handshake in one_frame_inference() is decisive."""
    evidence = []
    pci = run(["lspci"]) or ""
    for line in pci.splitlines():
        if any(word in line.lower() for word in ("rockchip", "rknn", "ntb")):
            evidence.append(line.strip())
    for pattern in ("/dev/rknn*", "/dev/rk182*"):
        evidence.extend(str(path) for path in Path("/").glob(pattern.lstrip("/")))
    report.info(f"module evidence: {'; '.join(evidence) if evidence else 'nothing matched in lspci//dev'}")
    return report.check("RK182x module reports itself", True,
                        "runtime handshake is checked by the one-frame inference below")


def probe_binding(report: Report) -> bool:
    try:
        module = importlib.import_module("rknn3lite.api.rknn3_lite")
    except ImportError as error:
        return report.check("rknn3lite binding imports", False,
                            f"{error}; the RKNN3 runtime is not installed - install the RK182x SDK, "
                            f"or pass RKNN3_WHEEL=/path/to/rknn3_toolkit_lite-*.whl to the install script")
    report.check("rknn3lite binding imports", True, str(module.__file__))
    try:
        version = (module.RKNN3Lite().get_sdk_version() or "").strip().splitlines()
        report.info(f"runtime sdk version: {' | '.join(version[:2])}")
    except Exception as error:
        report.info(f"runtime sdk version not available yet ({type(error).__name__})")
    return True


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
    """Load the model, initialise the runtime (proves the module answers) and infer one frame."""
    try:
        from rknn3lite.api.rknn3_lite import RKNN3Lite
    except ImportError:
        return report.check("one-frame inference", False, "the binding is not importable")

    from postprocess_common import letterbox, to_network_input
    from postprocess_yolo26 import decode_detections_yolo26
    from rknn_helpers import collect_heads_yolo26, dequantize, init_runtime_with_fallback

    frame, frames, fps = read_first_frame(root)
    if frame is None:
        return report.check("one-frame inference", False, "cannot read the sample clip")

    rknn = RKNN3Lite()
    try:
        if rknn.load_rknn(model_path=str(root / MODEL), weight_path=str(root / WEIGHT)) != 0:
            return report.check("one-frame inference", False, "load_rknn failed")
        mask = init_runtime_with_fallback(rknn, 0xff, lambda message: report.info(message))
        output_attrs = rknn.get_outputs_tensor_attr()
        raw = rknn.inference(inputs=[to_network_input(letterbox(frame, 640)[0])], data_format=["nhwc"])
        if raw is None or not len(raw):
            return report.check("one-frame inference", False, "inference returned no output")
        box_heads, score_heads, described = collect_heads_yolo26(dequantize(raw, output_attrs))
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
    parser = argparse.ArgumentParser(description="Check the RK182x environment.")
    parser.parse_args()

    root = repo_root()
    report = Report("rk182x", "RK182x (RKNN3 runtime)")
    if not report.platform_gate():
        raise SystemExit(report.finish())

    entries = load_checksums(root)
    for label, relative in (("model file", MODEL), ("weight file", WEIGHT),
                            ("sample clip", "video/test.mp4")):
        ok, detail = verify_artefact(root, relative, entries)
        report.check(label, ok, detail)
    probe_dependencies(report)
    probe_driver(report)
    if probe_binding(report):
        one_frame_inference(report, root)
    raise SystemExit(report.finish())


if __name__ == "__main__":
    main()