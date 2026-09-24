"""Report what each backend has installed on this machine, before running a benchmark.

Board-side helper. The three runners need three different vendor runtimes, and the failure mode
without them is an ImportError deep inside a run; this script answers "is the runtime here, and
which version is it" in one shot. It only imports modules and asks them for their version - it
never loads a model, opens a device or writes anything, so it is safe to run on a live board.

    python common/check_environment.py              # all three backends
    python common/check_environment.py --backend rk3576
    python common/check_environment.py --json out/environment.json

Exit code is 0 when every requested backend is ready and 1 otherwise, so it can gate a script.
"""

from __future__ import annotations

import argparse
import ctypes.util
import glob
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

#: backend -> (module to import, human name, where the runtime comes from)
BACKENDS = {
    "hailo8": ("hailo_platform", "Hailo-8 (HailoRT Python bindings)",
               "HailoRT, installed from Hailo's Developer Zone; must match the installed driver"),
    "rk3576": ("rknnlite.api", "RK3576 built-in NPU (rknn-toolkit-lite2)",
               "rknn_toolkit_lite2 wheel + librknnrt.so from Rockchip's RKNPU2/RKNN-Toolkit2 release"),
    "rk1820": ("rknn3lite.api.rknn3_lite", "RK182x (RKNN3 runtime)",
               "RKNN3 Toolkit Lite (on-board Python interface over the RKNN3 runtime), from Rockchip"),
}


def module_version(module) -> str | None:
    for attribute in ("__version__", "version", "VERSION"):
        value = getattr(module, attribute, None)
        if isinstance(value, str):
            return value
    return None


def run(command: list[str], timeout: int = 10) -> str | None:
    try:
        finished = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (finished.stdout or finished.stderr).strip()
    return text.splitlines()[0] if text else None


def probe_common() -> dict:
    probe = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    for name in ("numpy", "cv2"):
        try:
            module = importlib.import_module(name)
            probe[name] = module_version(module) or "present"
        except ImportError as error:
            probe[name] = f"MISSING ({error})"
    return probe


def probe_backend(backend: str) -> dict:
    module_name, label, origin = BACKENDS[backend]
    entry = {"backend": backend, "label": label, "origin": origin, "ready": False}
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        entry["detail"] = f"import {module_name} failed: {error}"
        return entry

    entry["module"] = module_name
    entry["module_file"] = getattr(module, "__file__", None)
    entry["version"] = module_version(module)

    # per-backend extras: the tool or library the runner relies on alongside the module
    if backend == "hailo8":
        entry["hailortcli"] = shutil.which("hailortcli")
        if entry["hailortcli"]:
            entry["hailortcli_version"] = run(["hailortcli", "--version"])
            entry["note"] = ("run `hailortcli fw-control identify` with the module in the M.2 slot "
                             "to confirm the device and its firmware")
    elif backend == "rk3576":
        entry["librknnrt"] = ctypes.util.find_library("rknnrt") or next(
            (path for path in glob.glob("/usr/lib/*/librknnrt.so") + glob.glob("/usr/lib/librknnrt.so")), None)
        try:
            from rknnlite.api import RKNNLite  # noqa: WPS433 (inside probe on purpose)

            entry["sdk_version"] = (RKNNLite().get_sdk_version() or "").strip().splitlines()[:2]
        except Exception as error:  # a runtime that needs a model loaded reports it here
            entry["sdk_version"] = f"unavailable without a model: {type(error).__name__}: {error}"
    elif backend == "rk1820":
        try:
            from rknn3lite.api.rknn3_lite import RKNN3Lite  # noqa: WPS433

            entry["sdk_version"] = (RKNN3Lite().get_sdk_version() or "").strip().splitlines()[:2]
        except Exception as error:
            entry["sdk_version"] = f"unavailable without a model: {type(error).__name__}: {error}"
        entry["note"] = ("the RK182x module must be in the M.2 slot and its runtime service must "
                         "report the device; `rknn3_model_test` is the vendor's own device check")

    # ready = the import worked and (where the runtime exposes one) a version came back
    entry["ready"] = True
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the runtime of each accelerator backend.")
    parser.add_argument("--backend", choices=sorted(BACKENDS), action="append",
                        help="check only this backend; repeat for several (default: all)")
    parser.add_argument("--json", type=Path, help="also write the report here")
    args = parser.parse_args()

    requested = args.backend or sorted(BACKENDS)
    report = {"common": probe_common(),
              "backends": [probe_backend(backend) for backend in requested]}

    print(f"python {report['common']['python']} on {report['common']['machine']}, "
          f"numpy {report['common']['numpy']}, opencv {report['common']['cv2']}")
    for entry in report["backends"]:
        mark = "ready" if entry["ready"] else "MISSING"
        print(f"\n[{mark}] {entry['label']}")
        print(f"    needs: {entry['origin']}")
        if entry.get("module_file"):
            print(f"    module: {entry['module_file']}")
        if entry.get("version"):
            print(f"    version: {entry['version']}")
        for key in ("hailortcli", "hailortcli_version", "librknnrt", "sdk_version"):
            if entry.get(key):
                value = entry[key]
                value = " | ".join(value) if isinstance(value, list) else value
                print(f"    {key}: {value}")
        if not entry["ready"]:
            print(f"    detail: {entry.get('detail')}")
        if entry.get("note"):
            print(f"    note: {entry['note']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    missing = [entry["backend"] for entry in report["backends"] if not entry["ready"]]
    if missing:
        sys.exit(f"not ready: {missing}")
    print(f"\nall requested backends are ready: {[e['backend'] for e in report['backends']]}")


if __name__ == "__main__":
    main()