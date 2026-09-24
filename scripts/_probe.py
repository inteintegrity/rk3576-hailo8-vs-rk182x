"""Shared helpers for the three per-backend environment checks.

This module is not a check itself: it resolves the repository root, verifies shipped artefacts
against checksums.sha256, reads the sample clip's first frame, and prints the PASS/FAIL report
the three `check-*.py` scripts share. Keeping it in one place is what stops the three checks
from drifting apart.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ARCHES = ("aarch64", "arm64")
REQUIRED_PYTHON = (3, 11)
SAMPLE_CLIP = "video/test.mp4"


def repo_root() -> Path:
    """The folder that holds common/, model/, wheels/ and video/."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "common" / "postprocess_yolo26.py").is_file():
            return candidate
    raise SystemExit("cannot locate the repository root above scripts/ (common/ is missing)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(root: Path) -> dict[str, str]:
    """Path -> sha256 from checksums.sha256 (models, the vendor wheel, the sample clip)."""
    path = Path(root) / "checksums.sha256"
    entries: dict[str, str] = {}
    if not path.is_file():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition("  ")
        if name:
            entries[name] = digest
    return entries


def verify_artefact(root: Path, relative: str, entries: dict[str, str]) -> tuple[bool, str]:
    """Check one shipped file exists and matches checksums.sha256."""
    path = Path(root) / relative
    if not path.is_file():
        return False, f"{relative} is missing"
    actual = sha256(path)
    expected = entries.get(relative)
    if expected is None:
        return True, f"{relative} present, sha256 {actual[:12]}... (not covered by checksums.sha256)"
    if actual != expected:
        return False, f"{relative} sha256 mismatch: {actual} expected {expected}"
    return True, f"{relative} sha256 ok ({path.stat().st_size:,} bytes)"


def which(command: str) -> str | None:
    return shutil.which(command)


def run(command: list[str], timeout: int = 20) -> str | None:
    try:
        finished = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (finished.stdout or finished.stderr).strip()
    return text or None


def read_first_frame(root: Path, relative: str = SAMPLE_CLIP):
    """The sample clip's first frame, as (frame, frames_in_clip, source_fps)."""
    import cv2

    path = Path(root) / relative
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None, 0, 0.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    ok, frame = capture.read()
    capture.release()
    return (frame if ok else None), total, fps


class Report:
    """PASS/FAIL report with one line per item and a single exit code."""

    def __init__(self, backend: str, title: str) -> None:
        self.backend = backend
        self.title = title
        self.failures = 0
        print(f"== {title} ({backend}) ==")
        print(f"   python {platform.python_version()} on {platform.machine()}, {platform.platform()}")

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"   [{mark}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            self.failures += 1
        return ok

    def info(self, text: str) -> None:
        print(f"   [info] {text}")

    def platform_gate(self) -> bool:
        """aarch64 and Python 3.11 - the two hard requirements of the bundled wheel."""
        ok = True
        machine = platform.machine()
        ok &= self.check("architecture is aarch64", machine in ARCHES,
                         "" if machine in ARCHES else f"found {machine}; this package targets RK3576 (aarch64 Linux)")
        version = sys.version_info[:2]
        ok &= self.check(f"python is {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}", version == REQUIRED_PYTHON,
                         "" if version == REQUIRED_PYTHON else
                         f"found {version[0]}.{version[1]}; the bundled wheel is CPython 3.11, "
                         f"run this through the venv the install script creates")
        return ok

    def finish(self) -> int:
        if self.failures:
            print(f"\n{self.failures} check(s) failed - {self.backend} is not ready")
            return 1
        print(f"\n{self.backend} is ready")
        return 0