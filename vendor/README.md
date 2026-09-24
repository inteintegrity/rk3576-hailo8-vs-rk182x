# vendor/

Everything the three environments need that is not already on the board. The install scripts pull
from here with `pip --no-index`, so a board with no network can still be set up, and the venvs are
self-contained: **no PYTHONPATH, no LD_LIBRARY_PATH, no system site-packages**.

| File in `vendor/wheels/` | Version | Used by |
|---|---|---|
| `numpy-1.26.4-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 1.26.4 | all three backends |
| `opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 4.11.0.86 | all three backends |
| `rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2.3.2 | RK3576 built-in NPU |
| `psutil-7.2.2-cp36-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | 7.2.2 | declared by rknn-toolkit-lite2 |
| `ruamel_yaml-0.19.1-py3-none-any.whl` | 0.19.1 | declared by rknn-toolkit-lite2 |
| `setuptools-75.8.0-py3-none-any.whl` | 75.8.0 | `rknnlite/api/rknn_lite.py` reads its version through `pkg_resources` |
| `hailort-4.23.0-cp311-cp311-linux_aarch64.whl` | 4.23.0 | Hailo-8 |
| `argcomplete-3.7.2-py3-none-any.whl` | 3.7.2 | declared by hailort |
| `contextlib2-21.6.0-py2.py3-none-any.whl` | 21.6.0 | declared by hailort |
| `future-1.0.0-py3-none-any.whl` | 1.0.0 | declared by hailort |
| `netaddr-1.3.0-py3-none-any.whl` | 1.3.0 | declared by hailort (Ethernet device helpers) |

Two declared dependencies are deliberately **not** bundled, and the installers work without them:

- `netifaces` (declared by hailort) is imported only by
  `hailo_platform/pyhailort/ethernet_utils.py`, which the PCIe import path never touches - it is for
  Hailo devices that enumerate over Ethernet. No CPython 3.11 aarch64 wheel of it exists on PyPI.
- `transformers` (declared by rknn3-toolkit-lite) is only used by its LLM helper
  (`rknn3lite/api/rknn3_lite_llm.py`), which `rknn3lite/__init__.py` does not import.

The dependency list of each backend was derived from the wheels' `Requires-Dist` metadata plus a
static scan of every import in the installed packages, and each list is installed with
`pip --no-index --find-links vendor/wheels`, so a missing file fails the install instead of surfacing
at run time.

All of these are CPython 3.11 aarch64 wheels or pure-Python wheels; the install scripts refuse to run on another
architecture or interpreter rather than failing later. Every file here is covered by
`checksums.sha256` in the repository root and verified by `scripts/verify-checksums.py` before an
install touches anything.

`vendor/rknn3/` is different: Rockchip distributes the RKNN3 Python binding as part of the RK182x
SDK rather than as a wheel. This repository therefore ships the portable CPython 3.11 aarch64
package taken from that SDK environment:

```
vendor/rknn3/rknn3lite-cp311-aarch64.tar.gz
```

The installer verifies the tarball through `checksums.sha256`, unpacks it into `.venvs/rk182x`, and
then proves it by running one real inference. Its fallback discovery logic remains for SDK images
that replace the shipped package deliberately, but a fresh clone no longer depends on an older
`~/rk1820_yolo/rknn3_env` directory.

## Kernel drivers

Nothing in `vendor/` and nothing in `scripts/` installs, replaces, or reloads a kernel driver. The
board keeps its own `pcie-rkep` (RK182x) and `hailo_pci` (Hailo-8) drivers and its own
`librknnrt.so` / `libhailort.so`; the checks only report what they find.

## Redistribution

The two vendor wheels that are not open-source packages need a decision before this repository is
published anywhere:

- **`hailort-4.23.0-cp311-cp311-linux_aarch64.whl`** - the HailoRT Python binding, distributed by
  Hailo through their Developer Zone. Its terms are Hailo's; check them before redistributing the
  file. (The board also needs Hailo's own `.deb` packages for the driver and `libhailort.so`, which
  are not part of this repository.)
- **`rknn3lite`** (inside `vendor/rknn3/rknn3lite-cp311-aarch64.tar.gz`) - Rockchip's on-board
  Python interface for the RK182x, distributed with the RK182x SDK. Check Rockchip's terms before
  redistributing the tarball.

The three aarch64 wheels from PyPI (NumPy, OpenCV headless) and the Rockchip RKNNLite2 wheel are
covered by their own licences, which are collected in the wheel metadata (`*.dist-info/LICENSE*`).
