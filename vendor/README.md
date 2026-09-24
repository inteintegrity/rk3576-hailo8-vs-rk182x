# vendor/

Everything the three environments need that is not already on the board. The install scripts pull
from here with `pip --no-index`, so a board with no network can still be set up, and the venvs are
self-contained: **no PYTHONPATH, no LD_LIBRARY_PATH, no system site-packages**.

| File in `vendor/wheels/` | Version | Size | Used by |
|---|---|---:|---|
| `numpy-1.26.4-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 1.26.4 | 14.2 MB | all three backends |
| `opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 4.11.0.86 | 29.5 MB | all three backends |
| `rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 2.3.2 | 569 KB | RK3576 built-in NPU |
| `hailort-4.23.0-cp311-cp311-linux_aarch64.whl` | 4.23.0 | 9.9 MB | Hailo-8 |

All four are CPython 3.11 aarch64 wheels; the install scripts refuse to run on another
architecture or interpreter rather than failing later. Every file here is covered by
`checksums.sha256` in the repository root and verified by `scripts/verify-checksums.py` before an
install touches anything.

`vendor/rknn3/` is different: the RKNN3 Python binding has no wheel in this repository, so
`scripts/install-rk182x.sh` builds one portable package from the environment that is already on the
board:

```
vendor/rknn3/rknn3lite-cp311-aarch64.tar.gz          produced by the installer, not shipped
vendor/rknn3/rknn3lite-cp311-aarch64.tar.gz.sha256   its digest
```

The installer looks for `rknn3lite` in the known board locations (`~/rk1820_yolo/rknn3_env`,
`~/rknn3_env`, `/opt/rknn3/rknn3_env`, or wherever the system interpreter imports it), packs the
package with its metadata, unpacks it into `.venvs/rk182x`, and then proves it by running one real
inference. The same tarball can be copied to another board; the generated files are git-ignored
because they are produced on the board, not authored here.

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
- **`rknn3lite`** (inside the generated tarball) - Rockchip's on-board Python interface for the
  RK182x, distributed with the RK182x SDK. Same question: check Rockchip's terms before shipping
  the tarball.

The three aarch64 wheels from PyPI (NumPy, OpenCV headless) and the Rockchip RKNNLite2 wheel are
covered by their own licences, which are collected in the wheel metadata (`*.dist-info/LICENSE*`).
