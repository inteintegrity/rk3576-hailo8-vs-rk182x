# Environment

Where each piece runs, the exact versions behind every record in `results/`, and what each of
the three backends needs installed - with the command that checks it.

## Board (RK3576, aarch64, Python 3.11.2)

Common to all three runners:

| Package | Version recorded in the runs |
|---|---|
| Python | 3.11.2 |
| numpy | 1.24.2 (Hailo-8 run), 1.26.4 (RK3576 run), 2.4.6 (RK182x run) |
| OpenCV | 4.6.0 (Hailo-8 run), 4.11.0 (RK3576 run), 5.0.0 (RK182x run) |

Per accelerator:

| Accelerator | Runtime | Version |
|---|---|---|
| Hailo-8 | HailoRT Python bindings, `InferModel` + `run_async` | 4.23.0 |
| RK3576 built-in NPU | `rknn-toolkit-lite2` / `librknnrt` | librknnrt 2.3.0 |
| RK182x | RKNN3 runtime, `rknn3lite.api.rknn3_lite.RKNN3Lite` | RKNN3 runtime 1.0.4 |

The vendor runtimes come from the vendor's installer and must match the installed driver:
`hailort` is version-locked to the HailoRT driver, `rknn-toolkit-lite2` ships with `librknnrt`,
and the RKNN3 runtime comes with the RK182x SDK. The runners import them inside `main()`, so
`--help`, the figure generators and the checks work on a host without them.

`python3 rk182x/rk1820/run_video_inference.py --help` etc. therefore runs on any machine with
numpy and OpenCV, which is what makes the code inspectable without the hardware.

## Installing each backend

The three runtimes are vendor packages, not PyPI eggs: each has to match the driver or firmware on
the board, and each is distributed through the vendor's own channel. What follows is what each
backend needs, where it comes from, how to check it, and what this board actually had. Nothing here
was executed by this project's scripts - the runners only import the result.

The one command that covers all three at once is the checker:

```bash
python3 common/check_environment.py            # all three backends, exit code 0 when all are ready
python3 common/check_environment.py --backend rk3576 --json out/environment.json
```

It imports each runtime, prints the module path, the version it reports and (on the board) the
helper binary or shared library next to it, and exits non-zero if anything is missing.

### Hailo-8

| | |
|---|---|
| runner imports | `hailo_platform` (`VDevice`, `HEF`, `FormatType`, `InferModel.run_async`) |
| also needed | Hailo's PCIe kernel driver for the M.2 card, and `hailortcli` for firmware checks |
| version rule | the HailoRT user-space package (and its Python bindings) must match the installed driver and firmware version exactly |
| where it comes from | Hailo's Developer Zone: the HailoRT packages and the AI Software Suite; the driver package is the board-specific one for the host's kernel. Hailo's own installation guide is `docs/GETTING_STARTED.rst` in [hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo) |
| install | Hailo does not publish these on PyPI: download the board's HailoRT `.deb` (or the AI Software Suite) and the PCIe driver for your kernel from the Developer Zone, install the driver first, then HailoRT, then confirm the Python bindings import. There is no public mirror, which is why this delivery cannot ship one |
| check | `hailortcli --version`, then `hailortcli fw-control identify` with the module in the M.2 slot (device, firmware, PCIe link), and `python3 -c "import hailo_platform"`; `python3 common/check_environment.py --backend hailo8` does the last two |
| on this board | HailoRT 4.23.0, module at `0000:01:00.0`, PCIe **Gen2 x1** (the module supports Gen3 x4) |

### RK3576 built-in NPU

| | |
|---|---|
| runner imports | `rknnlite.api.RKNNLite` from `rknn-toolkit-lite2` |
| also needed | `librknnrt.so` (the NPU runtime library, `librknnrt` 2.3.0 here) |
| version rule | the `rknn_toolkit_lite2` wheel must match the board's Python (3.11 here) and the `librknnrt.so` that ships with it; models must be compiled for the same runtime generation |
| where it comes from | Rockchip's [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2): the board wheel is `rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` (cp37-cp312 for the other Python versions) and the runtime library is `rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so`; the full release is on Rockchip's cloud drive (fetch code `rknn`). Toolkit2 v2.3.2 is current and lists RK3576 and Python 3.6-3.12 |
| install | `pip install <wheel>` on the board, then put `librknnrt.so` where the loader finds it (`sudo cp librknnrt.so /usr/lib/ && sudo ldconfig`). Wheel and library must come from the same release |
| check | `python3 common/check_environment.py --backend rk3576`, or `python3 -c "from rknnlite.api import RKNNLite; print(RKNNLite().get_sdk_version())"`; `ldconfig -p` lists it under `librknnrt.so` |
| on this board | librknnrt 2.3.0, two NPU cores (`core_mask` `0x3` selects both, `0x1`/`0x2` one each). The 2.3.2 wheel carries its own `librknnrt.so`; install the release whose runtime you want, keeping wheel and library together |

### RK182x

| | |
|---|---|
| runner imports | `rknn3lite.api.rknn3_lite.RKNN3Lite` (the on-board Python interface that wraps the RKNN3 runtime's C API) |
| also needed | the RKNN3 runtime on the board, and the module seated in the M.2 slot (the runtime reports the device) |
| version rule | **toolkit and runtime versions must match**: the source project's checklist records that a model compiled with toolkit 1.0.5 failed on this board's 1.0.4 runtime with `RKNN3_ERR_FAIL`, which is why everything here is 1.0.4 |
| where it comes from | Rockchip's [airockchip/rknn3-toolkit](https://github.com/airockchip/rknn3-toolkit): the board wheel is `rknn3-toolkit-lite/packages/rknn3_toolkit_lite-<version>-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` (1.1.0 is published, with a `requirements.txt` beside it), the runtime library is in `rknn3-runtime/`, and the SDK with firmware, documentation and pre-converted models is on Rockchip's cloud drive (access code `rknn`). The board vendor's own walkthrough: Seeed's [RK3588/RK182x RKNN3 deployment guide](https://sensecraft.seeed.cc/ai-lab/zh/tools/rk/rk3588-rk182x-rknn3-deploy) |
| install | `pip install <wheel>` on the board, matching the RKNN3 runtime installed there. **Keep toolkit and runtime on the same version**: this board runs runtime 1.0.4 and a model compiled with the 1.0.5 toolkit failed on it with `RKNN3_ERR_FAIL`. The published SDK has moved to V1.1.0 (its docs list Python 3.10/3.12), while the records here were produced with 1.0.4 on Python 3.11.2 |
| check | `python3 common/check_environment.py --backend rk1820`, or `python3 -c "from rknn3lite.api.rknn3_lite import RKNN3Lite; print(RKNN3Lite().get_sdk_version())"`; the vendor's `rknn3_model_test` benchmark is the device-level check |
| on this board | RKNN3 runtime 1.0.4, module at the M.2 slot; the model in `model/rk1820/` is compiled for **one** core, and the runtime refuses a mask that does not match, so the runners request a mask and record the one that was accepted |

`common/check_osd_figure.py` is deliberately not part of this list: it needs no accelerator, only
OpenCV (see below).

## Host side (figures and verification)

```
pip install -r requirements.txt
```

- `common/make_final_figures.py` needs matplotlib and only reads the JSON records.
- `common/make_osd_figure.py` and `common/check_osd_figure.py` need OpenCV.

### The one build-dependent step

`common/check_osd_figure.py` re-renders the on-screen banner with the same font metrics the clip
was drawn with and compares it pixel by pixel. OpenCV 4.x and 5.x render the Hershey font with
different metrics - the same two-line banner comes out 482x78 px under 4.x and 402x86 px under
5.x - so the check is only meaningful under the same OpenCV major version that drew the frame.
Each result JSON records that version (`environment.opencv`), and the script reports the devices
it cannot verify as "skipped" with the version to use instead of reporting a false failure.

All three screenshots in this folder are verified in `results/single_stream/frame200_check.json`:
`hailo8` and `rk3576_npu` under OpenCV 4.11 (the frames were drawn with 4.6 / 4.11, which render
identically here), `rk1820` under OpenCV 5.0. To reproduce, run the script once per major version
- the report accumulates both runs.

## Timing methodology

Every runner reports three layers, so host-side work is never presented as accelerator speed:

| Field in the JSON | What it measures |
|---|---|
| `python_infer_only_fps` / `device_service_fps` | the accelerator call itself (Hailo-8: the device service rate with 4 inferences in flight; RK3576 / RK182x: the per-call time, which includes the PCIe round trip on RK182x) |
| `python_pipeline_fps` (`pipeline_without_io`) | the accelerator call plus host decode and NMS. The Hailo-8 runner measures it over the whole loop iteration and therefore also includes the frame read and the letterbox; the two Rockchip runners exclude those two and record them separately in `read_letterbox_infer_decode`. `results/README.md` gives the like-for-like figures. |
| `python_end_to_end_fps` | + drawing and MP4 writing (the annotated pass) |

The multi-stream records report aggregate throughput instead: total processed frames per second
across all streams, measured in a window that starts when every worker has loaded its model and
warmed up and ends when the last worker finishes.