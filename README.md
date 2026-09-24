# RK3576: built-in NPU vs Hailo-8 vs RK182x

Run the same **YOLO26n** model on three accelerators connected to one reComputer RK3576:

- RK3576 built-in NPU
- Hailo-8 M.2 module
- RK182x / RK1820 M.2 module

The repository includes the three compiled models, a sample video, pinned aarch64 Python
dependencies and one install/run pair per backend. No Docker, `PYTHONPATH` or system Python
fallback is required.

## Quick start

Requirements common to every backend:

- reComputer RK3576 running aarch64 Linux
- CPython 3.11 with `venv`
- the board firmware's compatible accelerator driver and user-space runtime libraries

Hailo-8 and RK182x share the same M.2 slot, so install only one external module at a time.
The scripts do not install, replace or reload kernel drivers.

```bash
git clone https://github.com/inteintegrity/rk3576-hailo8-vs-rk182x.git
cd rk3576-hailo8-vs-rk182x
```

### RK3576 built-in NPU

```bash
bash scripts/install-rk3576.sh
bash scripts/run-rk3576.sh
```

### Hailo-8

Requires a Hailo-8 module and matching HailoRT 4.23 host driver/libraries.

```bash
bash scripts/install-hailo8.sh
bash scripts/run-hailo8.sh
```

### RK182x

Requires an RK182x module bound to the `pcie-rkep` driver and matching RKNN3 runtime libraries.
The CPython 3.11 aarch64 `rknn3lite` binding is included under `vendor/rknn3/`.

```bash
bash scripts/install-rk182x.sh
bash scripts/run-rk182x.sh
```

Each installer verifies the shipped files, creates a self-contained `.venvs/<backend>` environment,
installs dependencies without accessing PyPI, checks the driver/device state, and runs one real
inference on one frame of `video/test.mp4`.

Each runner defaults to `video/test.mp4`. Pass another clip as the first argument:

```bash
bash scripts/run-rk3576.sh /path/to/clip.mp4
```

Successful single-stream runs write:

```text
out/<backend>/annotated.mp4
out/<backend>/video_result.json
out/<backend>/run.log
```

## Multi-stream benchmark

Install the corresponding backend first, then call the benchmark through its venv:

```bash
.venvs/hailo8/bin/python Hailo/Hailo8/run_streams_aggregate.py \
  --hef model/Hailo/yolo26n_hailo8_official.hef \
  --video video/test.mp4 --streams 8 --frames 200 \
  --json out/hailo8_8stream.json

.venvs/rk3576/bin/python common/run_video_streams_benchmark.py \
  --backend rk3576 --model model/rk3576/yolo26n_rk3576_int8.rknn \
  --video video/test.mp4 --instances 1,2,4,8 --frames 200 \
  --json out/rk3576_multi.json

.venvs/rk182x/bin/python common/run_video_streams_benchmark.py \
  --backend rk1820 --model model/rk1820/yolo26n_rk1820_int8.rknn \
  --weight model/rk1820/yolo26n_rk1820_int8.weight \
  --video video/test.mp4 --instances 1,2,4,8 --frames 200 \
  --json out/rk1820_multi.json
```

## Published results

| Accelerator | Single-stream pipeline | 8-stream aggregate |
|---|---:|---:|
| Hailo-8 | **49.2 FPS** | 51.6 FPS |
| RK3576 built-in NPU | 23.5 FPS | 69.7 FPS |
| RK182x | 17.5 FPS | **88.6 FPS** |

![Single-stream comparison](results/figures/single_stream_bars.png)

![Multi-stream comparison](results/figures/multi_stream_scaling.png)

The compact machine-readable summary is in [`results/summary.json`](results/summary.json). Full
per-frame JSON, raw multi-stream records, screenshots, report articles and figure-generation tools
are published in the
[`v1.0.0` benchmark artifacts](https://github.com/inteintegrity/rk3576-hailo8-vs-rk182x/releases/tag/v1.0.0)
so ordinary clones remain small.

## Repository layout

```text
Hailo/Hailo8/       Hailo-8 runners
rk3576/             built-in NPU runner
rk182x/rk1820/      RK182x runner
common/             shared preprocessing, YOLO26 decode and benchmark code
model/              HEF, RKNN and RKNN3 weight artifacts
scripts/            install, environment-check and run entry points
vendor/             pinned aarch64 Python dependencies
video/test.mp4      shared sample clip
results/            compact summary and final charts
docs/ENVIRONMENT.md runtime versions and timing definitions
```

## Runtime boundary

Python dependencies are bundled, but kernel drivers remain part of the board image:

- RK3576 NPU: firmware `rknpu` driver and compatible `librknnrt.so`
- Hailo-8: HailoRT 4.23 PCIe driver and `libhailort.so`
- RK182x: `pcie-rkep` driver and RKNN3 runtime libraries

The checks distinguish a missing Python binding, an installed driver with no module in the slot,
and a present device that is not bound to its driver.

## Redistribution

`vendor/wheels/hailort-4.23.0-cp311-cp311-linux_aarch64.whl` comes from Hailo's Developer Zone,
and `vendor/rknn3/rknn3lite-cp311-aarch64.tar.gz` comes from Rockchip's RK182x SDK environment.
Verify the respective vendor redistribution terms before mirroring or republishing these files.

## License

No project license is attached. Contact the repository owner before reusing the code or bundled
vendor artifacts.
