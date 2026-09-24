# RK3576 accelerators head to head: built-in NPU vs Hailo-8 vs RK182x

One board (reComputer RK3576 devkit), one model (**YOLO26n**, COCO 80 classes, 640x640, INT8),
three accelerators. Everything under `results/` was produced with the code in this folder.

| Accelerator | Interface | Runtime | 1 stream: inference / full pipeline | 8 streams: aggregate |
|---|---|---|---:|---:|
| **Hailo-8** (M.2) | PCIe Gen2 x1 | HailoRT 4.23 `InferModel` pipeline | **51.2 / 49.2 FPS** | 51.6 FPS |
| **RK3576 built-in NPU** | on-die, 2 NPU cores | rknn-toolkit-lite2 (librknnrt 2.3.0) | 32.9 / 23.5 FPS | 69.7 FPS |
| **RK182x** (M.2) | PCIe, RKNN3 NTB | RKNN3 runtime 1.0.4 (`rknn3lite`) | 22.8 / 17.5 FPS | **88.6 FPS** |

**Short version.** On a single stream Hailo-8 wins: 51.2 FPS on the device, 49.2 FPS through the
whole host pipeline, one inference in 13.9 ms, and none of the SoC's NPU is used. With many
streams RK182x wins on total throughput (88.6 FPS at 8 streams, 1.27x the built-in NPU), while
the built-in NPU is the cheapest option and overtakes Hailo-8 above two streams. Hailo-8's
aggregate is flat (50-52 FPS at any stream count) because one module is one device: it serves
every stream at its full rate rather than adding capacity. Full write-ups:
[`docs/REPORT.md`](docs/REPORT.md) (Chinese, all tables),
[`docs/ARTICLE_zh.md`](docs/ARTICLE_zh.md), [`docs/ARTICLE_en.md`](docs/ARTICLE_en.md).

## Run it

On the board, one triple per backend - install once, then run any clip:

```bash
bash scripts/install-rk3576.sh     # RKNNLite2 wheel from wheels/, board numpy/OpenCV reused
bash scripts/run-rk3576.sh         # the bundled sample clip; pass a path for another clip

bash scripts/install-hailo8.sh     # uses the HailoRT already on the board
bash scripts/run-hailo8.sh

bash scripts/install-rk182x.sh     # uses the RKNN3 runtime already on the board
bash scripts/run-rk182x.sh
```

Each install creates `.venvs/<backend>` (a venv with `--system-site-packages`, so nothing is
downloaded), verifies the shipped files against `checksums.sha256` and then runs the backend's
check, which ends with **one real inference on one frame of the sample clip**. The run scripts
always use that venv and refuse to fall back to the system Python. `HAILORT_WHEEL=/path` and
`RKNN3_WHEEL=/path` install a locally held wheel instead of relying on the board's runtime; those
two wheels are deliberately not shipped here (`docs/ENVIRONMENT.md`).

**Every runner defaults to the clip that ships with this repository**, `video/test.mp4` (the one
behind every record in `results/`; the records name it `test_640.mp4`, the name it had during the
runs), so the video argument can simply be left out. Any other clip works too - it is letterboxed
to 640x640 automatically.

The multi-stream tools have no wrapper (they need more arguments), so call the runners directly -
still through a venv:

```bash
.venvs/rk3576/bin/python common/run_video_streams_benchmark.py --backend rk3576     --model model/rk3576/yolo26n_rk3576_int8.rknn --instances 1,2,4,8 --frames 200     --json out/rk3576_multi.json
```

The single-stream runners can also be called directly, with the same arguments they document
(`python3 rk3576/run_video_inference.py --help`); the wrappers just fill in the model paths and the
output directory:

```bash
# Hailo-8, one stream (4 inferences in flight)
python3 Hailo/Hailo8/run_video_inference.py --hef model/Hailo/yolo26n_hailo8_official.hef \
    --out-dir out/hailo8 --depth 4

# RK3576 built-in NPU, one stream
python3 rk3576/run_video_inference.py --model model/rk3576/yolo26n_rk3576_int8.rknn \
    --out-dir out/rk3576

# RK182x, one stream
python3 rk182x/rk1820/run_video_inference.py --model model/rk1820/yolo26n_rk1820_int8.rknn \
    --weight model/rk1820/yolo26n_rk1820_int8.weight \
    --out-dir out/rk1820

# Hailo-8, 8 streams on the one device (add --no-decode for the device-only rate)
python3 Hailo/Hailo8/run_streams_aggregate.py --hef model/Hailo/yolo26n_hailo8_official.hef \
    --streams 8 --frames 200 --depth 8 --json out/hailo8_8stream.json

# built-in NPU / RK182x, 1/2/4/8 streams, one process and one NPU core per stream
python3 common/run_video_streams_benchmark.py --backend rk3576 \
    --model model/rk3576/yolo26n_rk3576_int8.rknn \
    --instances 1,2,4,8 --frames 200 --json out/rk3576_multi.json
python3 common/run_video_streams_benchmark.py --backend rk1820 \
    --model model/rk1820/yolo26n_rk1820_int8.rknn \
    --weight model/rk1820/yolo26n_rk1820_int8.weight \
    --instances 1,2,4,8 --frames 300 --json out/rk1820_multi.json

# figures and the screenshot verification (host side, from the JSON records)
python3 common/make_final_figures.py
python3 common/make_osd_figure.py
python3 common/check_osd_figure.py
```

Requirements: [`requirements.txt`](requirements.txt) for the host side, [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)
for the runtime version behind each record. Each runner writes `video_result.json` (or the `--json`
file it is given); the three single-stream runners also write an annotated `annotated.mp4`, while
the two aggregate runners deliberately write no video (drawing and encoding would make the host the
bottleneck). The JSON carries per-frame detections and the timing of every layer, so a result can be
re-analysed without re-running the board.

## What makes the numbers comparable

- **One source model, one graph boundary.** All three graphs stop at the **same six raw detection
  heads** - YOLO26 emits 4-channel direct boxes plus 80-channel score logits, with no DFL and no
  NMS inside the graph. Decode and NMS run in **one shared host implementation**
  (`common/postprocess_common.py` + `common/postprocess_yolo26.py`), and all three runs read the
  same clip through the same letterbox code, so they are handed the same input tensor.
- **Each platform is driven through the API its vendor documents for throughput.** Hailo-8 uses
  `create_infer_model()` -> `configure()` -> `run_async()` with 4-8 inferences in flight
  (`common/hailo_pipeline.py`), which is what `hailortcli run` does internally; its own CLI reports
  52.53 FPS for the same HEF on this board, and this pipeline reaches 51.2. The two Rockchip parts
  use their Python bindings one frame per call, and the multi-stream runs use one process and one
  NPU core per stream.
- **Official artefacts where the vendor ships them.** Hailo-8 runs Hailo's prebuilt HEF; RK182x
  runs Rockchip's official quantization recipe (w8a8 with w16a16 score-branch subgraphs).
- **Layered accounting.** The accelerator call, the host decode/NMS and the drawing/encoding are
  measured and reported separately, so host-side cost is never presented as accelerator speed. The
  middle layer is not spelled out identically by the three runners (the Hailo-8 one includes the
  frame read and letterbox, the Rockchip ones record those in a separate field); `results/README.md`
  gives both, including the like-for-like figures.
- **Every run is traceable.** Each record stores the sha256 of the model files and of the clip it
  used, plus the runtime versions and per-frame detections.

## What is verified in this delivery

- The three model files match the `*_sha256` values stored inside the records, and every runner
  recomputes and prints those hashes at run time (`model/README.md`).
- `video/test.mp4` matches `input.video_sha256` in all three records, i.e. the shipped records
  were produced on the shipped clip (`video/test.mp4`, 640x640, 394 frames, 30 fps).
- `checksums.sha256` covers the four model files, the bundled RKNNLite2 wheel and the sample clip;
  `python3 scripts/verify-checksums.py` checks them (line-ending tolerant, unlike `sha256sum -c`),
  and every install script runs exactly that before touching the environment. Each `scripts/check-*.py` ends with one real inference on the sample clip.
- The copied records were checked field by field against the source project: no timing, per-frame
  detection, frame count or hash differs. The description/naming edits that were made are listed in
  `results/README.md`.
- All three screenshots were verified pixel by pixel against the text they are supposed to carry,
  under the OpenCV build that drew each one, and the expected text had to beat every one-character
  variant of itself (wrong digit, wrong device name). The same record holds the banner rectangle
  measured on screen and the comparison of each still with frame 200 of its annotated clip:
  `results/single_stream/frame200_check.json`.
- Both chart PNGs regenerate from the records with zero differing pixels
  (`python3 common/make_final_figures.py`), and `figures/runtime_osd.png` is exactly the three stills
  side by side (`python3 common/make_osd_figure.py`).
- Every headline figure in these documents was recomputed from the JSON records.

## Known limits

- This board's PCIe link is **Gen2 x1** while the Hailo-8 module supports Gen3 x4, so Hailo's
  published 155 FPS is unreachable here: its own HEF measures 52.5 FPS and this pipeline 51.2.
- There is no hardware decoder in this environment (the 4K source caps out at 16.5 FPS software
  decode), so the comparison uses a derived 640x640 clip. On a real deployment the RK3576 VPU
  should be used, and that decode cost is a host-side limit, not an accelerator one.
- The multi-stream runs deliberately exclude drawing and video encoding: they measure inference
  service capacity. Drawing plus MP4 writing costs about 15 ms per frame on this host (single
  stream: 49.2 -> 28.3 FPS), so eight annotated outputs would become host-bound.
- RK182x needs 4 streams before it beats the built-in NPU (1.09x) and 8 streams to lead by 1.27x;
  at 1-2 streams the PCIe round trip makes it the slower of the two.

## License

No license is attached; please contact the repository owner before reusing this code.