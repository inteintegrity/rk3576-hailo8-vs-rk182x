# RK3576 accelerators head to head: built-in NPU vs Hailo-8 vs RK182x

The same **YOLO26n** (COCO 80 classes, 640x640, INT8) driven on all three accelerators of one
RK3576 board (reComputer RK3576 devkit), measured single-stream and with many streams at once:

| Accelerator | Runtime | Single stream (pipeline) | 8-stream aggregate |
|---|---|---:|---:|
| **Hailo-8** (M.2) | HailoRT 4.23, official async pipeline | **49.2 FPS** | 51.6 FPS (flat: the device's own ceiling) |
| **RK182x** (M.2) | RKNN3 1.0.4 | 17.5 FPS | **88.6 FPS** (scales with cores) |
| **RK3576 built-in NPU** | rknn-toolkit-lite2 2.3.2 | 23.5 FPS | 49.4 FPS (capped at its 3 cores) |

**Short version:** for a single stream Hailo-8 wins (device service 51.2 FPS, 13.9 ms per
inference, and it keeps the work off the SoC's NPU); for multi-camera / 4+ concurrent streams
RK182x wins (88.6 FPS at 8 streams, 1.7x Hailo-8); the built-in NPU is free but loses on both.
Full data, figures and write-ups (Chinese and English) are under
[`results/final_benchmark/`](results/final_benchmark/SUMMARY.md) —
[SUMMARY.md](results/final_benchmark/SUMMARY.md),
[ARTICLE_en.md](results/final_benchmark/ARTICLE_en.md),
[ARTICLE_zh.md](results/final_benchmark/ARTICLE_zh.md).

## Why these numbers are comparable

- **One source model, one graph boundary.** All three graphs stop at the **same six raw
  detection heads** (YOLO26 emits 4-channel direct boxes plus 80-channel score logits, with no
  DFL and no NMS inside the accelerator). Decode and NMS run in **one shared host
  implementation** (`common/postprocess_yolo26.py`) and the letterboxed input is byte-identical.
- **Official artefacts per platform.** Hailo-8 runs Hailo's prebuilt HEF, RK182x uses Rockchip's
  official quantization recipe (w8a8 with w16a16 score-branch subgraphs), and the RK3576 model was
  converted by this project with the same graph boundary.
- **Hailo-8 goes through Hailo's recommended asynchronous pipeline**
  (`create_infer_model()` -> `configure()` -> `run_async()` with 4-8 inferences in flight, see
  `common/hailo_async.py`). That is what `hailortcli run` does internally and what Hailo's docs
  prescribe for peak throughput. With the older synchronous `InferVStreams.infer()` the same board
  and HEF measured only 33-35 FPS because the device idled while the host decoded; with the async
  pipeline it reaches 51.2 FPS against the 52.5 FPS Hailo's own CLI reports.
- **Layered accounting.** Inference only / + letterbox, decode, NMS / + drawing and MP4 writing
  are measured and reported separately, so host-side cost is never presented as accelerator speed.

## Repository layout

```
common/          Shared code: host-side decode (postprocess_yolo11/26), drawing, the HailoRT
                 async pipeline, RKNN helpers, figure and snapshot generators, pixel-level OSD
                 verification, and a board SSH helper (remote_ops.py)
Hailo/Hailo8/    Hailo-8 runners: single image, video, live camera, N-stream aggregate,
                 plus the async-pipeline validation and live-loop breakdown tools
rk182x/rk1820/   RK182x (RKNN3) runners and the official YOLO26 conversion recipe
rk3576/          Built-in NPU (RKNN2) runners, YOLO26 conversion, calibration set, stream derivation
results/
  final_benchmark/   Deliverables: SUMMARY.md, both articles, single-stream and aggregate JSON,
                     figures, verified runtime screenshots
  final_benchmark/_syncmethod/   The older synchronous-API numbers, kept for comparison
```

## Reproducing it

```bash
# 1) Source model: Ultralytics yolo26n.pt -> canonical ONNX (six raw detection heads)
python common/export_canonical_onnx.py --weights yolo26n.pt --output model/yolo26n/yolo26n.onnx

# 2) Calibration set: one shared set of letterboxed 640x640 images for both toolchains
python common/prepare_calibration.py --manifest <image-list> --output-dir <dir> --run-conversion

# 3) Toolchain artefacts: Hailo (.hef), RKNN3 (.rknn + .weight), RKNN2 (.rknn)
#    (Hailo Dataflow Compiler / rknn3-toolkit / rknn-toolkit2 required)

# 4) Board-side runs, e.g. Hailo-8 single stream (async depth 4) and 8-stream aggregate
python Hailo/Hailo8/run_video_inference.py --hef <yolo26n.hef> --video <clip.mp4> \
    --out-dir out --model-family yolo26 --async-depth 4
python Hailo/Hailo8/run_streams_aggregate.py --hef <yolo26n.hef> --video <clip.mp4> \
    --streams 8 --frames 200 --async-depth 8

# 5) Figures and the pixel-level OSD check (everything is generated from the JSON, never hand-edited)
python common/make_final_figures.py
python common/make_osd_figure.py
python common/check_osd_strip.py --clip <annotated.mp4> --record <result.json> \
    --device "RK3576 + Hailo-8" --device-fps 51.2
```

Board commands were run over SSH through `common/remote_ops.py`; the password is read from the
`RK_SSH_PASSWORD` environment variable only (never written to a file, never passed on a command
line), and the host from `RK_SSH_HOST`.

## Known limits

- This board's PCIe link is **Gen2 x1** while the Hailo-8 module supports Gen3 x4, so Hailo's
  published 155 FPS is unreachable here: its own HEF measures 52.5 FPS and this project's async
  pipeline 51.2 FPS.
- There is no hardware decoder in this environment (16.5 FPS ceiling on the 4K source), so the
  comparison uses a derived 640x640 clip.
- The two Rockchip parts currently only expose synchronous Python bindings
  (`rknn3_run_async` is not available in RKNN3 1.0.4 and rknnlite does not surface the
  rknn_run/rknn_wait pair), so their figures are synchronous-API numbers and not strictly
  like-for-like with Hailo-8's async pipeline.

## License

No license is attached; please contact the repository owner before reusing this code.