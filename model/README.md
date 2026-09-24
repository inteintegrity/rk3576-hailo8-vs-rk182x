# Model artefacts

Three files, one per accelerator. All of them stop at the **same graph boundary**: the six
raw YOLO26 detection heads (three strides x {box, score}), with no DFL, no in-graph NMS and no
post-processing baked into the graph. That is what makes the three runs comparable: decode and
NMS are the host's job in all three cases, and the host code is shared
(`common/postprocess_common.py` + `common/postprocess_yolo26.py`).

| File | Accelerator | Toolchain / origin | Size | sha256 |
|---|---|---|---:|---|
| `Hailo/yolo26n_hailo8_official.hef` | Hailo-8 | Hailo's prebuilt HEF for yolo26n (Hailo Model Zoo), INT8 | 8,755,512 B | `743de04972fbdba05140d18b9176e38620011134e9f79c7a7dd63b1d13eb90ae` |
| `rk3576/yolo26n_rk3576_int8.rknn` | RK3576 NPU | converted by this project with rknn-toolkit2 2.3.2, w8a8 | 7,692,400 B | `8750d6e73d2e98fa34b8f4d310180d2975f36110be2529225b960e26e7287c9e` |
| `rk1820/yolo26n_rk1820_int8.rknn` | RK182x | Rockchip's official YOLO26 quantization recipe (w8a8, score branch w16a16), compiled for **one** NPU core | 221,240 B | `77eedf56ad8647fb9cd033b665f9efc9e34c0a5dfe390bc9c526df1ee8cff301` |
| `rk1820/yolo26n_rk1820_int8.weight` | RK182x | companion weight file for the RKNN3 model | 3,536,896 B | `4bb04ca1800a823f70053b9991658ab47b2f9ad8b97b432d3b792e0a9604789f` |

The four hashes are the same ones stored inside the full per-device records in the `v1.0.0`
benchmark-artifacts Release asset (`model.*_sha256`). Each runner recomputes the hash at run time
and prints it.

Notes:

- The source graph for both converted models is a canonical ONNX exported from Ultralytics
  `yolo26n.pt` and cut at the six `one2one` head convolutions
  (`/model.23/one2one_cv2.{0,1,2}/.../Conv_output_0`, `/model.23/one2one_cv3.{0,1,2}/...`),
  exported from a 640x640 input with COCO's 80 classes. The ONNX itself, the calibration set
  (128 letterboxed COCO128 images, the same set for both Rockchip toolchains) and the conversion
  scripts are conversion intermediates and are not part of this delivery.
- The RK182x model was compiled for a single NPU core. RKNN3 requires `core_mask` to equal the
  compile-time `core_num`, so a wider mask is refused by the runtime and the runner falls back to
  the model's own core count. Splitting this one graph across 8 cores instead measured *slower*
  per inference, which is why the multi-stream runs give each stream its own core instead.
- The RK3576 NPU on this SoC has **two** cores. The single-stream record's `core_mask` field reads
  `0x7`, which is the value the runner was given and the runtime accepted at measurement time (the
  platform reports "NPU_CORE_0_1_2 is not supported and will be automatically replaced with
  NPU_CORE_0_1"). The multi-stream record was produced by a script whose core count was 3, so its
  streams requested masks `0x1` / `0x2` / `0x4`; the platform mapped the third onto a physical core
  as it does for the single-stream mask. The runner in this folder pins `0x1` / `0x2` for the two
  cores and records the mask that was actually accepted for every stream
  (`per_stream_core_mask_active`).
