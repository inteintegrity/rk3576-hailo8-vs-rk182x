# RK3576：片内 NPU vs Hailo-8 vs RK182x，同一份 YOLO 模型的实测对比

在一块 RK3576 板（reComputer RK3576 devkit）上，把**同一个 YOLO26n**（COCO 80 类，640×640，
INT8）分别放到三种加速器上跑，测单路与多路的真实吞吐：

| 加速器 | 接口 | 单路（管道） | 8 路聚合 |
|---|---|---:|---:|
| **Hailo-8**（M.2） | HailoRT 4.23，官方 async 管线 | **49.2 FPS** | 51.6 FPS（平，设备上限） |
| **RK182x**（M.2） | RKNN3 1.0.4 | 17.5 FPS | **88.6 FPS**（随核线性增长） |
| **RK3576 片内 NPU** | rknn-toolkit-lite2 2.3.2 | 23.5 FPS | 49.4 FPS（3 核封顶） |

结论：**单路看 Hailo-8**（设备侧 51.2 FPS，单帧 13.9 ms，且不占 SoC 的 NPU）；
**多相机/多路并发看 RK182x**（8 路 88.6 FPS，是 Hailo-8 的 1.7 倍）；
内置 NPU 免费但两条路都不占优。完整数据、图表与中英文文章见
[`results/final_benchmark/SUMMARY.md`](results/final_benchmark/SUMMARY.md)
（中文文章 [ARTICLE_zh.md](results/final_benchmark/ARTICLE_zh.md)、英文文章
[ARTICLE_en.md](results/final_benchmark/ARTICLE_en.md)）。

## 方法要点（为什么这些数字可信）

- **同一个源模型、同一个图边界**：三端都停在**同样的六个原始检测头**
  （YOLO26 是 4 通道直接框 + 80 通道分数，无 DFL/NMS 在加速器内），
  解码与 NMS 由**同一份主机代码**（`common/postprocess_yolo26.py`）执行，输入逐字节一致。
- **各用官方产物**：Hailo-8 用官方预编译 HEF，RK182x 用官方量化配方（w8a8 + score 分支 w16a16），
  RK3576 用同样图边界自行转换。
- **Hailo-8 走官方 async 管线**：`create_infer_model()` → `configure()` → `run_async()`，
  常驻 4–8 帧在飞（`common/hailo_async.py`）。这是 `hailortcli run` 内部的做法，
  也是 Hailo 文档给出达到峰值吞吐的方式——用老的同步 `InferVStreams.infer()` 只能测到
  33–35 FPS（设备在主机解码期间空转），换异步后同为 52.5 FPS 的官方 CLI 口径。
- **口径分层记录**：仅推理 / +letterbox+解码+NMS / +画框+写视频 三层分别测量，
  不把主机侧开销记到加速器头上。

## 目录结构

```
common/          共享代码：主机解码（postprocess_yolo11/26）、绘图、Hailo 异步管线、
                 RKNN 辅助、图与快照生成、OSD 文字校验、板子 SSH 辅助（remote_ops.py）
Hailo/Hailo8/    Hailo-8 运行脚本：单张图片、视频、实时摄像头、多路聚合、
                 HEF 编译脚本与配置（hailo_config/*.alls）
rk182x/rk1820/   RK182x（RKNN3）运行脚本与转换脚本（含官方 YOLO26 配方示例）
rk3576/          片内 NPU（RKNN2）运行脚本、派生视频、YOLO26 转换与校准集制作
results/
  final_benchmark/  最终结果：SUMMARY.md、中文/英文文章、单路与多路 JSON、图表、运行画面截图
  final_benchmark/_syncmethod/  旧同步口径的数据（用于对照）
```

## 复现步骤

```bash
# 1) 源模型：Ultralytics yolo26n.pt -> 规范 ONNX（六个原始检测头）
python common/export_canonical_onnx.py --weights yolo26n.pt --output model/yolo26n/yolo26n.onnx

# 2) 校准集（同一套 letterbox 640x640 图片喂给两条工具链）
python common/prepare_calibration.py --manifest <图片清单> --output-dir <目录> --run-conversion

# 3) 三个加速器的产物：Hailo(.hef) / RKNN3(.rknn+.weight) / RKNN2(.rknn)
bash Hailo/Hailo8/compile_hef.sh                # 需要 Hailo Dataflow Compiler
python rk182x/rk1820/convert_rknn3.py ...       # 需要 rknn3-toolkit
python rk3576/convert_yolo26.py ...             # 需要 rknn-toolkit2

# 4) 板端测试（示例：Hailo-8 单路，异步深度 4；多路用 run_streams_aggregate.py）
python Hailo/Hailo8/run_video_inference.py --hef <yolo26n.hef> --video <clip.mp4> \
    --out-dir out --model-family yolo26 --async-depth 4
python Hailo/Hailo8/run_streams_aggregate.py --hef <yolo26n.hef> --video <clip.mp4> \
    --streams 8 --frames 200 --async-depth 8

# 5) 图表与像素级 OSD 校验（全部由 JSON 生成，不手改数字）
python common/make_final_figures.py
python common/make_osd_figure.py
python common/check_osd_strip.py --clip <annotated.mp4> --record <result.json> --device "RK3576 + Hailo-8" --device-fps 51.2
```

板端操作通过 SSH 完成（`common/remote_ops.py`，密码只从环境变量 `RK_SSH_PASSWORD` 读取，
不写入文件、不出现在命令行）。

## 已知边界

- 板子 PCIe 链路是 **Gen2 x1**（模块支持 Gen3 x4），Hailo-8 官方标称的 155 FPS 在此不可达；
  实测其官方 HEF 为 52.5 FPS，我们的异步管线 51.2 FPS。
- 本环境无硬件解码器，4K 原片只有 16.5 FPS 上限；对比使用 640×640 派生视频。
- RK 两端的 Python 绑定没有异步接口，因此它们的数字是同步口径，与 Hailo 的异步口径不同源。

## 许可

未附许可协议；如需复用请先联系仓库作者。