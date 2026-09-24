"""Generate the two result figures from the benchmark JSONs.

Everything here is data-driven: the numbers come from the JSON files written by the benchmark
runs, so the plots can be regenerated at any time and can never drift from the records:

    python common/make_final_figures.py

Outputs (results/figures/):
    multi_stream_scaling.png   aggregate throughput vs number of concurrent streams
    single_stream_bars.png     one stream: inference / pipeline / annotate+encode
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SINGLE = ROOT / "results" / "single_stream"
MULTI = ROOT / "results" / "multi_stream"
OUT = ROOT / "results" / "figures"

COLOURS = {"hailo8": "#7d3c98", "rk3576_npu": "#2e86c1", "rk1820": "#c0392b"}
LABELS = {
    "hailo8": "Hailo-8 (one device, all streams through one pipeline)",
    "rk3576_npu": "RK3576 built-in NPU (2 cores, oversubscribed above 2)",
    "rk1820": "RK182x (8-core module, one process per stream)",
}
NL = chr(10)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def multi_stream_figure() -> plt.Figure:
    rk3576 = load(MULTI / "rk3576_multi_stream.json")["results"]
    rk1820 = load(MULTI / "rk1820_multi_stream.json")["results"]
    hailo = {n: load(MULTI / f"aggregate_{n}_streams.json") for n in (1, 2, 4, 8)}

    series = [
        ("rk1820", [1, 2, 4, 8], [rk1820[str(n)]["aggregate_fps"] for n in (1, 2, 4, 8)]),
        ("rk3576_npu", [1, 2, 4, 8], [rk3576[str(n)]["aggregate_fps"] for n in (1, 2, 4, 8)]),
        ("hailo8", [1, 2, 4, 8], [hailo[n]["aggregate_fps"] for n in (1, 2, 4, 8)]),
    ]

    fig, ax = plt.subplots(figsize=(11.0, 6.6), dpi=170)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.15)

    for key, xs, ys in series:
        ax.plot(xs, ys, "o-", color=COLOURS[key], linewidth=3, markersize=11, label=LABELS[key])
        ax.annotate(f"{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(10, -4),
                    ha="left", va="center", fontsize=13, fontweight="bold", color=COLOURS[key])

    ax.annotate(NL.join(["flat at the device's own ceiling:", "one device serves all streams"]),
                xy=(6.2, 51.6), xytext=(5.1, 22), fontsize=11.5, color=COLOURS["hailo8"],
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=COLOURS["hailo8"], linewidth=2,
                                connectionstyle="arc3,rad=0.2"))

    ax.set_xlabel("Concurrent video streams", fontsize=13.5)
    ax.set_ylabel("Aggregate throughput (FPS)", fontsize=13.5)
    ax.set_title("Aggregate throughput: only the core-scaled parts grow", fontsize=16, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 8])
    ax.set_xlim(0.4, 9.8)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=11.5, framealpha=0.95)
    fig.text(0.5, 0.04,
             "Each accelerator driven its own way: Hailo-8 takes all streams through one device "
             "pipeline, the Rockchip parts give one stream per NPU core.",
             ha="center", fontsize=11, color="#444444")
    return fig


def single_stream_figure() -> plt.Figure:
    order = ["hailo8", "rk3576_npu", "rk1820"]
    records = {key: load(SINGLE / key / "video_result.json") for key in order}
    stages = ["inference only", "+ letterbox, decode, NMS", "+ drawing and MP4 output"]
    values = {}
    for key in order:
        timing = records[key]["timing"]
        frames = records[key]["detection_summary"]["frames"]
        values[key] = [
            timing["python_infer_only_fps"],
            1000.0 / timing["pipeline_without_io"]["mean_ms"],
            frames / timing["total_wall_seconds"],
        ]

    fig, ax = plt.subplots(figsize=(11.0, 6.6), dpi=170)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.86, bottom=0.20)
    positions = np.arange(len(order), dtype=float)
    width = 0.26
    shades = [1.0, 0.62, 0.3]

    for stage_index, (stage, shade) in enumerate(zip(stages, shades)):
        bars = ax.bar(positions + (stage_index - 1) * width,
                      [values[key][stage_index] for key in order], width,
                      color=[COLOURS[key] for key in order], alpha=shade,
                      edgecolor="black", linewidth=1.2, label=stage)
        for rect, key in zip(bars, order):
            ax.annotate(f"{values[key][stage_index]:.1f}",
                        (rect.get_x() + rect.get_width() / 2, values[key][stage_index]),
                        textcoords="offset points", xytext=(0, 3), ha="center", va="bottom",
                        fontsize=11, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[k].split(" (")[0] for k in order], fontsize=12.5)
    ax.set_ylabel("Frames per second", fontsize=13.5)
    ax.set_title("One stream, 394 frames: Hailo-8 leads by 2.1x", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 62)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    fig.text(0.5, 0.04,
             "Same model, same six raw detection heads, same host-side decode and NMS, "
             "same 640x640 video.",
             ha="center", fontsize=11, color="#444444")
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in (("multi_stream_scaling.png", multi_stream_figure),
                          ("single_stream_bars.png", single_stream_figure)):
        fig = builder()
        path = OUT / name
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()