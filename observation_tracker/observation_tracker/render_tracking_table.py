"""Render publication-ready tables from the final tracking evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METHOD_NAMES = {
    "observation_tracker": "Observation Tracker",
    "ab3dmot": "AB3DMOT",
    "eagermot": "EagerMOT",
    "fusiontrack": "FusionTrack",
}
METHOD_ORDER = list(METHOD_NAMES)
HEADER_COLOR = "#18324A"
BEST_COLOR = "#DDF2E1"
ROW_COLORS = ("#F7F9FB", "#FFFFFF")


def read_rows(path: Path, sequence: str):
    with path.open(newline="") as stream:
        rows = {row["method"]: row for row in csv.DictReader(stream) if row["sequence"] == sequence}
    return [rows[method] for method in METHOD_ORDER]


def add_table(ax, title, rows, columns):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=9)
    values = [[METHOD_NAMES[row["method"]]] + [formatter(float(row[key])) for key, _, formatter, _ in columns]
              for row in rows]
    labels = ["Algorithm"] + [label for _, label, _, _ in columns]
    table = ax.table(cellText=values, colLabels=labels, cellLoc="center", loc="center",
                     bbox=[0, 0, 1, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for column in range(len(labels)):
        cell = table[0, column]
        cell.set_facecolor(HEADER_COLOR); cell.set_text_props(color="white", fontweight="bold")
    for row_index in range(len(rows)):
        for column in range(len(labels)):
            table[row_index + 1, column].set_facecolor(ROW_COLORS[row_index % 2])
        table[row_index + 1, 0].set_text_props(fontweight="bold")
    for column_index, (key, _, _, direction) in enumerate(columns, start=1):
        if direction is None:
            continue
        raw = [float(row[key]) for row in rows]
        best = (max if direction == "max" else min)(raw)
        for row_index, value in enumerate(raw, start=1):
            if abs(value - best) < 1e-12:
                table[row_index, column_index].set_facecolor(BEST_COLOR)
                table[row_index, column_index].set_text_props(fontweight="bold")
    for cell in table.get_celld().values():
        cell.set_edgecolor("#C7D0D9"); cell.set_linewidth(0.65)


def render_overall(metrics_path: Path, runtime_path: Path, output_dir: Path):
    rows = read_rows(metrics_path, "ALL")
    runtime = read_rows(runtime_path, "ALL")
    f3 = lambda value: f"{value:.3f}"
    f1 = lambda value: f"{value:.1f}"
    integer = lambda value: f"{int(value):,}"
    percent = lambda value: f"{value * 100:.1f}%"
    seconds = lambda value: f"{value:.2f}s"
    fig, axes = plt.subplots(4, 1, figsize=(19, 14.5), gridspec_kw={"height_ratios": [1, 1, 1.05, 1]})
    fig.suptitle("Overall 3D Multi-Object Tracking Performance", fontsize=22, fontweight="bold", y=0.99)
    add_table(axes[0], "A. Overall tracking and association", rows, [
        ("hota_3d", "HOTA ↑", f3, "max"), ("assa_3d", "AssA ↑", f3, "max"),
        ("deta_3d", "DetA ↑", f3, "max"), ("idf1_3d", "IDF1 ↑", f3, "max"),
        ("idsw_3d", "IDSW ↓", integer, "min"),
        ("fragmentation_3d", "Frag. ↓", integer, "min"), ("fn_3d", "FN ↓", integer, "min"),
    ])
    add_table(axes[1], "B. Miss recovery and ID persistence", rows, [
        ("track_recovery_rate", "Recovery ↑", percent, "max"),
        ("id_preservation_rate", "ID preservation ↑", percent, "max"),
        ("miss_episodes", "Miss episodes", integer, None),
        ("recovered_episodes", "Recovered ↑", integer, "max"),
        ("track_duration_mean_s", "Track duration mean ↑", seconds, "max"),
        ("track_duration_max_s", "Track duration max ↑", seconds, "max"),
    ])
    add_table(axes[2], "C. False tracks, ghosts, and lifecycle", rows, [
        ("false_track_count", "False tracks ↓", integer, "min"),
        ("tracker_fp", "Tracker FP ↓", integer, "min"),
        ("fp_propagation_rate", "FP propagation ↓", percent, "min"),
        ("ghost_lifetime_mean_s", "Ghost mean ↓", seconds, "min"),
        ("ghost_lifetime_max_s", "Ghost max ↓", seconds, "min"),
        ("termination_delay_mean_s", "Termination delay ↓", seconds, "min"),
        ("fp_track_frames", "FP track frames ↓", integer, "min"),
    ])
    add_table(axes[3], "D. Runtime performance", runtime, [
        ("core_latency_mean_ms", "Core latency mean ↓", lambda v: f"{v:.2f}ms", "min"),
        ("core_latency_p95_ms", "Core latency P95 ↓", lambda v: f"{v:.2f}ms", "min"),
        ("core_fps", "Core FPS ↑", f1, "max"),
    ])
    fig.text(0.01, 0.006,
             "HOTA/AssA: oriented 3D IoU, α=0.05–0.95. ID metrics: 3D IoU ≥ 0.5. "
             "Recovery/ghost: class-aware center matching. Termination: two GT exit events. Green: best value.",
             fontsize=9, color="#465563")
    fig.tight_layout(rect=[0, 0.025, 1, 0.975], h_pad=1.5)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"full_tracking_metrics_table.{extension}", dpi=350, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.evaluation_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    render_overall(args.evaluation_dir / "extended_metrics.csv",
                   args.evaluation_dir / "runtime_metrics.csv", output_dir)


if __name__ == "__main__":
    main()
