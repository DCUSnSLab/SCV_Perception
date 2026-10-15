"""Render a compact booktabs-style table and matching LaTeX source."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = [
    ("observation_tracker", "Observation Tracker"),
    ("ab3dmot", "AB3DMOT"),
    ("eagermot", "EagerMOT"),
    ("fusiontrack", "FusionTrack"),
]
METRICS = [
    ("hota_3d", "HOTA$\\uparrow$", ".3f", "max", "Tracking"),
    ("assa_3d", "AssA$\\uparrow$", ".3f", "max", "Tracking"),
    ("idf1_3d", "IDF1$\\uparrow$", ".3f", "max", "Tracking"),
    ("idsw_3d", "IDSW$\\downarrow$", ".0f", "min", "Association / Recovery"),
    ("fragmentation_3d", "Frag.$\\downarrow$", ".0f", "min", "Association / Recovery"),
    ("track_recovery_rate", "Rec.$\\uparrow$", ".1%", "max", "Association / Recovery"),
    ("id_preservation_rate", "IDP$\\uparrow$", ".1%", "max", "Association / Recovery"),
    ("fn_3d", "FN$\\downarrow$", ".0f", "min", "Error Suppression"),
    ("false_track_count", "FTrk$\\downarrow$", ".0f", "min", "Error Suppression"),
    ("tracker_fp", "FP$\\downarrow$", ".0f", "min", "Error Suppression"),
    ("fp_propagation_rate", "FP-Prop.$\\downarrow$", ".1%", "min", "Error Suppression"),
    ("core_latency_mean_ms", "Latency (ms)$\\downarrow$", ".2f", "min", "Efficiency"),
    ("core_fps", "FPS$\\uparrow$", ".1f", "max", "Efficiency"),
]


def load_rows(evaluation_dir: Path):
    with (evaluation_dir / "extended_metrics.csv").open(newline="") as stream:
        metrics = {row["method"]: row for row in csv.DictReader(stream) if row["sequence"] == "ALL"}
    with (evaluation_dir / "runtime_metrics.csv").open(newline="") as stream:
        runtime = {row["method"]: row for row in csv.DictReader(stream) if row["sequence"] == "ALL"}
    return [{**metrics[key], **runtime[key], "display_name": name} for key, name in METHODS]


def ranks(rows, key, direction):
    values = [float(row[key]) for row in rows]
    unique = sorted(set(values), reverse=direction == "max")
    return unique[0], unique[1] if len(unique) > 1 else unique[0]


def formatted(value, spec):
    return format(value, spec)


def render(rows, output_dir: Path):
    widths = [2.25] + [1.0] * len(METRICS)
    edges = [0]
    for width in widths:
        edges.append(edges[-1] + width)
    total_width = edges[-1]
    fig, ax = plt.subplots(figsize=(19.5, 5.1))
    ax.set_xlim(0, total_width); ax.set_ylim(-0.95, 6.35); ax.axis("off")
    centers = [(edges[index] + edges[index + 1]) / 2 for index in range(len(widths))]

    ax.hlines(6.05, 0, total_width, color="black", linewidth=1.8)
    groups = []
    start = 1
    for index, metric in enumerate(METRICS, start=1):
        if not groups or groups[-1][0] != metric[4]:
            groups.append([metric[4], index, index])
        else:
            groups[-1][2] = index
    for name, first, last in groups:
        ax.text((edges[first] + edges[last + 1]) / 2, 5.68, name, ha="center", va="center",
                fontsize=11.5, fontweight="bold")
        ax.hlines(5.43, edges[first] + 0.05, edges[last + 1] - 0.05, color="black", linewidth=0.7)
    ax.text(centers[0], 5.15, "Method", ha="center", va="center", fontsize=11.5, fontweight="bold")
    for column, (_, label, _, _, _) in enumerate(METRICS, start=1):
        ax.text(centers[column], 5.15, label.replace("$\\uparrow$", " ↑").replace("$\\downarrow$", " ↓"),
                ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.hlines(4.78, 0, total_width, color="black", linewidth=1.2)

    for row_index, row in enumerate(rows):
        y = 4.22 - row_index * 0.92
        ax.text(edges[0] + 0.08, y, row["display_name"], ha="left", va="center", fontsize=11)
        for column, (key, _, spec, direction, _) in enumerate(METRICS, start=1):
            value = float(row[key]); best, second = ranks(rows, key, direction)
            is_best, is_second = value == best, value == second and value != best
            ax.text(centers[column], y, formatted(value, spec), ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if is_best else "normal")
            if is_second:
                ax.hlines(y - 0.17, centers[column] - 0.22, centers[column] + 0.22,
                          color="black", linewidth=0.8)
    ax.hlines(0.98, 0, total_width, color="black", linewidth=1.8)
    ax.text(0, 0.48, "Best values are bold; second-best values are underlined.", fontsize=9.5, ha="left")
    ax.text(0, 0.05,
            "HOTA/AssA: oriented 3D IoU averaged over α=0.05–0.95; ID metrics: 3D IoU ≥ 0.5. "
            "Rec.: recovery rate; IDP: ID preservation; FTrk: false-track count.",
            fontsize=9.1, ha="left")
    ax.text(0, -0.38,
            "Recovery and false-track metrics use class-aware center matching. Latency excludes detector inference.",
            fontsize=9.1, ha="left")
    fig.tight_layout(pad=0.4)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"paper_tracking_performance_table.{extension}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def latex_value(value, spec, best, second):
    text = formatted(value, spec)
    if value == best:
        return f"\\textbf{{{text}}}"
    if value == second:
        return f"\\underline{{{text}}}"
    return text


def write_latex(rows, output_dir: Path):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Overall performance comparison of the four 3D multi-object tracking algorithms.}",
        r"\label{tab:overall_tracking_performance}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccccccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Tracking} & \multicolumn{4}{c}{Association / Recovery} & \multicolumn{4}{c}{Error Suppression} & \multicolumn{2}{c}{Efficiency} \\" ,
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}\cmidrule(lr){13-14}",
        r"Method & HOTA$\uparrow$ & AssA$\uparrow$ & IDF1$\uparrow$ & IDSW$\downarrow$ & Frag.$\downarrow$ & Rec.$\uparrow$ & IDP$\uparrow$ & FN$\downarrow$ & FTrk$\downarrow$ & FP$\downarrow$ & FP-Prop.$\downarrow$ & Latency (ms)$\downarrow$ & FPS$\uparrow$ \\ ",
        r"\midrule",
    ]
    for row in rows:
        values = []
        for key, _, spec, direction, _ in METRICS:
            best, second = ranks(rows, key, direction)
            values.append(latex_value(float(row[key]), spec, best, second))
        lines.append(row["display_name"] + " & " + " & ".join(values) + " \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{1mm}",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize Best values are bold and second-best values are underlined. HOTA and AssA use oriented 3D IoU averaged over $\alpha=0.05{:}0.05{:}0.95$; ID metrics use 3D IoU $\geq0.5$. Rec.: track recovery rate; IDP: ID preservation rate; FTrk: false-track count. Latency excludes detector inference.",
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ])
    (output_dir / "paper_tracking_performance_table.tex").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.evaluation_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.evaluation_dir)
    render(rows, output_dir)
    write_latex(rows, output_dir)


if __name__ == "__main__":
    main()
