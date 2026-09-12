"""Redraw the published team results in arXiv:2601.21600v2, Tables 1–2.

Run from any directory: python assets/plot_results.py
Requires matplotlib. Source numbers live in paper-results.json; no benchmark
is executed by this script. The PNG is a convenient raster preview.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main() -> None:
    root = Path(__file__).resolve().parent
    data = json.loads((root / "paper-results.json").read_text())
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "svg.hashsalt": "core-paper-results-v2",
        "axes.edgecolor": "#cbd5e1",
        "text.color": "#172b40",
        "axes.labelcolor": "#475569",
        "xtick.color": "#64748b",
        "ytick.color": "#172b40",
    })
    colors = {"base": "#9baac0", "sd_e2": "#e1a34a", "core": "#07857d"}
    labels = {"base": "Base + oracle", "sd_e2": "SD-E² + oracle", "core": "CoRe + oracle"}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
    fig.patch.set_facecolor("#ffffff")
    fig.subplots_adjust(left=0.075, right=0.97, top=0.70, bottom=0.17, wspace=0.25)
    fig.text(0.04, 0.93, "Learning together, reasoning independently", fontsize=21, weight="bold")
    fig.text(0.04, 0.875, "Published oracle Team Pass@2 (%)  ·  2 models × 2 samples = 4 candidates per problem", fontsize=11, color="#475569")
    legend = [Patch(facecolor=colors[key], label=labels[key]) for key in colors]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.033, 0.84), ncol=3, frameon=False, fontsize=11)

    for ax, pair in zip(axes, data["pairs"]):
        ax.set_facecolor("#ffffff")
        y_positions = list(range(len(data["datasets"])))
        for offset, key in zip((-0.23, 0, 0.23), colors):
            ys = [y + offset for y in y_positions]
            bars = ax.barh(ys, pair[key], height=0.19, color=colors[key], zorder=3)
            for bar, value in zip(bars, pair[key]):
                ax.text(value + 1.1, bar.get_y() + bar.get_height() / 2,
                        f"{value:.2f}", va="center", fontsize=9,
                        color="#075b56" if key == "core" else "#475569",
                        weight="bold" if key == "core" else "normal")
        ax.set_yticks(y_positions, data["datasets"], fontsize=11, weight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 110)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", color="#e8edf2", linewidth=0.8, zorder=0)
        ax.tick_params(axis="both", length=0)
        ax.tick_params(axis="y", pad=8)
        ax.set_xlabel("Oracle Team Pass@2 (%)", labelpad=9, fontsize=10)
        ax.set_title(pair["label"], loc="left", fontsize=14, weight="bold", pad=16)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.text(0.04, 0.065, "Source: arXiv:2601.21600v2, Tables 1–2. Redrawn from published values; no new benchmark run.", fontsize=9, color="#64748b")
    fig.text(0.04, 0.03, "Qwen: Qwen2.5-3B-Instruct + Qwen3-4B-Instruct. Reasoning: Phi-4-mini-reasoning + Ministral-3-3B-Reasoning.", fontsize=9, color="#64748b")
    fig.savefig(root / "paper-results.svg", facecolor="white", metadata={"Date": None, "Description": data["metric"] + "; " + data["source"]})
    fig.savefig(root / "paper-results.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
