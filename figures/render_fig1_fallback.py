"""Fallback Fig.1 renderer (matplotlib) when mermaid-cli is unavailable."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = Path(__file__).resolve().parent


def box(ax, x, y, w, h, text, color):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color,
        edgecolor="#333",
        linewidth=1.2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    box(ax, 1.5, 12.2, 7, 1.2, "Layer I (Why): When does recursive FT yield\nauditable bounded gains?", "#e1f5fe")
    box(ax, 1.5, 10.7, 7, 1.2, "Layer II (What): Failure-directed data curation\nin few-shot chemistry", "#fff3e0")
    box(ax, 1.5, 9.2, 7, 1.2, "Layer III (How): Fixed split, recipe lock,\npre-registered contrasts", "#e8f5e9")
    for y0, y1 in [(12.2, 11.9), (10.7, 10.4)]:
        ax.annotate("", xy=(5, y1), xytext=(5, y0), arrowprops=dict(arrowstyle="->", color="#555"))

    box(
        ax,
        1.2,
        5.6,
        7.6,
        3.2,
        "Failure-Directed Curation Loop\n"
        "Train (LoRA) → Evaluate → Identify Failures → Curate A_fail\n"
        "→ Augment & Train (loop)\n"
        "Control: A_rand (nominal K-matched)",
        "#f3e5f5",
    )
    ax.annotate("", xy=(5, 8.8), xytext=(5, 9.2), arrowprops=dict(arrowstyle="->", color="#555"))

    box(ax, 0.4, 3.2, 2.9, 1.8, "In-Split (RQ1)\n+66.60pp", "#e8f5e9")
    box(ax, 3.55, 3.2, 2.9, 1.8, "Neg. Ctrl (RQ2)\nAll fail", "#fff3e0")
    box(ax, 6.7, 3.2, 2.9, 1.8, "Hold-Out (RQ3)\n0% / 0%", "#ffebee")
    ax.annotate("", xy=(5, 5.0), xytext=(5, 5.6), arrowprops=dict(arrowstyle="->", color="#555"))

    box(
        ax,
        1.5,
        1.0,
        7,
        1.6,
        "Verdict: Bounded data-centric refinement\neffective in-domain, NOT cross-split",
        "#ffebee",
    )
    ax.annotate("", xy=(5, 2.6), xytext=(5, 3.2), arrowprops=dict(arrowstyle="->", color="#555"))

    ax.set_title("Figure 1. Method pipeline", fontsize=12, pad=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_method_pipeline.pdf", bbox_inches="tight")
    fig.savefig(out / "fig1_method_pipeline.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "fig1_method_pipeline.svg", bbox_inches="tight")
    print("wrote fig1 pdf/png/svg")


if __name__ == "__main__":
    main()
