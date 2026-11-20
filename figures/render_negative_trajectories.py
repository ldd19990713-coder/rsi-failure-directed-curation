"""
Render negative-control trajectory illustrations (Appendix E1, E2) as
publication-ready PDF / SVG.

Data provenance (IMPORTANT)
---------------------------
The per-round accuracy values plotted here are SCHEMATIC illustrations of the
qualitative patterns reported in §5.1 (M3) and §5.2 (P2a); they are NOT
extracted from any individual seed's training log. Per-seed peak / final /
drawdown aggregates are reported in Tables 2, 3, C1–C6.

    * P2a schematic: [15, 22, 18, 14, 10, 7, 4] — captures the early-peak-
      at-R2 followed by monotonic decline described in §5.2.
    * M3  schematic: [12, 18, 18, 18, 18, 18, 18] — captures the hard-lock
      at the R2 peak checkpoint described in §5.1.

We label both panels as "Schematic trajectory" so readers do not mistake the
illustration for a reported seed.

Usage
-----
    python figures/render_negative_trajectories.py

Outputs
-------
    figures/figE1_p2a_scaling.{pdf,svg,png}
    figures/figE2_m3_rollback.{pdf,svg,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE

# Journal-friendly styling: serif font, grayscale-safe palette.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ROUNDS = np.arange(1, 8)
ROUND_LABELS = [f"R{i}" for i in ROUNDS]


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}")
    plt.close(fig)
    for ext in ("pdf", "svg", "png"):
        p = OUT_DIR / f"{stem}.{ext}"
        print(f"{ext.upper():>3}  {p}  ({p.stat().st_size:,} bytes)")


def fig_e1_p2a() -> None:
    # Schematic P2a trajectory — early peak at R2, then monotonic collapse.
    p2a_schematic = np.array([15.0, 22.0, 18.0, 14.0, 10.0, 7.0, 4.0])

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(ROUNDS, p2a_schematic, "-o", color="#2c2c2c",
            label="P2a: naive scaling (schematic)")
    ax.axvline(x=2, color="#b00020", linestyle=":", alpha=0.7,
               label="Early peak at R2")
    ax.set_xticks(ROUNDS)
    ax.set_xticklabels(ROUND_LABELS)
    ax.set_xlabel("Training round")
    ax.set_ylabel("In-split accuracy (%)")
    ax.set_ylim(0, 28)
    ax.set_title("P2a naive scaling — catastrophic forgetting")

    ax.annotate(
        "monotonic decline",
        xy=(5, 8.5), xytext=(3.0, 24),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
    )
    # Top-right schematic watermark.
    ax.text(0.98, 0.98, "schematic\ntrajectory",
            transform=ax.transAxes, fontsize=7, color="#888",
            ha="right", va="top", style="italic")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    _save(fig, "figE1_p2a_scaling")


def fig_e2_m3() -> None:
    # Schematic M3 trajectory — hard lock at the R2 peak checkpoint.
    m3_schematic = np.array([12.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0])

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(ROUNDS, m3_schematic, "-s", color="#2c2c2c",
            label="M3: hard rollback (schematic)")
    ax.axhline(y=18.0, color="#b00020", linestyle=":", alpha=0.7,
               label="Locked at R2 peak (18.0%)")
    ax.axvline(x=2, color="#b00020", linestyle=":", alpha=0.4)
    ax.set_xticks(ROUNDS)
    ax.set_xticklabels(ROUND_LABELS)
    ax.set_xlabel("Training round")
    ax.set_ylabel("In-split accuracy (%)")
    ax.set_ylim(0, 25)
    ax.set_title("M3 hard rollback — premature lock")

    ax.annotate(
        "flatline after rollback",
        xy=(5, 18.0), xytext=(5.0, 6.0),
        fontsize=8, ha="center",
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
    )
    ax.text(0.98, 0.98, "schematic\ntrajectory",
            transform=ax.transAxes, fontsize=7, color="#888",
            ha="right", va="top", style="italic")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    _save(fig, "figE2_m3_rollback")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_e1_p2a()
    fig_e2_m3()
    print("\nAll outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
