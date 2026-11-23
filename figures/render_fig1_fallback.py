"""Fig.1 renderer aligned to RSI_paper_v9 wording (JCIM framing).

Text sources of truth:
  Layer I–III  ← §1.2
  Loop title   ← §1.1 / §3.3 “failure-directed reaction-data curation”
  Gates        ← RQ1–RQ3 + §4–§6 headline results
  Verdict      ← Figure 1 caption / §8
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent

X0, W = 0.55, 8.9
XC = X0 + W / 2


def rounded(ax, x, y, w, h, facecolor, edge="#2f2f2f", lw=1.15, radius=0.12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edge,
        linewidth=lw,
        mutation_aspect=0.35,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=9.2, weight="regular", color="#1a1a1a"):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=1.28,
        family="DejaVu Sans",
    )


def chip(ax, x, y, w, h, text):
    rounded(ax, x, y, w, h, facecolor="#ffffff", edge="#666666", lw=0.7, radius=0.08)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color="#333333",
        family="DejaVu Sans",
    )


def v_arrow(ax, y_from, y_to, color="#5a5a5a"):
    ax.annotate(
        "",
        xy=(XC, y_to),
        xytext=(XC, y_from),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=12),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 9.0), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Band A: three-layer stack (§1.2) ---
    layer_h = 0.78
    layer_gap = 0.16
    y_l1 = 10.85
    y_l2 = y_l1 - layer_h - layer_gap
    y_l3 = y_l2 - layer_h - layer_gap

    # Wording matched to §1.2 Layer I / II / III headers + one-line gist
    layers = [
        (
            y_l1,
            "#D9ECF8",
            "Layer I · Why",
            "Bounded iterative fine-tuning:\nwhen are gains auditable for a chemical predictor?",
        ),
        (
            y_l2,
            "#F7E6D0",
            "Layer II · What",
            "Failure-directed reaction-data curation\n(few-shot exact-SMILES product prediction)",
        ),
        (
            y_l3,
            "#DCEFDD",
            "Layer III · How",
            "Chemical LLM evaluation protocol:\nleakage-controlled split · recipe lock · pre-registered contrasts",
        ),
    ]
    for y, color, tag, body in layers:
        rounded(ax, X0, y, W, layer_h, facecolor=color, radius=0.10)
        chip(ax, X0 + 0.12, y + layer_h - 0.42, 1.78, 0.30, tag)
        label(ax, XC + 0.40, y + layer_h / 2 - 0.04, body, size=8.2)

    v_arrow(ax, y_l1, y_l2 + layer_h)
    v_arrow(ax, y_l2, y_l3 + layer_h)

    # --- Band B: curation loop (§3.3 / method) ---
    loop_top = y_l3 - 0.26
    v_arrow(ax, y_l3, loop_top)

    loop_h = 3.40
    loop_y = loop_top - loop_h
    rounded(ax, X0, loop_y, W, loop_h, facecolor="#F4ECF8", edge="#9A7BB0", lw=1.25, radius=0.14)
    label(
        ax,
        XC,
        loop_y + loop_h - 0.28,
        "Failure-Directed Reaction-Data Curation Loop",
        size=9.4,
        weight="bold",
        color="#4A2F5C",
    )
    # Match §4 / Fig.1 caption: A_rand = same pool, nominal K-matched
    label(
        ax,
        XC,
        loop_y + loop_h - 0.56,
        "Control: A_rand  (same pool · nominal K-matched)",
        size=7.3,
        color="#555555",
    )

    steps = [
        "Train\n(LoRA)",
        "Evaluate\n(in-split)",
        "Identify\nfailures",
        "Curate\nA_fail",
        "Augment\n& retrain",
    ]
    n = len(steps)
    node_w, node_h = 1.42, 0.95
    span = W - 0.55
    start_x = X0 + 0.28
    step_y = loop_y + 1.15
    xs = [start_x + i * (span - node_w) / (n - 1) for i in range(n)]
    colors_steps = ["#E8DAEF", "#D6EAF8", "#FCF3CF", "#D5F5E3", "#FADBD8"]

    for i, (x, text, c) in enumerate(zip(xs, steps, colors_steps)):
        rounded(ax, x, step_y, node_w, node_h, facecolor=c, edge="#555555", lw=1.0, radius=0.10)
        label(ax, x + node_w / 2, step_y + node_h / 2, text, size=7.5)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.02, step_y + node_h / 2),
                xytext=(x + node_w + 0.02, step_y + node_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#5a5a5a", lw=1.0, mutation_scale=9),
            )

    ax.annotate(
        "",
        xy=(xs[0] + node_w / 2, step_y - 0.02),
        xytext=(xs[-1] + node_w / 2, step_y - 0.02),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#7D3C98",
            lw=1.15,
            mutation_scale=10,
            connectionstyle="arc3,rad=0.42",
        ),
    )
    # §3.2.1 recipe locking
    label(ax, XC, loop_y + 0.38, "repeat under locked recipe (7 rounds)", size=7.2, color="#7D3C98")

    # --- Band C: evaluation-protocol gates (RQ1–RQ3) ---
    gate_h = 1.58
    gate_gap = 0.55  # vertical space for fan-out arrows only (no extra center arrow)
    gate_y = loop_y - gate_gap - gate_h
    gap = 0.18
    gate_w = (W - 2 * gap) / 3
    # Results match Tables / §4–§6: +66.60pp; M3/P2a/P1 fail; hold-out 0%/0%
    gates = [
        ("In-Split · RQ1", "A_fail >> A_rand\n+66.60 pp (N=5)", "#D5F5E3", "#1E8449"),
        ("Neg. Ctrl · RQ2", "M3 rollback · P2a scaling\n· P1 LR — all fail", "#FCF3CF", "#9A7D0A"),
        ("Hold-Out · RQ3", "key-disjoint n=31\nbase & LoRA: 0% / 0%", "#FADBD8", "#922B21"),
    ]
    for i, (title, body, fill, accent) in enumerate(gates):
        gx = X0 + i * (gate_w + gap)
        rounded(ax, gx, gate_y, gate_w, gate_h, facecolor=fill, edge=accent, lw=1.2, radius=0.10)
        label(ax, gx + gate_w / 2, gate_y + gate_h - 0.30, title, size=7.4, weight="bold", color=accent)
        label(ax, gx + gate_w / 2, gate_y + 0.58, body, size=7.5, color="#222222")

    for i in range(3):
        gx = X0 + i * (gate_w + gap) + gate_w / 2
        ax.annotate(
            "",
            xy=(gx, gate_y + gate_h + 0.02),
            xytext=(XC, loop_y - 0.02),
            arrowprops=dict(arrowstyle="-|>", color="#8a8a8a", lw=0.95, mutation_scale=8),
        )

    # --- Band D: verdict (Fig.1 caption / §8) ---
    verd_h = 0.88
    verd_y = 0.32
    v_arrow(ax, gate_y, verd_y + verd_h)
    rounded(ax, X0, verd_y, W, verd_h, facecolor="#FDEDEC", edge="#C0392B", lw=1.25, radius=0.12)
    label(
        ax,
        XC,
        verd_y + verd_h / 2 + 0.12,
        "Verdict: bounded in-domain refinement",
        size=8.8,
        weight="bold",
        color="#7B241C",
    )
    label(
        ax,
        XC,
        verd_y + verd_h / 2 - 0.22,
        "effective under protocol fidelity  ·  not cross-split generalization",
        size=7.6,
        color="#7B241C",
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    for ext, kw in [("pdf", {}), ("png", {"dpi": 220}), ("svg", {})]:
        fig.savefig(OUT / f"fig1_method_pipeline.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)
    print("wrote fig1 pdf/png/svg (V9-aligned)")


if __name__ == "__main__":
    main()
