#!/usr/bin/env python3
"""
plot_eval.py — render the evaluation figures used by the READMEs.

Three figures, each written as a light and a dark PNG (the palette is stepped
for each surface, not flipped automatically):

  eval_full118  where the 118 validation answers land: compile rate and
                behavior match by optimization level, plus the outcome split.
  eval_epochs   how pass rate and compile rate move across the three epochs of
                one training run.
  eval_probes   the same adapter measured with one fixed probe list versus
                per-program probes.

All numbers are read from eval_results/, not typed in: the script re-renders
the figures after a new evaluation without anyone editing constants.

Usage:
  python3 plot_eval.py
  python3 plot_eval.py --out-dir docs
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "eval_results")
DEFAULT_OUT = os.path.join(BASE_DIR, "docs")

# Which summary file belongs to which epoch of the night run. The epoch-3 row
# is the plain `night-final` evaluation with the fixed 10 probes; the other two
# use the same probes, so the three points are comparable.
EPOCH_SUMMARIES = [
    (1, "summary_night-epoch1-*.json"),
    (2, "summary_night-epoch2__*.json"),
    (3, "summary_night-final-20261022-095409.json"),
]
FIXED_SUMMARY = "summary_night-final-20261022-095409.json"
PROBES_SUMMARY = "summary_night-final-cases__*.json"
PROBES_PREDICTIONS = "predictions_night-final-cases__*.jsonl"

# Palette: same surfaces and series hue as plot_training.py, plus one lighter
# step of the same hue for the second series and a four-step ramp for the
# ordered outcome segments.
PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "strong": "#2a78d6",
        "soft": "#7aa9e2",
        "faint": "#c3d9f2",
        "palest": "#e6eef9",
        "warm": "#b45309",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "strong": "#3987e5",
        "soft": "#2b5f9c",
        "faint": "#223b58",
        "palest": "#1e2c3b",
        "warm": "#d98a3d",
    },
}


def load_summary(pattern):
    paths = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    if not paths:
        raise SystemExit(f"no summary matching {pattern} in {RESULTS}")
    with open(paths[0], "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(pattern):
    paths = sorted(glob.glob(os.path.join(RESULTS, pattern)))
    if not paths:
        raise SystemExit(f"no predictions matching {pattern} in {RESULTS}")
    with open(paths[0], "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hbar(ax, x0, y0, width, height, color, rx=0.9, ry=0.12, edge=None):
    """Horizontal bar from x0 to x0+width, rounded at the far end only.

    A bar must sit flat on its baseline: rounding all four corners lifts it
    off the axis and breaks the reading of the chart. `rx`/`ry` are the corner
    radii in data units, because the x and y axes never share a scale."""
    if width <= 0:
        return
    rx = min(rx, width)
    ry = min(ry, height / 2)
    y1 = y0 + height
    verts = [(x0, y0), (x0 + width - rx, y0),
             (x0 + width, y0), (x0 + width, y0 + ry),
             (x0 + width, y1 - ry),
             (x0 + width, y1), (x0 + width - rx, y1),
             (x0, y1), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                           edgecolor=edge if edge else color,
                           linewidth=0.8 if edge else 0.0,
                           joinstyle="round"))


def vbar(ax, x0, y0, width, height, color, rx=0.06, ry=0.9, edge=None):
    """Vertical bar from y0 up to y0+height, rounded at the top end only."""
    if height <= 0:
        return
    rx = min(rx, width / 2)
    ry = min(ry, height)
    x1 = x0 + width
    verts = [(x0, y0), (x0, y0 + height - ry),
             (x0, y0 + height), (x0 + rx, y0 + height),
             (x1 - rx, y0 + height),
             (x1, y0 + height), (x1, y0 + height - ry),
             (x1, y0), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                           edgecolor=edge if edge else color,
                           linewidth=0.8 if edge else 0.0,
                           joinstyle="round"))


def style_axes(ax, palette, ylabel=None, xlabel=None):
    """Recessive frame: no top/right spine, no tick marks, muted labels."""
    ax.set_facecolor(palette["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(palette["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=palette["muted"], labelsize=9,
                   length=0, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color=palette["secondary"],
                      labelpad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color=palette["secondary"],
                      labelpad=8)


def title_block(fig, palette, title, subtitle):
    fig.suptitle(title, x=0.065, y=0.965, ha="left", va="top",
                 fontsize=13, fontweight="semibold", color=palette["primary"])
    fig.text(0.065, 0.885, subtitle, ha="left", va="top",
             fontsize=9.5, color=palette["secondary"])


# --------------------------------------------------------------------------
# Figure 1: the 118-pair split, by optimization level and by outcome
# --------------------------------------------------------------------------

def render_full118(summary, predictions, out_path, palette):
    surface = palette["surface"]
    n = len(predictions)
    passed = sum(1 for r in predictions if r["passed"])
    compiled = sum(1 for r in predictions if r["cand_compiled"])
    extracted = sum(1 for r in predictions if r["extracted"])
    differed = compiled - passed
    no_compile = extracted - compiled
    no_code = n - extracted

    fig = plt.figure(figsize=(9.8, 4.3), dpi=170)
    fig.patch.set_facecolor(surface)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.42)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    style_axes(ax1, palette, xlabel="Share of the 118 pairs, %")
    style_axes(ax2, palette, xlabel="Share of the 118 pairs, %")

    # --- left: compile vs behavior, overall and per -O level
    groups = ["All", "-O0", "-O2"]
    rows = [summary["by_opt"][k] for k in ("O0", "O2")]
    compile_vals = [summary["compile_rate"] * 100] + [r["compile_rate"] * 100
                                                      for r in rows]
    pass_vals = [summary["pass_rate"] * 100] + [r["pass_rate"] * 100
                                                for r in rows]

    h = 0.30
    gap = 0.06
    for i, name in enumerate(groups):
        y_compile = len(groups) - 1 - i + gap / 2
        y_pass = len(groups) - 1 - i - gap / 2 - h
        hbar(ax1, 0, y_compile, compile_vals[i], h, palette["soft"],
             rx=1.1, ry=h / 2)
        hbar(ax1, 0, y_pass, pass_vals[i], h, palette["strong"],
             rx=1.1, ry=h / 2)
        ax1.text(compile_vals[i] + 1.4, y_compile + h / 2,
                 f"{compile_vals[i]:.1f}", va="center", ha="left",
                 fontsize=9, color=palette["secondary"])
        ax1.text(pass_vals[i] + 1.4, y_pass + h / 2,
                 f"{pass_vals[i]:.1f}", va="center", ha="left",
                 fontsize=9, fontweight="semibold", color=palette["primary"])

    ax1.set_ylim(-0.35, len(groups) - 0.35)
    ax1.set_xlim(0, 100)
    ax1.set_yticks([len(groups) - 1 - i + h / 2 for i in range(len(groups))])
    ax1.set_yticklabels(groups, fontsize=10, color=palette["primary"])
    ax1.grid(axis="x", color=palette["grid"], linewidth=1.0)
    ax1.set_axisbelow(True)
    ax1.legend(
        handles=[Patch(facecolor=palette["soft"], label="Compiles"),
                 Patch(facecolor=palette["strong"], label="Same behavior")],
        loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False, fontsize=9,
        handlelength=1.2, handleheight=0.9, labelspacing=0.45, ncol=2,
        columnspacing=1.6, labelcolor=palette["secondary"])

    # --- right: the four outcomes as one 100% bar
    segments = [
        ("passed", passed, palette["strong"]),
        ("compiled, differs", differed, palette["soft"]),
        ("did not compile", no_compile, palette["faint"]),
        ("no C block", no_code, palette["palest"]),
    ]
    x = 0.0
    y0, height = 0.15, 0.5
    gap_px = 0.6
    for i, (name, count, color) in enumerate(segments):
        share = count / n * 100
        edge = palette["axis"] if color is palette["palest"] else None
        hbar(ax2, x, y0, share - gap_px, height, color, rx=1.1,
             ry=height / 3, edge=edge)
        center = x + (share - gap_px) / 2
        # Count inside the segment, name above it: identity never rests on
        # color alone, and no legend is needed in a narrow second panel.
        ax2.text(center, y0 + height / 2, f"{count}", ha="center", va="center",
                 fontsize=10, fontweight="semibold",
                 color=palette["surface"] if i < 2 else palette["primary"])
        # Names alternate between two rows: the last two segments are narrow
        # and their names would otherwise run into each other.
        ax2.text(center, y0 + height + (0.10 if i % 2 == 0 else 0.27), name,
                 ha="right" if i == 3 else "center", va="bottom", fontsize=8.5,
                 color=palette["secondary"])
        x += share

    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.grid(axis="x", color=palette["grid"], linewidth=1.0)
    ax2.set_axisbelow(True)

    title_block(fig, palette,
                "Full validation: 118 unseen pairs",
                "LoRA night-20261022 final | 2,579 training pairs, 3 epochs | "
                "per-program differential probes")
    fig.subplots_adjust(left=0.085, right=0.975, top=0.75, bottom=0.14)
    fig.savefig(out_path, facecolor=surface)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Figure 2: one training run, three epochs
# --------------------------------------------------------------------------

def render_epochs(points, out_path, palette):
    surface = palette["surface"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=170)
    fig.patch.set_facecolor(surface)
    style_axes(ax, palette, ylabel="Share of the 118 pairs, %",
               xlabel="Epoch of the night-20261022 run")

    epochs = [p[0] for p in points]
    pass_vals = [p[1] for p in points]
    compile_vals = [p[2] for p in points]

    width = 0.34
    for i, e in enumerate(epochs):
        vbar(ax, e - width / 2, 0, width, pass_vals[i], palette["strong"],
             rx=width / 2, ry=1.4)
        ax.text(e, pass_vals[i] + 2.2, f"{pass_vals[i]:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="semibold",
                color=palette["primary"])

    ax.plot(epochs, compile_vals, color=palette["warm"], linewidth=2.0,
            marker="o", markersize=7, markerfacecolor=palette["warm"],
            markeredgecolor=surface, markeredgewidth=2.0, zorder=4,
            solid_capstyle="round")
    for e, v in zip(epochs, compile_vals):
        ax.annotate(f"{v:.1f}", (e, v), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9,
                    color=palette["warm"])

    ax.legend(handles=[Patch(facecolor=palette["strong"],
                             label="Same behavior"),
                       Line2D([], [], color=palette["warm"], linewidth=2.0,
                              marker="o", markersize=7, label="Compiles")],
              loc="upper left", frameon=False, fontsize=9,
              labelcolor=palette["secondary"], ncol=2,
              bbox_to_anchor=(0.0, 1.0), handlelength=1.4)

    ax.set_xticks(epochs)
    ax.set_xticklabels([f"epoch {e}" for e in epochs], fontsize=10,
                       color=palette["primary"])
    ax.set_xlim(min(epochs) - 0.6, max(epochs) + 0.6)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color=palette["grid"], linewidth=1.0)
    ax.set_axisbelow(True)

    title_block(fig, palette,
                "More epochs, more correct behavior",
                "118 unseen pairs, fixed 10 probes | compile rate moves "
                "noisily, behavior match climbs")
    fig.subplots_adjust(left=0.085, right=0.975, top=0.75, bottom=0.15)
    fig.savefig(out_path, facecolor=surface)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# Figure 3: fixed probes vs per-program probes
# --------------------------------------------------------------------------

def render_probes(fixed, probes, probes_ge2, out_path, palette):
    surface = palette["surface"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=170)
    fig.patch.set_facecolor(surface)
    style_axes(ax, palette, ylabel="Pairs with matching behavior, %")

    bars = [
        ("same 10 probes\nfor every program", fixed),
        ("probes chosen\nper program", probes),
        ("per program,\n2 or more probes", probes_ge2),
    ]
    width = 0.46
    for i, (name, value) in enumerate(bars):
        color = palette["strong"] if i == 1 else palette["faint"]
        edge = None if i == 1 else palette["axis"]
        vbar(ax, i - width / 2, 0, width, value, color, rx=width / 2, ry=1.4,
             edge=edge)
        ax.text(i, value + 1.6, f"{value:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="semibold",
                color=palette["primary"] if i == 1 else palette["secondary"])

    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9.5,
                       color=palette["primary"], linespacing=1.4)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_ylim(0, 38)
    ax.grid(axis="y", color=palette["grid"], linewidth=1.0)
    ax.set_axisbelow(True)

    title_block(fig, palette,
                "The probe set changes the number",
                "Same adapter, same 118 answers — only the differential "
                "probes differ")
    fig.subplots_adjust(left=0.115, right=0.975, top=0.74, bottom=0.19)
    fig.savefig(out_path, facecolor=surface)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Render evaluation figures")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    probes_summary = load_summary(PROBES_SUMMARY)
    fixed_summary = load_summary(FIXED_SUMMARY)
    predictions = load_predictions(PROBES_PREDICTIONS)

    informative = [r for r in predictions
                   if r["informative"] and r["cases_total"] >= 2]
    probes_ge2 = (sum(1 for r in informative if r["passed"])
                  / len(informative) * 100)

    points = []
    for epoch, pattern in EPOCH_SUMMARIES:
        s = load_summary(pattern)
        points.append((epoch, s["pass_rate"] * 100, s["compile_rate"] * 100))

    for mode, palette in PALETTES.items():
        render_full118(probes_summary, predictions,
                       os.path.join(args.out_dir, f"eval_full118_{mode}.png"),
                       palette)
        render_epochs(points,
                      os.path.join(args.out_dir, f"eval_epochs_{mode}.png"),
                      palette)
        render_probes(fixed_summary["pass_rate"] * 100,
                      probes_summary["pass_rate"] * 100,
                      probes_ge2,
                      os.path.join(args.out_dir, f"eval_probes_{mode}.png"),
                      palette)


if __name__ == "__main__":
    main()
