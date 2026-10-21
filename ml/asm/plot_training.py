#!/usr/bin/env python3
"""
plot_training.py — render training-loss curves for the README.

Two data sources are supported:
  * train_state.json written by train.py (contains the full `history` list);
  * a training log, from which lines like
        epoch 1 step 20/91 loss=0.6123 lr=1.50e-04 elapsed=42.1s
    are parsed. Useful while a run is still in progress, because
    train_state.json is written only at the end.

The script renders a light and a dark PNG (the palette is stepped for each
surface, not flipped automatically), suitable for a README <picture> block.

Usage:
  python3 plot_training.py
  python3 plot_training.py --state checkpoints/lora-full/train_state.json \
      --title "Full run: loss on assistant tokens" \
      --subtitle "723 pairs | 2 epochs | 182 optimizer steps" \
      --stem training_loss_full
  python3 plot_training.py --log /tmp/train_full.log --stem training_loss_full
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE = os.path.join(BASE_DIR, "checkpoints", "lora-smoke",
                             "train_state.json")
DEFAULT_OUT = os.path.join(BASE_DIR, "docs")

# Reference data-viz palette (light / dark surfaces are validated separately).
PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": "#3987e5",
    },
}

LOG_RE = re.compile(
    r"epoch (\d+) step (\d+)/(\d+) loss=([0-9.]+)")


def load_state(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = [h for h in data.get("history", [])
               if "step" in h and "loss" in h]
    if not history:
        raise SystemExit(f"no loss history in {path}")
    return history


def load_log(path):
    history = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            history.append({
                "epoch": int(m.group(1)),
                "step": int(m.group(2)),
                "loss": float(m.group(4)),
            })
    if not history:
        raise SystemExit(f"no 'epoch ... step ... loss=' lines in {path}")
    # The log repeats the last step of an epoch; keep the last occurrence.
    dedup = {}
    for h in history:
        dedup[h["step"]] = h
    return [dedup[k] for k in sorted(dedup)]


def render(history, out_path, title, subtitle, palette):
    surface = palette["surface"]
    grid = palette["grid"]
    primary = palette["primary"]
    secondary = palette["secondary"]
    muted = palette["muted"]
    series = palette["series"]

    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=170)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    # Epoch boundaries as recessive hairlines, with small muted labels.
    epochs = []
    for h in history:
        if not epochs or epochs[-1] != h["epoch"]:
            epochs.append(h["epoch"])
    if len(epochs) > 1:
        bounds = []
        for i in range(1, len(epochs)):
            prev_epoch = epochs[i - 1]
            last_prev = max(h["step"] for h in history
                            if h["epoch"] == prev_epoch)
            bounds.append(last_prev + 0.5)
        for b in bounds:
            ax.axvline(b, color=grid, linewidth=1.0, zorder=1)
        edges = [min(steps)] + bounds + [max(steps)]
        for i, epoch in enumerate(epochs):
            mid = (edges[i] + edges[i + 1]) / 2
            ax.text(mid, 0.985, f"epoch {epoch}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5, color=muted)

    ax.plot(steps, losses, color=series, linewidth=2.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=3)

    # End marker: 2px ring in the surface color, value in text ink.
    ax.scatter([steps[-1]], [losses[-1]], s=42, color=series,
               edgecolors=surface, linewidths=2.0, zorder=4)
    ax.annotate(f"{losses[-1]:.2f}", (steps[-1], losses[-1]),
                textcoords="offset points", xytext=(9, 4),
                fontsize=10, fontweight="semibold", color=primary)

    ax.grid(axis="y", color=grid, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(palette["axis"])
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(axis="both", colors=muted, labelsize=9,
                   length=0, pad=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_ylim(0, max(losses) * 1.12)
    ax.set_xlim(min(steps), max(steps) + max(1, max(steps) * 0.02))

    ax.set_xlabel("Optimizer step", fontsize=9.5, color=secondary, labelpad=8)
    ax.set_ylabel("Cross-entropy loss", fontsize=9.5, color=secondary,
                  labelpad=8)

    fig.suptitle(title, x=0.075, y=0.965, ha="left", va="top",
                 fontsize=13, fontweight="semibold", color=primary)
    fig.text(0.075, 0.885, subtitle, ha="left", va="top",
             fontsize=9.5, color=secondary)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.80, bottom=0.15)

    fig.savefig(out_path, facecolor=surface)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Render training-loss curves (light + dark PNG)")
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help="train_state.json written by train.py")
    ap.add_argument("--log", help="training log to parse instead of --state")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--stem", default="training_loss")
    ap.add_argument("--title", default="LoRA smoke run: loss on assistant tokens")
    ap.add_argument("--subtitle",
                    default="60 pairs | 3 epochs | 45 optimizer steps | "
                            "LoRA r=16 | lr 2e-4")
    args = ap.parse_args()

    history = load_log(args.log) if args.log else load_state(args.state)
    os.makedirs(args.out_dir, exist_ok=True)
    for mode, palette in PALETTES.items():
        out = os.path.join(args.out_dir, f"{args.stem}_{mode}.png")
        render(history, out, args.title, args.subtitle, palette)


if __name__ == "__main__":
    main()
