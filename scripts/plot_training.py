"""Training curves from checkpoints/<run>/log.csv -> assets/training.png

    python scripts/plot_training.py main no_push
"""

import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"main": "#2de2e6", "no_push": "#ff5fa2"}
LABELS = {"main": "with kicks", "no_push": "without kicks"}


def load(run):
    with open(ROOT / "checkpoints" / run / "log.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    runs = sys.argv[1:] or ["main"]
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=110)
    for ax in axes:
        ax.set_facecolor("#0d0b1a")
        ax.grid(alpha=0.15)
    for run in runs:
        log = load(run)
        x = log["env_steps"] / 1e6
        c, label = COLORS.get(run, "#ffd166"), LABELS.get(run, run)
        axes[0].plot(x, log["ep_return"], color=c, lw=2, label=label)
        axes[1].plot(x, log["track_lin_vel"], color=c, lw=2, label=label)
        axes[2].plot(x, 100 * log["fall_rate"], color=c, lw=2, label=label)
    axes[0].set_title("episode return")
    axes[1].set_title("velocity tracking (1 = perfect)")
    axes[2].set_title("episodes ending in a fall (%)")
    for ax in axes:
        ax.set_xlabel("million env steps")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "training.png", facecolor="#0d0b1a")
    print("saved assets/training.png")


if __name__ == "__main__":
    main()
