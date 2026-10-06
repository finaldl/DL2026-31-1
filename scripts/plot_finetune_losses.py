"""Plot fine-tuning training loss from the TensorBoard logs in runs/.

One panel per label budget; each backbone shows its 3-seed mean (solid) and the
individual seeds (faint). Stage 1 = epochs 1-10, Stage 2 = epochs 11-40.

Usage (from repo root):  python scripts/plot_finetune_losses.py
Needs: tensorboard, matplotlib, numpy (no GPU, no torch).
"""
import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_DIR = "runs"
OUTPUT = "plots/finetune_loss_curves.png"
BACKBONES = {"resnet50": ("ResNet-50", "#2a78d6"), "dinov2": ("DINOv2 ViT-S/14", "#eb6834")}
K_VALUES = ["5", "10", "25", "all"]
SEEDS = [0, 1, 2]


def load_curve(run_dir):
    """Return Stage 1 + Stage 2 loss per epoch from the newest event file in run_dir."""
    event_file = sorted(glob.glob(os.path.join(run_dir, "events*")))[-1]  # last file = grid run
    acc = EventAccumulator(event_file)
    acc.Reload()
    stage1 = [e.value for e in acc.Scalars("Loss/Stage1")]
    stage2 = [e.value for e in acc.Scalars("Loss/Stage2")]
    return np.array(stage1 + stage2)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, k in zip(axes, K_VALUES):
        ax.set_facecolor("#fcfcfb")
        for backbone, (label, color) in BACKBONES.items():
            curves = [load_curve(f"{RUNS_DIR}/{backbone}_k{k}_seed{s}") for s in SEEDS]
            epochs = np.arange(1, len(curves[0]) + 1)
            for c in curves:
                ax.plot(epochs, c, color=color, linewidth=1, alpha=0.25)
            ax.plot(epochs, np.mean(curves, axis=0), color=color, linewidth=2, label=label)
        ax.axvline(10.5, color="#8a8984", linestyle="--", linewidth=1)
        ax.axhline(math.log(37), color="#8a8984", linestyle=":", linewidth=1)
        ax.text(11.2, 4.25, "backbone\nunfrozen", fontsize=8, color="#52514e", va="top")
        ax.set_title(f"k = {k}" + (" (5,912 imgs)" if k == "all" else f" ({int(k) * 37} imgs)"),
                     fontsize=11, color="#0b0b0b")
        ax.set_xlabel("epoch", color="#52514e")
        ax.set_xlim(1, 40)
        ax.set_ylim(0, 4.4)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")
    axes[0].set_ylabel("training loss (cross-entropy)", color="#52514e")
    axes[0].text(39.5, math.log(37) + 0.06, "random guess = ln 37", fontsize=8,
                 color="#52514e", ha="right", va="bottom")
    axes[-1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.suptitle("Fine-tuning training loss on Pets: Stage 1 (head only, ep 1-10) → Stage 2 (backbone unfrozen, ep 11-40)"
                 "  ·  mean of 3 seeds, faint = single seeds", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
