#!/usr/bin/env python3
"""Plot Linear Probe vs Fine-tune on Pets -> plots/finetune_vs_frozen.png.

Standalone script: only reads results/master_table_v2.csv and writes one PNG.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INPUT = "results/master_table_v2.csv"
OUTPUT = "plots/finetune_vs_frozen.png"
DATASET = "pets"
BUDGETS = ["5", "10", "25", "all"]
BACKBONES = {
    "resnet50": ("#1f77b4", "ResNet-50 (supervised)"),
    "dinov2_vits14": ("#d62728", "DINOv2 ViT-S/14 (self-supervised)"),
}
METHODS = [("linear_probe", "-", "s", "Linear Probe"), ("finetune", "--", "^", "Fine-tune")]


def main():
    frame = pd.read_csv(INPUT)
    frame["k_shot"] = frame["k_shot"].astype(str)
    frame = frame[
        (frame["dataset"] == DATASET)
        & frame["method"].isin([m[0] for m in METHODS])
        & frame["k_shot"].isin(BUDGETS)
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for backbone, (color, blabel) in BACKBONES.items():
        for method, linestyle, marker, mlabel in METHODS:
            g = frame[(frame["backbone"] == backbone) & (frame["method"] == method)].copy()
            g["_x"] = g["k_shot"].map({k: i for i, k in enumerate(BUDGETS)})
            g = g.sort_values("_x")
            ax.errorbar(
                g["_x"], g["acc_mean"] * 100, yerr=g["acc_std"] * 100,
                color=color, linestyle=linestyle, marker=marker,
                linewidth=2.2, markersize=8, capsize=3,
                label=f"{blabel} - {mlabel}",
            )
    ax.set_xticks(range(len(BUDGETS)))
    ax.set_xticklabels(BUDGETS)
    ax.set_xlabel("Label budget (images per class)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Fine-tune vs Linear Probe - Pets (mean ± std, 3-5 seeds)")
    ax.set_ylim(75, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
