#!/usr/bin/env python3
"""Plot mean accuracy and standard-deviation bars from master_table_v2.csv."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BUDGET_X = {"1": 1, "2": 2, "5": 5, "10": 10, "25": 25, "50": 50, "all": 100}
STYLE = {
    ("knn", "resnet50"): {"color": "#1f77b4", "marker": "o", "linestyle": "--"},
    ("linear_probe", "resnet50"): {"color": "#1f77b4", "marker": "s"},
    ("knn", "dinov2_vits14"): {"color": "#d62728", "marker": "o", "linestyle": "--"},
    ("linear_probe", "dinov2_vits14"): {"color": "#d62728", "marker": "s"},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/master_table_v2.csv")
    parser.add_argument("--output-dir", default="plots/eval_v2")
    return parser.parse_args()


def plot_dataset(frame, dataset, output_path):
    figure, axis = plt.subplots(figsize=(9, 6))
    subset = frame[frame["dataset"] == dataset]
    for (method, backbone), group in subset.groupby(["method", "backbone"]):
        group = group.copy()
        group["_x"] = group["k_shot"].astype(str).map(BUDGET_X)
        if group["_x"].isna().any():
            raise ValueError(f"Unsupported label budget in {dataset}/{method}/{backbone}")
        group = group.sort_values("_x")
        style = STYLE.get((method, backbone), {"marker": "o"})
        axis.errorbar(
            group["_x"],
            group["acc_mean"] * 100,
            yerr=group["acc_std"] * 100,
            capsize=3,
            linewidth=2,
            markersize=7,
            label=f"{method} {backbone}",
            **style,
        )
    axis.set_xscale("log")
    axis.set_xticks(list(BUDGET_X.values()))
    axis.set_xticklabels(list(BUDGET_X.keys()))
    axis.set_xlabel("Label budget (images per class)")
    axis.set_ylabel("Top-1 Accuracy (%)")
    axis.set_title(f"Frozen-feature evaluation — {dataset}")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=9)
    axis.set_ylim(bottom=0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {"dataset", "backbone", "method", "k_shot", "acc_mean", "acc_std"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{args.input}: missing columns {sorted(missing)}")
    os.makedirs(args.output_dir, exist_ok=True)
    for dataset in sorted(frame["dataset"].unique()):
        output_path = os.path.join(args.output_dir, f"frozen_curves_{dataset}.png")
        plot_dataset(frame, dataset, output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
