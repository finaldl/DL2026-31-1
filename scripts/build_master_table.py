#!/usr/bin/env python3
"""Normalize experiment CSVs and build long and presentation-wide summaries."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


KEY_COLUMNS = ["dataset", "backbone", "method", "k_shot", "seed"]
BACKBONE_ALIASES = {"dinov2": "dinov2_vits14"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", default="results/frozen_grid_v2.csv")
    parser.add_argument("--finetune", default=None)
    parser.add_argument("--output", default="results/master_table_v2.csv")
    parser.add_argument("--output-wide", default="results/master_table_wide_v2.csv")
    return parser.parse_args()


def budget_sort_key(value):
    return (1, float("inf")) if str(value).lower() == "all" else (0, int(value))


def _normalize_accuracy(series, source):
    values = pd.to_numeric(series, errors="raise").astype(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{source}: accuracy contains invalid values")
    if (values > 1).any():
        if (values <= 100).all() and (values > 1).all():
            values = values / 100.0
        else:
            raise ValueError(f"{source}: accuracy mixes incompatible units")
    return values


def normalize_results(path, source_kind):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {source_kind} results: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path}: result file is empty")

    budget_column = next(
        (name for name in ("k_shot", "label_budget", "k") if name in frame.columns),
        None,
    )
    if budget_column is None:
        raise ValueError(f"{path}: missing one of k_shot/label_budget/k")
    frame = frame.rename(columns={budget_column: "k_shot"})
    missing = [name for name in KEY_COLUMNS + ["accuracy"] if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    frame = frame.copy()
    frame["k_shot"] = frame["k_shot"].map(
        lambda value: "all" if str(value).lower() == "all" else str(int(value))
    )
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["backbone"] = frame["backbone"].replace(BACKBONE_ALIASES)
    frame["accuracy"] = _normalize_accuracy(frame["accuracy"], path)
    duplicates = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        duplicate_keys = frame.loc[duplicates, KEY_COLUMNS].drop_duplicates()
        raise ValueError(
            f"{path}: duplicate experiment keys:\n"
            f"{duplicate_keys.to_string(index=False)}"
        )
    return frame[KEY_COLUMNS + ["accuracy"]]


def aggregate_results(frame):
    summary = (
        frame.groupby(["dataset", "backbone", "method", "k_shot"], sort=False)[
            "accuracy"
        ]
        .agg(acc_mean="mean", acc_std="std", n_seeds="count")
        .reset_index()
    )
    summary["acc_std"] = summary["acc_std"].fillna(0.0)
    summary["_budget_order"] = summary["k_shot"].map(budget_sort_key)
    summary = summary.sort_values(
        ["dataset", "method", "backbone", "_budget_order"]
    ).drop(columns="_budget_order")
    return summary


def build_wide(summary):
    presentation = summary.copy()
    presentation["label"] = presentation.apply(
        lambda row: (
            f"{row.acc_mean * 100:.2f} ± {row.acc_std * 100:.2f} "
            f"(n={int(row.n_seeds)})"
        ),
        axis=1,
    )
    budgets = sorted(presentation["k_shot"].unique(), key=budget_sort_key)
    wide = presentation.pivot(
        index=["method", "backbone", "dataset"], columns="k_shot", values="label"
    ).reset_index()
    return wide[["method", "backbone", "dataset", *budgets]]


def main():
    args = parse_args()
    frames = [normalize_results(args.frozen, "frozen")]
    if args.finetune:
        frames.append(normalize_results(args.finetune, "finetune"))
    combined = pd.concat(frames, ignore_index=True)
    cross_source_duplicates = combined.duplicated(KEY_COLUMNS, keep=False)
    if cross_source_duplicates.any():
        raise ValueError("Combined inputs contain duplicate experiment keys")

    summary = aggregate_results(combined)
    wide = build_wide(summary)
    for path in (args.output, args.output_wide):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    summary.to_csv(args.output, index=False)
    wide.to_csv(args.output_wide, index=False)
    print(f"Wrote {len(summary)} summary rows to {args.output}")
    print(f"Wrote presentation table to {args.output_wide}")


if __name__ == "__main__":
    main()
