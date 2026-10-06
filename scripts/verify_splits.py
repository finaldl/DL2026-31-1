#!/usr/bin/env python3
"""Verify the committed k-shot split files in splits/.

The split files are fixed artifacts: they cannot be regenerated from
src/sampler.py, so reproducibility means checking them rather than rebuilding
them. Checks, per dataset and seed:

  - every budget in config/default.yaml has a file, with no duplicate indices
  - |k| = k x |k=1| (same number of images per class at every budget)
  - nested: k=1 subset of k=2 subset of ... subset of k=all
  - k=all is exactly range(N), so every index points into the train(val) split
    and none can point into the separately stored test split
  - different seeds give different subsets

With --features-dir (needs torch and the cached train/trainval features), it
also checks that each budget holds exactly k images of every class.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_FEATURES = {
    "pets": "resnet50_pets_trainval.pt",
    "eurosat": "resnet50_eurosat_train.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--datasets", nargs="+", default=["pets", "eurosat"])
    parser.add_argument(
        "--features-dir",
        default=None,
        help="optional: also check per-class counts against cached labels",
    )
    return parser.parse_args()


def load_split(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("train", data.get("indices"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of indices")
    return data


def load_train_labels(features_dir, dataset):
    from src.evaluate_frozen import _load_raw_cache

    raw = _load_raw_cache(os.path.join(features_dir, TRAIN_FEATURES[dataset]))
    return raw["labels"]


def check_dataset(dataset, budgets, seeds, splits_dir, labels=None):
    errors = []
    numeric = [b for b in budgets if str(b) != "all"]
    subsets_k1 = []
    for seed in seeds:
        splits = {}
        for budget in budgets:
            path = os.path.join(splits_dir, f"{dataset}_seed{seed}_k{budget}.json")
            if not os.path.exists(path):
                errors.append(f"missing {path}")
                continue
            indices = load_split(path)
            if len(indices) != len(set(indices)):
                errors.append(f"{path}: duplicate indices")
            splits[str(budget)] = indices
        if len(splits) != len(budgets):
            continue

        per_class = len(splits[str(numeric[0])]) // int(numeric[0])
        for budget in numeric:
            if len(splits[str(budget)]) != int(budget) * per_class:
                errors.append(
                    f"{dataset} seed {seed} k={budget}: size {len(splits[str(budget)])}, "
                    f"expected {int(budget) * per_class}"
                )

        order = [str(b) for b in numeric] + ["all"]
        for small, big in zip(order, order[1:]):
            if not set(splits[small]) <= set(splits[big]):
                errors.append(f"{dataset} seed {seed}: k={small} is not a subset of k={big}")

        n_train = len(splits["all"])
        if sorted(splits["all"]) != list(range(n_train)):
            errors.append(f"{dataset} seed {seed}: k=all is not exactly range({n_train})")

        if labels is not None:
            if len(labels) != n_train:
                errors.append(f"{dataset}: {len(labels)} cached labels, k=all has {n_train}")
            else:
                for budget in numeric:
                    counts = {}
                    for index in splits[str(budget)]:
                        label = int(labels[index])
                        counts[label] = counts.get(label, 0) + 1
                    if set(counts.values()) != {int(budget)}:
                        errors.append(
                            f"{dataset} seed {seed} k={budget}: per-class counts "
                            f"{sorted(set(counts.values()))}, expected {budget}"
                        )

        subsets_k1.append(sorted(splits[str(numeric[0])]))
        print(
            f"{dataset} seed {seed}: sizes "
            + ", ".join(str(len(splits[b])) for b in order)
            + f" | n_train={n_train}"
        )

    if len(subsets_k1) == len(seeds) and len({tuple(s) for s in subsets_k1}) < len(seeds):
        errors.append(f"{dataset}: two seeds share the same k={numeric[0]} subset")
    return errors


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    budgets, seeds = config["label_budgets"], config["seeds"]

    errors = []
    for dataset in args.datasets:
        labels = load_train_labels(args.features_dir, dataset) if args.features_dir else None
        errors += check_dataset(dataset, budgets, seeds, args.splits_dir, labels)

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    checked = "structure + per-class counts" if args.features_dir else "structure"
    print(f"OK: all splits valid ({checked})")


if __name__ == "__main__":
    main()
