#!/usr/bin/env python3
"""Run the frozen-feature evaluation matrix with resumable CSV output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate_frozen import evaluate_one, load_features, load_split_indices  # noqa: E402


METHODS = ("knn", "linear_probe")
BACKBONE_ALIASES = {"dinov2": "dinov2_vits14", "dinov2_vits14": "dinov2_vits14"}
FEATURE_NAME_ALIASES = {
    "dinov2_vits14": ("dinov2_vits14", "dinov2"),
    "resnet50": ("resnet50",),
}
FIELDNAMES = [
    "dataset",
    "backbone",
    "method",
    "k_shot",
    "seed",
    "accuracy",
    "f1_macro",
    "num_train_samples",
    "num_test_samples",
    "run_id",
    "k_actual",
    "n_iter",
    "preprocess",
    "knn_k",
    "knn_weights",
    "lp_C",
    "lp_max_iter",
    "eval_time_sec",
]


def load_project_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = ("label_budgets", "seeds", "backbones")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{path}: missing config fields {missing}")
    return config


def canonical_backbone(name):
    return BACKBONE_ALIASES.get(name, name)


def normalize_budget(value):
    text = str(value).strip().lower()
    return "all" if text == "all" else int(text)


def stable_run_id(configuration):
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--output", default="results/frozen_grid_v2.csv")
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--datasets", nargs="+", default=["pets", "eurosat"])
    parser.add_argument("--budgets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument(
        "--knn-weights", choices=("uniform", "distance"), default="uniform"
    )
    parser.add_argument("--lp-C", type=float, default=1.0)
    parser.add_argument("--lp-max-iter", type=int, default=1000)
    parser.add_argument(
        "--preprocess", choices=("l2", "standard", "none"), default="l2"
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def find_feature_cache(features_dir, backbone, dataset):
    names = FEATURE_NAME_ALIASES.get(backbone, (backbone,))
    for name in names:
        for extension in (".pt", ".npz", ".pkl", ".pickle"):
            candidate = os.path.join(features_dir, f"{name}_{dataset}{extension}")
            if os.path.isfile(candidate):
                return candidate
    expected = ", ".join(f"{name}_{dataset}.*" for name in names)
    raise FileNotFoundError(f"Missing feature cache in {features_dir}: {expected}")


def read_completed_run_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                f"{path} has a legacy or incompatible header. Preserve it and choose "
                "a new --output path instead of appending mixed schemas."
            )
        rows = list(reader)
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(f"{path} contains duplicate run_id values")
    return set(run_ids)


def run_configuration(args, project_config):
    backbones = [
        canonical_backbone(value)
        for value in (args.backbones or project_config["backbones"])
    ]
    datasets = list(args.datasets)
    budgets = [
        normalize_budget(value)
        for value in (args.budgets or project_config["label_budgets"])
    ]
    seeds = list(args.seeds or project_config["seeds"])
    if args.quick:
        backbones, datasets, budgets, seeds = (
            backbones[:1], datasets[:1], budgets[:1], seeds[:1]
        )
    return backbones, datasets, budgets, seeds


def main():
    args = parse_args()
    if args.knn_k <= 0 or args.lp_C <= 0 or args.lp_max_iter <= 0:
        raise SystemExit("knn-k, lp-C and lp-max-iter must be positive")
    project_config = load_project_config(args.config)
    backbones, datasets, budgets, seeds = run_configuration(args, project_config)
    expected_runs = len(backbones) * len(datasets) * len(budgets) * len(seeds) * len(METHODS)

    output_directory = os.path.dirname(args.output)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    completed = read_completed_run_ids(args.output)
    output_exists = os.path.exists(args.output)
    feature_cache = {}
    failures = []
    completed_this_run = 0
    started = time.perf_counter()

    with open(args.output, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not output_exists:
            writer.writeheader()

        for backbone in backbones:
            for dataset in datasets:
                cache_key = (backbone, dataset)
                try:
                    feature_path = find_feature_cache(args.features_dir, backbone, dataset)
                    feature_cache[cache_key] = load_features(feature_path)
                except Exception as exc:
                    failures.append(f"{backbone}/{dataset}: {exc}")
                    continue
                features = feature_cache[cache_key]

                for budget in budgets:
                    for seed in seeds:
                        try:
                            indices = load_split_indices(
                                args.splits_dir,
                                dataset,
                                seed,
                                budget,
                                n_train=len(features["train_labels"]),
                            )
                        except Exception as exc:
                            failures.append(f"{backbone}/{dataset}/k={budget}/seed={seed}: {exc}")
                            continue

                        for method in METHODS:
                            identity = {
                                "dataset": dataset,
                                "backbone": backbone,
                                "method": method,
                                "k_shot": str(budget),
                                "seed": seed,
                                "preprocess": args.preprocess,
                                "knn_k": args.knn_k,
                                "knn_weights": args.knn_weights,
                                "lp_C": args.lp_C,
                                "lp_max_iter": args.lp_max_iter,
                            }
                            run_id = stable_run_id(identity)
                            if run_id in completed:
                                continue
                            try:
                                result = evaluate_one(
                                    features,
                                    indices,
                                    method,
                                    knn_k=args.knn_k,
                                    knn_weights=args.knn_weights,
                                    lp_C=args.lp_C,
                                    lp_max_iter=args.lp_max_iter,
                                    preprocess=args.preprocess,
                                )
                            except Exception as exc:
                                failures.append(
                                    f"{method}/{backbone}/{dataset}/k={budget}/seed={seed}: {exc}"
                                )
                                continue
                            row = {
                                **identity,
                                "accuracy": round(result["accuracy"], 8),
                                "f1_macro": round(result["f1_macro"], 8),
                                "num_train_samples": result["num_train_samples"],
                                "num_test_samples": result["num_test_samples"],
                                "run_id": run_id,
                                "k_actual": result.get("k_actual", ""),
                                "n_iter": result.get("n_iter", ""),
                                "eval_time_sec": round(result["eval_time_sec"], 4),
                            }
                            writer.writerow(row)
                            handle.flush()
                            completed.add(run_id)
                            completed_this_run += 1

    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}", file=sys.stderr)
        raise SystemExit(f"Evaluation incomplete: {len(failures)} configuration error(s)")

    elapsed = time.perf_counter() - started
    print(
        f"Evaluation complete: expected={expected_runs}, new={completed_this_run}, "
        f"output={args.output}, elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
