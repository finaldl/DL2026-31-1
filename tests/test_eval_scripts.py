from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script_module(name, filename):
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class RunnerIntegrationTests(unittest.TestCase):
    def test_quick_grid_schema_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            features = temporary / "features"
            splits = temporary / "splits"
            features.mkdir()
            splits.mkdir()
            train_x = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
            train_y = np.array([0, 0, 1, 1])
            np.savez(
                features / "resnet50_pets.npz",
                train_features=train_x,
                train_labels=train_y,
                test_features=train_x,
                test_labels=train_y,
            )
            (splits / "pets_seed0_k1.json").write_text(
                json.dumps([0, 2]), encoding="utf-8"
            )
            config = temporary / "config.yaml"
            config.write_text(
                "label_budgets: [1]\nseeds: [0]\nbackbones: [resnet50]\n",
                encoding="utf-8",
            )
            output = temporary / "grid.csv"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_frozen_grid.py"),
                "--config",
                str(config),
                "--features-dir",
                str(features),
                "--splits-dir",
                str(splits),
                "--output",
                str(output),
                "--datasets",
                "pets",
            ]
            environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True)
            subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True)

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["method"] for row in rows}, {"knn", "linear_probe"})
            self.assertEqual(len({row["run_id"] for row in rows}), 2)
            self.assertTrue(all(row["knn_weights"] == "uniform" for row in rows))


class AggregatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script_module("build_master_table", "build_master_table.py")

    def test_legacy_units_and_column_names_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            frozen = directory / "frozen.csv"
            finetune = directory / "finetune.csv"
            pd.DataFrame(
                [
                    {"dataset": "pets", "backbone": "dinov2", "method": "knn", "label_budget": 5, "seed": 0, "accuracy": 0.8},
                    {"dataset": "pets", "backbone": "dinov2", "method": "knn", "label_budget": 5, "seed": 1, "accuracy": 0.9},
                ]
            ).to_csv(frozen, index=False)
            pd.DataFrame(
                [
                    {"dataset": "pets", "backbone": "resnet50", "method": "finetune", "k": 5, "seed": 0, "accuracy": 80.0},
                    {"dataset": "pets", "backbone": "resnet50", "method": "finetune", "k": 5, "seed": 1, "accuracy": 90.0},
                ]
            ).to_csv(finetune, index=False)
            combined = pd.concat(
                [
                    self.module.normalize_results(frozen, "frozen"),
                    self.module.normalize_results(finetune, "finetune"),
                ],
                ignore_index=True,
            )
            summary = self.module.aggregate_results(combined)
            self.assertEqual(len(summary), 2)
            self.assertTrue(np.allclose(summary["acc_mean"], 0.85))
            self.assertIn("dinov2_vits14", set(summary["backbone"]))

    def test_duplicate_experiment_keys_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            row = {
                "dataset": "pets",
                "backbone": "resnet50",
                "method": "finetune",
                "k": 5,
                "seed": 0,
                "accuracy": 80,
            }
            pd.DataFrame([row, row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate experiment keys"):
                self.module.normalize_results(path, "finetune")


if __name__ == "__main__":
    unittest.main()
