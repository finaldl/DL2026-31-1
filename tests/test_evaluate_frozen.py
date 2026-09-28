from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluate_frozen import (
    evaluate_one,
    knn_eval,
    linear_probe_eval,
    load_features,
    load_split_indices,
    validate_split_indices,
)
from src.metrics import confusion_matrix, macro_f1, random_baseline, top1_accuracy


class MetricsTests(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(top1_accuracy([0, 1, 1], [0, 0, 1]), 2 / 3)
        self.assertAlmostEqual(macro_f1([0, 1, 1], [0, 0, 1]), 2 / 3)
        matrix, classes = confusion_matrix([0, 1, 1], [0, 0, 1])
        np.testing.assert_array_equal(matrix, [[1, 0], [1, 1]])
        self.assertEqual(classes, [0, 1])
        self.assertEqual(random_baseline(4), 0.25)

    def test_empty_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            top1_accuracy([], [])
        with self.assertRaises(ValueError):
            random_baseline(0)


class FeatureAndSplitTests(unittest.TestCase):
    def test_npz_loading_and_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez(
                path,
                train_features=np.eye(2, dtype=np.float32),
                train_labels=np.array([0, 1]),
                test_features=np.eye(2, dtype=np.float32),
                test_labels=np.array([0, 1]),
            )
            loaded = load_features(path)
            self.assertEqual(loaded["dim"], 2)
            self.assertEqual(loaded["n_classes"], 2)

    def test_split_loading_checks_duplicates_and_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "pets_seed0_k1.json"
            split_path.write_text(json.dumps([0, 2]), encoding="utf-8")
            np.testing.assert_array_equal(
                load_split_indices(directory, "pets", 0, 1, n_train=3), [0, 2]
            )
            with self.assertRaises(ValueError):
                validate_split_indices([0, 0], 3)
            with self.assertRaises(ValueError):
                validate_split_indices([3], 3)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.train_x = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
        self.train_y = np.array([0, 0, 1, 1])

    def test_knn_uses_majority_vote_by_default(self):
        result = knn_eval(
            self.train_x, self.train_y, self.train_x, self.train_y, k=1
        )
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["f1_macro"], 1.0)
        self.assertEqual(result["k_actual"], 1)

    def test_linear_probe_supports_installed_sklearn(self):
        result = linear_probe_eval(
            self.train_x, self.train_y, self.train_x, self.train_y
        )
        self.assertGreaterEqual(result["accuracy"], 0.5)
        self.assertGreater(result["n_iter"], 0)

    def test_evaluate_one_includes_schema_counts(self):
        features = {
            "train_features": self.train_x,
            "train_labels": self.train_y,
            "test_features": self.train_x,
            "test_labels": self.train_y,
        }
        result = evaluate_one(features, [0, 1, 2, 3], "knn", knn_k=1)
        self.assertEqual(result["num_train_samples"], 4)
        self.assertEqual(result["num_test_samples"], 4)


if __name__ == "__main__":
    unittest.main()
