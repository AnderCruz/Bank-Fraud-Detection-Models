"""Synthetic unit tests for paired validation AP uncertainty analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src import analyze_ap_uncertainty as analysis


class PairedApUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
        base = np.linspace(0.05, 0.95, len(self.labels))
        self.scores = {
            "random_forest": base[::-1].copy(),
            "mlp": base.copy(),
            "lstm": np.roll(base, 1),
            "gru": np.roll(base, 2),
            "xgboost": np.roll(base, 3),
        }

    def test_paired_row_indices_are_shared_positions_for_every_score_vector(self):
        indices = analysis.sample_indices(
            self.labels, np.random.default_rng(14), "paired_row"
        )
        self.assertEqual(len(indices), len(self.labels))
        for values in self.scores.values():
            self.assertEqual(len(values[indices]), len(self.labels))
        np.testing.assert_array_equal(self.labels[indices], self.labels.take(indices))

    def test_difference_sign_is_first_model_minus_second(self):
        indices = np.arange(len(self.labels))
        expected = (
            analysis.average_precision_score(self.labels, self.scores["random_forest"])
            - analysis.average_precision_score(self.labels, self.scores["mlp"])
        )
        observed = analysis._ap_difference(
            self.labels, self.scores, indices, "random_forest", "mlp"
        )
        self.assertAlmostEqual(observed, float(expected), places=14)

    def test_weighted_tie_group_ap_matches_expanded_sklearn_ap(self):
        indices = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7])
        multiplicities = np.bincount(indices, minlength=len(self.labels))
        for name, values in self.scores.items():
            fast = analysis._AveragePrecisionIndex(self.labels, values).score(multiplicities)
            expected = analysis.average_precision_score(
                self.labels[indices], values[indices]
            )
            self.assertAlmostEqual(fast, float(expected), places=14)

    def test_class_stratified_sample_preserves_observed_class_counts(self):
        indices = analysis.sample_indices(
            self.labels, np.random.default_rng(9), "paired_class_stratified_row"
        )
        self.assertEqual(len(indices), len(self.labels))
        np.testing.assert_array_equal(
            np.bincount(self.labels[indices], minlength=2),
            np.bincount(self.labels, minlength=2),
        )

    def test_moving_blocks_are_contiguous_and_replicate_is_exact_size(self):
        class FixedStarts:
            def __init__(self):
                self.starts = [0, 6, 13, 3, 10]

            def integers(self, low, high, size):
                starts = np.asarray(self.starts[:size])
                if len(starts) != size or not ((starts >= low) & (starts < high)).all():
                    raise AssertionError("test block start is out of range")
                return starts

        labels = np.tile(np.array([0, 0, 1, 0], dtype=np.int8), 5)[:18]
        indices = analysis.sample_indices(
            labels, FixedStarts(), "paired_moving_block", 4
        )
        expected = np.array([
            0, 1, 2, 3,
            6, 7, 8, 9,
            13, 14, 15, 16,
            3, 4, 5, 6,
            10, 11,
        ])
        np.testing.assert_array_equal(indices, expected)
        self.assertEqual(len(indices), len(labels))
        # The final block is truncated to the exact replicate size.
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < len(labels)))

    def test_invalid_replicates_are_counted_by_reason(self):
        labels = np.array([1, 0, 0, 0], dtype=np.int8)
        scores = {name: np.linspace(0.1, 0.9, 4) for name in self.scores}
        with mock.patch.object(
            analysis, "sample_indices", return_value=np.array([1, 2, 3, 1])
        ):
            result = analysis._run_method(
                labels, scores, 3, np.random.default_rng(2), "paired_row"
            )
        self.assertEqual(result["requested_replicates"], 3)
        self.assertEqual(result["valid_replicates"], 0)
        self.assertEqual(result["invalid_replicates"], 3)
        self.assertEqual(result["invalid_replicates_by_reason"], {"no_positive_labels": 3})
        for comparison in result["comparisons"].values():
            self.assertIsNone(comparison["percentile_interval"])

    def test_bootstrap_is_deterministic_and_reports_all_schemes(self):
        labels = np.tile(np.array([0, 0, 1, 0], dtype=np.int8), 100)
        scores = {name: np.linspace(0.001, 0.999, len(labels)) for name in self.scores}
        first = analysis.run_bootstrap(labels, scores, replicates=4, seed=107)
        second = analysis.run_bootstrap(labels, scores, replicates=4, seed=107)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(
            [item["block_length_rows"] for item in first],
            [None, None, 10, 68, 372],
        )

    def test_provenance_records_input_hashes_source_and_resampling_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            score = root / "scores.parquet"
            report = root / "evaluation.json"
            manifest = root / "manifest.json"
            source = root / "analysis.py"
            for path, data in (
                (score, b"score-bytes"),
                (report, b"report-bytes"),
                (manifest, b"manifest-bytes"),
                (source, b"source-bytes"),
            ):
                path.write_bytes(data)
            with (
                mock.patch.object(analysis, "SCORE_PATH", score),
                mock.patch.object(analysis, "EVALUATION_PATH", report),
                mock.patch.object(analysis, "MANIFEST_PATH", manifest),
                mock.patch.object(analysis, "SOURCE_PATH", source),
            ):
                provenance = analysis.collect_provenance(seed=41, replicates=19)
            self.assertEqual(
                provenance["input_hashes"]["score_table_sha256"],
                analysis.sha256_file(score),
            )
            self.assertEqual(
                provenance["input_hashes"]["evaluation_report_sha256"],
                analysis.sha256_file(report),
            )
            self.assertEqual(
                provenance["input_hashes"]["analysis_manifest_sha256"],
                analysis.sha256_file(manifest),
            )
            self.assertEqual(provenance["analysis_source"]["sha256"], analysis.sha256_file(source))
            self.assertEqual(provenance["random_seed"], 41)
            self.assertEqual(provenance["requested_replicates_per_method"], 19)
            self.assertEqual(
                provenance["sampling"]["moving_block"]["block_lengths_rows"],
                [10, 68, 372],
            )


if __name__ == "__main__":
    unittest.main()
