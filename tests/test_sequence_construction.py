"""Focused tests for temporal sequence construction."""

import unittest

import pandas as pd

from src.train_models import make_sequences


class SequenceConstructionTests(unittest.TestCase):
    def test_windows_stay_within_input_split_and_target_final_row(self) -> None:
        train_features = pd.DataFrame({"feature": [10, 11, 12, 13]})
        train_labels = pd.Series([0, 1, 0, 1])
        validation_features = pd.DataFrame({"feature": [100, 101, 102, 103]})
        validation_labels = pd.Series([1, 0, 1, 0])

        train_windows, train_targets = make_sequences(
            train_features, train_labels, length=3
        )
        validation_windows, validation_targets = make_sequences(
            validation_features, validation_labels, length=3
        )

        self.assertEqual(train_windows.tolist(), [[[10.0], [11.0], [12.0]], [[11.0], [12.0], [13.0]]])
        self.assertEqual(train_targets.tolist(), [0, 1])
        self.assertEqual(
            validation_windows.tolist(),
            [[[100.0], [101.0], [102.0]], [[101.0], [102.0], [103.0]]],
        )
        self.assertEqual(validation_targets.tolist(), [1, 0])
        self.assertTrue((train_windows < 100).all())
        self.assertTrue((validation_windows >= 100).all())


if __name__ == "__main__":
    unittest.main()
