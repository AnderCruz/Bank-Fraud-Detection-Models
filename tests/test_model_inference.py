"""Synthetic tests for the shared canonical inference implementation."""

from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src import inference, train_models


ROOT = Path(__file__).resolve().parents[1]
FEATURES = list(inference.EXPECTED_FEATURES)
CANONICAL_ARTIFACTS = {
    name: ROOT / spec.artifact_path
    for name, spec in inference.CANONICAL_MODELS.items()
}


def synthetic_raw_features(rows: int = 13) -> pd.DataFrame:
    """Build deterministic ordered inputs without reading any dataset."""
    rng = np.random.default_rng(20261009)
    values = rng.normal(size=(rows, len(FEATURES))).astype(np.float64)
    amount = np.linspace(1.0, 500.0, rows)
    values[:, 28] = amount
    values[:, 29] = np.log1p(amount)
    values[:, 30] = np.arange(rows) * 7.0
    values[:, 31] = np.arange(rows) % 24
    return pd.DataFrame(values, columns=FEATURES, index=pd.Index(range(200, 200 + rows)))


class CanonicalInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_features = synthetic_raw_features()
        cls.loaded = {
            name: inference.load_model(name)
            for name in inference.CANONICAL_MODELS
        }

    def test_canonical_artifacts_and_metadata_are_verified(self):
        for name, loaded_model in self.loaded.items():
            with self.subTest(model=name):
                spec = inference.CANONICAL_MODELS[name]
                self.assertTrue(CANONICAL_ARTIFACTS[name].is_file())
                self.assertEqual(loaded_model.metadata["model_name"], name)
                self.assertEqual(loaded_model.metadata["features"], FEATURES)
                self.assertEqual(loaded_model.spec, spec)

    def test_missing_artifact_and_metadata_fail_clearly(self):
        original = inference.CANONICAL_MODELS["random_forest"]
        with tempfile.TemporaryDirectory() as temporary:
            missing = replace(
                original,
                artifact_path=f"{temporary}/missing.joblib",
                metadata_path=f"{temporary}/missing_metadata.json",
            )
            with patch.dict(inference.CANONICAL_MODELS, {"random_forest": missing}):
                with self.assertRaisesRegex(FileNotFoundError, "artifact is missing"):
                    inference.load_model("random_forest")

            existing_artifact_missing_metadata = replace(
                original,
                artifact_path=original.artifact_path,
                metadata_path=f"{temporary}/missing_metadata.json",
            )
            with patch.dict(
                inference.CANONICAL_MODELS,
                {"random_forest": existing_artifact_missing_metadata},
            ):
                with self.assertRaisesRegex(FileNotFoundError, "metadata is missing"):
                    inference.load_model("random_forest")

    def test_artifact_and_metadata_hash_mismatches_fail(self):
        original = inference.CANONICAL_MODELS["random_forest"]
        for field in ("artifact_sha256", "metadata_sha256"):
            with self.subTest(field=field):
                mismatched = replace(original, **{field: "0" * 64})
                with patch.dict(
                    inference.CANONICAL_MODELS,
                    {"random_forest": mismatched},
                ):
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        inference.load_model("random_forest")

    def test_required_scaler_is_hash_checked_and_missing_scaler_fails(self):
        with patch.object(inference, "SCALER_PATH", "data/processed/missing_scaler.joblib"):
            with self.assertRaisesRegex(FileNotFoundError, "RobustScaler is missing"):
                inference.load_model("logistic_regression")
        with patch.object(inference, "SCALER_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "RobustScaler SHA-256 mismatch"):
                inference.load_model("logistic_regression")

    def test_unknown_model_and_feature_order_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model name"):
            inference.load_model("unknown_model")
        model = self.loaded["random_forest"]
        reordered = self.raw_features[FEATURES[::-1]]
        with self.assertRaisesRegex(ValueError, "feature names or order"):
            inference.predict_scores(model, reordered)

    def test_missing_or_unknown_preprocessing_metadata_is_rejected(self):
        for metadata in ({}, {"configuration": {"input": "mystery representation"}}):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "Preprocessing metadata is missing"):
                    inference._recorded_preprocessing("random_forest", metadata)

    def test_class_one_probability_uses_estimator_class_mapping(self):
        class ReversedClassEstimator:
            classes_ = np.array([1, 0])

            def predict_proba(self, features):
                self.asserted_features = features
                return np.array([[0.83, 0.17], [0.24, 0.76]])

        estimator = ReversedClassEstimator()
        spec = inference.CANONICAL_MODELS["random_forest"]
        loaded_model = inference.LoadedModel(
            name="random_forest",
            estimator=estimator,
            metadata={},
            spec=spec,
            artifact_path=CANONICAL_ARTIFACTS["random_forest"],
            metadata_path=ROOT / spec.metadata_path,
        )
        features = self.raw_features.iloc[:2]
        result = inference.predict_scores(loaded_model, features)
        np.testing.assert_array_equal(result.scores, np.array([0.83, 0.24]))
        self.assertTrue(result.row_indices.equals(features.index))

    def test_classical_inference_uses_raw_or_saved_scaled_inputs(self):
        representations = {
            "random_forest": "unscaled",
            "xgboost": "unscaled",
            "logistic_regression": "scaled",
            "lightgbm": "unscaled",
        }
        for name, representation in representations.items():
            with self.subTest(model=name):
                loaded_model = self.loaded[name]
                self.assertEqual(loaded_model.spec.preprocessing, representation)
                result = inference.predict_scores(loaded_model, self.raw_features)
                self.assertEqual(result.score_type, "fraud_probability")
                self.assertEqual(result.scores.shape, (len(self.raw_features),))
                self.assertTrue(np.isfinite(result.scores).all())
                self.assertTrue(((result.scores >= 0) & (result.scores <= 1)).all())
                self.assertTrue(result.row_indices.equals(self.raw_features.index))

                estimator = loaded_model.estimator
                inference_features = self.raw_features
                if representation == "scaled":
                    inference_features = pd.DataFrame(
                        loaded_model.scaler.transform(self.raw_features),
                        columns=FEATURES,
                        index=self.raw_features.index,
                    )
                class_one = list(estimator.classes_).index(1)
                expected = estimator.predict_proba(inference_features)[:, class_one]
                np.testing.assert_allclose(result.scores, expected)

        scaler = self.loaded["logistic_regression"].scaler
        center_before = scaler.center_.copy()
        scale_before = scaler.scale_.copy()
        inference.predict_scores(self.loaded["logistic_regression"], self.raw_features)
        np.testing.assert_array_equal(scaler.center_, center_before)
        np.testing.assert_array_equal(scaler.scale_, scale_before)
        self.assertIsNone(self.loaded["random_forest"].scaler)

    def test_neural_probabilities_strict_load_and_eval_mode(self):
        for name in ("mlp", "lstm", "gru"):
            with self.subTest(model=name):
                loaded_model = self.loaded[name]
                self.assertFalse(loaded_model.estimator.training)
                result = inference.predict_scores(loaded_model, self.raw_features)
                self.assertEqual(result.score_type, "fraud_probability")
                self.assertTrue(np.isfinite(result.scores).all())
                self.assertTrue(((result.scores >= 0) & (result.scores <= 1)).all())

        mlp = self.loaded["mlp"]
        scaled = mlp.scaler.transform(self.raw_features)
        with torch.inference_mode():
            expected = torch.sigmoid(
                mlp.estimator(torch.as_tensor(scaled, dtype=torch.float32))
            ).squeeze(1).numpy()
        np.testing.assert_allclose(
            inference.predict_scores(mlp, self.raw_features).scores, expected
        )

    def test_autoencoder_returns_one_per_row_mean_squared_error(self):
        loaded_model = self.loaded["autoencoder"]
        self.assertFalse(loaded_model.estimator.training)
        result = inference.predict_scores(loaded_model, self.raw_features)
        self.assertEqual(result.score_type, "anomaly_score")
        self.assertEqual(result.scores.shape, (len(self.raw_features),))
        self.assertTrue(np.isfinite(result.scores).all())
        self.assertTrue((result.scores >= 0).all())

        scaled = loaded_model.scaler.transform(self.raw_features)
        values = torch.as_tensor(scaled, dtype=torch.float32)
        with torch.inference_mode():
            reconstruction, _ = loaded_model.estimator(values)
            expected = ((reconstruction - values) ** 2).mean(dim=1).numpy()
        np.testing.assert_allclose(result.scores, expected)

    def test_temporal_outputs_match_window_count_and_final_row_alignment(self):
        labels = pd.Series(np.arange(len(self.raw_features)), index=self.raw_features.index)
        expected_windows, expected_targets = train_models.make_sequences(
            self.raw_features,
            labels,
            length=10,
        )
        self.assertEqual(expected_windows.shape, (4, 10, 32))
        self.assertEqual(expected_targets.tolist(), list(range(9, 13)))
        np.testing.assert_allclose(expected_windows[:, -1, :], self.raw_features.to_numpy()[9:])

        scaled_features = pd.DataFrame(
            self.loaded["lstm"].scaler.transform(self.raw_features),
            columns=FEATURES,
            index=self.raw_features.index,
        )
        direct_windows = train_models.make_feature_sequences(scaled_features, 10)
        self.assertEqual(direct_windows.shape, (4, 10, 32))

        for name in ("lstm", "gru"):
            with self.subTest(model=name):
                result = inference.predict_scores(self.loaded[name], self.raw_features)
                self.assertEqual(result.scores.shape, (len(self.raw_features) - 9,))
                self.assertTrue(result.row_indices.equals(self.raw_features.index[9:]))
                self.assertTrue(np.isfinite(result.scores).all())
                self.assertTrue(((result.scores >= 0) & (result.scores <= 1)).all())
                with torch.inference_mode():
                    logits = self.loaded[name].estimator(
                        torch.as_tensor(direct_windows, dtype=torch.float32)
                    )
                    expected_scores = torch.sigmoid(logits).numpy()
                np.testing.assert_allclose(result.scores, expected_scores)

        too_short = inference.predict_scores(
            self.loaded["lstm"], self.raw_features.iloc[:9]
        )
        self.assertEqual(too_short.scores.shape, (0,))
        self.assertEqual(len(too_short.row_indices), 0)

    def test_training_runs_and_artifact_imports_remain_distinct(self):
        for name, spec in inference.CANONICAL_MODELS.items():
            with self.subTest(model=name):
                self.assertNotEqual(
                    spec.training_run_id is not None,
                    spec.artifact_import_run_id is not None,
                )
                if spec.artifact_import_run_id:
                    run = inference.MlflowClient().get_run(spec.artifact_import_run_id)
                    self.assertEqual(run.data.tags.get("record_type"), "artifact_import")
                    self.assertEqual(run.data.tags.get("training_run_recovered"), "false")
                    self.assertNotIn("run.selected_models", run.data.params)
                    self.assertFalse(run.data.metrics)
                else:
                    run = inference.MlflowClient().get_run(spec.training_run_id)
                    self.assertEqual(run.data.tags.get("record_type"), None)
                    self.assertEqual(
                        run.data.tags.get("pipeline_run_id"), spec.pipeline_run_id
                    )

        self.assertIn("not conclusively linked", self.loaded["lightgbm"].spec.provenance_limitation)


if __name__ == "__main__":
    unittest.main()
