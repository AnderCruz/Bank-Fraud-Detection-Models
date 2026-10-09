"""Synthetic tests for the validation-only evaluation protocol."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src import evaluate_models, inference


def synthetic_features(rows: int = 12) -> pd.DataFrame:
    values = np.arange(rows * 32, dtype=np.float32).reshape(rows, 32) / 100.0
    return pd.DataFrame(
        values,
        columns=inference.EXPECTED_FEATURES,
        index=pd.Index(np.arange(500, 500 + rows) * 3),
    )


class IdentityScaler:
    def transform(self, frame):
        return frame.to_numpy()


class ProbabilityEstimator:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        probability = np.linspace(0.1, 0.9, len(features))
        return np.column_stack((1 - probability, probability))


class NeuralScorer(torch.nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def forward(self, batch):
        if self.kind == "autoencoder":
            reconstruction = batch + 0.1
            return reconstruction, reconstruction
        if batch.ndim == 3:
            batch = batch[:, -1, :]
        return batch[:, :1] / 10.0


def synthetic_loaded_models() -> dict[str, inference.LoadedModel]:
    models = {}
    for name, spec in inference.CANONICAL_MODELS.items():
        neural = spec.implementation in {
            "FraudMLP", "FraudAutoencoder", "LSTMClassifier", "GRUClassifier"
        }
        metadata = {
            "configuration": {"sequence_length": 10},
            "provenance": {"run": {"run_id": spec.pipeline_run_id}},
        }
        models[name] = inference.LoadedModel(
            name=name,
            estimator=NeuralScorer(name).eval() if neural else ProbabilityEstimator(),
            metadata=metadata,
            spec=spec,
            artifact_path=Path(spec.artifact_path),
            metadata_path=Path(spec.metadata_path),
            scaler=IdentityScaler() if spec.preprocessing == "scaled" else None,
        )
    return models


class ValidationEvaluationTests(unittest.TestCase):
    def test_validation_loader_reads_only_validation_files_and_checks_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / evaluate_models.VALIDATION_FEATURES_PATH).parent.mkdir(parents=True)
            (root / evaluate_models.VALIDATION_FEATURES_PATH).write_bytes(b"features")
            (root / evaluate_models.VALIDATION_LABELS_PATH).write_bytes(b"labels")
            features = synthetic_features()
            labels = pd.DataFrame({"Class": [0, 1] * 6}, index=features.index)
            parsed_bytes = []

            def read_parquet(buffer):
                contents = buffer.getvalue()
                parsed_bytes.append(contents)
                return features if contents == b"features" else labels

            with patch.object(evaluate_models.pd, "read_parquet", side_effect=read_parquet):
                loaded_features, loaded_labels, hashes = evaluate_models.load_validation_data(root)

            self.assertEqual(parsed_bytes, [b"features", b"labels"])
            self.assertEqual(tuple(loaded_features.columns), inference.EXPECTED_FEATURES)
            self.assertEqual(len(loaded_labels), len(features))
            self.assertEqual(
                hashes,
                {
                    "data/processed/X_val.parquet": hashlib.sha256(b"features").hexdigest(),
                    "data/processed/y_val.parquet": hashlib.sha256(b"labels").hexdigest(),
                },
            )

            reordered = features[list(inference.EXPECTED_FEATURES)[::-1]]
            with self.assertRaisesRegex(ValueError, "feature names or order"):
                evaluate_models.build_evaluation_record(
                    reordered, loaded_labels, hashes, synthetic_loaded_models()
                )

    def test_metrics_use_average_precision_and_handle_missing_classes(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = evaluate_models.calculate_ranking_metrics(labels, scores)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertIsNone(metrics["undefined_reason"])

        one_class = evaluate_models.calculate_ranking_metrics(np.zeros(4), scores)
        self.assertIsNone(one_class["average_precision"])
        self.assertIsNone(one_class["roc_auc"])
        self.assertTrue(one_class["undefined_reason"])

    def test_ranking_metrics_reject_invalid_labels_lengths_and_scores(self):
        scores = np.array([0.1, 0.9])
        invalid_labels = (
            [0, 0.5],
            [0, np.nan],
            [0, np.inf],
            [0, 2],
        )
        for labels in invalid_labels:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_models.calculate_ranking_metrics(labels, scores)

        with self.assertRaisesRegex(ValueError, "lengths do not match"):
            evaluate_models.calculate_ranking_metrics([0, 1, 0], scores)
        with self.assertRaisesRegex(ValueError, "non-finite values"):
            evaluate_models.calculate_ranking_metrics([0, 1], [0.1, np.nan])

    def test_native_common_cohorts_and_temporal_target_alignment(self):
        features = synthetic_features(12)
        labels = pd.Series([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], index=features.index)
        hashes = {"data/processed/X_val.parquet": "a", "data/processed/y_val.parquet": "b"}
        record = evaluate_models.build_evaluation_record(
            features, labels, hashes, synthetic_loaded_models()
        )

        self.assertEqual(record["evaluation_split"]["native_row_count"], 12)
        self.assertEqual(record["evaluation_split"]["common_row_count"], 3)
        self.assertEqual(record["evaluation_split"]["method"], "chronological, using the existing 70/15/15 partition")
        self.assertIn("not an untouched or blind holdout", record["holdout_disclosure"])
        self.assertIn("not trapezoidal area", record["metric_definitions"]["average_precision"])
        for name, model_record in record["models"].items():
            expected_native_rows = 3 if name in evaluate_models.TEMPORAL_MODELS else 12
            self.assertEqual(
                record["native_cohort_results"]["models"][name]["row_count"],
                expected_native_rows,
            )
            self.assertEqual(
                record["common_cohort_results"]["models"][name]["row_count"], 3
            )
            self.assertEqual(model_record["artifact_path"], inference.CANONICAL_MODELS[name].artifact_path)
            self.assertEqual(model_record["artifact_sha256"], inference.CANONICAL_MODELS[name].artifact_sha256)
            self.assertEqual(model_record["metadata_sha256"], inference.CANONICAL_MODELS[name].metadata_sha256)
            self.assertNotEqual(
                model_record["training_run_id"] is not None,
                model_record["artifact_import_run_id"] is not None,
            )
        temporal = record["models"]["lstm"]
        self.assertEqual(temporal["temporal_alignment"]["sequence_length"], 10)
        self.assertEqual(temporal["temporal_alignment"]["first_n_input_rows_without_prediction"], 9)
        self.assertEqual(temporal["temporal_alignment"]["first_target_row_index"], features.index[9])
        self.assertEqual(temporal["temporal_alignment"]["last_target_row_index"], features.index[-1])
        self.assertEqual(
            record["native_cohort_results"]["models"]["lstm"]["target_class_counts"],
            {"0": 1, "1": 2},
        )

        autoencoder = record["models"]["autoencoder"]
        self.assertEqual(autoencoder["score_type"], "anomaly_score")
        self.assertIn("not a calibrated fraud probability", record["score_interpretation"]["autoencoder"])
        self.assertEqual(
            record["native_cohort_results"]["models"]["autoencoder"]["row_count"],
            len(features),
        )
        json.dumps(record)

    def test_temporal_score_alignment_rejects_missing_common_rows(self):
        indices = pd.Index([5, 6])
        result = inference.ScoreResult(np.array([0.2, 0.8]), "fraud_probability", indices)
        with self.assertRaisesRegex(ValueError, "cannot score every row"):
            evaluate_models._scores_for_rows(result, pd.Index([5, 7]))

    def test_common_cohort_alignment_failure_propagates_from_record_builder(self):
        features = synthetic_features()
        labels = pd.Series([0, 1] * 6, index=features.index)
        models = synthetic_loaded_models()
        predict_scores = inference.predict_scores

        def omit_one_xgboost_row(loaded_model, model_features):
            result = predict_scores(loaded_model, model_features)
            if loaded_model.name == "xgboost":
                return inference.ScoreResult(
                    result.scores[:-1], result.score_type, result.row_indices[:-1]
                )
            return result

        with patch.object(inference, "predict_scores", side_effect=omit_one_xgboost_row):
            with self.assertRaisesRegex(ValueError, "cannot score every row"):
                evaluate_models.build_evaluation_record(
                    features,
                    labels,
                    {
                        "data/processed/X_val.parquet": "features-hash",
                        "data/processed/y_val.parquet": "labels-hash",
                    },
                    models,
                )

    def test_serialized_record_is_deterministic_except_id_and_timestamp(self):
        features = synthetic_features()
        labels = pd.Series([0, 1] * 6, index=features.index)
        hashes = {
            "data/processed/X_val.parquet": "features-hash",
            "data/processed/y_val.parquet": "labels-hash",
        }
        first = evaluate_models.build_evaluation_record(
            features, labels, hashes, synthetic_loaded_models()
        )
        second = evaluate_models.build_evaluation_record(
            features, labels, hashes, synthetic_loaded_models()
        )
        for record in (first, second):
            record.pop("evaluation_id")
            record.pop("created_at_utc")
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )


if __name__ == "__main__":
    unittest.main()
