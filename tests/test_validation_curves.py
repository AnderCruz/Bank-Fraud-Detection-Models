"""Synthetic tests for saved validation scores and PR curves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from src import analyze_validation_curves as analysis
from src import inference


def features_frame(rows: int = 12) -> pd.DataFrame:
    values = np.arange(rows * 32, dtype=np.float32).reshape(rows, 32) / 50.0
    return pd.DataFrame(
        values,
        columns=inference.EXPECTED_FEATURES,
        index=pd.Index(np.arange(900, 900 + rows) * 11),
    )


class IdentityScaler:
    def transform(self, frame):
        return frame.to_numpy()


class SyntheticClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        positive = np.linspace(0.05, 0.95, len(frame))
        return np.column_stack((1.0 - positive, positive))


class SyntheticNetwork(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, batch):
        if self.name == "autoencoder":
            reconstructed = batch + 0.05
            return reconstructed, reconstructed
        if batch.ndim == 3:
            batch = batch[:, -1, :]
        return batch[:, :1] / 20.0


def loaded_models():
    result = {}
    for name, spec in inference.CANONICAL_MODELS.items():
        is_torch = spec.implementation in {
            "FraudMLP", "FraudAutoencoder", "LSTMClassifier", "GRUClassifier"
        }
        result[name] = inference.LoadedModel(
            name=name,
            estimator=SyntheticNetwork(name).eval() if is_torch else SyntheticClassifier(),
            metadata={"configuration": {"sequence_length": 10}},
            spec=spec,
            artifact_path=Path(spec.artifact_path),
            metadata_path=Path(spec.metadata_path),
            scaler=IdentityScaler() if spec.preprocessing == "scaled" else None,
        )
    return result


class ValidationCurveTests(unittest.TestCase):
    def setUp(self):
        self.features = features_frame()
        self.labels = pd.Series(
            [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
            index=self.features.index,
        )
        self.scores, self.model_info = analysis.build_score_table(
            self.features, self.labels, loaded_models()
        )

    def test_score_table_preserves_indices_and_temporal_eligibility(self):
        self.assertEqual(self.scores["observation_id"].tolist(), self.features.index.tolist())
        self.assertEqual(self.scores["true_label"].tolist(), self.labels.tolist())
        self.assertEqual(self.scores["in_common_cohort"].sum(), 3)
        np.testing.assert_array_equal(
            self.scores.loc[self.scores["in_common_cohort"], "observation_id"],
            self.features.index[9:],
        )
        for name in inference.CANONICAL_MODELS:
            score_column = self.model_info[name]["score_column"]
            if name in {"lstm", "gru"}:
                self.assertTrue(self.scores[score_column].iloc[:9].isna().all())
                self.assertTrue(self.scores[score_column].iloc[9:].notna().all())
                self.assertEqual(self.model_info[name]["temporal_alignment"]["sequence_length"], 10)
            else:
                self.assertTrue(self.scores[score_column].notna().all())
        self.assertEqual(self.model_info["autoencoder"]["score_type"], "anomaly_score")
        for name in analysis.SUPERVISED_MODELS:
            self.assertEqual(self.model_info[name]["score_type"], "fraud_probability")

    def test_pr_coordinates_use_continuous_scores_for_both_cohorts(self):
        curves = analysis.build_curve_data(
            self.scores, self.model_info, "synthetic-evaluation-id"
        )
        self.assertEqual(set(curves["cohorts"]), {"native", "common"})
        self.assertEqual(set(curves["cohorts"]["common"]), set(inference.CANONICAL_MODELS))
        self.assertEqual(len([m for m in curves["cohorts"]["common"] if m != "autoencoder"]), 7)
        self.assertIn("not a calibrated fraud probability", curves["autoencoder_interpretation"])

        for cohort in ("native", "common"):
            for name, curve in curves["cohorts"][cohort].items():
                score_column = self.model_info[name]["score_column"]
                mask = self.scores["in_common_cohort"] if cohort == "common" else self.scores[score_column].notna()
                labels = self.scores.loc[mask, "true_label"].to_numpy()
                values = self.scores.loc[mask, score_column].to_numpy()
                expected_precision, expected_recall, _ = precision_recall_curve(labels, values)
                self.assertEqual(curve["precision"], expected_precision.tolist())
                self.assertEqual(curve["recall"], expected_recall.tolist())
                self.assertEqual(
                    curve["average_precision"], average_precision_score(labels, values)
                )
                self.assertEqual(curve["score_type"], self.model_info[name]["score_type"])

    def test_report_provenance_is_bound_to_report_and_canonical_registry(self):
        report_id = analysis.EVALUATION_REPORT.stem.removeprefix("validation_evaluation_")
        report = {
            "record_type": "validation_evaluation",
            "evaluation_id": report_id,
            "evaluation_split": {
                "features": list(inference.EXPECTED_FEATURES),
                "features_sha256": "features-digest",
                "labels_sha256": "labels-digest",
            },
            "models": {
                name: {
                    "artifact_path": spec.artifact_path,
                    "artifact_sha256": spec.artifact_sha256,
                    "metadata_path": spec.metadata_path,
                    "metadata_sha256": spec.metadata_sha256,
                }
                for name, spec in inference.CANONICAL_MODELS.items()
            },
        }
        report_bytes = json.dumps(report).encode()
        provenance = analysis.validate_evaluation_report(report, report_bytes)
        self.assertEqual(provenance["evaluation_report_sha256"], hashlib.sha256(report_bytes).hexdigest())
        self.assertEqual(provenance["validation_features_sha256"], "features-digest")

    def test_saved_outputs_retain_score_rows_and_hashes(self):
        curves = analysis.build_curve_data(self.scores, self.model_info, "synthetic-id")
        input_provenance = {
            "evaluation_report_path": "reports/example.json",
            "evaluation_report_id": "synthetic-id",
            "evaluation_report_sha256": "report-hash",
            "validation_features_sha256": "features-hash",
            "validation_labels_sha256": "labels-hash",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "analysis-v1"
            paths = analysis._write_outputs(
                output, self.scores, curves, self.model_info, input_provenance
            )
            restored = pd.read_parquet(paths["score_table"])
            pd.testing.assert_frame_equal(restored, self.scores)
            manifest = json.loads(paths["manifest"].read_text())
            self.assertEqual(
                manifest["score_table"]["sha256"],
                hashlib.sha256(paths["score_table"].read_bytes()).hexdigest(),
            )
            self.assertEqual(len(paths["figures"]), 8)
            self.assertTrue(all(path.is_file() for path in paths["figures"]))


if __name__ == "__main__":
    unittest.main()
