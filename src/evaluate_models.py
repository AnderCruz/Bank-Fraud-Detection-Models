"""Reproducible validation-only evaluation for canonical fraud models.

This entry point reads only the existing validation feature and label files.
It does not fit models, refit preprocessing, or write to MLflow.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import logging
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import inference


ROOT = inference.ROOT
logger = logging.getLogger(__name__)
MODEL_ORDER = tuple(inference.CANONICAL_MODELS)
VALIDATION_FEATURES_PATH = Path("data/processed/X_val.parquet")
VALIDATION_LABELS_PATH = Path("data/processed/y_val.parquet")
VALIDATION_SPLIT_ID = "chronological_validation_split"
TEMPORAL_MODELS = ("lstm", "gru")
METRIC_DEFINITIONS = {
    "average_precision": (
        "sklearn.metrics.average_precision_score over continuous scores; "
        "this is not trapezoidal area under the precision-recall curve"
    ),
    "roc_auc": (
        "sklearn.metrics.roc_auc_score over continuous scores; undefined when "
        "the cohort contains only one target class"
    ),
}


def load_validation_data(
    root: Path = ROOT,
) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """Load and validate only the project's persisted validation split."""
    features_path = root / VALIDATION_FEATURES_PATH
    labels_path = root / VALIDATION_LABELS_PATH
    for path in (features_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required validation data is missing: {path}")

    # Read each source once so hashes and parsed frames refer to identical bytes.
    features_bytes = features_path.read_bytes()
    labels_bytes = labels_path.read_bytes()
    data_hashes = {
        str(VALIDATION_FEATURES_PATH): hashlib.sha256(features_bytes).hexdigest(),
        str(VALIDATION_LABELS_PATH): hashlib.sha256(labels_bytes).hexdigest(),
    }
    features = pd.read_parquet(io.BytesIO(features_bytes))
    labels_frame = pd.read_parquet(io.BytesIO(labels_bytes))
    if tuple(features.columns) != inference.EXPECTED_FEATURES:
        raise ValueError("Validation feature names or order violate the canonical contract")
    if "Class" not in labels_frame.columns:
        raise ValueError("Validation labels must contain the 'Class' column")
    labels = labels_frame["Class"]
    if len(features) != len(labels) or not features.index.equals(labels.index):
        raise ValueError("Validation feature and label rows or indices do not match")
    if not features.index.is_unique:
        raise ValueError("Validation row index must be unique for score alignment")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("Validation targets must be non-missing binary labels 0/1")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Validation features contain non-finite values")
    return features, labels.astype("int64"), data_hashes


def class_counts(labels: pd.Series) -> dict[str, int]:
    """Return explicit counts for both binary classes, including absent classes."""
    counts = labels.value_counts()
    return {str(label): int(counts.get(label, 0)) for label in (0, 1)}


def calculate_ranking_metrics(
    labels: pd.Series | np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """Calculate ranking metrics; return nulls when a cohort lacks either class."""
    try:
        raw_target = np.asarray(labels, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("Targets must be finite binary labels 0/1") from error
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if raw_target.shape != values.shape:
        raise ValueError("Target and score lengths do not match")
    if not np.isfinite(raw_target).all():
        raise ValueError("Targets must be finite binary labels 0/1")
    if not np.isin(raw_target, (0.0, 1.0)).all():
        raise ValueError("Targets must contain only binary labels 0/1")
    if not np.isfinite(values).all():
        raise ValueError("Scores contain non-finite values")
    target = raw_target.astype(np.int64)
    if len(np.unique(target)) < 2:
        return {
            "average_precision": None,
            "roc_auc": None,
            "undefined_reason": "Both target classes are required for this evaluation.",
        }
    return {
        "average_precision": float(average_precision_score(target, values)),
        "roc_auc": float(roc_auc_score(target, values)),
        "undefined_reason": None,
    }


def _labels_for_rows(labels: pd.Series, row_indices: pd.Index) -> pd.Series:
    """Align labels by the original row index and fail on missing/ambiguous rows."""
    if not labels.index.is_unique or not row_indices.is_unique:
        raise ValueError("Unique row indices are required for cohort alignment")
    positions = labels.index.get_indexer(row_indices)
    if (positions < 0).any():
        raise ValueError("A model score refers to a row outside the validation labels")
    return labels.iloc[positions]


def _scores_for_rows(
    result: inference.ScoreResult,
    requested_indices: pd.Index,
) -> np.ndarray:
    """Select scores in the requested original-row order."""
    if not result.row_indices.is_unique or not requested_indices.is_unique:
        raise ValueError("Unique row indices are required for score alignment")
    positions = result.row_indices.get_indexer(requested_indices)
    if (positions < 0).any():
        raise ValueError("A model cannot score every row in the common cohort")
    return result.scores[positions]


def _json_index_value(index: pd.Index, position: int) -> Any:
    """Convert a pandas/numpy index scalar to a JSON-native value."""
    value = index[position]
    return value.item() if isinstance(value, np.generic) else value


def _git_value(*arguments: str) -> str | None:
    """Read Git provenance without making evaluation depend on Git availability."""
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True, capture_output=True,
            text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _environment() -> dict[str, Any]:
    dependencies: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "lightgbm"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "dependencies": dependencies,
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": (
                None if (status := _git_value("status", "--porcelain")) is None
                else bool(status)
            ),
        },
    }


def _model_record(
    name: str,
    loaded: inference.LoadedModel,
    result: inference.ScoreResult,
) -> dict[str, Any]:
    spec = loaded.spec
    metadata_run = loaded.metadata.get("provenance", {}).get("run", {})
    return {
        "model_name": name,
        "artifact_path": spec.artifact_path,
        "artifact_sha256": spec.artifact_sha256,
        "metadata_path": spec.metadata_path,
        "metadata_sha256": spec.metadata_sha256,
        "scaler": (
            {"path": inference.SCALER_PATH, "sha256": inference.SCALER_SHA256}
            if spec.preprocessing == "scaled" else None
        ),
        "features": list(inference.EXPECTED_FEATURES),
        "preprocessing": spec.preprocessing,
        "score_type": result.score_type,
        "training_run_id": spec.training_run_id,
        "artifact_import_run_id": spec.artifact_import_run_id,
        "pipeline_run_id": spec.pipeline_run_id,
        "provenance_caveat": spec.provenance_limitation,
        "metadata_run_id": metadata_run.get("run_id"),
        "temporal_alignment": (
            {
                "sequence_length": int(loaded.metadata["configuration"]["sequence_length"]),
                "target": "final row of each ordered sequence",
                "first_n_input_rows_without_prediction": int(
                    loaded.metadata["configuration"]["sequence_length"] - 1
                ),
                "first_target_row_index": (
                    _json_index_value(result.row_indices, 0)
                    if len(result.row_indices) else None
                ),
                "last_target_row_index": (
                    _json_index_value(result.row_indices, -1)
                    if len(result.row_indices) else None
                ),
            }
            if name in TEMPORAL_MODELS else None
        ),
    }


def build_evaluation_record(
    features: pd.DataFrame,
    labels: pd.Series,
    data_hashes: dict[str, str],
    loaded_models: dict[str, inference.LoadedModel] | None = None,
) -> dict[str, Any]:
    """Score all canonical models and return a JSON-serializable record."""
    if tuple(features.columns) != inference.EXPECTED_FEATURES:
        raise ValueError("Validation feature names or order violate the canonical contract")
    if len(features) != len(labels) or not features.index.equals(labels.index):
        raise ValueError("Validation feature and label rows or indices do not match")
    if not features.index.is_unique:
        raise ValueError("Unique validation row indices are required")
    if loaded_models is None:
        loaded_models = {name: inference.load_model(name) for name in MODEL_ORDER}
    if tuple(loaded_models) != MODEL_ORDER:
        raise ValueError("Loaded model names must match canonical order and include all models")

    score_results = {
        name: inference.predict_scores(loaded_models[name], features)
        for name in MODEL_ORDER
    }
    lstm_rows = score_results["lstm"].row_indices
    gru_rows = score_results["gru"].row_indices
    if not lstm_rows.equals(gru_rows):
        raise ValueError("LSTM and GRU validation target rows do not match")
    common_indices = lstm_rows
    _labels_for_rows(labels, common_indices)

    model_records = {
        name: _model_record(name, loaded_models[name], score_results[name])
        for name in MODEL_ORDER
    }
    native_results = {}
    common_results = {}
    for name in MODEL_ORDER:
        result = score_results[name]
        native_labels = _labels_for_rows(labels, result.row_indices)
        common_labels = _labels_for_rows(labels, common_indices)
        native_results[name] = {
            "row_count": int(len(result.row_indices)),
            "target_class_counts": class_counts(native_labels),
            "metrics": calculate_ranking_metrics(native_labels, result.scores),
        }
        common_results[name] = {
            "row_count": int(len(common_indices)),
            "target_class_counts": class_counts(common_labels),
            "metrics": calculate_ranking_metrics(
                common_labels, _scores_for_rows(result, common_indices)
            ),
        }
    return {
        "evaluation_id": uuid.uuid4().hex,
        "record_type": "validation_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": {
            "identifier": VALIDATION_SPLIT_ID,
            "method": "chronological, using the existing 70/15/15 partition",
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "validation_end_fraction": 0.85,
            "features_path": str(VALIDATION_FEATURES_PATH),
            "labels_path": str(VALIDATION_LABELS_PATH),
            "features_sha256": data_hashes[str(VALIDATION_FEATURES_PATH)],
            "labels_sha256": data_hashes[str(VALIDATION_LABELS_PATH)],
            "feature_count": len(inference.EXPECTED_FEATURES),
            "features": list(inference.EXPECTED_FEATURES),
            "native_row_count": int(len(features)),
            "native_target_class_counts": class_counts(labels),
            "common_row_count": int(len(common_indices)),
            "common_target_class_counts": class_counts(_labels_for_rows(labels, common_indices)),
            "common_cohort_definition": "validation rows targeted by complete length-10 temporal windows",
        },
        "native_cohort_results": {
            "definition": "all validation rows supported by each model",
            "models": native_results,
        },
        "common_cohort_results": {
            "definition": "validation rows targeted by both temporal models' complete length-10 windows",
            "row_count": int(len(common_indices)),
            "target_class_counts": class_counts(_labels_for_rows(labels, common_indices)),
            "models": common_results,
        },
        "holdout_disclosure": (
            "This evaluation reads validation data only. The test split was used in prior "
            "analysis and is not an untouched or blind holdout."
        ),
        "score_interpretation": {
            "autoencoder": "reconstruction-error anomaly score; not a calibrated fraud probability",
            "supervised_models": "fraud probability for class 1",
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "metric_implementation_versions": {
            "scikit-learn": importlib.metadata.version("scikit-learn"),
        },
        "environment": _environment(),
        "models": model_records,
    }


def evaluate_validation(root: Path = ROOT) -> dict[str, Any]:
    """Load only the persisted validation split and build its evaluation record."""
    features, labels, hashes = load_validation_data(root)
    return build_evaluation_record(features, labels, hashes)


def main() -> Path:
    """Write a uniquely named validation evaluation record; no MLflow run is created."""
    record = evaluate_validation()
    output_directory = ROOT / "reports"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"validation_evaluation_{record['evaluation_id']}.json"
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Validation evaluation record saved: %s", output_path.name)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
