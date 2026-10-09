"""Generate validation-only per-observation scores and PR-curve outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src import evaluate_models, inference


ROOT = inference.ROOT
logger = logging.getLogger(__name__)
EVALUATION_REPORT = Path(
    "reports/validation_evaluation_1d24658eef4f4241b4f4e4c1e3a00578.json"
)
SUPERVISED_MODELS = tuple(
    name for name, spec in inference.CANONICAL_MODELS.items()
    if spec.score_type == "fraud_probability"
)
ANOMALY_MODELS = ("autoencoder",)
SCORE_FILENAME = "validation_scores.parquet"
CURVE_FILENAME = "precision_recall_curves.json"
MANIFEST_FILENAME = "analysis_manifest.json"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_evaluation_report(report: dict[str, Any], report_bytes: bytes) -> dict[str, Any]:
    """Confirm the supplied aggregate report matches current canonical inputs."""
    if report.get("record_type") != "validation_evaluation":
        raise ValueError("The referenced JSON is not a validation evaluation record")
    if report.get("evaluation_id") != EVALUATION_REPORT.stem.removeprefix(
        "validation_evaluation_"
    ):
        raise ValueError("Evaluation report identifier does not match its configured path")
    split = report.get("evaluation_split", {})
    if tuple(split.get("features", ())) != inference.EXPECTED_FEATURES:
        raise ValueError("Evaluation report feature order differs from the canonical contract")
    if set(report.get("models", {})) != set(inference.CANONICAL_MODELS):
        raise ValueError("Evaluation report does not contain the eight canonical models")
    for name, spec in inference.CANONICAL_MODELS.items():
        item = report["models"][name]
        if (
            item.get("artifact_path") != spec.artifact_path
            or item.get("artifact_sha256") != spec.artifact_sha256
            or item.get("metadata_path") != spec.metadata_path
            or item.get("metadata_sha256") != spec.metadata_sha256
        ):
            raise ValueError(f"Evaluation report canonical artifact identity mismatch: {name}")
    return {
        "evaluation_report_path": str(EVALUATION_REPORT),
        "evaluation_report_id": report["evaluation_id"],
        "evaluation_report_sha256": _sha256_bytes(report_bytes),
        "validation_features_sha256": split["features_sha256"],
        "validation_labels_sha256": split["labels_sha256"],
    }


def build_score_table(
    features: pd.DataFrame,
    labels: pd.Series,
    loaded_models: dict[str, inference.LoadedModel] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Build a compact wide score table retaining each original row index once."""
    if tuple(features.columns) != inference.EXPECTED_FEATURES:
        raise ValueError("Validation feature names or order violate the canonical contract")
    if len(features) != len(labels) or not features.index.equals(labels.index):
        raise ValueError("Validation feature and label rows or indices do not match")
    if not features.index.is_unique:
        raise ValueError("Unique validation row indices are required")
    if loaded_models is None:
        loaded_models = {
            name: inference.load_model(name) for name in inference.CANONICAL_MODELS
        }
    if tuple(loaded_models) != tuple(inference.CANONICAL_MODELS):
        raise ValueError("Loaded model names must match canonical order and include all models")

    results = {
        name: inference.predict_scores(loaded_models[name], features)
        for name in inference.CANONICAL_MODELS
    }
    common_indices = results["lstm"].row_indices
    if not common_indices.equals(results["gru"].row_indices):
        raise ValueError("LSTM and GRU validation target rows do not match")
    if not common_indices.is_unique or (features.index.get_indexer(common_indices) < 0).any():
        raise ValueError("Temporal target rows do not map uniquely to validation observations")

    # One row per validation observation avoids repeating labels across models.
    scores = pd.DataFrame({
        "observation_id": features.index.to_numpy(copy=True),
        "true_label": labels.to_numpy(dtype=np.int64, copy=True),
        "in_common_cohort": features.index.isin(common_indices),
    })
    model_provenance: dict[str, dict[str, Any]] = {}
    for name, spec in inference.CANONICAL_MODELS.items():
        result = results[name]
        if result.score_type != spec.score_type:
            raise ValueError(f"Score type mismatch for {name}")
        if not result.row_indices.is_unique:
            raise ValueError(f"Score rows are not unique for {name}")
        positions = features.index.get_indexer(result.row_indices)
        if (positions < 0).any() or len(positions) != len(result.scores):
            raise ValueError(f"Scores cannot be aligned to validation observations for {name}")
        column = f"score_{name}"
        values = np.full(len(features), np.nan, dtype=np.float64)
        values[positions] = result.scores
        scores[column] = values
        model_provenance[name] = {
            "score_column": column,
            "score_type": spec.score_type,
            "artifact_path": spec.artifact_path,
            "artifact_sha256": spec.artifact_sha256,
            "metadata_path": spec.metadata_path,
            "metadata_sha256": spec.metadata_sha256,
            "preprocessing": spec.preprocessing,
            "pipeline_run_id": spec.pipeline_run_id,
            "training_run_id": spec.training_run_id,
            "artifact_import_run_id": spec.artifact_import_run_id,
            "provenance_caveat": spec.provenance_limitation,
            "native_row_count": int(len(result.row_indices)),
            "common_row_count": int(len(common_indices)),
            "temporal_alignment": (
                {
                    "sequence_length": int(
                        loaded_models[name].metadata["configuration"]["sequence_length"]
                    ),
                    "target": "final row of each ordered sequence",
                    "first_n_input_rows_without_prediction": int(
                        loaded_models[name].metadata["configuration"]["sequence_length"] - 1
                    ),
                }
                if name in {"lstm", "gru"} else None
            ),
        }
        if spec.preprocessing == "scaled":
            model_provenance[name]["scaler_path"] = inference.SCALER_PATH
            model_provenance[name]["scaler_sha256"] = inference.SCALER_SHA256
    return scores, model_provenance


def build_curve_data(
    scores: pd.DataFrame,
    model_provenance: dict[str, dict[str, Any]],
    evaluation_report_id: str,
) -> dict[str, Any]:
    """Calculate continuous-score precision/recall coordinates by cohort."""
    if "observation_id" not in scores or "true_label" not in scores:
        raise ValueError("Score table must contain observation_id and true_label")
    if not scores["observation_id"].is_unique:
        raise ValueError("Score table observation identifiers must be unique")
    curve_groups: dict[str, Any] = {}
    for cohort in ("common", "native"):
        cohort_curves: dict[str, Any] = {}
        common_mask = scores["in_common_cohort"].to_numpy(dtype=bool)
        for name, model in model_provenance.items():
            column = model["score_column"]
            has_score = scores[column].notna().to_numpy()
            mask = common_mask & has_score if cohort == "common" else has_score
            labels = scores.loc[mask, "true_label"].to_numpy(dtype=np.int64)
            values = scores.loc[mask, column].to_numpy(dtype=np.float64)
            if len(labels) == 0 or len(np.unique(labels)) < 2:
                raise ValueError(f"Both target classes are required for {cohort} PR curve: {name}")
            precision, recall, _ = precision_recall_curve(labels, values)
            cohort_curves[name] = {
                "model_name": name,
                "score_type": model["score_type"],
                "score_column": column,
                "observation_count": int(len(labels)),
                "target_class_counts": evaluate_models.class_counts(pd.Series(labels)),
                "average_precision": float(average_precision_score(labels, values)),
                "precision": precision.astype(float).tolist(),
                "recall": recall.astype(float).tolist(),
                "coordinate_definition": "paired precision and recall array entries; thresholds are not selected",
            }
        curve_groups[cohort] = cohort_curves
    return {
        "record_type": "validation_precision_recall_coordinates",
        "evaluation_report_id": evaluation_report_id,
        "metric": "Average Precision and precision_recall_curve coordinates from continuous scores",
        "average_precision_definition": evaluate_models.METRIC_DEFINITIONS["average_precision"],
        "common_cohort_definition": "rows with scores from both LSTM and GRU complete length-10 windows",
        "native_cohort_definition": "all rows supported by each model; temporal models omit the first nine rows",
        "autoencoder_interpretation": "reconstruction-error anomaly score; not a calibrated fraud probability",
        "cohorts": curve_groups,
    }


def _plot_group(
    curves: dict[str, Any],
    cohort: str,
    model_names: tuple[str, ...],
    output_path: Path,
    model_provenance: dict[str, dict[str, Any]],
) -> None:
    """Save a labeled, publication-resolution PR plot in PNG and SVG formats."""
    figure, axis = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for position, name in enumerate(model_names):
        curve = curves["cohorts"][cohort][name]
        if name == "autoencoder":
            label = f"Autoencoder anomaly score (AP={curve['average_precision']:.4f})"
        else:
            label = f"{name.replace('_', ' ').title()} (AP={curve['average_precision']:.4f})"
        axis.plot(
            curve["recall"], curve["precision"],
            color=colors(position % 10), linewidth=1.8, label=label,
        )
    row_counts = [curves["cohorts"][cohort][name]["observation_count"] for name in model_names]
    cohort_label = "Common cohort" if cohort == "common" else "Native cohorts"
    axis.set_title(f"Validation Precision–Recall Curves — {cohort_label}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
    axis.legend(loc="best", frameon=True, fontsize=8)
    if model_names == ANOMALY_MODELS:
        axis.text(
            0.02, 0.02,
            "Score is reconstruction error (anomaly ranking), not fraud probability.",
            transform=axis.transAxes, fontsize=8, va="bottom",
        )
    if cohort == "native" and any(count != row_counts[0] for count in row_counts[1:]):
        axis.text(
            0.02, 0.98,
            "Native sample sizes differ; temporal models omit the first nine rows.",
            transform=axis.transAxes, fontsize=8, va="top",
        )
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _write_outputs(
    output_dir: Path,
    scores: pd.DataFrame,
    curves: dict[str, Any],
    model_provenance: dict[str, dict[str, Any]],
    input_provenance: dict[str, Any],
) -> dict[str, Path]:
    """Persist one score table, curve coordinates, figures, and their manifest."""
    output_dir.mkdir(parents=True, exist_ok=False)
    scores_path = output_dir / SCORE_FILENAME
    curves_path = output_dir / CURVE_FILENAME
    scores.to_parquet(scores_path, index=False)
    curves_path.write_text(
        json.dumps(curves, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plot_specs = (
        ("common", SUPERVISED_MODELS, "pr_common_supervised"),
        ("common", ANOMALY_MODELS, "pr_common_autoencoder_anomaly"),
        ("native", SUPERVISED_MODELS, "pr_native_supervised"),
        ("native", ANOMALY_MODELS, "pr_native_autoencoder_anomaly"),
    )
    plot_paths: list[Path] = []
    for cohort, names, stem in plot_specs:
        base = output_dir / stem
        _plot_group(curves, cohort, names, base, model_provenance)
        plot_paths.extend((base.with_suffix(".png"), base.with_suffix(".svg")))

    manifest = {
        "record_type": "validation_precision_recall_analysis",
        "analysis_id": output_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_report": input_provenance,
        "validation_split": {
            "identifier": evaluate_models.VALIDATION_SPLIT_ID,
            "feature_path": str(evaluate_models.VALIDATION_FEATURES_PATH),
            "label_path": str(evaluate_models.VALIDATION_LABELS_PATH),
            "feature_sha256": input_provenance["validation_features_sha256"],
            "label_sha256": input_provenance["validation_labels_sha256"],
            "feature_order": list(inference.EXPECTED_FEATURES),
        },
        "score_table": {
            "path": SCORE_FILENAME,
            "sha256": _sha256_file(scores_path),
            "format": "wide Parquet; one row per validation observation",
            "observation_id_column": "observation_id (original validation DataFrame index)",
            "label_column": "true_label",
            "common_cohort_column": "in_common_cohort",
            "model_score_columns": {
                name: model["score_column"] for name, model in model_provenance.items()
            },
        },
        "curve_coordinates": {
            "path": CURVE_FILENAME,
            "sha256": _sha256_file(curves_path),
            "precision_recall_curve_implementation": "sklearn.metrics.precision_recall_curve",
        },
        "figures": [
            {"path": path.name, "sha256": _sha256_file(path)} for path in plot_paths
        ],
        "models": model_provenance,
        "cohort_counts": {
            "native_validation_observations": int(len(scores)),
            "common_observations": int(scores["in_common_cohort"].sum()),
            "common_fraud_cases": int(
                scores.loc[scores["in_common_cohort"], "true_label"].sum()
            ),
        },
        "score_semantics": {
            "supervised_models": "fraud probability for class 1",
            "autoencoder": "reconstruction-error anomaly score; not a probability",
        },
        "threshold_policy": "No threshold selected; only continuous-score PR coordinates are saved.",
        "software": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "pyarrow")
        },
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "score_table": scores_path,
        "curve_coordinates": curves_path,
        "manifest": manifest_path,
        "figures": plot_paths,
    }


def run_analysis() -> tuple[Path, dict[str, Path]]:
    """Generate auditable validation scores and PR outputs without training."""
    report_path = ROOT / EVALUATION_REPORT
    report_bytes = report_path.read_bytes()
    evaluation_report = json.loads(report_bytes)
    input_provenance = validate_evaluation_report(evaluation_report, report_bytes)
    features, labels, data_hashes = evaluate_models.load_validation_data(ROOT)
    if data_hashes[str(evaluate_models.VALIDATION_FEATURES_PATH)] != input_provenance[
        "validation_features_sha256"
    ] or data_hashes[str(evaluate_models.VALIDATION_LABELS_PATH)] != input_provenance[
        "validation_labels_sha256"
    ]:
        raise ValueError("Current validation data hashes differ from the recorded evaluation")
    loaded_models = {
        name: inference.load_model(name) for name in inference.CANONICAL_MODELS
    }
    scores, model_provenance = build_score_table(features, labels, loaded_models)
    curves = build_curve_data(scores, model_provenance, evaluation_report["evaluation_id"])

    analysis_id = uuid.uuid4().hex
    output_dir = ROOT / "reports" / "validation_analysis" / evaluation_report["evaluation_id"] / analysis_id
    outputs = _write_outputs(
        output_dir, scores, curves, model_provenance, input_provenance
    )
    logger.info("Validation curve analysis saved under reports/validation_analysis/%s/%s",
                evaluation_report["evaluation_id"], analysis_id)
    return output_dir, outputs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_analysis()


if __name__ == "__main__":
    main()
