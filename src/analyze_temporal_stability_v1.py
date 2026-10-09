"""Summarize temporal validation stability from the saved score table only."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ID = "1d24658eef4f4241b4f4e4c1e3a00578"
SOURCE_ANALYSIS_ID = "d034e61aaf394581ab9ffb8e3c471089"
SOURCE_ANALYSIS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "validation_analysis"
    / EVALUATION_ID
    / SOURCE_ANALYSIS_ID
)
SCORE_TABLE_PATH = SOURCE_ANALYSIS_DIR / "validation_scores.parquet"
SOURCE_MANIFEST_PATH = SOURCE_ANALYSIS_DIR / "analysis_manifest.json"
EVALUATION_REPORT_PATH = PROJECT_ROOT / "reports" / f"validation_evaluation_{EVALUATION_ID}.json"
MODEL_NAMES = (
    "random_forest",
    "xgboost",
    "logistic_regression",
    "lightgbm",
    "mlp",
    "autoencoder",
    "lstm",
    "gru",
)
EXPECTED_COHORT_ROWS = 42_712
EXPECTED_FRAUDS = 56
EXPECTED_LEGITIMATE = 42_656
EXPECTED_FIRST_ID = 199_373
EXPECTED_LAST_ID = 242_084
SEGMENT_SIZE = 10_678
AGGREGATE_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(arguments: list[str]) -> str | None:
    """Read Git provenance safely; return None when Git is unavailable."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def package_version(distribution: str) -> str | None:
    """Read a distribution version through standard package metadata."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def validate_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, str]]:
    """Verify the saved score table and its provenance before metric work."""
    for path in (SCORE_TABLE_PATH, SOURCE_MANIFEST_PATH, EVALUATION_REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved validation input is missing: {path}")

    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    evaluation_report = json.loads(EVALUATION_REPORT_PATH.read_text(encoding="utf-8"))
    input_hashes = {
        "score_table_sha256": sha256_file(SCORE_TABLE_PATH),
        "source_analysis_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "evaluation_report_sha256": sha256_file(EVALUATION_REPORT_PATH),
    }

    if evaluation_report.get("evaluation_id") != EVALUATION_ID:
        raise ValueError("Evaluation report ID does not match the expected evaluation.")
    if source_manifest.get("analysis_id") != SOURCE_ANALYSIS_ID:
        raise ValueError("Source analysis manifest ID does not match the expected analysis.")
    if source_manifest.get("evaluation_report", {}).get("evaluation_report_id") != EVALUATION_ID:
        raise ValueError("Source analysis manifest references a different evaluation report.")
    if source_manifest.get("score_table", {}).get("sha256") != input_hashes["score_table_sha256"]:
        raise ValueError("Saved score table hash does not match its source manifest.")
    if source_manifest.get("evaluation_report", {}).get("evaluation_report_sha256") != input_hashes["evaluation_report_sha256"]:
        raise ValueError("Evaluation report hash does not match its source manifest.")

    # Read only the persisted validation-score table; no split dataset or model is loaded.
    score_table = pd.read_parquet(SCORE_TABLE_PATH)
    required_columns = {
        "observation_id",
        "true_label",
        "in_common_cohort",
        *(f"score_{name}" for name in MODEL_NAMES),
    }
    missing = sorted(required_columns.difference(score_table.columns))
    if missing:
        raise ValueError(f"Validation score table is missing required columns: {missing}")

    common = score_table.loc[score_table["in_common_cohort"]].copy()
    if len(common) != EXPECTED_COHORT_ROWS:
        raise ValueError(f"Expected {EXPECTED_COHORT_ROWS} common rows; found {len(common)}.")
    if not common["observation_id"].is_monotonic_increasing or not common["observation_id"].is_unique:
        raise ValueError("Common-cohort observation IDs are not unique and chronological.")
    ids = common["observation_id"].to_numpy(dtype=np.int64)
    expected_ids = np.arange(EXPECTED_FIRST_ID, EXPECTED_LAST_ID + 1, dtype=np.int64)
    if not np.array_equal(ids, expected_ids):
        raise ValueError("Common-cohort observation ID range/order differs from expectations.")
    labels = common["true_label"].to_numpy()
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Common-cohort labels are not finite binary values.")
    label_counts = pd.Series(labels).value_counts().to_dict()
    if int(label_counts.get(1, 0)) != EXPECTED_FRAUDS or int(label_counts.get(0, 0)) != EXPECTED_LEGITIMATE:
        raise ValueError(f"Unexpected common-cohort class counts: {label_counts}")
    for name in MODEL_NAMES:
        values = common[f"score_{name}"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Common-cohort scores for {name} contain missing or non-finite values.")

    expected_report_counts = evaluation_report["common_cohort_results"]
    if expected_report_counts.get("row_count") != EXPECTED_COHORT_ROWS:
        raise ValueError("Evaluation report common-cohort row count does not match the score table.")
    normalized_counts = {str(key): value for key, value in label_counts.items()}
    if expected_report_counts.get("target_class_counts") != normalized_counts:
        raise ValueError("Evaluation report class counts do not match the score table.")

    # Confirm that only the expected leading rows lack temporal-model scores.
    native = score_table.sort_values("observation_id", kind="stable")
    for name in ("lstm", "gru"):
        missing_positions = np.flatnonzero(native[f"score_{name}"].isna().to_numpy())
        if not np.array_equal(missing_positions, np.arange(9)):
            raise ValueError(f"Unexpected missing-score alignment for {name}: {missing_positions.tolist()}")

    return common, source_manifest, evaluation_report, input_hashes


def compute_analysis(
    common: pd.DataFrame,
    source_manifest: dict[str, Any],
    evaluation_report: dict[str, Any],
    input_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calculate four contiguous segment summaries and verify aggregates."""
    segment_records: list[dict[str, Any]] = []
    for segment_number in range(4):
        start = segment_number * SEGMENT_SIZE
        stop = start + SEGMENT_SIZE
        segment = common.iloc[start:stop]
        if len(segment) != SEGMENT_SIZE:
            raise ValueError(f"Segment {segment_number + 1} has an unexpected row count.")
        labels = segment["true_label"].to_numpy(dtype=np.int64)
        frauds = int(labels.sum())
        segment_records.append(
            {
                "segment": segment_number + 1,
                "row_start_in_common_cohort": start,
                "row_end_in_common_cohort_exclusive": stop,
                "first_observation_id": int(segment["observation_id"].iloc[0]),
                "last_observation_id": int(segment["observation_id"].iloc[-1]),
                "row_count": len(segment),
                "fraud_count": frauds,
                "legitimate_count": int(len(segment) - frauds),
                "fraud_prevalence": frauds / len(segment),
                "percentage_of_common_cohort_frauds": 100.0 * frauds / EXPECTED_FRAUDS,
                "models": {},
            }
        )

    if sum(item["row_count"] for item in segment_records) != EXPECTED_COHORT_ROWS:
        raise ValueError("Segment row counts do not sum to the common-cohort size.")
    if sum(item["fraud_count"] for item in segment_records) != EXPECTED_FRAUDS:
        raise ValueError("Segment fraud counts do not sum to the common-cohort fraud count.")
    if sum(item["legitimate_count"] for item in segment_records) != EXPECTED_LEGITIMATE:
        raise ValueError("Segment legitimate counts do not sum to the common-cohort legitimate count.")

    aggregate: dict[str, Any] = {}
    for name in MODEL_NAMES:
        score_column = f"score_{name}"
        all_labels = common["true_label"].to_numpy(dtype=np.int64)
        all_scores = common[score_column].to_numpy(dtype=np.float64)
        aggregate_ap = float(average_precision_score(all_labels, all_scores))
        aggregate_auc = float(roc_auc_score(all_labels, all_scores))
        report_metrics = evaluation_report["common_cohort_results"]["models"][name]["metrics"]
        ap_difference = aggregate_ap - float(report_metrics["average_precision"])
        auc_difference = aggregate_auc - float(report_metrics["roc_auc"])
        if abs(ap_difference) > AGGREGATE_TOLERANCE or abs(auc_difference) > AGGREGATE_TOLERANCE:
            raise ValueError(
                f"Aggregate metric reconciliation failed for {name}: "
                f"AP delta={ap_difference}, ROC-AUC delta={auc_difference}"
            )
        aggregate[name] = {
            "average_precision": aggregate_ap,
            "roc_auc": aggregate_auc,
            "ap_rank": None,
            "aggregate_ap_report_difference": ap_difference,
            "aggregate_roc_auc_report_difference": auc_difference,
            "score_type": source_manifest["models"][name]["score_type"],
            "preprocessing": source_manifest["models"][name]["preprocessing"],
        }

        for segment_record in segment_records:
            start = segment_record["row_start_in_common_cohort"]
            stop = segment_record["row_end_in_common_cohort_exclusive"]
            segment = common.iloc[start:stop]
            labels = segment["true_label"].to_numpy(dtype=np.int64)
            scores = segment[score_column].to_numpy(dtype=np.float64)
            ap_value = float(average_precision_score(labels, scores))
            auc_value = None
            if np.unique(labels).size == 2:
                auc_value = float(roc_auc_score(labels, scores))
            segment_record["models"][name] = {
                "average_precision": ap_value,
                "roc_auc": auc_value,
                "roc_auc_unavailable_reason": None if auc_value is not None else "segment contains only one class",
                "ap_rank": None,
            }

    def assign_ranks(items: dict[str, dict[str, Any]], metric_key: str, rank_key: str) -> None:
        ordered = sorted(items, key=lambda model: (-items[model][metric_key], model))
        for rank, name in enumerate(ordered, start=1):
            items[name][rank_key] = rank

    assign_ranks(aggregate, "average_precision", "ap_rank")
    for segment_record in segment_records:
        assign_ranks(segment_record["models"], "average_precision", "ap_rank")

    summaries: dict[str, Any] = {}
    for name in MODEL_NAMES:
        ap_values = [segment["models"][name]["average_precision"] for segment in segment_records]
        auc_values = [segment["models"][name]["roc_auc"] for segment in segment_records]
        available_auc = [value for value in auc_values if value is not None]
        segment_ranks = [segment["models"][name]["ap_rank"] for segment in segment_records]
        summaries[name] = {
            "aggregate_ap_rank": aggregate[name]["ap_rank"],
            "segment_ap_ranks": segment_ranks,
            "segment_ap_rank_min": min(segment_ranks),
            "segment_ap_rank_max": max(segment_ranks),
            "segment_ap_min": min(ap_values),
            "segment_ap_max": max(ap_values),
            "segment_ap_range": max(ap_values) - min(ap_values),
            "segment_roc_auc_min": min(available_auc) if available_auc else None,
            "segment_roc_auc_max": max(available_auc) if available_auc else None,
            "segment_roc_auc_range": max(available_auc) - min(available_auc) if available_auc else None,
            "segments_without_defined_roc_auc": [
                segment["segment"] for segment in segment_records if segment["models"][name]["roc_auc"] is None
            ],
            "segment_count": len(segment_records),
            "aggregate_rank_is_segment_top_rank": [rank == 1 for rank in segment_ranks],
        }

    # Recompute segment metrics independently from each saved slice as a regression check.
    for segment_record in segment_records:
        segment = common.iloc[
            segment_record["row_start_in_common_cohort"]:
            segment_record["row_end_in_common_cohort_exclusive"]
        ]
        labels = segment["true_label"].to_numpy(dtype=np.int64)
        for name in MODEL_NAMES:
            scores = segment[f"score_{name}"].to_numpy(dtype=np.float64)
            independently_computed_ap = float(average_precision_score(labels, scores))
            stored_ap = segment_record["models"][name]["average_precision"]
            if abs(independently_computed_ap - stored_ap) > AGGREGATE_TOLERANCE:
                raise ValueError(f"Independent segment AP verification failed for {name}, segment {segment_record['segment']}.")
            stored_auc = segment_record["models"][name]["roc_auc"]
            if np.unique(labels).size == 2:
                independently_computed_auc = float(roc_auc_score(labels, scores))
                if stored_auc is None or abs(independently_computed_auc - stored_auc) > AGGREGATE_TOLERANCE:
                    raise ValueError(f"Independent segment ROC-AUC verification failed for {name}, segment {segment_record['segment']}.")
            elif stored_auc is not None:
                raise ValueError("ROC-AUC should be unavailable for a single-class segment.")

    report = {
        "record_type": "validation_temporal_stability_analysis",
        "evaluation_id": EVALUATION_ID,
        "source_analysis_id": SOURCE_ANALYSIS_ID,
        "analysis_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": {
            "definition": "Rows in the saved common cohort with scores from all eight models.",
            "observation_count": len(common),
            "first_observation_id": int(common["observation_id"].iloc[0]),
            "last_observation_id": int(common["observation_id"].iloc[-1]),
            "class_counts": {str(key): int(value) for key, value in common["true_label"].value_counts().sort_index().items()},
            "ordering": "Original observation ID order; no shuffling or stratification.",
        },
        "segmentation": {
            "method": "Four contiguous equal-sized slices of the ordered common cohort.",
            "segment_size": SEGMENT_SIZE,
            "segments": segment_records,
        },
        "metric_definitions": {
            "average_precision": "sklearn.metrics.average_precision_score on continuous saved scores and binary labels.",
            "roc_auc": "sklearn.metrics.roc_auc_score on continuous saved scores when both classes are present; otherwise null.",
            "ranking": "Descending AP; deterministic model-name tie-break if values tie.",
        },
        "aggregate_common_cohort_metrics": aggregate,
        "stability_summaries": summaries,
        "interpretation": {
            "aggregate_rank_persistence": "Segment ranks are descriptive; rank changes do not establish model superiority.",
            "temporal_variation": "Segment differences alone do not establish concept drift or statistical significance.",
            "small_positive_counts": "Only 56 frauds are distributed across four segments; segment-level metrics are sensitive to a small number of positive examples.",
            "temporal_model_semantics": "LSTM and GRU windows follow global chronological transaction order. No customer/account/merchant/device identifiers are available to define entity histories.",
            "test_set_disclosure": evaluation_report.get("holdout_disclosure"),
        },
        "input_hashes": input_hashes,
        "provenance_classification": {
            name: {
                "training_run_id": source_manifest["models"][name].get("training_run_id"),
                "artifact_import_run_id": source_manifest["models"][name].get("artifact_import_run_id"),
                "pipeline_run_id": source_manifest["models"][name].get("pipeline_run_id"),
                "artifact_path": source_manifest["models"][name]["artifact_path"],
                "artifact_sha256": source_manifest["models"][name]["artifact_sha256"],
                "metadata_path": source_manifest["models"][name]["metadata_path"],
                "metadata_sha256": source_manifest["models"][name]["metadata_sha256"],
                "score_type": source_manifest["models"][name]["score_type"],
                "preprocessing": source_manifest["models"][name]["preprocessing"],
                "provenance_caveat": source_manifest["models"][name].get("provenance_caveat"),
            }
            for name in MODEL_NAMES
        },
        "limitations": [
            "The validation observations were used for model and checkpoint selection.",
            "Only 56 fraud observations are available in the common cohort.",
            "Segment metrics are descriptive and are not uncertainty intervals.",
            "Temporal variation does not by itself demonstrate concept drift.",
            "The test split was examined in prior project work and is not an untouched or blind holdout.",
        ],
    }

    markdown = render_markdown(report)
    return report, {"markdown": markdown, "source_hashes": input_hashes}


def render_markdown(report: dict[str, Any]) -> str:
    """Render concise tables while retaining full precision in the JSON record."""
    segments = report["segmentation"]["segments"]
    aggregate = report["aggregate_common_cohort_metrics"]
    summaries = report["stability_summaries"]
    lines = [
        "# Four-Segment Temporal Stability Analysis",
        "",
        f"- Evaluation ID: `{report['evaluation_id']}`",
        f"- Source curve-analysis ID: `{report['source_analysis_id']}`",
        f"- Common cohort: {report['cohort']['observation_count']:,} observations, "
        f"{report['cohort']['class_counts'].get('1', 0)} frauds, "
        f"{report['cohort']['class_counts'].get('0', 0):,} legitimate transactions",
        "- This analysis uses only saved validation scores and labels; no model was loaded and no inference was run.",
        "- Scores are continuous. No threshold was selected.",
        "",
        "## Segment composition",
        "",
        "| Segment | Observation IDs | Rows | Fraud | Legitimate | Fraud prevalence | Share of cohort frauds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for segment in segments:
        lines.append(
            f"| {segment['segment']} | {segment['first_observation_id']}–{segment['last_observation_id']} "
            f"| {segment['row_count']:,} | {segment['fraud_count']} | {segment['legitimate_count']:,} "
            f"| {segment['fraud_prevalence']:.6%} | {segment['percentage_of_common_cohort_frauds']:.2f}% |"
        )

    for metric_name, metric_key in (("AP", "average_precision"), ("ROC-AUC", "roc_auc")):
        lines.extend(
            [
                "",
                f"## {metric_name} by segment",
                "",
                "| Model | Segment 1 | Segment 2 | Segment 3 | Segment 4 | Min | Max | Range |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in MODEL_NAMES:
            values = [segment["models"][name][metric_key] for segment in segments]
            available = [value for value in values if value is not None]
            fmt = lambda value: "N/A" if value is None else f"{value:.6f}"
            low = min(available) if available else None
            high = max(available) if available else None
            value_range = high - low if low is not None and high is not None else None
            lines.append(
                f"| {name} | " + " | ".join(fmt(value) for value in values) +
                f" | {fmt(low)} | {fmt(high)} | {fmt(value_range)} |"
            )

    lines.extend(
        [
            "",
            "## Aggregate common-cohort metrics and AP ranks",
            "",
            "| Model | AP | ROC-AUC | Aggregate AP rank | Segment AP ranks (1–4) | Segment AP range | Segment ROC-AUC range |",
            "|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for name in MODEL_NAMES:
        summary = summaries[name]
        lines.append(
            f"| {name} | {aggregate[name]['average_precision']:.12f} | {aggregate[name]['roc_auc']:.12f} "
            f"| {summary['aggregate_ap_rank']} | {summary['segment_ap_ranks']} "
            f"| {summary['segment_ap_min']:.6f}–{summary['segment_ap_max']:.6f} "
            f"| {summary['segment_roc_auc_min']:.6f}–{summary['segment_roc_auc_max']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "The segment metrics are descriptive summaries of four contiguous portions of the common validation cohort. "
            "They do not establish statistical significance, model superiority, or concept drift. There are only 56 frauds "
            "across the full cohort, so segment metrics are sensitive to a small number of positives. AP and ROC-AUC can "
            "rank models differently because they summarize different aspects of ranking performance. The Autoencoder "
            "produces a reconstruction-error anomaly score, not a calibrated fraud probability. LSTM and GRU sequences "
            "use global chronological transaction order; without entity identifiers they cannot be interpreted as customer histories.",
            "",
            str(report["interpretation"]["test_set_disclosure"]),
            "",
            "No production winner or operating threshold is selected.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Validate inputs, produce a new report directory, and write hashes."""
    common, source_manifest, evaluation_report, input_hashes = validate_inputs()
    report, rendered = compute_analysis(common, source_manifest, evaluation_report, input_hashes)
    run_id = report["analysis_id"]
    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "validation_analysis"
        / "temporal_stability_v1"
        / EVALUATION_ID
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    report_path = output_dir / "temporal_stability_report.json"
    markdown_path = output_dir / "temporal_stability_report.md"
    manifest_path = output_dir / "analysis_manifest.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(rendered["markdown"], encoding="utf-8")

    try:
        dirty = bool(git_value(["status", "--porcelain"]))
    except Exception:
        dirty = None
    manifest = {
        "record_type": "validation_temporal_stability_manifest",
        "analysis_id": run_id,
        "evaluation_id": EVALUATION_ID,
        "source_analysis_id": SOURCE_ANALYSIS_ID,
        "created_at_utc": report["created_at_utc"],
        "inputs": {
            "score_table_path": str(SCORE_TABLE_PATH.relative_to(PROJECT_ROOT)),
            "score_table_sha256": input_hashes["score_table_sha256"],
            "source_analysis_manifest_path": str(SOURCE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "source_analysis_manifest_sha256": input_hashes["source_analysis_manifest_sha256"],
            "evaluation_report_path": str(EVALUATION_REPORT_PATH.relative_to(PROJECT_ROOT)),
            "evaluation_report_sha256": input_hashes["evaluation_report_sha256"],
            "data_used": "Only the saved validation score table, its labels, and report/manifest metadata; no split files were loaded.",
        },
        "outputs": {
            "report_json": {"path": report_path.name, "sha256": sha256_file(report_path)},
            "report_markdown": {"path": markdown_path.name, "sha256": sha256_file(markdown_path)},
        },
        "source": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pyarrow": package_version("pyarrow"),
        },
        "git": {
            "revision": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["branch", "--show-current"]),
            "dirty": dirty,
        },
        "command": ".venv/bin/python src/analyze_temporal_stability_v1.py",
        "method": {
            "segments": 4,
            "segment_size": SEGMENT_SIZE,
            "ordering": "original observation_id ascending",
            "shuffle": False,
            "stratification": False,
            "aggregate_metric_tolerance": AGGREGATE_TOLERANCE,
            "threshold_selected": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir.relative_to(PROJECT_ROOT)), "manifest": manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
