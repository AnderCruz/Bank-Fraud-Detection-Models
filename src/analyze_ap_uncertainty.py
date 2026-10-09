"""Exploratory paired AP-difference bootstrap using saved validation scores only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ID = "1d24658eef4f4241b4f4e4c1e3a00578"
ANALYSIS_ID = "d034e61aaf394581ab9ffb8e3c471089"
BASE_DIR = ROOT / "reports" / "validation_analysis" / EVALUATION_ID / ANALYSIS_ID
SCORE_PATH = BASE_DIR / "validation_scores.parquet"
EVALUATION_PATH = ROOT / "reports" / f"validation_evaluation_{EVALUATION_ID}.json"
MANIFEST_PATH = BASE_DIR / "analysis_manifest.json"
SOURCE_PATH = Path(__file__).resolve()
MODEL_COLUMNS = {
    "random_forest": "score_random_forest",
    "mlp": "score_mlp",
    "lstm": "score_lstm",
    "gru": "score_gru",
    "xgboost": "score_xgboost",
}
COMPARISONS = (
    ("random_forest", "mlp"),
    ("random_forest", "lstm"),
    ("random_forest", "gru"),
    ("mlp", "lstm"),
    ("random_forest", "xgboost"),
)
BLOCK_LENGTHS = (10, 68, 372)
DEFAULT_SEED = 20261009
DEFAULT_REPLICATES = 1000
CONFIDENCE_LEVEL = 0.95


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a provenance input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_provenance(seed: int, replicates: int) -> dict[str, Any]:
    """Capture source/input identity and the controlled resampling settings."""
    git_status = _git_value("status", "--porcelain")
    try:
        source_path = str(SOURCE_PATH.relative_to(ROOT))
    except ValueError:
        source_path = SOURCE_PATH.name
    return {
        "input_hashes": {
            "score_table_sha256": sha256_file(SCORE_PATH),
            "evaluation_report_sha256": sha256_file(EVALUATION_PATH),
            "analysis_manifest_sha256": sha256_file(MANIFEST_PATH),
        },
        "analysis_source": {
            "path": source_path,
            "sha256": sha256_file(SOURCE_PATH),
            "git_revision": _git_value("rev-parse", "HEAD"),
            "git_dirty": None if git_status is None else bool(git_status),
        },
        "software": {
            "python": __import__("platform").python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "random_seed": int(seed),
        "requested_replicates_per_method": int(replicates),
        "confidence_level": CONFIDENCE_LEVEL,
        "comparisons": [
            {"first": first, "second": second, "definition": "AP(first) - AP(second)"}
            for first, second in COMPARISONS
        ],
        "sampling": {
            "paired": True,
            "row": "n row indices sampled uniformly with replacement",
            "class_stratified": (
                "positive and negative indices sampled separately with replacement; "
                "conditions on the observed class counts"
            ),
            "moving_block": {
                "block_lengths_rows": list(BLOCK_LENGTHS),
                "algorithm": (
                    "sample contiguous non-circular blocks by choosing each start "
                    "uniformly from 0..n-L; concatenate until at least n rows, then "
                    "truncate to exactly n; use the resulting indices jointly for all models"
                ),
                "ordering": "chronological order of the common-cohort score table",
                "invalid_replicate": (
                    "exclude from interval calculation and count explicitly if the "
                    "replicate contains no positive or no negative labels"
                ),
            },
            "invalid_replicate_policy": (
                "A replicate is valid only when both binary classes are represented; "
                "invalid replicates are counted by reason and are not silently omitted."
            ),
        },
        "interpretation": (
            "Exploratory percentile intervals conditional on the existing validation "
            "cohort and selected models; not independent generalization estimates."
        ),
    }


def load_common_scores() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read only the persisted validation score table and select its common cohort."""
    frame = pd.read_parquet(SCORE_PATH)
    required = {"observation_id", "true_label", "in_common_cohort", *MODEL_COLUMNS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Saved validation score table is missing columns: {sorted(missing)}")
    if frame["observation_id"].duplicated().any():
        raise ValueError("Saved validation observation identifiers must be unique")
    common = frame.loc[frame["in_common_cohort"].astype(bool)].copy()
    if not common["observation_id"].is_monotonic_increasing:
        raise ValueError("Common-cohort observations must remain in chronological row order")
    labels_raw = common["true_label"].to_numpy()
    if not np.isfinite(labels_raw).all() or not np.isin(labels_raw, (0, 1)).all():
        raise ValueError("Common-cohort labels must be finite binary values")
    labels = labels_raw.astype(np.int8)
    scores: dict[str, np.ndarray] = {}
    for name, column in MODEL_COLUMNS.items():
        values = common[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Common-cohort scores must be finite for {name}")
        scores[name] = values
    if len(np.unique(labels)) != 2:
        raise ValueError("Both target classes are required in the common cohort")
    return common["observation_id"].to_numpy(), labels, scores


def sample_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
    method: str,
    block_length: int | None = None,
) -> np.ndarray:
    """Produce one shared resample of source-row positions."""
    n = len(labels)
    if method == "paired_row":
        return rng.integers(0, n, size=n, dtype=np.int64)
    if method == "paired_class_stratified_row":
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        return np.concatenate((
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        )).astype(np.int64, copy=False)
    if method == "paired_moving_block":
        if block_length is None or not 1 <= block_length <= n:
            raise ValueError("Moving-block bootstrap requires a valid block length")
        block_count = (n + block_length - 1) // block_length
        starts = rng.integers(0, n - block_length + 1, size=block_count)
        offsets = np.arange(block_length, dtype=np.int64)
        return (starts[:, None] + offsets[None, :]).reshape(-1)[:n].astype(
            np.int64, copy=False
        )
    raise ValueError(f"Unknown bootstrap method: {method}")


def _ap_difference(
    labels: np.ndarray, scores: dict[str, np.ndarray], indices: np.ndarray,
    first: str, second: str,
) -> float:
    sampled_labels = labels[indices]
    ap_first = average_precision_score(sampled_labels, scores[first][indices])
    ap_second = average_precision_score(sampled_labels, scores[second][indices])
    return float(ap_first - ap_second)


class _AveragePrecisionIndex:
    """Fast exact AP for integer row multiplicities using fixed score tie groups."""

    def __init__(self, labels: np.ndarray, values: np.ndarray):
        _, inverse = np.unique(values, return_inverse=True)
        self.group_count = int(inverse.max()) + 1
        self.group_id = self.group_count - 1 - inverse
        self.labels = labels.astype(np.float64, copy=False)

    def score(self, row_multiplicities: np.ndarray) -> float:
        positive = np.bincount(
            self.group_id,
            weights=row_multiplicities * self.labels,
            minlength=self.group_count,
        )
        negative = np.bincount(
            self.group_id,
            weights=row_multiplicities * (1.0 - self.labels),
            minlength=self.group_count,
        )
        total_positive = positive.sum()
        if total_positive <= 0:
            raise ValueError("Average Precision requires at least one positive label")
        cumulative_positive = np.cumsum(positive)
        cumulative_total = cumulative_positive + np.cumsum(negative)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        return float(np.sum((positive / total_positive) * precision))


def _run_method(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    replicates: int,
    rng: np.random.Generator,
    method: str,
    block_length: int | None = None,
) -> dict[str, Any]:
    """Resample paired rows and preserve explicit invalid-replicate accounting."""
    observed = {
        f"{first}_vs_{second}": float(
            average_precision_score(labels, scores[first])
            - average_precision_score(labels, scores[second])
        )
        for first, second in COMPARISONS
    }
    samples = {key: [] for key in observed}
    comparison_models = sorted({name for pair in COMPARISONS for name in pair})
    ap_indexes = {
        name: _AveragePrecisionIndex(labels, scores[name])
        for name in comparison_models
    }
    invalid: dict[str, int] = {}
    for _ in range(replicates):
        indices = sample_indices(labels, rng, method, block_length)
        sample_labels = labels[indices]
        classes = np.unique(sample_labels)
        if len(classes) < 2:
            reason = "no_positive_labels" if classes[0] == 0 else "no_negative_labels"
            invalid[reason] = invalid.get(reason, 0) + 1
            continue
        # Compute each model AP once per replicate, then form all paired differences.
        multiplicities = np.bincount(indices, minlength=len(labels))
        replicate_ap = {
            name: ap_indexes[name].score(multiplicities)
            for name in comparison_models
        }
        for first, second in COMPARISONS:
            key = f"{first}_vs_{second}"
            samples[key].append(replicate_ap[first] - replicate_ap[second])

    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    results: dict[str, Any] = {}
    valid_count = replicates - sum(invalid.values())
    for key, observed_difference in observed.items():
        values = np.asarray(samples[key], dtype=np.float64)
        interval = (
            [float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))]
            if len(values) else None
        )
        results[key] = {
            "observed_ap_difference": observed_difference,
            "percentile_interval": interval,
            "valid_replicates": int(len(values)),
        }
    return {
        "method": method,
        "block_length_rows": block_length,
        "requested_replicates": int(replicates),
        "valid_replicates": int(valid_count),
        "invalid_replicates": int(sum(invalid.values())),
        "invalid_replicates_by_reason": invalid,
        "comparisons": results,
    }


def run_bootstrap(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Run prespecified paired resampling methods in a reproducible order."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be a one-dimensional binary array")
    if len(np.unique(labels)) != 2:
        raise ValueError("Observed cohort must contain both classes")
    if any(len(np.asarray(values)) != len(labels) for values in scores.values()):
        raise ValueError("Every model score vector must align with the labels")
    if set(name for pair in COMPARISONS for name in pair) - set(scores):
        raise ValueError("Scores are missing a model required by the prespecified comparisons")
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("Model scores must be finite")

    rng = np.random.default_rng(seed)
    results = [
        _run_method(labels, scores, replicates, rng, "paired_row"),
        _run_method(labels, scores, replicates, rng, "paired_class_stratified_row"),
    ]
    results.extend(
        _run_method(
            labels, scores, replicates, rng, "paired_moving_block", block_length
        )
        for block_length in BLOCK_LENGTHS
    )
    return results


def build_report(replicates: int, seed: int) -> dict[str, Any]:
    """Build the reproducible uncertainty record from the saved score table."""
    ids, labels, scores = load_common_scores()
    report = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if report.get("evaluation_id") != EVALUATION_ID:
        raise ValueError("Evaluation report ID does not match the score-table hierarchy")
    if manifest.get("evaluation_report", {}).get("evaluation_report_id") != EVALUATION_ID:
        raise ValueError("Analysis manifest does not identify the expected evaluation report")
    evaluation_hash = sha256_file(EVALUATION_PATH)
    if manifest.get("evaluation_report", {}).get("evaluation_report_sha256") != evaluation_hash:
        raise ValueError("Analysis manifest evaluation-report hash does not match the input file")
    if manifest.get("score_table", {}).get("sha256") != sha256_file(SCORE_PATH):
        raise ValueError("Analysis manifest score-table hash does not match the input file")
    results = run_bootstrap(labels, scores, replicates, seed)
    return {
        "record_type": "paired_validation_ap_uncertainty",
        "analysis_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_id": EVALUATION_ID,
        "source_analysis_id": ANALYSIS_ID,
        "cohort": {
            "definition": "saved common cohort: rows with scores from both length-10 temporal models",
            "observation_count": int(len(ids)),
            "fraud_count": int(labels.sum()),
            "non_fraud_count": int((labels == 0).sum()),
            "first_observation_id": int(ids[0]),
            "last_observation_id": int(ids[-1]),
        },
        "metric": "sklearn.metrics.average_precision_score",
        "interval": {
            "type": "percentile",
            "confidence_level": CONFIDENCE_LEVEL,
            "interpretation": "Exploratory resampling summary, conditional on this validation cohort.",
        },
        "results": results,
        "provenance": collect_provenance(seed, replicates),
        "limitations": [
            "Validation observations were used for model/checkpoint selection.",
            "The common validation cohort has only 56 fraud observations.",
            "Temporal dependence may remain; block lengths are sensitivity anchors, not estimated dependence scales.",
            "The test split has been examined previously and is not an independent blind holdout.",
            "Intervals do not establish superiority or independent generalization performance.",
        ],
    }


def write_report(report: dict[str, Any], output_root: Path | None = None) -> Path:
    """Write a new, non-overwriting versioned JSON report."""
    output_root = output_root or ROOT / "reports" / "validation_analysis" / "ap_uncertainty_v1"
    output_dir = output_root / EVALUATION_ID / report["analysis_id"]
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "paired_ap_uncertainty.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    path = write_report(build_report(args.replicates, args.seed))
    print(path.relative_to(ROOT))
    return path


if __name__ == "__main__":
    main()
