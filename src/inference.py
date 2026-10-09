"""Canonical artifact loading and inference for the eight fraud models.

The registry pins local artifact and metadata bytes. Training runs and imported
artifact records are represented separately; an import record never implies
that the referenced model was trained during that MLflow run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src import train_models


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FEATURES = tuple(
    [*(f"V{i}" for i in range(1, 29)), "Amount", "Amount_log", "Time", "Time_hour"]
)
SCALER_PATH = "data/processed/robust_scaler.joblib"
SCALER_SHA256 = "1e9363ad8f9f03f58ddcdb8532ab7d5a08e5ee20ebe7afdc8db941be52ec5067"
INFERENCE_BATCH_SIZE = 1024


@dataclass(frozen=True)
class ModelSpec:
    """Immutable artifact identity and lineage for a canonical model version."""

    artifact_path: str
    metadata_path: str
    artifact_sha256: str
    metadata_sha256: str
    preprocessing: str
    implementation: str
    score_type: str
    pipeline_run_id: str
    training_run_id: str | None = None
    artifact_import_run_id: str | None = None
    provenance_limitation: str | None = None


CANONICAL_MODELS: dict[str, ModelSpec] = {
    "random_forest": ModelSpec(
        "models/917ec001800745a6bc3e81dd11d9a929/random_forest.joblib",
        "models/917ec001800745a6bc3e81dd11d9a929/random_forest_metadata.json",
        "59fb399c9833c8b95550ed4be825dcae7e433418cdf9f76c9daa55501365fcc6",
        "6a56b874cc830311effe285e8ae78bf5663b7e3540270076dc422f72fd5968db",
        "unscaled", "RandomForestClassifier", "fraud_probability",
        "917ec001800745a6bc3e81dd11d9a929",
        training_run_id="3f8608746b5b486cadd61a1f6331e5eb",
    ),
    "xgboost": ModelSpec(
        "models/3fb3fda08f984b899e99498798ba83ce/xgboost.joblib",
        "models/3fb3fda08f984b899e99498798ba83ce/xgboost_metadata.json",
        "eb122352abcf9c1bf2ca5ae4f2637fb94ba02f7926363512c6c701b76694f041",
        "e775d2df6fc688846ff8d68fa28d080f5d023245cabb9c1acea2f0771e3e2275",
        "unscaled", "XGBClassifier", "fraud_probability",
        "3fb3fda08f984b899e99498798ba83ce",
        artifact_import_run_id="08edc67966a846158e472f4f71a0f83d",
        provenance_limitation=(
            "No original training MLflow run was recovered. Validation AP is a "
            "pre-existing metadata value and was not recomputed during import."
        ),
    ),
    "logistic_regression": ModelSpec(
        "models/3fb3fda08f984b899e99498798ba83ce/logistic_regression.joblib",
        "models/3fb3fda08f984b899e99498798ba83ce/logistic_regression_metadata.json",
        "3078b0f0471239bb8a22e3b5054bd8fd9351c7ecd3a81bebc85e45fcd3b9a317",
        "229302cfbb08fb6ccb30baa2600dc16fbeff317238fa44b64b933d6562a38d9c",
        "scaled", "LogisticRegression", "fraud_probability",
        "3fb3fda08f984b899e99498798ba83ce",
        artifact_import_run_id="1ae816295848406da9be50a8b62192c1",
        provenance_limitation=(
            "No original training MLflow run was recovered. Validation AP is a "
            "pre-existing metadata value and was not recomputed during import."
        ),
    ),
    "lightgbm": ModelSpec(
        "models/3fb3fda08f984b899e99498798ba83ce/lightgbm.joblib",
        "models/3fb3fda08f984b899e99498798ba83ce/lightgbm_metadata.json",
        "1c54a4254b8500c5529c2f35b5255bae1c5a57cfc748eb61161c3e8c9077b812",
        "d6e3093a61e2dd2b4dfbef3dddf5106a2dfd0a1ad0faa0da3984c78a4b9549fd",
        "unscaled", "LGBMClassifier", "fraud_probability",
        "3fb3fda08f984b899e99498798ba83ce",
        artifact_import_run_id="11c360d936f04152b886480a74ba16c0",
        provenance_limitation=(
            "No original training MLflow run was recovered. Validation AP is a "
            "pre-existing metadata value and was not recomputed during import. "
            "The root-level LightGBM binary differs by SHA-256. The aggregate binary is "
            "the forward canonical choice, but the historical validation AP is not "
            "conclusively linked to this binary."
        ),
    ),
    "mlp": ModelSpec(
        "models/05e4fc0601b34589b84053e1b3f8c683/mlp_state_dict.pt",
        "models/05e4fc0601b34589b84053e1b3f8c683/mlp_metadata.json",
        "6642bf6c9b6431e37df7cc3310e3136325d9fd3a9045b193f29ba328f4a3855c",
        "f2bcdaf49141eb5354e4d0673792ba6016d89d68f1d6547d347fca4c0da61e98",
        "scaled", "FraudMLP", "fraud_probability",
        "05e4fc0601b34589b84053e1b3f8c683",
        training_run_id="b7393416190340ad826a9f3ad25cd377",
    ),
    "autoencoder": ModelSpec(
        "models/881087b271b240b192d2385e49c3251a/autoencoder_state_dict.pt",
        "models/881087b271b240b192d2385e49c3251a/autoencoder_metadata.json",
        "403cf1e980e9b4b26f5a54148847d6b6612b062289e185361a129653621615b6",
        "3d44162f99c3bdecf1e7c076669b393d9df3705778b44cb91ccf7c32e5b9e834",
        "scaled", "FraudAutoencoder", "anomaly_score",
        "881087b271b240b192d2385e49c3251a",
        training_run_id="9d337a314b6848dbb5b232ad46395673",
    ),
    "lstm": ModelSpec(
        "models/136349275cab4fa482792ceef86d9ba2/lstm_state_dict.pt",
        "models/136349275cab4fa482792ceef86d9ba2/lstm_metadata.json",
        "090fc43d7541880c542a16783962e8f32470d3f382008db054553ecddaeee532",
        "eb737750db1e0c075822484cdeb589241619b2591781035493640e2c80c65be0",
        "scaled", "LSTMClassifier", "fraud_probability",
        "136349275cab4fa482792ceef86d9ba2",
        training_run_id="e02d4e87207f400a90a14bddf3c124bc",
    ),
    "gru": ModelSpec(
        "models/35a1905ea8634a169817e321ea9237b8/gru_state_dict.pt",
        "models/35a1905ea8634a169817e321ea9237b8/gru_metadata.json",
        "91e48688ebe1c86c1e334fdf6d273b9c0b6a7fe11d4eb38842a438ca58e8057e",
        "5018744f58b225bb1952eb71e2d14e210757491df84c407212fdfb8f572c88ab",
        "scaled", "GRUClassifier", "fraud_probability",
        "35a1905ea8634a169817e321ea9237b8",
        training_run_id="9410c67e5ad4493886cbe44d28735c32",
    ),
}


@dataclass
class LoadedModel:
    """Loaded estimator and its validated preprocessing/lineage contract."""

    name: str
    estimator: Any
    metadata: dict[str, Any]
    spec: ModelSpec
    artifact_path: Path
    metadata_path: Path
    scaler: Any | None = None


@dataclass(frozen=True)
class ScoreResult:
    """Per-input scores and the original rows to which those scores apply."""

    scores: np.ndarray
    score_type: str
    row_indices: pd.Index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_preprocessing(model_name: str, metadata: dict[str, Any]) -> str:
    model_record = (
        metadata.get("provenance", {})
        .get("run", {})
        .get("configuration", {})
        .get("models", {})
        .get(model_name, {})
    )
    representation = model_record.get("input_representation")
    if representation is None:
        value = metadata.get("configuration", {}).get("input", "")
        if isinstance(value, str) and value.startswith("scaled"):
            representation = "scaled"
        elif isinstance(value, str) and value.startswith("unscaled"):
            representation = "unscaled"
        else:
            raise ValueError(f"Preprocessing metadata is missing for {model_name}")
    if representation not in {"scaled", "unscaled"}:
        raise ValueError(f"Unsupported preprocessing metadata for {model_name}: {representation}")
    return representation


def _verify_mlflow_lineage(name: str, spec: ModelSpec) -> None:
    """Check the existing run record without changing MLflow state."""
    client = MlflowClient()
    run_id = spec.training_run_id or spec.artifact_import_run_id
    if run_id is None:
        raise ValueError(f"{name} has no configured lineage record")
    try:
        run = client.get_run(run_id)
    except Exception as error:
        raise RuntimeError(f"Cannot read configured MLflow record for {name}: {run_id}") from error
    if run.info.status != "FINISHED":
        raise ValueError(f"Configured MLflow record for {name} is not FINISHED")

    if spec.training_run_id is not None:
        if run.data.tags.get("record_type") == "artifact_import":
            raise ValueError(f"Training run for {name} is tagged as an artifact import")
        if run.data.tags.get("pipeline_run_id") != spec.pipeline_run_id:
            raise ValueError(f"Training pipeline ID mismatch for {name}")
        try:
            selection = json.loads(run.data.params["run.selected_models"])
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"Training selection missing or invalid for {name}") from error
        if selection != [name]:
            raise ValueError(f"Configured training run did not select only {name}")
    else:
        expected_tags = {
            "record_type": "artifact_import",
            "training_run_recovered": "false",
            "model_name": name,
            "artifact_sha256": spec.artifact_sha256,
            "metadata_sha256": spec.metadata_sha256,
            "source_artifact_path": spec.artifact_path,
            "source_metadata_path": spec.metadata_path,
            "metadata_pipeline_run_id": spec.pipeline_run_id,
        }
        for tag, expected in expected_tags.items():
            if run.data.tags.get(tag) != expected:
                raise ValueError(f"Artifact-import record tag {tag} mismatch for {name}")
        if "run.selected_models" in run.data.params or run.data.metrics:
            raise ValueError(f"Artifact import for {name} is incorrectly recorded as training")


def _make_torch_model(name: str, configuration: dict[str, Any]) -> torch.nn.Module:
    if name == "mlp":
        return train_models.FraudMLP(
            dimensions=configuration["architecture"],
            dropout=configuration["dropout"],
            activation=configuration["activation"],
        )
    if name == "autoencoder":
        return train_models.FraudAutoencoder(
            dimensions=configuration["architecture"],
            hidden_activation=configuration["hidden_activation"],
        )
    if name in {"lstm", "gru"}:
        architecture = configuration["architecture"]
        arguments = {
            key: architecture[key]
            for key in (
                "input_size", "hidden_size", "num_layers",
                "batch_first", "bidirectional",
            )
        }
        model_class = (
            train_models.LSTMClassifier if name == "lstm"
            else train_models.GRUClassifier
        )
        return model_class(**arguments)
    raise ValueError(f"No PyTorch architecture is registered for {name}")


def load_model(model_name: str) -> LoadedModel:
    """Load one pinned canonical model after verifying bytes, metadata, lineage."""
    try:
        spec = CANONICAL_MODELS[model_name]
    except KeyError as error:
        raise ValueError(f"Unknown model name: {model_name}") from error

    artifact_path = ROOT / spec.artifact_path
    metadata_path = ROOT / spec.metadata_path
    for path, label in ((artifact_path, "artifact"), (metadata_path, "metadata")):
        if not path.is_file():
            raise FileNotFoundError(f"Canonical {label} is missing for {model_name}: {path}")
    if _sha256(artifact_path) != spec.artifact_sha256:
        raise ValueError(f"Canonical artifact SHA-256 mismatch for {model_name}")
    if _sha256(metadata_path) != spec.metadata_sha256:
        raise ValueError(f"Canonical metadata SHA-256 mismatch for {model_name}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_name") != model_name:
        raise ValueError(f"Metadata model identity mismatch for {model_name}")
    if tuple(metadata.get("features", ())) != EXPECTED_FEATURES:
        raise ValueError(f"Metadata feature order mismatch for {model_name}")
    if _recorded_preprocessing(model_name, metadata) != spec.preprocessing:
        raise ValueError(f"Metadata preprocessing mismatch for {model_name}")
    metadata_pipeline_id = (
        metadata.get("provenance", {}).get("run", {}).get("run_id")
    )
    if metadata_pipeline_id != spec.pipeline_run_id:
        raise ValueError(f"Metadata pipeline ID mismatch for {model_name}")

    _verify_mlflow_lineage(model_name, spec)

    if spec.implementation in {
        "FraudMLP", "FraudAutoencoder", "LSTMClassifier", "GRUClassifier"
    }:
        estimator = _make_torch_model(model_name, metadata["configuration"])
        state = torch.load(artifact_path, map_location="cpu", weights_only=True)
        estimator.load_state_dict(state, strict=True)
        estimator.eval()
    else:
        estimator_types = {
            "RandomForestClassifier": RandomForestClassifier,
            "XGBClassifier": XGBClassifier,
            "LogisticRegression": LogisticRegression,
            "LGBMClassifier": LGBMClassifier,
        }
        estimator_type = estimator_types.get(spec.implementation)
        if estimator_type is None:
            raise ValueError(f"Unsupported registered implementation: {spec.implementation}")
        estimator = joblib.load(artifact_path)
        if not isinstance(estimator, estimator_type):
            raise TypeError(f"Loaded {model_name} artifact has unexpected type")

    scaler = None
    if spec.preprocessing == "scaled":
        scaler_path = ROOT / SCALER_PATH
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Fitted RobustScaler is missing: {scaler_path}")
        if _sha256(scaler_path) != SCALER_SHA256:
            raise ValueError("Fitted RobustScaler SHA-256 mismatch")
        scaler = joblib.load(scaler_path)
        if getattr(scaler, "n_features_in_", None) != len(EXPECTED_FEATURES):
            raise ValueError("Fitted RobustScaler has an unexpected feature count")
        fitted_features = getattr(scaler, "feature_names_in_", None)
        if fitted_features is not None and tuple(fitted_features) != EXPECTED_FEATURES:
            raise ValueError("Fitted RobustScaler feature order mismatch")

    return LoadedModel(
        name=model_name,
        estimator=estimator,
        metadata=metadata,
        spec=spec,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        scaler=scaler,
    )


def _validate_scores(scores: np.ndarray, expected_rows: int, score_type: str) -> None:
    if scores.shape != (expected_rows,):
        raise ValueError(
            f"Expected {expected_rows} {score_type} scores; received shape {scores.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{score_type} output contains non-finite values")
    if score_type == "fraud_probability" and (
        (scores < 0.0).any() or (scores > 1.0).any()
    ):
        raise ValueError("Fraud probabilities must lie in [0, 1]")
    if score_type == "anomaly_score" and (scores < 0.0).any():
        raise ValueError("Anomaly scores must be non-negative")


def predict_scores(loaded_model: LoadedModel, features: pd.DataFrame) -> ScoreResult:
    """Generate aligned fraud probabilities or per-row anomaly scores.

    Input columns must already match the canonical feature order. Scaled models
    use the verified saved RobustScaler; temporal outputs align to each window's
    final input row. The function never fits or mutates a preprocessing object.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if tuple(features.columns) != EXPECTED_FEATURES:
        raise ValueError("Input feature names or order do not match the canonical 32 features")
    if features.empty and loaded_model.name not in {"lstm", "gru"}:
        raise ValueError("At least one input row is required")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Input features contain non-finite values")

    values = features
    if loaded_model.spec.preprocessing == "scaled":
        if loaded_model.scaler is None:
            raise ValueError(f"A fitted scaler was not loaded for {loaded_model.name}")
        values = pd.DataFrame(
            loaded_model.scaler.transform(features),
            columns=EXPECTED_FEATURES,
            index=features.index,
        )

    row_indices = features.index
    if loaded_model.name in {"lstm", "gru"}:
        sequence_length = int(
            loaded_model.metadata["configuration"]["sequence_length"]
        )
        if sequence_length != 10:
            raise ValueError(f"Unexpected {loaded_model.name} sequence length: {sequence_length}")
        row_indices = features.index[sequence_length - 1:]
        if len(features) < sequence_length:
            scores = np.empty((0,), dtype=np.float64)
            _validate_scores(scores, len(row_indices), loaded_model.spec.score_type)
            return ScoreResult(scores, loaded_model.spec.score_type, row_indices)
        model_values = train_models.make_feature_sequences(values, sequence_length)
    else:
        model_values = values.to_numpy(dtype=np.float32, copy=True)

    if loaded_model.name in {"random_forest", "xgboost", "logistic_regression", "lightgbm"}:
        probabilities = loaded_model.estimator.predict_proba(values)
        classes = list(loaded_model.estimator.classes_)
        if 1 not in classes:
            raise ValueError(f"Model {loaded_model.name} has no fraud class 1")
        scores = np.asarray(probabilities[:, classes.index(1)], dtype=np.float64)
    else:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(model_values), INFERENCE_BATCH_SIZE):
                batch = torch.as_tensor(
                    model_values[start:start + INFERENCE_BATCH_SIZE], dtype=torch.float32
                )
                output = loaded_model.estimator(batch)
                if loaded_model.name == "autoencoder":
                    reconstruction, _ = output
                    batch_scores = ((reconstruction - batch) ** 2).mean(dim=1)
                else:
                    batch_scores = torch.sigmoid(output)
                chunks.append(batch_scores.cpu().numpy().reshape(-1))
        scores = (
            np.concatenate(chunks).astype(np.float64, copy=False)
            if chunks else np.empty((0,), dtype=np.float64)
        )

    _validate_scores(scores, len(row_indices), loaded_model.spec.score_type)
    return ScoreResult(scores, loaded_model.spec.score_type, row_indices)
