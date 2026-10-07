"""Train the evaluated fraud models from the existing chronological splits.

Only the training and validation datasets are read. The test split is never
loaded or evaluated by this artifact generation script.
"""

from __future__ import annotations

import json
import importlib.metadata
import logging
import platform
import random
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)
MODEL_EXECUTION_ORDER = (
    "random_forest",
    "xgboost",
    "logistic_regression",
    "lightgbm",
    "mlp",
    "autoencoder",
    "lstm",
    "gru",
)


@dataclass
class RunSettings:
    """Settings that apply to an entire training run."""

    seed: int = 42
    device: str = "auto"
    selected_models: list[str] = field(
        default_factory=lambda: list(MODEL_EXECUTION_ORDER)
    )


@dataclass
class DataSettings:
    """Repository-relative paths for the existing train/validation splits."""

    processed_dir: str = "data/processed"
    x_train: str = "X_train.parquet"
    x_val: str = "X_val.parquet"
    x_train_scaled: str = "X_train_scaled.parquet"
    x_val_scaled: str = "X_val_scaled.parquet"
    y_train: str = "y_train.parquet"
    y_val: str = "y_val.parquet"


@dataclass
class FeatureSettings:
    """Ordered modeling feature contract shared by all model families."""

    names: list[str] = field(default_factory=lambda: [
        *[f"V{i}" for i in range(1, 29)],
        "Amount", "Amount_log", "Time", "Time_hour",
    ])


@dataclass
class ModelSettings:
    """Existing per-model parameters and input/training distinctions."""

    parameters: dict[str, Any] = field(default_factory=dict)
    input_representation: str = "unscaled"
    loss: str | None = None
    checkpoint_metric: str | None = None
    positive_class_weighted: bool = False
    training_rows: str | None = None
    shuffle_training_batches: bool | None = None


def default_model_settings() -> dict[str, ModelSettings]:
    """Return the current model configurations without changing their values."""
    return {
        "random_forest": ModelSettings(
            parameters={"n_estimators": 200, "n_jobs": -1},
        ),
        "xgboost": ModelSettings(
            parameters={
                "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "objective": "binary:logistic", "eval_metric": "logloss",
                "n_jobs": -1,
            },
        ),
        "logistic_regression": ModelSettings(
            parameters={"max_iter": 1000},
            input_representation="scaled",
        ),
        "lightgbm": ModelSettings(
            parameters={
                "n_estimators": 200, "num_leaves": 31, "learning_rate": 0.1,
                "min_child_samples": 100, "colsample_bytree": 0.8,
                "objective": "binary", "n_jobs": -1, "verbosity": -1,
            },
        ),
        "mlp": ModelSettings(
            parameters={
                "dimensions": [32, 64, 32, 1], "activation": "ReLU",
                "dropout": 0.20,
            },
            input_representation="scaled",
            loss="BCEWithLogitsLoss",
            checkpoint_metric="Average Precision (AP)",
            positive_class_weighted=True,
            shuffle_training_batches=True,
        ),
        "autoencoder": ModelSettings(
            parameters={
                "dimensions": [32, 16, 8, 16, 32],
                "hidden_activation": "ReLU",
            },
            input_representation="scaled",
            loss="MSELoss",
            checkpoint_metric="validation reconstruction MSE",
            training_rows="legitimate training transactions only",
            shuffle_training_batches=True,
        ),
        "lstm": ModelSettings(
            parameters={
                "input_size": 32, "hidden_size": 64, "num_layers": 1,
                "batch_first": True, "bidirectional": False,
            },
            input_representation="scaled",
            loss="BCEWithLogitsLoss",
            checkpoint_metric="Average Precision (AP)",
            positive_class_weighted=True,
            shuffle_training_batches=False,
        ),
        "gru": ModelSettings(
            parameters={
                "input_size": 32, "hidden_size": 64, "num_layers": 1,
                "batch_first": True, "bidirectional": False,
            },
            input_representation="scaled",
            loss="BCEWithLogitsLoss",
            checkpoint_metric="Average Precision (AP)",
            positive_class_weighted=True,
            shuffle_training_batches=False,
        ),
    }


@dataclass
class TrainingSettings:
    """Shared optimizer, batching, stopping, and validation settings."""

    optimizer: str = "Adam"
    learning_rate: float = 0.001
    batch_size: int = 1024
    max_epochs: int = 50
    early_stopping_patience: int = 7
    sequence_length: int = 10
    validation_metric: str = "Average Precision (AP)"
    autoencoder_checkpoint_metric: str = "validation reconstruction MSE"


@dataclass
class OutputSettings:
    """Artifact naming and historical comparison values."""

    model_dir: str = "models"
    artifact_formats: dict[str, str] = field(default_factory=lambda: {
        "sklearn": "joblib", "pytorch": "state_dict",
    })
    artifact_suffixes: dict[str, str] = field(default_factory=lambda: {
        "sklearn": ".joblib", "pytorch": "_state_dict.pt",
    })
    metadata_suffix: str = "_metadata.json"
    # These AP values are historical references, not runtime selection inputs.
    reference_validation_ap: dict[str, float] = field(default_factory=lambda: {
        "random_forest": 0.858000,
        "mlp": 0.857221,
        "lstm": 0.848276,
        "gru": 0.843658,
        "xgboost": 0.828400,
        "logistic_regression": 0.784900,
        "lightgbm": 0.681400,
        "autoencoder": 0.285407,
    })


@dataclass
class RunConfig:
    """Complete JSON-serializable configuration for one training run."""

    run: RunSettings = field(default_factory=RunSettings)
    data: DataSettings = field(default_factory=DataSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    models: dict[str, ModelSettings] = field(default_factory=default_model_settings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    output: OutputSettings = field(default_factory=OutputSettings)


DEFAULT_CONFIG = RunConfig()


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Convert the dataclass configuration into JSON-compatible values."""
    return asdict(config)


def serialize_config(config: RunConfig) -> str:
    """Serialize configuration deterministically for future run tracking."""
    return json.dumps(config_to_dict(config), indent=2, sort_keys=True)


def _git_command(*arguments: str) -> str | None:
    """Run a bounded Git query, returning None when repository data is unavailable."""
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, check=True, capture_output=True,
            text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def collect_provenance(config: RunConfig,
                       selected_models: list[str],
                       run_id: str) -> dict[str, Any]:
    """Collect runtime, dependency, Git, and existing run configuration details."""
    distribution_names = (
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "torch",
    )
    dependencies: dict[str, str | None] = {}
    for distribution_name in distribution_names:
        try:
            dependencies[distribution_name] = importlib.metadata.version(
                distribution_name
            )
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution_name] = None

    status = _git_command("status", "--porcelain")
    return {
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
        },
        "dependencies": dependencies,
        "git": {
            "commit": _git_command("rev-parse", "HEAD"),
            "branch": _git_command("branch", "--show-current"),
            "dirty": None if status is None else bool(status),
        },
        "run": {
            "run_id": run_id,
            "seed": config.run.seed,
            "device": config.run.device,
            "selected_models": selected_models,
            "configuration": config_to_dict(config),
        },
    }


def resolve_device(device_setting: str) -> torch.device:
    """Preserve automatic CUDA selection while allowing an explicit device."""
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def select_features(settings: ModelSettings, unscaled: pd.DataFrame,
                    scaled: pd.DataFrame) -> pd.DataFrame:
    """Select the current scaled or unscaled representation for a model."""
    if settings.input_representation == "scaled":
        return scaled
    if settings.input_representation == "unscaled":
        return unscaled
    raise ValueError(f"Unsupported input representation: {settings.input_representation}")


def resolve_selected_models(config: RunConfig) -> list[str]:
    """Validate model names and return them in the configured canonical order."""
    selected = config.run.selected_models
    if not selected:
        raise ValueError("At least one model must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError("Selected model names must not contain duplicates")
    unknown = [name for name in selected if name not in config.models]
    if unknown:
        raise ValueError(f"Unknown model name(s): {', '.join(unknown)}")
    selected_set = set(selected)
    return [name for name in config.models if name in selected_set]


def set_seed(seed: int) -> None:
    """Set the random generators used by the notebook training procedures."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits(config: RunConfig = DEFAULT_CONFIG) -> tuple[pd.DataFrame, ...]:
    """Load only the train and validation feature/label parquet files."""
    data_dir = ROOT / config.data.processed_dir
    logger.info("Loading training and validation data")
    x_train = pd.read_parquet(data_dir / config.data.x_train)
    x_val = pd.read_parquet(data_dir / config.data.x_val)
    x_train_scaled = pd.read_parquet(data_dir / config.data.x_train_scaled)
    x_val_scaled = pd.read_parquet(data_dir / config.data.x_val_scaled)
    y_train = pd.read_parquet(data_dir / config.data.y_train)["Class"].astype("int64")
    y_val = pd.read_parquet(data_dir / config.data.y_val)["Class"].astype("int64")
    for name, frame in (("X_train", x_train), ("X_val", x_val),
                        ("X_train_scaled", x_train_scaled), ("X_val_scaled", x_val_scaled)):
        if frame.columns.tolist() != config.features.names:
            raise ValueError(f"{name} feature columns do not match the expected ordered list")
        logger.info("Loaded %s with shape %s", name, frame.shape)
    logger.info("Loaded y_train with shape %s", y_train.shape)
    logger.info("Loaded y_val with shape %s", y_val.shape)
    return x_train, x_val, x_train_scaled, x_val_scaled, y_train, y_val


def write_metadata(name: str, config: RunConfig, model_configuration: dict[str, Any],
                   metric: str, score: float, best_epoch: int | None = None,
                   extra: dict | None = None,
                   provenance: dict[str, Any] | None = None,
                   output_dir: Path | None = None) -> None:
    """Write model provenance and the newly measured validation metric."""
    model_settings = config.models[name]
    scaled_inputs = model_settings.input_representation == "scaled"
    feature_file = config.data.x_train_scaled if scaled_inputs else config.data.x_train
    validation_file = config.data.x_val_scaled if scaled_inputs else config.data.x_val
    record = {
        "model_name": name,
        "configuration": model_configuration,
        "random_seed": config.run.seed,
        "features": config.features.names,
        "training_dataset": [
            f"{config.data.processed_dir}/{feature_file}",
            f"{config.data.processed_dir}/{config.data.y_train}",
        ],
        "validation_dataset": [
            f"{config.data.processed_dir}/{validation_file}",
            f"{config.data.processed_dir}/{config.data.y_val}",
        ],
        "validation_metric": metric,
        "reproduced_validation_metric": score,
        "reference_validation_ap": config.output.reference_validation_ap[name],
        "best_epoch": best_epoch,
    }
    if provenance is not None:
        record["provenance"] = provenance
    if extra:
        record.update(extra)
    model_dir = output_dir or ROOT / config.output.model_dir
    (model_dir / f"{name}{config.output.metadata_suffix}").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Saved metadata file: %s", f"{name}{config.output.metadata_suffix}")


class FraudMLP(nn.Module):
    """The two-hidden-layer classifier defined in notebook 03."""
    def __init__(self, dimensions: list[int], dropout: float,
                 activation: str) -> None:
        super().__init__()
        activation_type = getattr(nn, activation)
        self.fc1 = nn.Linear(dimensions[0], dimensions[1])
        self.relu1 = activation_type()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dimensions[1], dimensions[2])
        self.relu2 = activation_type()
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(dimensions[2], dimensions[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.output(x)


class FraudAutoencoder(nn.Module):
    """The 32-16-8-16-32 reconstruction network from notebook 03."""
    def __init__(self, dimensions: list[int], hidden_activation: str) -> None:
        super().__init__()
        activation_type = getattr(nn, hidden_activation)
        self.encoder = nn.Sequential(
            nn.Linear(dimensions[0], dimensions[1]), activation_type(),
            nn.Linear(dimensions[1], dimensions[2]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dimensions[2], dimensions[3]), activation_type(),
            nn.Linear(dimensions[3], dimensions[4]),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        return self.decoder(latent), latent


class LSTMClassifier(nn.Module):
    """Single-layer unidirectional LSTM that classifies the final step."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 batch_first: bool, bidirectional: bool) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=batch_first, bidirectional=bidirectional,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.output_layer(hidden[-1]).squeeze(1)


class GRUClassifier(nn.Module):
    """Single-layer unidirectional GRU that classifies the final step."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 batch_first: bool, bidirectional: bool) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=batch_first, bidirectional=bidirectional,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.output_layer(hidden[-1]).squeeze(1)


def train_supervised_torch(model: nn.Module, train_loader: DataLoader,
                           val_loader: DataLoader, y_train: np.ndarray,
                           device: torch.device, training: TrainingSettings,
                           loss_name: str, checkpoint_metric: str,
                           positive_class_weighted: bool) -> tuple[dict, int, float]:
    """Train with positive-class weighted BCE and retain best validation AP."""
    model = model.to(device)
    if loss_name != "BCEWithLogitsLoss":
        raise ValueError(f"Unsupported supervised loss: {loss_name}")
    if checkpoint_metric != training.validation_metric:
        raise ValueError("Supervised checkpoint metric must match the validation metric setting")
    pos_weight = None
    if positive_class_weighted:
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / (y_train == 1).sum()],
            dtype=torch.float32, device=device,
        )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer_type = getattr(torch.optim, training.optimizer)
    optimizer = optimizer_type(model.parameters(), lr=training.learning_rate)
    best_state = None
    best_ap = -np.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, training.max_epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(features)
            # The MLP returns (batch, 1), while the notebook compares squeezed logits.
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(features)
            sample_count += len(features)

        # Validation AP is the notebook's checkpoint criterion and uses no test data.
        model.eval()
        probabilities, targets_all = [], []
        with torch.no_grad():
            for features, targets in val_loader:
                logits = model(features.to(device))
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                probabilities.extend(torch.sigmoid(logits).cpu().numpy())
                targets_all.extend(targets.numpy())
        val_ap = average_precision_score(targets_all, probabilities)
        logger.info("%s epoch %02d: train_loss=%.6f, val_AP=%.6f",
                    model.__class__.__name__, epoch,
                    total_loss / sample_count, val_ap)
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            logger.info("%s checkpoint improved at epoch %d: validation AP=%.6f",
                        model.__class__.__name__, best_epoch, best_ap)
        else:
            stale_epochs += 1
        if stale_epochs >= training.early_stopping_patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    logger.info("%s best epoch=%d, best validation AP=%.6f",
                model.__class__.__name__, best_epoch, best_ap)
    return best_state, best_epoch, float(best_ap)


def make_sequences(features: pd.DataFrame, labels: pd.Series,
                   length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows within one split; target the final transaction."""
    values = features.to_numpy(dtype=np.float32)
    targets = labels.to_numpy(dtype=np.int64)
    windows = np.stack([values[i:i + length]
                        for i in range(len(values) - length + 1)])
    return windows, targets[length - 1:]


def main(config: RunConfig = DEFAULT_CONFIG) -> None:
    """Fit the selected reference models and write weights plus metadata."""
    # Resolve selection before creating output directories or reading datasets.
    run_id = uuid.uuid4().hex
    selected_models = resolve_selected_models(config)
    selected_model_set = set(selected_models)
    logger.info("Training run started")
    logger.info("Selected models: %s", selected_models)
    provenance = collect_provenance(config, selected_models, run_id)
    logger.info("Runtime and run provenance collected")
    if config.output.artifact_formats["sklearn"] != "joblib":
        raise ValueError("The current scikit-learn artifact format must remain joblib")
    if config.output.artifact_formats["pytorch"] != "state_dict":
        raise ValueError("The current PyTorch artifact format must remain state_dict")
    model_dir = ROOT / config.output.model_dir / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run ID: %s", run_id)
    logger.info("Run output directory configured")
    set_seed(config.run.seed)
    x_train, x_val, xs_train, xs_val, y_train, y_val = load_splits(config)

    # The reference comparison uses unweighted baseline variants except LightGBM,
    # where the regularized unweighted validation-selected candidate is recorded.
    rf_config = config.models["random_forest"]
    xgb_config = config.models["xgboost"]
    logistic_config = config.models["logistic_regression"]
    lightgbm_config = config.models["lightgbm"]
    sklearn_models = {
        "random_forest": (
            RandomForestClassifier(**rf_config.parameters, random_state=config.run.seed),
            select_features(rf_config, x_train, xs_train),
            select_features(rf_config, x_val, xs_val),
            {**rf_config.parameters, "random_state": config.run.seed, "class_weight": None},
        ),
        "xgboost": (
            XGBClassifier(**xgb_config.parameters, random_state=config.run.seed),
            select_features(xgb_config, x_train, xs_train),
            select_features(xgb_config, x_val, xs_val),
            {**xgb_config.parameters, "random_state": config.run.seed, "class_weight": None},
        ),
        "logistic_regression": (
            LogisticRegression(**logistic_config.parameters, random_state=config.run.seed),
            select_features(logistic_config, x_train, xs_train),
            select_features(logistic_config, x_val, xs_val),
            {**logistic_config.parameters, "random_state": config.run.seed,
             "class_weight": None,
             "input": f"{logistic_config.input_representation} features"},
        ),
        "lightgbm": (
            LGBMClassifier(**lightgbm_config.parameters, random_state=config.run.seed),
            select_features(lightgbm_config, x_train, xs_train),
            select_features(lightgbm_config, x_val, xs_val),
            {**lightgbm_config.parameters, "random_state": config.run.seed,
             "max_depth": -1, "class_weight": None},
        ),
    }
    sklearn_format = config.output.artifact_formats["sklearn"]
    for name, (estimator, train_x, val_x, model_metadata) in sklearn_models.items():
        if name not in selected_model_set:
            continue
        logger.info("Training model: %s", name)
        estimator.fit(train_x, y_train)
        ap = average_precision_score(y_val, estimator.predict_proba(val_x)[:, 1])
        if sklearn_format != "joblib":
            raise ValueError(f"Unsupported scikit-learn artifact format: {sklearn_format}")
        artifact_path = model_dir / f"{name}{config.output.artifact_suffixes['sklearn']}"
        joblib.dump(estimator, artifact_path)
        logger.info("Saved model artifact: %s", artifact_path.name)
        write_metadata(
            name, config, model_metadata, config.training.validation_metric,
            float(ap), provenance=provenance, output_dir=model_dir,
        )
        logger.info("%s validation AP=%.6f (reference AP=%.6f)", name, ap,
                    config.output.reference_validation_ap[name])

    # Prepare shared neural settings only when at least one neural model is selected.
    train_values_by_model: dict[str, torch.Tensor] = {}
    val_values_by_model: dict[str, torch.Tensor] = {}
    selected_neural_models = selected_model_set.intersection(
        {"mlp", "autoencoder", "lstm", "gru"}
    )
    if selected_neural_models:
        device = resolve_device(config.run.device)
        logger.info("Training device: %s", device)
        train_targets = torch.from_numpy(y_train.to_numpy(dtype=np.float32, copy=True))
        val_targets = torch.from_numpy(y_val.to_numpy(dtype=np.float32, copy=True))
    for name in ("mlp", "autoencoder"):
        if name not in selected_model_set:
            continue
        model_settings = config.models[name]
        train_frame = select_features(model_settings, x_train, xs_train)
        val_frame = select_features(model_settings, x_val, xs_val)
        train_values_by_model[name] = torch.from_numpy(
            train_frame.to_numpy(dtype=np.float32, copy=True)
        )
        val_values_by_model[name] = torch.from_numpy(
            val_frame.to_numpy(dtype=np.float32, copy=True)
        )

    if "mlp" in selected_model_set:
        # Train the scaled tabular MLP and checkpoint by validation Average Precision.
        logger.info("Training model: mlp")
        mlp_config = config.models["mlp"]
        mlp_train_values = train_values_by_model["mlp"]
        mlp_val_values = val_values_by_model["mlp"]
        train_gen = torch.Generator().manual_seed(config.run.seed)
        train_loader = DataLoader(
            TensorDataset(mlp_train_values, train_targets),
            batch_size=config.training.batch_size,
            shuffle=bool(mlp_config.shuffle_training_batches),
            generator=train_gen,
        )
        val_loader = DataLoader(
            TensorDataset(mlp_val_values, val_targets),
            batch_size=config.training.batch_size,
            shuffle=False,
        )
        mlp_dimensions = mlp_config.parameters["dimensions"]
        mlp = FraudMLP(
            mlp_dimensions,
            mlp_config.parameters["dropout"],
            mlp_config.parameters["activation"],
        )
        mlp_state, mlp_epoch, mlp_ap = train_supervised_torch(
            mlp, train_loader, val_loader, y_train.to_numpy(), device,
            config.training, mlp_config.loss, mlp_config.checkpoint_metric,
            mlp_config.positive_class_weighted,
        )
        torch.save(
            mlp_state,
            model_dir / f"mlp{config.output.artifact_suffixes['pytorch']}",
        )
        logger.info("Saved model artifact: %s",
                    f"mlp{config.output.artifact_suffixes['pytorch']}")
        write_metadata("mlp", config, {
            "architecture": mlp_dimensions, "activation": mlp_config.parameters["activation"],
            "dropout": mlp_config.parameters["dropout"], "loss": mlp_config.loss,
            "positive_class_weight": "training negatives / positives",
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "max_epochs": config.training.max_epochs,
            "early_stopping_patience": config.training.early_stopping_patience,
            "input": f"{mlp_config.input_representation} features",
            "training_batches_shuffled": mlp_config.shuffle_training_batches,
        }, config.training.validation_metric, mlp_ap, mlp_epoch,
            provenance=provenance, output_dir=model_dir)
        logger.info("mlp validation AP=%.6f (reference AP=%.6f)", mlp_ap,
                    config.output.reference_validation_ap["mlp"])

    if "autoencoder" in selected_model_set:
        # Autoencoder validation reconstruction MSE selects the state; validation
        # labels are used only afterward to report anomaly-ranking AP for reference.
        logger.info("Training model: autoencoder")
        autoencoder_config = config.models["autoencoder"]
        if autoencoder_config.training_rows != "legitimate training transactions only":
            raise ValueError("The Autoencoder must use legitimate training transactions only")
        ae_train_values = train_values_by_model["autoencoder"][y_train.to_numpy() == 0]
        ae_val_values = val_values_by_model["autoencoder"].to(device)
        ae_loader = DataLoader(
            TensorDataset(ae_train_values),
            batch_size=config.training.batch_size,
            shuffle=bool(autoencoder_config.shuffle_training_batches),
            generator=torch.Generator().manual_seed(config.run.seed),
        )
        ae_dimensions = autoencoder_config.parameters["dimensions"]
        autoencoder = FraudAutoencoder(
            ae_dimensions, autoencoder_config.parameters["hidden_activation"],
        ).to(device)
        optimizer_type = getattr(torch.optim, config.training.optimizer)
        ae_optimizer = optimizer_type(
            autoencoder.parameters(), lr=config.training.learning_rate,
        )
        if autoencoder_config.loss != "MSELoss":
            raise ValueError(f"Unsupported Autoencoder loss: {autoencoder_config.loss}")
        if autoencoder_config.checkpoint_metric != config.training.autoencoder_checkpoint_metric:
            raise ValueError("Autoencoder checkpoint metric must match the shared training setting")
        mse = getattr(nn, autoencoder_config.loss)()
        best_ae_loss, best_ae_epoch, stale = np.inf, 0, 0
        best_ae_state = None
        for epoch in range(1, config.training.max_epochs + 1):
            autoencoder.train()
            for (batch,) in ae_loader:
                batch = batch.to(device)
                ae_optimizer.zero_grad()
                reconstruction, _ = autoencoder(batch)
                loss = mse(reconstruction, batch)
                loss.backward()
                ae_optimizer.step()
            autoencoder.eval()
            with torch.no_grad():
                val_reconstruction, _ = autoencoder(ae_val_values)
                val_loss = float(mse(val_reconstruction, ae_val_values).item())
            logger.info("FraudAutoencoder epoch %02d: val_MSE=%.6f", epoch, val_loss)
            if val_loss < best_ae_loss:
                best_ae_loss, best_ae_epoch, stale = val_loss, epoch, 0
                best_ae_state = {key: value.detach().cpu().clone()
                                 for key, value in autoencoder.state_dict().items()}
                logger.info("Autoencoder checkpoint improved at epoch %d: "
                            "validation reconstruction MSE=%.6f",
                            best_ae_epoch, best_ae_loss)
            else:
                stale += 1
            if stale >= config.training.early_stopping_patience:
                break
        assert best_ae_state is not None
        autoencoder.load_state_dict(best_ae_state)
        logger.info("Autoencoder best epoch=%d, best validation reconstruction MSE=%.6f",
                    best_ae_epoch, best_ae_loss)
        with torch.no_grad():
            val_reconstruction, _ = autoencoder(ae_val_values)
            # Per-row mean squared reconstruction error is the anomaly score.
            ae_scores = ((val_reconstruction - ae_val_values) ** 2).mean(dim=1).cpu().numpy()
        ae_ap = float(average_precision_score(y_val, ae_scores))
        torch.save(
            best_ae_state,
            model_dir / f"autoencoder{config.output.artifact_suffixes['pytorch']}",
        )
        logger.info("Saved model artifact: %s",
                    f"autoencoder{config.output.artifact_suffixes['pytorch']}")
        write_metadata("autoencoder", config, {
            "architecture": ae_dimensions,
            "hidden_activation": autoencoder_config.parameters["hidden_activation"],
            "loss": "mean squared error", "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "max_epochs": config.training.max_epochs,
            "early_stopping_patience": config.training.early_stopping_patience,
            "training_rows": autoencoder_config.training_rows,
            "input": f"{autoencoder_config.input_representation} features",
            "training_batches_shuffled": autoencoder_config.shuffle_training_batches,
            "checkpoint_selection_metric": autoencoder_config.checkpoint_metric,
            "best_validation_reconstruction_mse": best_ae_loss,
        }, "Average Precision (AP) from per-row reconstruction MSE", ae_ap,
            best_ae_epoch, provenance=provenance, output_dir=model_dir)
        logger.info("autoencoder validation AP=%.6f (reference AP=%.6f)", ae_ap,
                    config.output.reference_validation_ap["autoencoder"])

    # Temporal windows are made independently within each split, matching the
    # notebook and preventing windows from crossing chronological boundaries.
    for name, model_class in (("lstm", LSTMClassifier), ("gru", GRUClassifier)):
        if name not in selected_model_set:
            continue
        logger.info("Training model: %s", name)
        temporal_config = config.models[name]
        temporal_train_frame = select_features(temporal_config, x_train, xs_train)
        temporal_val_frame = select_features(temporal_config, x_val, xs_val)
        seq_train, seq_y_train = make_sequences(
            temporal_train_frame, y_train, config.training.sequence_length,
        )
        seq_val, seq_y_val = make_sequences(
            temporal_val_frame, y_val, config.training.sequence_length,
        )
        seq_train_t = torch.from_numpy(seq_train)
        seq_val_t = torch.from_numpy(seq_val)
        seq_y_train_t = torch.from_numpy(seq_y_train.astype(np.float32))
        seq_y_val_t = torch.from_numpy(seq_y_val.astype(np.float32))
        temporal_train_loader = DataLoader(
            TensorDataset(seq_train_t, seq_y_train_t),
            batch_size=config.training.batch_size,
            shuffle=bool(temporal_config.shuffle_training_batches),
        )
        temporal_val_loader = DataLoader(
            TensorDataset(seq_val_t, seq_y_val_t),
            batch_size=config.training.batch_size,
            shuffle=False,
        )
        model = model_class(**temporal_config.parameters)
        state, epoch, ap = train_supervised_torch(
            model, temporal_train_loader, temporal_val_loader,
            seq_y_train, device, config.training, temporal_config.loss,
            temporal_config.checkpoint_metric,
            temporal_config.positive_class_weighted,
        )
        torch.save(
            state,
            model_dir / f"{name}{config.output.artifact_suffixes['pytorch']}",
        )
        logger.info("Saved model artifact: %s",
                    f"{name}{config.output.artifact_suffixes['pytorch']}")
        write_metadata(name, config, {
            "architecture": {
                **temporal_config.parameters,
                "output": "single linear logit",
            },
            "sequence_length": config.training.sequence_length,
            "sequence_target": "Class of final transaction",
            "sequence_splits": "constructed independently per split",
            "loss": temporal_config.loss,
            "positive_class_weight": "training sequence negatives / positives",
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "max_epochs": config.training.max_epochs,
            "early_stopping_patience": config.training.early_stopping_patience,
            "training_batches_shuffled": temporal_config.shuffle_training_batches,
            "input": f"{temporal_config.input_representation} features",
        }, config.training.validation_metric, ap, epoch,
            provenance=provenance, output_dir=model_dir)
        logger.info("%s validation AP=%.6f (reference AP=%.6f)", name, ap,
                    config.output.reference_validation_ap[name])

    logger.info("Training run completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
