"""Train the evaluated fraud models from the existing chronological splits.

Only the training and validation datasets are read. The test split is never
loaded or evaluated by this artifact generation script.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

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
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
SEED = 42
FEATURES = [f"V{i}" for i in range(1, 29)] + [
    "Amount", "Amount_log", "Time", "Time_hour"
]
REFERENCE_AP = {
    "random_forest": 0.858000,
    "mlp": 0.857221,
    "lstm": 0.848276,
    "gru": 0.843658,
    "xgboost": 0.828400,
    "logistic_regression": 0.784900,
    "lightgbm": 0.681400,
    "autoencoder": 0.285407,
}


def set_seed(seed: int = SEED) -> None:
    """Set the random generators used by the notebook training procedures."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits() -> tuple[pd.DataFrame, ...]:
    """Load only the train and validation feature/label parquet files."""
    x_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    x_val = pd.read_parquet(DATA_DIR / "X_val.parquet")
    x_train_scaled = pd.read_parquet(DATA_DIR / "X_train_scaled.parquet")
    x_val_scaled = pd.read_parquet(DATA_DIR / "X_val_scaled.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")["Class"].astype("int64")
    y_val = pd.read_parquet(DATA_DIR / "y_val.parquet")["Class"].astype("int64")
    for name, frame in (("X_train", x_train), ("X_val", x_val),
                        ("X_train_scaled", x_train_scaled), ("X_val_scaled", x_val_scaled)):
        if frame.columns.tolist() != FEATURES:
            raise ValueError(f"{name} feature columns do not match the expected ordered list")
    return x_train, x_val, x_train_scaled, x_val_scaled, y_train, y_val


def write_metadata(name: str, config: dict, metric: str, score: float,
                   best_epoch: int | None = None, extra: dict | None = None) -> None:
    """Write model provenance and the newly measured validation metric."""
    scaled_inputs = name in {"logistic_regression", "mlp", "autoencoder", "lstm", "gru"}
    feature_file = "X_train_scaled.parquet" if scaled_inputs else "X_train.parquet"
    validation_file = "X_val_scaled.parquet" if scaled_inputs else "X_val.parquet"
    record = {
        "model_name": name,
        "configuration": config,
        "random_seed": SEED,
        "features": FEATURES,
        "training_dataset": [f"data/processed/{feature_file}", "data/processed/y_train.parquet"],
        "validation_dataset": [f"data/processed/{validation_file}", "data/processed/y_val.parquet"],
        "validation_metric": metric,
        "reproduced_validation_metric": score,
        "reference_validation_ap": REFERENCE_AP[name],
        "best_epoch": best_epoch,
    }
    if extra:
        record.update(extra)
    (MODEL_DIR / f"{name}_metadata.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


class FraudMLP(nn.Module):
    """The two-hidden-layer classifier defined in notebook 03."""
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.20)
        self.output = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.output(x)


class FraudAutoencoder(nn.Module):
    """The 32-16-8-16-32 reconstruction network from notebook 03."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        return self.decoder(latent), latent


class LSTMClassifier(nn.Module):
    """Single-layer unidirectional LSTM that classifies the final step."""
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(32, 64, num_layers=1, batch_first=True, bidirectional=False)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.output_layer(hidden[-1]).squeeze(1)


class GRUClassifier(nn.Module):
    """Single-layer unidirectional GRU that classifies the final step."""
    def __init__(self) -> None:
        super().__init__()
        self.gru = nn.GRU(32, 64, num_layers=1, batch_first=True, bidirectional=False)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.output_layer(hidden[-1]).squeeze(1)


def train_supervised_torch(model: nn.Module, train_loader: DataLoader,
                           val_loader: DataLoader, y_train: np.ndarray,
                           device: torch.device, max_epochs: int = 50,
                           patience: int = 7) -> tuple[dict, int, float]:
    """Train with positive-class weighted BCE and retain best validation AP."""
    model = model.to(device)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_state = None
    best_ap = -np.inf
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
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
        print(f"{model.__class__.__name__} epoch {epoch:02d}: "
              f"train_loss={total_loss / sample_count:.6f}, val_AP={val_ap:.6f}")
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return best_state, best_epoch, float(best_ap)


def make_sequences(features: pd.DataFrame, labels: pd.Series,
                   length: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows within one split; target the final transaction."""
    values = features.to_numpy(dtype=np.float32)
    targets = labels.to_numpy(dtype=np.int64)
    windows = np.stack([values[i:i + length]
                        for i in range(len(values) - length + 1)])
    return windows, targets[length - 1:]


def main() -> None:
    """Fit all eight reference models and write weights plus metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    set_seed()
    x_train, x_val, xs_train, xs_val, y_train, y_val = load_splits()

    # The reference comparison uses unweighted baseline variants except LightGBM,
    # where the regularized unweighted validation-selected candidate is recorded.
    sklearn_models = {
        "random_forest": (
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            x_train, x_val,
            {"n_estimators": 200, "random_state": 42, "n_jobs": -1, "class_weight": None},
        ),
        "xgboost": (
            XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          objective="binary:logistic", eval_metric="logloss",
                          random_state=42, n_jobs=-1),
            x_train, x_val,
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8,
             "objective": "binary:logistic", "eval_metric": "logloss",
             "random_state": 42, "n_jobs": -1, "class_weight": None},
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            xs_train, xs_val,
            {"max_iter": 1000, "random_state": 42, "class_weight": None,
             "input": "scaled features"},
        ),
        "lightgbm": (
            LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1,
                           min_child_samples=100, colsample_bytree=0.8,
                           objective="binary", random_state=42, n_jobs=-1,
                           verbosity=-1),
            x_train, x_val,
            {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.1,
             "min_child_samples": 100, "colsample_bytree": 0.8,
             "objective": "binary", "random_state": 42, "n_jobs": -1,
             "verbosity": -1, "max_depth": -1, "class_weight": None},
        ),
    }
    for name, (estimator, train_x, val_x, config) in sklearn_models.items():
        estimator.fit(train_x, y_train)
        ap = average_precision_score(y_val, estimator.predict_proba(val_x)[:, 1])
        joblib.dump(estimator, MODEL_DIR / f"{name}.joblib")
        write_metadata(name, config, "Average Precision (AP)", float(ap))
        print(f"{name}: validation AP={ap:.6f} (reference {REFERENCE_AP[name]:.6f})")

    # Tabular neural networks follow the notebook's scaled feature representation.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_values = torch.from_numpy(xs_train.to_numpy(dtype=np.float32, copy=True))
    val_values = torch.from_numpy(xs_val.to_numpy(dtype=np.float32, copy=True))
    train_targets = torch.from_numpy(y_train.to_numpy(dtype=np.float32, copy=True))
    val_targets = torch.from_numpy(y_val.to_numpy(dtype=np.float32, copy=True))
    train_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(train_values, train_targets),
                              batch_size=1024, shuffle=True, generator=train_gen)
    val_loader = DataLoader(TensorDataset(val_values, val_targets),
                            batch_size=1024, shuffle=False)

    mlp = FraudMLP()
    mlp_state, mlp_epoch, mlp_ap = train_supervised_torch(
        mlp, train_loader, val_loader, y_train.to_numpy(), device)
    torch.save(mlp_state, MODEL_DIR / "mlp_state_dict.pt")
    write_metadata("mlp", {
        "architecture": [32, 64, 32, 1], "activation": "ReLU",
        "dropout": 0.20, "loss": "BCEWithLogitsLoss",
        "positive_class_weight": "training negatives / positives",
        "optimizer": "Adam", "learning_rate": 0.001, "batch_size": 1024,
        "max_epochs": 50, "early_stopping_patience": 7,
        "input": "scaled features", "training_batches_shuffled": True,
    }, "Average Precision (AP)", mlp_ap, mlp_epoch)
    print(f"mlp: validation AP={mlp_ap:.6f} (reference {REFERENCE_AP['mlp']:.6f})")

    # Autoencoder validation reconstruction MSE selects the state; validation
    # labels are used only afterward to report anomaly-ranking AP for reference.
    ae_train_values = train_values[y_train.to_numpy() == 0]
    ae_loader = DataLoader(TensorDataset(ae_train_values), batch_size=1024,
                           shuffle=True, generator=torch.Generator().manual_seed(SEED))
    autoencoder = FraudAutoencoder().to(device)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    mse = nn.MSELoss()
    best_ae_loss, best_ae_epoch, stale = np.inf, 0, 0
    best_ae_state = None
    for epoch in range(1, 51):
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
            val_reconstruction, _ = autoencoder(val_values.to(device))
            val_loss = float(mse(val_reconstruction, val_values.to(device)).item())
        print(f"FraudAutoencoder epoch {epoch:02d}: val_MSE={val_loss:.6f}")
        if val_loss < best_ae_loss:
            best_ae_loss, best_ae_epoch, stale = val_loss, epoch, 0
            best_ae_state = {key: value.detach().cpu().clone()
                             for key, value in autoencoder.state_dict().items()}
        else:
            stale += 1
        if stale >= 7:
            break
    assert best_ae_state is not None
    autoencoder.load_state_dict(best_ae_state)
    with torch.no_grad():
        val_reconstruction, _ = autoencoder(val_values.to(device))
        # Per-row mean squared reconstruction error is the anomaly score.
        ae_scores = ((val_reconstruction - val_values.to(device)) ** 2).mean(dim=1).cpu().numpy()
    ae_ap = float(average_precision_score(y_val, ae_scores))
    torch.save(best_ae_state, MODEL_DIR / "autoencoder_state_dict.pt")
    write_metadata("autoencoder", {
        "architecture": [32, 16, 8, 16, 32], "hidden_activation": "ReLU",
        "loss": "mean squared error", "optimizer": "Adam",
        "learning_rate": 0.001, "batch_size": 1024,
        "max_epochs": 50, "early_stopping_patience": 7,
        "training_rows": "legitimate training transactions only",
        "input": "scaled features", "training_batches_shuffled": True,
        "checkpoint_selection_metric": "validation reconstruction MSE",
        "best_validation_reconstruction_mse": best_ae_loss,
    }, "Average Precision (AP) from per-row reconstruction MSE", ae_ap,
       best_ae_epoch)
    print(f"autoencoder: validation AP={ae_ap:.6f} (reference {REFERENCE_AP['autoencoder']:.6f})")

    # Temporal windows are made independently within each split, matching the
    # notebook and preventing windows from crossing chronological boundaries.
    seq_train, seq_y_train = make_sequences(xs_train, y_train)
    seq_val, seq_y_val = make_sequences(xs_val, y_val)
    seq_train_t = torch.from_numpy(seq_train)
    seq_val_t = torch.from_numpy(seq_val)
    seq_y_train_t = torch.from_numpy(seq_y_train.astype(np.float32))
    seq_y_val_t = torch.from_numpy(seq_y_val.astype(np.float32))
    temporal_train_loader = DataLoader(
        TensorDataset(seq_train_t, seq_y_train_t), batch_size=1024, shuffle=False)
    temporal_val_loader = DataLoader(
        TensorDataset(seq_val_t, seq_y_val_t), batch_size=1024, shuffle=False)
    for name, model_class in (("lstm", LSTMClassifier), ("gru", GRUClassifier)):
        model = model_class()
        state, epoch, ap = train_supervised_torch(
            model, temporal_train_loader, temporal_val_loader,
            seq_y_train, device)
        torch.save(state, MODEL_DIR / f"{name}_state_dict.pt")
        write_metadata(name, {
            "architecture": {"input_size": 32, "hidden_size": 64,
                             "num_layers": 1, "batch_first": True,
                             "bidirectional": False, "output": "single linear logit"},
            "sequence_length": 10, "sequence_target": "Class of final transaction",
            "sequence_splits": "constructed independently per split",
            "loss": "BCEWithLogitsLoss",
            "positive_class_weight": "training sequence negatives / positives",
            "optimizer": "Adam", "learning_rate": 0.001,
            "batch_size": 1024, "max_epochs": 50,
            "early_stopping_patience": 7, "training_batches_shuffled": False,
            "input": "scaled features",
        }, "Average Precision (AP)", ap, epoch)
        print(f"{name}: validation AP={ap:.6f} (reference {REFERENCE_AP[name]:.6f})")

    print("\nArtifacts written under models/. Test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
