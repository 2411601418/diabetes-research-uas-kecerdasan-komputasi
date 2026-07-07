from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET = "Diabetes_binary"
DATASET_NAME = "diabetes_012_health_indicators_BRFSS2015.csv"
SOURCE_TARGET = "Diabetes_012"
CROSS_DATASET_NAME = "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
CROSS_TARGET = "Diabetes_binary"
FEATURE_GROUPS = {
    "health_condition": ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk", "GenHlth", "MentHlth", "PhysHlth"],
    "lifestyle": ["BMI", "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
    "healthcare_access": ["CholCheck", "AnyHealthcare", "NoDocbcCost"],
    "demographic": ["Sex", "Age", "Education", "Income"],
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_path(user_path: str | None) -> Path:
    candidates = []
    if user_path:
        candidates.append(Path(user_path))

    root = project_root()
    candidates.extend(
        [
            root / "data" / DATASET_NAME,
            root / DATASET_NAME,
            root.parent / DATASET_NAME,
            Path.cwd() / DATASET_NAME,
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Dataset tidak ditemukan. Lokasi yang dicek:\n{checked}")


def resolve_cross_dataset_path(user_path: str | None = None) -> Path | None:
    root = project_root()
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    candidates.extend(
        [
            root / "data" / CROSS_DATASET_NAME,
            root / CROSS_DATASET_NAME,
            root.parent / CROSS_DATASET_NAME,
            Path.cwd() / CROSS_DATASET_NAME,
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def make_output_dirs(root: Path) -> dict[str, Path]:
    output = root / "outputs"
    dirs = {
        "output": output,
        "plots": output / "plots",
        "models": output / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if SOURCE_TARGET in df.columns:
        X = df.drop(columns=[SOURCE_TARGET])
        y = (df[SOURCE_TARGET].astype(int) > 0).astype(int)
    elif TARGET in df.columns:
        X = df.drop(columns=[TARGET])
        y = df[TARGET].astype(int)
    else:
        raise ValueError(f"Kolom target '{SOURCE_TARGET}' atau '{TARGET}' tidak ditemukan.")
    processed_df = X.copy()
    processed_df[TARGET] = y
    return processed_df, X, y


def load_cross_dataset(path: Path, expected_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(f"Cross dataset tidak memiliki fitur berikut: {missing}")
    X = df[expected_features].copy()
    if CROSS_TARGET in df.columns:
        y = df[CROSS_TARGET].astype(int)
    elif SOURCE_TARGET in df.columns:
        y = (df[SOURCE_TARGET].astype(int) > 0).astype(int)
    else:
        raise ValueError(f"Kolom target '{CROSS_TARGET}' atau '{SOURCE_TARGET}' tidak ditemukan pada cross dataset.")
    return df, X, y


def save_eda(df: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    feature_summary = df.describe().T
    feature_summary["missing"] = df.isna().sum()
    feature_summary.to_csv(out_dirs["output"] / "feature_summary.csv")

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Distribusi Kelas Diabetes")
    plt.xlabel("Diabetes_binary")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "class_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(14, 11))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, square=False, cbar_kws={"shrink": 0.7})
    plt.title("Heatmap Korelasi Fitur")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "correlation_heatmap.png", dpi=160)
    plt.close()


if TORCH_AVAILABLE:

    class TorchTabularClassifier(BaseEstimator, ClassifierMixin):
        def __init__(
            self,
            architecture: str = "mlp",
            hidden_dim: int = 64,
            hidden_dim_2: int = 32,
            residual_blocks: int = 2,
            dropout: float = 0.2,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            batch_size: int = 256,
            epochs: int = 25,
            patience: int = 6,
            random_state: int = RANDOM_STATE,
            verbose: bool = False,
        ):
            self.architecture = architecture
            self.hidden_dim = hidden_dim
            self.hidden_dim_2 = hidden_dim_2
            self.residual_blocks = residual_blocks
            self.dropout = dropout
            self.lr = lr
            self.weight_decay = weight_decay
            self.batch_size = batch_size
            self.epochs = epochs
            self.patience = patience
            self.random_state = random_state
            self.verbose = verbose

        def _build_model(self, n_features: int) -> torch.nn.Module:
            if self.architecture == "residual_mlp":
                return ResidualMLP(
                    n_features=n_features,
                    hidden_dim=self.hidden_dim,
                    residual_blocks=self.residual_blocks,
                    dropout=self.dropout,
                )
            return MLP(
                n_features=n_features,
                hidden_dim=self.hidden_dim,
                hidden_dim_2=self.hidden_dim_2,
                dropout=self.dropout,
            )

        def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchTabularClassifier":
            set_seed(self.random_state)
            X_np = np.asarray(X, dtype=np.float32)
            y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            self.classes_ = np.array([0, 1])
            self.n_features_in_ = X_np.shape[1]

            X_train, X_val, y_train, y_val = train_test_split(
                X_np, y_np, test_size=0.15, stratify=y_np.ravel(), random_state=self.random_state
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device_ = str(device)
            self.model_ = self._build_model(X_np.shape[1]).to(device)
            optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            criterion = torch.nn.BCEWithLogitsLoss()

            train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            generator = torch.Generator().manual_seed(self.random_state)
            loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                generator=generator,
            )
            X_val_t = torch.from_numpy(X_val).to(device)
            y_val_t = torch.from_numpy(y_val).to(device)

            best_loss = math.inf
            best_state = None
            wait = 0
            self.history_ = []
            for epoch in range(self.epochs):
                self.model_.train()
                losses = []
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(self.model_(xb), yb)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                self.model_.eval()
                with torch.no_grad():
                    val_loss = criterion(self.model_(X_val_t), y_val_t).item()
                train_loss = float(np.mean(losses))
                self.history_.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

            if best_state is not None:
                self.model_.load_state_dict(best_state)
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            X_np = np.asarray(X, dtype=np.float32)
            device = torch.device(self.device_)
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(torch.from_numpy(X_np).to(device))
                probs_1 = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            return np.column_stack([1 - probs_1, probs_1])

        def predict(self, X: np.ndarray) -> np.ndarray:
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


    class MLP(torch.nn.Module):
        def __init__(self, n_features: int, hidden_dim: int, hidden_dim_2: int, dropout: float):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_features, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim_2),
                torch.nn.BatchNorm1d(hidden_dim_2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim_2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class ResidualBlock(torch.nn.Module):
        def __init__(self, hidden_dim: int, dropout: float):
            super().__init__()
            self.block = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )
            self.activation = torch.nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.activation(x + self.block(x))


    class ResidualMLP(torch.nn.Module):
        def __init__(self, n_features: int, hidden_dim: int, residual_blocks: int, dropout: float):
            super().__init__()
            layers: list[torch.nn.Module] = [
                torch.nn.Linear(n_features, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
            ]
            for _ in range(residual_blocks):
                layers.append(ResidualBlock(hidden_dim, dropout))
            layers.extend([torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim, 1)])
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

else:

    class TorchTabularClassifier(BaseEstimator, ClassifierMixin):
        def __init__(
            self,
            architecture: str = "mlp",
            hidden_dim: int = 64,
            hidden_dim_2: int = 32,
            residual_blocks: int = 2,
            dropout: float = 0.2,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            batch_size: int = 256,
            epochs: int = 25,
            patience: int = 6,
            random_state: int = RANDOM_STATE,
            verbose: bool = False,
        ):
            self.architecture = architecture
            self.hidden_dim = hidden_dim
            self.hidden_dim_2 = hidden_dim_2
            self.residual_blocks = residual_blocks
            self.dropout = dropout
            self.lr = lr
            self.weight_decay = weight_decay
            self.batch_size = batch_size
            self.epochs = epochs
            self.patience = patience
            self.random_state = random_state
            self.verbose = verbose

        def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchTabularClassifier":
            if self.architecture == "residual_mlp":
                hidden = tuple([self.hidden_dim] * max(2, self.residual_blocks + 1))
            else:
                hidden = (self.hidden_dim, self.hidden_dim_2)
            self.model_ = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                alpha=self.weight_decay,
                batch_size=self.batch_size,
                learning_rate_init=self.lr,
                max_iter=self.epochs,
                early_stopping=True,
                n_iter_no_change=self.patience,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self.model_.fit(X, y)
            self.classes_ = self.model_.classes_
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return self.model_.predict_proba(X)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self.model_.predict(X)


@dataclass
class EvaluationResult:
    name: str
    estimator: Any
    metrics: dict[str, float]
    train_seconds: float
    inference_seconds: float


def probability_scores(estimator: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return estimator.predict(X)


def evaluate_estimator(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    start = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = estimator.predict(X_test)
    y_score = probability_scores(estimator, X_test)
    inference_seconds = time.perf_counter() - start

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }
    return EvaluationResult(name, estimator, metrics, train_seconds, inference_seconds)


def build_baseline_models(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1200,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=220,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Histogram Gradient Boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.06,
                        l2_regularization=0.01,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    TorchTabularClassifier(
                        architecture="mlp",
                        hidden_dim=64,
                        hidden_dim_2=32,
                        dropout=0.20,
                        lr=1e-3,
                        epochs=args.deep_epochs,
                        patience=max(4, args.deep_epochs // 4),
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Residual MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    TorchTabularClassifier(
                        architecture="residual_mlp",
                        hidden_dim=96,
                        residual_blocks=2,
                        dropout=0.15,
                        lr=8e-4,
                        epochs=args.deep_epochs,
                        patience=max(4, args.deep_epochs // 4),
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def model_family(best_name: str) -> str:
    if best_name == "Logistic Regression":
        return "logreg"
    if best_name == "Random Forest":
        return "rf"
    if best_name == "Histogram Gradient Boosting":
        return "hgb"
    if best_name == "MLP":
        return "mlp"
    if best_name == "Residual MLP":
        return "residual_mlp"
    raise ValueError(best_name)


def random_individual(n_features: int, family: str) -> dict[str, Any]:
    mask = np.random.rand(n_features) < np.random.uniform(0.45, 0.95)
    if not mask.any():
        mask[np.random.randint(0, n_features)] = True
    params: dict[str, Any] = {}
    if family == "logreg":
        params = {"C": 10 ** np.random.uniform(-2.5, 1.5)}
    elif family == "rf":
        params = {
            "n_estimators": int(np.random.choice([120, 180, 240, 320])),
            "max_depth": random.choice([None, 5, 8, 12, 16]),
            "min_samples_leaf": int(np.random.choice([1, 2, 4, 8])),
            "max_features": random.choice(["sqrt", "log2", None]),
        }
    elif family == "hgb":
        params = {
            "learning_rate": 10 ** np.random.uniform(-2.0, -0.55),
            "max_iter": int(np.random.choice([100, 160, 220, 300])),
            "max_leaf_nodes": int(np.random.choice([15, 31, 45, 63])),
            "l2_regularization": 10 ** np.random.uniform(-4, 0.5),
        }
    else:
        params = {
            "hidden_dim": int(np.random.choice([32, 64, 96, 128])),
            "hidden_dim_2": int(np.random.choice([16, 32, 64])),
            "residual_blocks": int(np.random.choice([1, 2, 3])),
            "dropout": float(np.random.uniform(0.05, 0.45)),
            "lr": 10 ** np.random.uniform(-4, -2.3),
            "weight_decay": 10 ** np.random.uniform(-6, -2.5),
            "batch_size": int(np.random.choice([128, 256, 512])),
        }
    return {"mask": mask, "params": params, "fitness": None}


def build_ga_estimator(family: str, params: dict[str, Any], deep_epochs: int) -> Pipeline:
    if family == "logreg":
        model = LogisticRegression(
            C=params["C"],
            max_iter=1200,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    elif family == "rf":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif family == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"],
            l2_regularization=params["l2_regularization"],
            random_state=RANDOM_STATE,
        )
    else:
        model = TorchTabularClassifier(
            architecture=family,
            hidden_dim=params["hidden_dim"],
            hidden_dim_2=params["hidden_dim_2"],
            residual_blocks=params["residual_blocks"],
            dropout=params["dropout"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            epochs=deep_epochs,
            patience=max(3, deep_epochs // 3),
            random_state=RANDOM_STATE,
        )
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def mutate(ind: dict[str, Any], family: str, mutation_rate: float) -> dict[str, Any]:
    child = {"mask": ind["mask"].copy(), "params": dict(ind["params"]), "fitness": None}
    flips = np.random.rand(child["mask"].size) < mutation_rate
    child["mask"][flips] = ~child["mask"][flips]
    if not child["mask"].any():
        child["mask"][np.random.randint(0, child["mask"].size)] = True

    if random.random() < 0.6:
        fresh = random_individual(child["mask"].size, family)["params"]
        key = random.choice(list(fresh.keys()))
        child["params"][key] = fresh[key]
    return child


def crossover(parent_a: dict[str, Any], parent_b: dict[str, Any]) -> dict[str, Any]:
    mask_pick = np.random.rand(parent_a["mask"].size) < 0.5
    mask = np.where(mask_pick, parent_a["mask"], parent_b["mask"])
    if not mask.any():
        mask[np.random.randint(0, mask.size)] = True
    params = {}
    for key in parent_a["params"]:
        params[key] = parent_a["params"][key] if random.random() < 0.5 else parent_b["params"][key]
    return {"mask": mask, "params": params, "fitness": None}


def run_ga_optimization(
    family: str,
    feature_names: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    args: argparse.Namespace,
    out_dirs: dict[str, Path],
) -> tuple[EvaluationResult, dict[str, Any], pd.DataFrame]:
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    population = [random_individual(len(feature_names), family) for _ in range(args.ga_population)]
    history = []

    def fitness(ind: dict[str, Any]) -> float:
        if ind["fitness"] is not None:
            return ind["fitness"]
        cols = [feature_names[i] for i, keep in enumerate(ind["mask"]) if keep]
        estimator = build_ga_estimator(family, ind["params"], args.ga_deep_epochs)
        estimator.fit(X_sub[cols], y_sub)
        pred = estimator.predict(X_val[cols])
        score = probability_scores(estimator, X_val[cols])
        f1 = f1_score(y_val, pred, zero_division=0)
        auc = roc_auc_score(y_val, score)
        feature_penalty = 0.015 * (len(cols) / len(feature_names))
        ind["fitness"] = (0.7 * f1) + (0.3 * auc) - feature_penalty
        return ind["fitness"]

    for generation in range(args.ga_generations):
        population.sort(key=fitness, reverse=True)
        best = population[0]
        history.append(
            {
                "generation": generation + 1,
                "best_fitness": fitness(best),
                "selected_features": int(best["mask"].sum()),
            }
        )

        elites = population[: max(2, args.ga_population // 5)]
        children = [dict(mask=e["mask"].copy(), params=dict(e["params"]), fitness=e["fitness"]) for e in elites]
        while len(children) < args.ga_population:
            contenders = random.sample(population[: max(4, len(population) // 2)], 2)
            child = crossover(contenders[0], contenders[1])
            child = mutate(child, family, args.ga_mutation_rate)
            children.append(child)
        population = children

    population.sort(key=fitness, reverse=True)
    best = population[0]
    selected_cols = [feature_names[i] for i, keep in enumerate(best["mask"]) if keep]
    best_estimator = build_ga_estimator(family, best["params"], args.deep_epochs)
    result = evaluate_estimator(
        "GA Optimized " + family,
        best_estimator,
        X_train[selected_cols],
        y_train,
        X_test[selected_cols],
        y_test,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dirs["output"] / "ga_history.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.plot(history_df["generation"], history_df["best_fitness"], marker="o")
    plt.title("Konvergensi Genetic Algorithm")
    plt.xlabel("Generasi")
    plt.ylabel("Best Fitness")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "ga_convergence.png", dpi=160)
    plt.close()

    ga_info = {
        "family": family,
        "params": best["params"],
        "selected_features": selected_cols,
        "fitness": float(best["fitness"]),
    }
    with open(out_dirs["output"] / "ga_best_config.json", "w", encoding="utf-8") as f:
        json.dump(ga_info, f, indent=2)
    return result, ga_info, history_df


def plot_results(
    results: list[EvaluationResult],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dirs: dict[str, Path],
    feature_sets: dict[str, list[str]] | None = None,
) -> None:
    rows = []
    for result in results:
        row = {"model": result.name, **result.metrics}
        rows.append(row)
    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)

    plt.figure(figsize=(9, 5))
    melted = metrics_df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"])
    sns.barplot(data=melted, x="value", y="model", hue="variable")
    plt.xlim(0, 1)
    plt.title("Perbandingan Metrik Model")
    plt.xlabel("Skor")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "model_comparison.png", dpi=160)
    plt.close()

    for result in results:
        features = feature_sets.get(result.name, list(X_test.columns)) if feature_sets else list(X_test.columns)
        pred = result.estimator.predict(X_test[features])
        cm = confusion_matrix(y_test, pred)
        display = ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"])
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {result.name}")
        plt.tight_layout()
        safe_name = result.name.lower().replace(" ", "_").replace("/", "_")
        plt.savefig(out_dirs["plots"] / f"confusion_matrix_{safe_name}.png", dpi=160)
        plt.close()


def save_permutation_importance(
    result: EvaluationResult,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dirs: dict[str, Path],
    selected_features: list[str] | None = None,
) -> None:
    features = selected_features or list(X_test.columns)
    X_eval = X_test[features]
    importance = permutation_importance(
        result.estimator,
        X_eval,
        y_test,
        scoring="f1",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(out_dirs["output"] / "permutation_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df.head(15), x="importance_mean", y="feature")
    plt.title("Top 15 Permutation Importance")
    plt.xlabel("Mean F1 Decrease")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "permutation_importance.png", dpi=160)
    plt.close()


def hgb_pipeline(**overrides: Any) -> Pipeline:
    params = {
        "max_iter": 220,
        "learning_rate": 0.06,
        "l2_regularization": 0.01,
        "random_state": RANDOM_STATE,
    }
    params.update(overrides)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingClassifier(**params)),
        ]
    )


def run_ablation_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dirs: dict[str, Path],
) -> pd.DataFrame:
    rows = []
    feature_sets = {"all_features": list(X_train.columns)}
    for group_name, group_features in FEATURE_GROUPS.items():
        feature_sets[f"without_{group_name}"] = [c for c in X_train.columns if c not in group_features]

    for setting, columns in feature_sets.items():
        result = evaluate_estimator(
            setting,
            hgb_pipeline(),
            X_train[columns],
            y_train,
            X_test[columns],
            y_test,
        )
        rows.append(
            {
                "setting": setting,
                "n_features": len(columns),
                **result.metrics,
                "train_seconds": result.train_seconds,
            }
        )

    ablation_df = pd.DataFrame(rows)
    baseline_f1 = float(ablation_df.loc[ablation_df["setting"] == "all_features", "f1"].iloc[0])
    ablation_df["f1_delta_vs_all_features"] = ablation_df["f1"] - baseline_f1
    ablation_df.to_csv(out_dirs["output"] / "ablation_study.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=ablation_df.sort_values("f1"), x="f1", y="setting")
    plt.title("Ablation Study - Histogram Gradient Boosting")
    plt.xlabel("F1-score")
    plt.ylabel("")
    plt.xlim(max(0, ablation_df["f1"].min() - 0.02), min(1, ablation_df["f1"].max() + 0.02))
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "ablation_study.png", dpi=160)
    plt.close()
    return ablation_df


def run_hyperparameter_sensitivity(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_dirs: dict[str, Path],
) -> pd.DataFrame:
    experiments = []
    for value in [0.03, 0.06, 0.10, 0.15]:
        experiments.append(("learning_rate", value, {"learning_rate": value}))
    for value in [15, 31, 45, 63]:
        experiments.append(("max_leaf_nodes", value, {"max_leaf_nodes": value}))

    rows = []
    for param_name, param_value, overrides in experiments:
        result = evaluate_estimator(
            f"{param_name}={param_value}",
            hgb_pipeline(**overrides),
            X_train,
            y_train,
            X_test,
            y_test,
        )
        rows.append(
            {
                "parameter": param_name,
                "value": param_value,
                **result.metrics,
                "train_seconds": result.train_seconds,
            }
        )

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df.to_csv(out_dirs["output"] / "hyperparameter_sensitivity.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    for param_name, part in sensitivity_df.groupby("parameter"):
        plt.plot(part["value"].astype(float), part["f1"], marker="o", label=param_name)
    plt.title("Hyperparameter Sensitivity - Histogram Gradient Boosting")
    plt.xlabel("Nilai Hyperparameter")
    plt.ylabel("F1-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "hyperparameter_sensitivity.png", dpi=160)
    plt.close()
    return sensitivity_df


def run_cross_dataset_evaluation(
    results: list[EvaluationResult],
    X_cross: pd.DataFrame,
    y_cross: pd.Series,
    out_dirs: dict[str, Path],
    feature_sets: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    rows = []
    for result in results:
        features = feature_sets.get(result.name, list(X_cross.columns)) if feature_sets else list(X_cross.columns)
        start = time.perf_counter()
        pred = result.estimator.predict(X_cross[features])
        score = probability_scores(result.estimator, X_cross[features])
        inference_seconds = time.perf_counter() - start
        rows.append(
            {
                "model": result.name,
                "accuracy": accuracy_score(y_cross, pred),
                "precision": precision_score(y_cross, pred, zero_division=0),
                "recall": recall_score(y_cross, pred, zero_division=0),
                "f1": f1_score(y_cross, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_cross, score),
                "inference_seconds": inference_seconds,
            }
        )

    cross_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    cross_df.to_csv(out_dirs["output"] / "cross_dataset_evaluation.csv", index=False)

    plt.figure(figsize=(9, 5))
    melted = cross_df.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"])
    sns.barplot(data=melted, x="value", y="model", hue="variable")
    plt.xlim(0, 1)
    plt.title("Cross-Dataset Evaluation pada BRFSS 2015 Diabetes 012")
    plt.xlabel("Skor")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_dirs["plots"] / "cross_dataset_evaluation.png", dpi=160)
    plt.close()
    return cross_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eksperimen komparatif ML, DL, dan GA untuk diabetes BRFSS 2015.")
    parser.add_argument("--data", default=None, help="Path ke CSV dataset.")
    parser.add_argument("--cross-data", default=None, help="Path ke CSV dataset eksternal untuk cross-dataset evaluation.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--deep-epochs", type=int, default=25)
    parser.add_argument("--ga-generations", type=int, default=8)
    parser.add_argument("--ga-population", type=int, default=12)
    parser.add_argument("--ga-mutation-rate", type=float, default=0.08)
    parser.add_argument("--ga-deep-epochs", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="Mempercepat eksperimen untuk smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.deep_epochs = min(args.deep_epochs, 8)
        args.ga_generations = min(args.ga_generations, 3)
        args.ga_population = min(args.ga_population, 6)
        args.ga_deep_epochs = min(args.ga_deep_epochs, 4)

    set_seed(RANDOM_STATE)
    root = project_root()
    out_dirs = make_output_dirs(root)
    dataset_path = resolve_dataset_path(args.data)
    df, X, y = load_data(dataset_path)
    save_eda(df, out_dirs)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"Dataset: {dataset_path}")
    print(f"Shape: {df.shape}, train: {X_train.shape}, test: {X_test.shape}")
    if TORCH_AVAILABLE:
        print(f"Device PyTorch: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    else:
        print("Device PyTorch: unavailable, using sklearn MLP fallback")

    results: list[EvaluationResult] = []
    for name, estimator in build_baseline_models(args).items():
        print(f"Training {name}...")
        result = evaluate_estimator(name, estimator, X_train, y_train, X_test, y_test)
        results.append(result)
        print(
            f"{name}: f1={result.metrics['f1']:.4f}, "
            f"roc_auc={result.metrics['roc_auc']:.4f}, "
            f"time={result.train_seconds:.2f}s"
        )

    best_baseline = max(results, key=lambda r: (r.metrics["f1"], r.metrics["roc_auc"]))
    family = model_family(best_baseline.name)
    print(f"Best baseline: {best_baseline.name}. Running GA optimization for {family}...")
    ga_result, ga_info, _ = run_ga_optimization(
        family,
        list(X.columns),
        X_train,
        y_train,
        X_test,
        y_test,
        args,
        out_dirs,
    )
    results.append(ga_result)

    metrics_rows = []
    for result in results:
        row = {
            "model": result.name,
            **result.metrics,
            "train_seconds": result.train_seconds,
            "inference_seconds": result.inference_seconds,
        }
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(out_dirs["output"] / "metrics.csv", index=False)

    final_best = max(results, key=lambda r: (r.metrics["f1"], r.metrics["roc_auc"]))
    selected_features = ga_info["selected_features"] if final_best.name.startswith("GA Optimized") else None
    feature_sets = {ga_result.name: ga_info["selected_features"]}
    plot_results(results, X_test, y_test, out_dirs, feature_sets=feature_sets)
    save_permutation_importance(final_best, X_test, y_test, out_dirs, selected_features=selected_features)
    ablation_df = run_ablation_study(X_train, y_train, X_test, y_test, out_dirs)
    sensitivity_df = run_hyperparameter_sensitivity(X_train, y_train, X_test, y_test, out_dirs)
    cross_dataset_path = resolve_cross_dataset_path(args.cross_data)
    cross_df = pd.DataFrame()
    cross_info: dict[str, Any] | str
    if cross_dataset_path:
        cross_raw_df, X_cross, y_cross = load_cross_dataset(cross_dataset_path, list(X.columns))
        cross_df = run_cross_dataset_evaluation(results, X_cross, y_cross, out_dirs, feature_sets=feature_sets)
        original_cross_target = CROSS_TARGET if CROSS_TARGET in cross_raw_df.columns else SOURCE_TARGET
        cross_info = {
            "dataset": str(cross_dataset_path),
            "rows": int(cross_raw_df.shape[0]),
            "original_target": original_cross_target,
            "original_target_distribution": cross_raw_df[original_cross_target].value_counts().sort_index().to_dict(),
            "binary_target_mapping": "Jika target Diabetes_012, nilai 0 menjadi 0 dan nilai 1 atau 2 menjadi 1. Jika target Diabetes_binary, nilai target digunakan langsung.",
            "binary_target_distribution": y_cross.value_counts().sort_index().to_dict(),
            "metrics": cross_df.to_dict(orient="records"),
        }
    else:
        cross_info = "Tidak dilakukan karena file cross dataset tidak ditemukan."

    with open(out_dirs["models"] / "best_model.pkl", "wb") as f:
        pickle.dump(final_best.estimator, f)
    joblib.dump({"columns": list(X.columns), "target": TARGET}, out_dirs["models"] / "metadata.joblib")

    summary = {
        "dataset": str(dataset_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_mapping": "Dataset utama memakai Diabetes_012 yang dikonversi menjadi biner: 0 tetap 0, sedangkan 1 dan 2 menjadi 1.",
        "target_distribution": y.value_counts().sort_index().to_dict(),
        "best_baseline": best_baseline.name,
        "best_final": final_best.name,
        "ga_info": ga_info,
        "metrics": metrics_df.to_dict(orient="records"),
        "ablation_study": ablation_df.to_dict(orient="records"),
        "hyperparameter_sensitivity": sensitivity_df.to_dict(orient="records"),
        "cross_dataset_evaluation": cross_info,
    }
    with open(out_dirs["output"] / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal ranking:")
    print(metrics_df.to_string(index=False))
    if not cross_df.empty:
        print("\nCross-dataset ranking:")
        print(cross_df.to_string(index=False))
    print(f"\nOutputs saved to: {out_dirs['output']}")


if __name__ == "__main__":
    main()
