from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "significant_mistake"
EXCLUDED_FEATURES = {
    "game_id",
    "game_date",
    "ply",
    "fen_before",
    "fen_after",
    "san",
    "uci",
    "best_move_uci",
    "eval_after_cp",
    "cpl",
    "quality",
    "significant_mistake",
    "mistake_cpl_threshold",
    "engine_config_hash",
    "source_url",
    "white",
    "black",
    "result",
}


class TrainingDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    metadata_path: Path
    metrics: dict[str, float | None]
    train_rows: int
    test_rows: int


def chronological_game_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise TrainingDataError("Training dataset is empty")
    if TARGET not in df.columns:
        raise TrainingDataError(f"Training dataset is missing target column {TARGET}")
    if df[TARGET].nunique(dropna=True) < 2:
        raise TrainingDataError("Training data must contain both target classes")
    if "game_id" not in df.columns or "game_date" not in df.columns:
        raise TrainingDataError("Training data must contain game_id and game_date")
    if not 0 < test_fraction < 1:
        raise TrainingDataError("test_fraction must be between 0 and 1")

    parsed_dates = pd.to_datetime(df["game_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise TrainingDataError("Every training game must have a valid game_date")

    games = (
        df[["game_id", "game_date"]]
        .drop_duplicates("game_id")
        .assign(game_date=lambda x: pd.to_datetime(x["game_date"], errors="coerce"))
        .sort_values(["game_date", "game_id"], na_position="last")
    )
    if len(games) < 3:
        raise TrainingDataError("At least 3 games are required for chronological splitting")
    test_games_count = max(1, math.ceil(len(games) * test_fraction))
    if test_games_count >= len(games):
        test_games_count = len(games) - 1
    test_ids = set(games.tail(test_games_count)["game_id"])
    train = df[~df["game_id"].isin(test_ids)].copy()
    test = df[df["game_id"].isin(test_ids)].copy()
    if train.empty or test.empty:
        raise TrainingDataError("Chronological split produced an empty partition")
    if train[TARGET].nunique(dropna=True) < 2:
        raise TrainingDataError("Training partition must contain both target classes")
    return train, test


def _feature_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_FEATURES and df[column].notna().any()
    ]
    if not columns:
        raise TrainingDataError("No usable model features remain after leakage filtering")
    return columns


def _dataset_hash(df: pd.DataFrame, feature_columns: list[str]) -> str:
    subset = df[["game_id", "game_date", TARGET, *feature_columns]].copy()
    payload = subset.to_json(orient="split", date_format="iso", default_handler=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _safe_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float | None]:
    both_classes = y_true.nunique(dropna=True) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, probabilities)) if both_classes else None,
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def train_model(
    df: pd.DataFrame,
    model_dir: Path,
    *,
    test_fraction: float = 0.2,
) -> TrainingResult:
    train, test = chronological_game_split(df, test_fraction=test_fraction)
    feature_columns = _feature_columns(df)
    x_train = train[feature_columns].copy()
    x_test = test[feature_columns].copy()
    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    categorical = [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(df[column])
        or isinstance(df[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in feature_columns if column not in categorical]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ],
        remainder="drop",
    )
    model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        verbosity=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = _safe_metrics(y_test, probabilities)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "mistake_model.joblib"
    metadata_path = model_dir / "mistake_model.metadata.json"
    joblib.dump(pipeline, model_path)

    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    importances = pipeline.named_steps["model"].feature_importances_.tolist()
    feature_importance = sorted(
        (
            {"feature": feature, "importance": int(importance)}
            for feature, importance in zip(transformed_names, importances)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    cutoff = pd.to_datetime(test["game_date"], errors="coerce").min()
    threshold_values = (
        sorted(df["mistake_cpl_threshold"].dropna().unique().tolist())
        if "mistake_cpl_threshold" in df.columns
        else []
    )
    metadata = {
        "dataset_version": 1,
        "mistake_cpl_threshold": threshold_values[0] if len(threshold_values) == 1 else None,
        "feature_columns": feature_columns,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "target": TARGET,
        "test_fraction": test_fraction,
        "chronological_test_start": cutoff.isoformat() if not pd.isna(cutoff) else None,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "dataset_hash": _dataset_hash(df, feature_columns),
        "metrics": metrics,
        "feature_importance": feature_importance,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return TrainingResult(
        model_path=model_path,
        metadata_path=metadata_path,
        metrics=metrics,
        train_rows=len(train),
        test_rows=len(test),
    )
