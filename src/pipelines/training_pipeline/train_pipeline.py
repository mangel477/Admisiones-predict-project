"""Training pipeline for the graduate admissions prediction project.

The stages run in this order: read the processed features written by the feature
pipeline, separate a test set, fit the model on the training split, evaluate it on
both sides of that split, and store the trained model together with its metrics.

This is where the model-dependent transformations live. Imputation, scaling and
encoding are parameterized by the training data, so they are fitted here and only
on the training split, and they travel inside the serialized pipeline: whoever
loads the artifact feeds it raw-scale columns and the preprocessing comes along.

The architecture is the one selected in ``notebooks/5-models``. Choosing the model
family is research and stays in that notebook; this pipeline trains the model that
was chosen, so the same data always produces the same artifact.

Run it standalone with::

    python src/pipelines/training_pipeline/train_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

LOR_COLUMN = "LOR "
TARGET_COLUMN = "Chance of Admit "
NUMERIC_COLUMNS = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    LOR_COLUMN,
    "CGPA",
]
BINARY_COLUMNS = ["Research"]

# Split and architecture as fixed in notebooks/5-models/05.seleccion-modelo.
RANDOM_STATE = 42
TEST_SIZE = 0.2
STRATIFICATION_BINS = 5
RIDGE_ALPHA = 10.0

FEATURE_RELATIVE_PATH = Path("data/04_feature/admisiones_features.parquet")
MODEL_RELATIVE_PATH = Path("models/modelo-seleccion-admisiones.joblib")
METRICS_RELATIVE_PATH = Path("data/08_reporting/metricas_entrenamiento.json")


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until the directory holding ``pyproject.toml`` is found."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found above {current}")


def load_features(path: Path) -> pd.DataFrame:
    """Read the feature table produced by the feature pipeline."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No feature table at {path}. Run the feature pipeline first: "
            "python src/pipelines/feature_pipeline/feature_pipeline.py"
        )
    logger.info("Reading features from %s", path)
    features = pd.read_parquet(path, engine="pyarrow")
    logger.debug("Feature table shape: %s", features.shape)
    return features


def split_features_and_label(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the predictors the model takes from the value it has to predict."""
    predictors = features[[*NUMERIC_COLUMNS, *BINARY_COLUMNS]]
    label = features[TARGET_COLUMN]
    return predictors, label


def split_train_test(
    predictors: pd.DataFrame, label: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out a test set, stratified by quantiles of the target.

    The target is continuous, so it is binned into quantiles only to stratify: with
    a few hundred records a blind split can leave one end of the range on one side.
    """
    bins = pd.qcut(label, q=STRATIFICATION_BINS, labels=False, duplicates="drop")
    x_train, x_test, y_train, y_test = train_test_split(
        predictors,
        label,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=bins,
    )
    logger.info("Split: %d training records, %d test records", len(x_train), len(x_test))
    return x_train, x_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Declare the model-dependent transformations, unfitted.

    ``OrdinalEncoder`` runs before its imputer on purpose: an unseen category becomes
    NaN instead of raising, and the imputer then resolves it.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("binary", binary, BINARY_COLUMNS),
        ]
    )


def build_model() -> Pipeline:
    """Assemble the selected architecture: preprocessing plus regressor, unfitted.

    Both travel in the same object so the artifact carries everything a prediction
    needs, and so the preprocessing can never be fitted outside the training split.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", Ridge(alpha=RIDGE_ALPHA, random_state=RANDOM_STATE)),
        ]
    )


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the model on the training split alone."""
    logger.info("Training Ridge(alpha=%s) on %d records", RIDGE_ALPHA, len(x_train))
    model = build_model()
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: Any, predictors: pd.DataFrame, label: pd.Series) -> dict[str, float]:
    """Score the model on one split.

    MAE is the primary metric because it reads in the units of the target: an error
    of 0.047 is 4.7 percentage points of admission chance. RMSE weighs the large
    misses more heavily, and R2 says how much of the variance is explained.
    """
    predictions = model.predict(predictors)
    return {
        "mae": float(mean_absolute_error(label, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(label, predictions))),
        "r2": float(r2_score(label, predictions)),
    }


def collect_metrics(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Build the evaluation report stored next to the model.

    Both sides of the split are scored: the gap between them is what exposes
    overfitting, and a report with test alone cannot show it.
    """
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "family": type(model.named_steps["model"]).__name__,
            "alpha": RIDGE_ALPHA,
            "random_state": RANDOM_STATE,
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratification_bins": STRATIFICATION_BINS,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
        },
        "metrics": {
            "train": evaluate_model(model, x_train, y_train),
            "test": evaluate_model(model, x_test, y_test),
        },
    }
    test = report["metrics"]["test"]
    logger.info(
        "Test metrics — MAE: %.4f | RMSE: %.4f | R2: %.4f", test["mae"], test["rmse"], test["r2"]
    )
    return report


def save_model(model: Pipeline, path: Path) -> Path:
    """Store the trained pipeline, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model stored in %s", path)
    return path


def save_metrics(report: dict[str, Any], path: Path) -> Path:
    """Store the evaluation report as JSON, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Evaluation results stored in %s", path)
    return path


def run_pipeline(
    features_path: Path, model_path: Path, metrics_path: Path
) -> tuple[Pipeline, dict[str, Any]]:
    """Run the whole training pipeline: read, split, train, evaluate, persist."""
    predictors, label = split_features_and_label(load_features(features_path))
    x_train, x_test, y_train, y_test = split_train_test(predictors, label)
    model = train_model(x_train, y_train)
    report = collect_metrics(model, x_train, x_test, y_train, y_test)
    save_model(model, model_path)
    save_metrics(report, metrics_path)
    return model, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments of the standalone entry point."""
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--features-path",
        type=Path,
        default=project_root / FEATURE_RELATIVE_PATH,
        help="Parquet file of the feature layer to read.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / MODEL_RELATIVE_PATH,
        help="File where the trained model is stored.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=project_root / METRICS_RELATIVE_PATH,
        help="File where the evaluation results are stored.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of the pipeline logs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point used when the script is executed on its own."""
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    run_pipeline(args.features_path, args.model_path, args.metrics_path)


if __name__ == "__main__":
    main()
