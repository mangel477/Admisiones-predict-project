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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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

# Cross-validation over the training split. The target is continuous, so the folds
# are stratified on its quantiles: with 320 records and ten folds, a blind split can
# hand a fold a skewed slice of the range. The bins only assign folds — every model
# is still fitted on the continuous target.
CV_FOLDS = 10

# Thresholds that turn the three sets of numbers into a verdict. They are deliberately
# loose: their job is to catch a model that is clearly broken, not to grade a good one.
OVERFITTING_GAP_RATIO = 0.25
UNDERFITTING_IMPROVEMENT = 1.2
TEST_REPRESENTATIVE_STD = 2.0

# Checks that make training unsound when they fail: a record shared by both sides
# turns the test metrics into a memory test, so the run aborts. Drift between the
# two sides is reported as a warning instead — it is a signal worth investigating,
# not proof that the split cannot be used.
INDEX_OVERLAP_CHECK = "Index overlap"
RECORD_OVERLAP_CHECK = "Record overlap"
LEAKAGE_CHECKS = frozenset({INDEX_OVERLAP_CHECK, RECORD_OVERLAP_CHECK})


class TrainTestSplitError(Exception):
    """Raised when the train/test split leaks information between both sides."""


FEATURE_RELATIVE_PATH = Path("data/04_feature/admisiones_features.parquet")
MODEL_RELATIVE_PATH = Path("src/model/modelo-seleccion-admisiones.joblib")
METRICS_RELATIVE_PATH = Path("src/model/metricas_entrenamiento.json")


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


def _as_plain_numeric(predictors: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    """Join one side of the split into the flat numeric frame the drift report takes."""
    joined = pd.concat([predictors, label], axis="columns")
    return joined.apply(pd.to_numeric, errors="coerce").astype("float64")


def _measure_leakage(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Count records the two sides share, by index and by content.

    Leakage is not a statistical question, it is a set intersection, so it is
    measured exactly here instead of being estimated by a library. Content overlap
    is the one that matters: two different index labels can still carry the same
    candidate, and the model would then be scored on a record it memorized.
    """
    shared_index = sorted(set(x_train.index) & set(x_test.index))
    train_records = _as_plain_numeric(x_train, y_train)
    test_records = _as_plain_numeric(x_test, y_test)
    traced = test_records.merge(train_records.drop_duplicates(), how="left", indicator=True)
    shared_records = int((traced["_merge"] == "both").sum())
    return {
        "shared_index_records": len(shared_index),
        "shared_record_count": shared_records,
        "shared_record_share": round(shared_records / max(len(test_records), 1), 4),
    }


def _measure_drift(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compare the distribution of every column between the two sides with evidently.

    The label is measured too: a split whose features match but whose target does
    not is still a split the test metrics cannot be trusted on.

    evidently is imported here rather than at the top of the module because it costs
    about three seconds to import, and every caller would pay it — including
    ``--help`` and every test collection — for a function most of them never reach.
    """
    from evidently import DataDefinition, Dataset, Report  # noqa: PLC0415
    from evidently.metrics import DriftedColumnsCount, ValueDrift  # noqa: PLC0415

    columns = [*NUMERIC_COLUMNS, *BINARY_COLUMNS, TARGET_COLUMN]
    definition = DataDefinition(
        numerical_columns=[*NUMERIC_COLUMNS, TARGET_COLUMN],
        categorical_columns=list(BINARY_COLUMNS),
    )
    report = Report(metrics=[*[ValueDrift(column=name) for name in columns], DriftedColumnsCount()])
    result = report.run(
        current_data=Dataset.from_pandas(
            _as_plain_numeric(x_test, y_test), data_definition=definition
        ),
        reference_data=Dataset.from_pandas(
            _as_plain_numeric(x_train, y_train), data_definition=definition
        ),
    )

    measured = result.dict()["metrics"]
    per_column: list[dict[str, Any]] = []
    for entry in measured[:-1]:
        config = entry["config"]
        score = float(entry["value"])
        threshold = float(config["threshold"])
        per_column.append(
            {
                "name": config["column"],
                "method": config["method"],
                "score": round(score, 6),
                "threshold": threshold,
                "drifted": score < threshold,
            }
        )
    summary = measured[-1]["value"]
    return {
        "drifted_columns": int(summary["count"]),
        "drifted_share": float(summary["share"]),
        "columns": per_column,
    }


def validate_train_test_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Check the split for leakage and for distribution differences between sides.

    Leakage is fatal: if a record appears on both sides, the test metrics measure
    memory instead of generalization, so every number downstream is a lie. Those
    checks raise :class:`TrainTestSplitError` before any model is fitted.

    Drift is reported as a warning. With a stratified split it should not appear,
    and when it does it points at the data rather than at the split, so the run
    continues and the report carries the evidence.
    """
    logger.info("Validating the train/test split")
    leakage = _measure_leakage(x_train, x_test, y_train, y_test)
    drift = _measure_drift(x_train, x_test, y_train, y_test)

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    overlaps = (
        (
            INDEX_OVERLAP_CHECK,
            leakage["shared_index_records"],
            f"{leakage['shared_index_records']} record(s) appear on both sides by index",
        ),
        (
            RECORD_OVERLAP_CHECK,
            leakage["shared_record_count"],
            f"{leakage['shared_record_count']} test record(s) also appear in training, "
            f"{leakage['shared_record_share']:.2%} of the test set",
        ),
    )
    for name, count, details in overlaps:
        failed = count > 0
        checks.append(
            {
                "name": name,
                "status": "failed" if failed else "passed",
                "severity": "error",
                "details": details if failed else "",
            }
        )
        if failed:
            errors.append(f"{name}: {details}")

    for column in drift["columns"]:
        details = f"{column['method']} = {column['score']} below threshold {column['threshold']}"
        checks.append(
            {
                "name": f"Drift: {column['name']}",
                "status": "failed" if column["drifted"] else "passed",
                "severity": "warning",
                "details": details if column["drifted"] else "",
            }
        )
        if column["drifted"]:
            warnings.append(f"drift detected in {column['name']} — {details}")

    report = {
        "passed": not errors,
        "leakage": leakage,
        "drift": drift,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
    for warning in warnings:
        logger.warning("Train/test split: %s", warning)
    if errors:
        raise TrainTestSplitError(
            "\n".join(["The train/test split leaks information between both sides:", *errors])
        )
    logger.info(
        "Train/test split is sound: no shared records, %d of %d columns drifted",
        drift["drifted_columns"],
        len(drift["columns"]),
    )
    return report


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


def _summarize_scores(scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Turn the raw fold scores into a mean and a spread per metric.

    The spread matters as much as the mean: a good average over folds that disagree
    wildly describes a model whose quality depends on which records it happened to
    see, and that is not a model anyone can trust in production.
    """
    return {
        "mae": {
            "mean": float(-scores["test_neg_mean_absolute_error"].mean()),
            "std": float(scores["test_neg_mean_absolute_error"].std()),
        },
        "rmse": {
            "mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
            "std": float(scores["test_neg_root_mean_squared_error"].std()),
        },
        "r2": {
            "mean": float(scores["test_r2"].mean()),
            "std": float(scores["test_r2"].std()),
        },
    }


def cross_validate_model(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Cross-validate the architecture over the training split, with a measured floor.

    A single held-out test set gives one number with no sense of its own uncertainty.
    Cross-validation gives a distribution instead, and the spread across folds is what
    says whether the held-out number was luck.

    The same folds also score a regressor that always predicts the training mean. That
    floor is what makes underfitting measurable rather than a matter of opinion, and
    computing it here keeps it honest: it is never a constant copied from a notebook.
    """
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    bins = pd.qcut(y_train, q=STRATIFICATION_BINS, labels=False, duplicates="drop")
    scoring = ["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"]

    logger.info("Cross-validating over %d stratified folds of the training split", CV_FOLDS)
    model_scores = cross_validate(
        build_model(), x_train, y_train, cv=list(folds.split(x_train, bins)), scoring=scoring
    )
    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    baseline_scores = cross_validate(
        baseline, x_train, y_train, cv=list(folds.split(x_train, bins)), scoring=scoring
    )

    metrics = _summarize_scores(model_scores)
    logger.info(
        "Cross-validation — MAE: %.4f ± %.4f | RMSE: %.4f ± %.4f | R2: %.4f ± %.4f",
        metrics["mae"]["mean"],
        metrics["mae"]["std"],
        metrics["rmse"]["mean"],
        metrics["rmse"]["std"],
        metrics["r2"]["mean"],
        metrics["r2"]["std"],
    )
    return {
        "strategy": type(folds).__name__,
        "folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "stratification_bins": STRATIFICATION_BINS,
        "metrics": metrics,
        "baseline": _summarize_scores(baseline_scores),
    }


def diagnose_fit(
    train: dict[str, float],
    cross_validation: dict[str, Any],
    test: dict[str, float],
) -> dict[str, Any]:
    """Read underfitting, overfitting and test representativeness off the three sets.

    Every verdict carries the number it came from. A diagnosis without its measurement
    is an opinion, and an opinion cannot be re-checked on the next run.
    """
    cv_mae = cross_validation["metrics"]["mae"]
    baseline_mae = cross_validation["baseline"]["mae"]["mean"]

    generalization_gap = cv_mae["mean"] - train["mae"]
    gap_ratio = generalization_gap / train["mae"] if train["mae"] else 0.0
    baseline_improvement = baseline_mae / cv_mae["mean"] if cv_mae["mean"] else 0.0
    spread = cv_mae["std"] or 1e-12
    test_distance = abs(test["mae"] - cv_mae["mean"]) / spread

    overfitting = gap_ratio > OVERFITTING_GAP_RATIO
    underfitting = baseline_improvement < UNDERFITTING_IMPROVEMENT
    representative = test_distance <= TEST_REPRESENTATIVE_STD

    actions: list[str] = []
    if overfitting:
        actions.append(
            f"The model fits training {gap_ratio:.0%} better than unseen folds: raise the "
            "regularization strength, drop the least informative features, or gather more "
            "records."
        )
    if underfitting:
        actions.append(
            f"Cross-validated error is only {baseline_improvement:.2f}x better than "
            "predicting the mean: the architecture is too simple for this signal. Try a "
            "model family that captures interactions, or engineer features that carry more."
        )
    if not representative:
        actions.append(
            f"Test error sits {test_distance:.1f} standard deviations from the "
            "cross-validated mean: the held-out set is not representative. Re-draw the "
            "split with another seed and confirm the numbers move together."
        )

    diagnosis = {
        "overfitting": overfitting,
        "underfitting": underfitting,
        "test_is_representative": representative,
        "generalization_gap": generalization_gap,
        "generalization_gap_ratio": gap_ratio,
        "baseline_improvement": baseline_improvement,
        "test_distance_in_std": test_distance,
        "actions": actions,
    }
    logger.info(
        "Fit diagnosis — gap train/CV: %.4f (%.1f%%) | vs mean baseline: %.2fx | "
        "test at %.1f sigma of CV",
        generalization_gap,
        gap_ratio * 100,
        baseline_improvement,
        test_distance,
    )
    for action in actions:
        logger.warning("Fit diagnosis: %s", action)
    return diagnosis


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
    train_metrics = evaluate_model(model, x_train, y_train)
    test_metrics = evaluate_model(model, x_test, y_test)
    cross_validation = cross_validate_model(x_train, y_train)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
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
            "train": train_metrics,
            "cross_validation": cross_validation,
            "test": test_metrics,
        },
        "diagnosis": diagnose_fit(train_metrics, cross_validation, test_metrics),
    }
    test = report["metrics"]["test"]
    logger.info(
        "Test metrics — MAE: %.4f | RMSE: %.4f | R2: %.4f",
        test["mae"],
        test["rmse"],
        test["r2"],
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
    split_report = validate_train_test_split(x_train, x_test, y_train, y_test)
    model = train_model(x_train, y_train)
    report = collect_metrics(model, x_train, x_test, y_train, y_test)
    report["split_validation"] = split_report
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
