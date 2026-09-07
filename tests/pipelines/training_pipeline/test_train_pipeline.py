"""Unit tests for the admissions training pipeline."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pipelines.training_pipeline.train_pipeline import (
    BINARY_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    RIDGE_ALPHA,
    TARGET_COLUMN,
    TEST_SIZE,
    build_model,
    collect_metrics,
    evaluate_model,
    find_project_root,
    load_features,
    main,
    parse_args,
    run_pipeline,
    save_metrics,
    save_model,
    split_features_and_label,
    split_train_test,
    train_model,
)

SYNTHETIC_ROWS = 60
METRIC_NAMES = ("mae", "rmse", "r2")
PERFECT_R2 = 1.0
MINIMUM_LEARNED_R2 = 0.9
EXPECTED_TEST_ROWS = round(SYNTHETIC_ROWS * TEST_SIZE)


def synthetic_features(rows: int = SYNTHETIC_ROWS) -> pd.DataFrame:
    """Build a feature table shaped like the one the feature pipeline produces."""
    rng = np.random.default_rng(0)
    cgpa = rng.uniform(6.8, 9.9, rows).round(2)
    frame = pd.DataFrame(
        {
            "GRE Score": pd.array(rng.integers(290, 341, rows), dtype="Int64"),
            "TOEFL Score": pd.array(rng.integers(92, 121, rows), dtype="Int64"),
            "University Rating": pd.array(rng.integers(1, 6, rows), dtype="Int64"),
            "SOP": rng.integers(2, 11, rows) / 2,
            "LOR ": rng.integers(2, 11, rows) / 2,
            "CGPA": cgpa,
            "Research": pd.array(rng.integers(0, 2, rows).astype(bool), dtype="boolean"),
        }
    )
    # A signal the model can actually learn, so the metrics are not noise.
    frame[TARGET_COLUMN] = ((cgpa - 6.8) / 3.1 * 0.6 + 0.35).round(4)
    return frame


def write_features(path: Path, frame: pd.DataFrame) -> Path:
    """Persist a feature table where the pipeline expects to find it."""
    frame.to_parquet(path, index=False)
    return path


class TestLoadFeatures:
    """Reading the processed features of the previous pipeline."""

    def test_reads_the_feature_table(self, tmp_path: Path) -> None:
        source = write_features(tmp_path / "features.parquet", synthetic_features())

        loaded = load_features(source)

        assert len(loaded) == SYNTHETIC_ROWS
        assert list(loaded.columns) == [*NUMERIC_COLUMNS, *BINARY_COLUMNS, TARGET_COLUMN]

    def test_preserves_the_dtypes_the_model_expects(self, tmp_path: Path) -> None:
        source = write_features(tmp_path / "features.parquet", synthetic_features())

        loaded = load_features(source)

        assert loaded["GRE Score"].dtype == "Int64"
        assert loaded["Research"].dtype == "boolean"

    def test_reports_a_missing_feature_table(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="feature pipeline"):
            load_features(tmp_path / "does-not-exist.parquet")


class TestSplitFeaturesAndLabel:
    """Separating the predictors from what the model has to predict."""

    def test_keeps_the_label_out_of_the_predictors(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())

        assert TARGET_COLUMN not in predictors.columns
        assert label.name == TARGET_COLUMN

    def test_exposes_the_predictors_the_model_was_designed_for(self) -> None:
        predictors, _ = split_features_and_label(synthetic_features())

        assert list(predictors.columns) == [*NUMERIC_COLUMNS, *BINARY_COLUMNS]


class TestSplitTrainTest:
    """The train/test separation."""

    def test_respects_the_configured_proportion(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())

        x_train, x_test, _, _ = split_train_test(predictors, label)

        assert len(x_test) == EXPECTED_TEST_ROWS
        assert len(x_train) == SYNTHETIC_ROWS - EXPECTED_TEST_ROWS

    def test_keeps_predictors_and_labels_aligned(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())

        x_train, x_test, y_train, y_test = split_train_test(predictors, label)

        assert list(x_train.index) == list(y_train.index)
        assert list(x_test.index) == list(y_test.index)

    def test_leaks_no_record_between_the_two_sides(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())

        x_train, x_test, _, _ = split_train_test(predictors, label)

        assert not set(x_train.index) & set(x_test.index)

    def test_is_reproducible(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())

        first, _, _, _ = split_train_test(predictors, label)
        second, _, _, _ = split_train_test(predictors, label)

        assert list(first.index) == list(second.index)


class TestBuildModel:
    """The architecture chosen in the model selection notebook."""

    def test_wraps_the_preprocessing_and_the_regressor(self) -> None:
        model = build_model()

        assert isinstance(model, Pipeline)
        assert list(model.named_steps) == ["preprocessor", "model"]

    def test_uses_the_selected_hyperparameters(self) -> None:
        model = build_model()

        assert model.named_steps["model"].alpha == RIDGE_ALPHA
        assert model.named_steps["model"].random_state == RANDOM_STATE

    def test_returns_a_fresh_unfitted_model_every_time(self) -> None:
        assert build_model() is not build_model()


class TestTrainModel:
    """Fitting, where the model-dependent transformations are parameterized."""

    def test_returns_a_fitted_pipeline(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())
        x_train, x_test, y_train, _ = split_train_test(predictors, label)

        model = train_model(x_train, y_train)

        assert len(model.predict(x_test)) == len(x_test)

    def test_fits_the_scaler_on_the_training_split_alone(self) -> None:
        """Fitting on everything would leak the test distribution into the model."""
        predictors, label = split_features_and_label(synthetic_features())
        x_train, _, y_train, _ = split_train_test(predictors, label)

        model = train_model(x_train, y_train)

        scaler = model.named_steps["preprocessor"].named_transformers_["numeric"]["scaler"]
        assert scaler.mean_ == pytest.approx(
            x_train[NUMERIC_COLUMNS].astype(float).mean().to_numpy()
        )

    def test_learns_the_signal_in_the_data(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())
        x_train, x_test, y_train, y_test = split_train_test(predictors, label)

        model = train_model(x_train, y_train)

        assert evaluate_model(model, x_test, y_test)["r2"] > MINIMUM_LEARNED_R2


class TestEvaluateModel:
    """The metrics reported for the trained model."""

    def test_reports_every_expected_metric(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())
        x_train, x_test, y_train, y_test = split_train_test(predictors, label)
        model = train_model(x_train, y_train)

        metrics = evaluate_model(model, x_test, y_test)

        assert tuple(metrics) == METRIC_NAMES
        assert all(np.isfinite(value) for value in metrics.values())

    def test_scores_a_perfect_prediction_as_such(self) -> None:
        class PerfectModel:
            def predict(self, features: pd.DataFrame) -> np.ndarray:
                return features["CGPA"].to_numpy(dtype=float)

        frame = synthetic_features()
        metrics = evaluate_model(PerfectModel(), frame, frame["CGPA"].astype(float))

        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(PERFECT_R2)

    def test_reports_the_error_in_the_units_of_the_target(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())
        x_train, x_test, y_train, y_test = split_train_test(predictors, label)
        model = train_model(x_train, y_train)

        metrics = evaluate_model(model, x_test, y_test)

        assert 0 <= metrics["mae"] <= 1
        assert metrics["rmse"] >= metrics["mae"]


class TestCollectMetrics:
    """The evaluation report that gets stored next to the model."""

    def test_reports_both_sides_of_the_split(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())
        splits = split_train_test(predictors, label)
        model = train_model(splits[0], splits[2])

        report = collect_metrics(model, *splits)

        assert tuple(report["metrics"]) == ("train", "test")
        assert tuple(report["metrics"]["test"]) == METRIC_NAMES

    def test_records_how_the_model_and_the_split_were_configured(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())
        splits = split_train_test(predictors, label)
        model = train_model(splits[0], splits[2])

        report = collect_metrics(model, *splits)

        assert report["model"]["alpha"] == RIDGE_ALPHA
        assert report["split"]["random_state"] == RANDOM_STATE
        assert report["split"]["test_rows"] == EXPECTED_TEST_ROWS
        assert report["split"]["train_rows"] == SYNTHETIC_ROWS - EXPECTED_TEST_ROWS


class TestSaveMetrics:
    """Persisting the evaluation results."""

    def test_writes_readable_json(self, tmp_path: Path) -> None:
        destination = tmp_path / "reporting" / "metrics.json"

        save_metrics({"metrics": {"test": {"mae": 0.05}}}, destination)

        assert json.loads(destination.read_text())["metrics"]["test"]["mae"] == pytest.approx(0.05)

    def test_creates_the_parent_directory(self, tmp_path: Path) -> None:
        destination = tmp_path / "reporting" / "metrics.json"

        save_metrics({}, destination)

        assert destination.exists()


class TestSaveModel:
    """Persisting the trained model."""

    def test_writes_the_artifact(self, tmp_path: Path) -> None:
        predictors, label = split_features_and_label(synthetic_features())
        x_train, _, y_train, _ = split_train_test(predictors, label)
        destination = tmp_path / "models" / "model.joblib"

        save_model(train_model(x_train, y_train), destination)

        assert destination.exists()

    def test_the_reloaded_model_predicts_the_same(self, tmp_path: Path) -> None:
        predictors, label = split_features_and_label(synthetic_features())
        x_train, x_test, y_train, _ = split_train_test(predictors, label)
        model = train_model(x_train, y_train)
        destination = tmp_path / "model.joblib"

        save_model(model, destination)
        reloaded = joblib.load(destination)

        assert reloaded.predict(x_test) == pytest.approx(model.predict(x_test))


class TestRunPipeline:
    """The autonomous entry point."""

    def test_trains_evaluates_and_stores_everything(self, tmp_path: Path) -> None:
        source = write_features(tmp_path / "features.parquet", synthetic_features())
        model_path = tmp_path / "model.joblib"
        metrics_path = tmp_path / "metrics.json"

        model, report = run_pipeline(source, model_path, metrics_path)

        assert model_path.exists()
        assert metrics_path.exists()
        assert report["metrics"]["test"]["r2"] > MINIMUM_LEARNED_R2
        assert joblib.load(model_path).predict(
            synthetic_features().head(1)[model.feature_names_in_]
        )

    def test_main_runs_end_to_end(self, tmp_path: Path) -> None:
        source = write_features(tmp_path / "features.parquet", synthetic_features())
        model_path = tmp_path / "model.joblib"
        metrics_path = tmp_path / "metrics.json"

        main(
            [
                "--features-path",
                str(source),
                "--model-path",
                str(model_path),
                "--metrics-path",
                str(metrics_path),
                "--log-level",
                "ERROR",
            ]
        )

        assert model_path.exists()
        assert json.loads(metrics_path.read_text())["metrics"]["test"]["mae"] >= 0


class TestCommandLine:
    """Default paths of the standalone script."""

    def test_finds_the_project_root_from_any_directory(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        nested = tmp_path / "src" / "pipelines"
        nested.mkdir(parents=True)

        assert find_project_root(nested) == tmp_path.resolve()

    def test_fails_when_there_is_no_project_above(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"pyproject\.toml"):
            find_project_root(tmp_path)

    def test_defaults_point_at_the_project_layers(self) -> None:
        args = parse_args([])

        assert args.features_path.parts[-2:] == ("04_feature", "admisiones_features.parquet")
        assert args.model_path.suffix == ".joblib"
        assert args.metrics_path.suffix == ".json"
