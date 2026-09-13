"""Unit tests for the admissions training pipeline."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pipelines.training_pipeline import train_pipeline
from pipelines.training_pipeline.train_pipeline import (
    BINARY_COLUMNS,
    CV_FOLDS,
    MINIMUM_FOLDS,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    RIDGE_ALPHA,
    TARGET_COLUMN,
    TEST_SIZE,
    TrainTestSplitError,
    build_model,
    collect_metrics,
    cross_validate_model,
    diagnose_fit,
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
    validate_train_test_split,
)

SYNTHETIC_ROWS = 100
METRIC_NAMES = ("mae", "rmse", "r2")
PERFECT_R2 = 1.0
MINIMUM_LEARNED_R2 = 0.9
EXPECTED_TEST_ROWS = round(SYNTHETIC_ROWS * TEST_SIZE)

# The split checks need enough records to say anything meaningful about a distribution.
VALIDATION_ROWS = 300
LEAKED_RECORDS = 40
# Small enough that the smallest quantile bin cannot supply ten folds.
SMALL_TRAINING_ROWS = 50


def realistic_features(rows: int = VALIDATION_ROWS) -> pd.DataFrame:
    """Build a feature table whose signal carries noise, like real admissions data.

    A target that is a pure function of one column makes the correlation checks fire
    on the fixture itself, which would tell us nothing about the split.
    """
    rng = np.random.default_rng(7)
    cgpa = rng.uniform(6.8, 9.9, rows).round(2)
    gre = rng.integers(290, 341, rows)
    frame = pd.DataFrame(
        {
            "GRE Score": pd.array(gre, dtype="Int64"),
            "TOEFL Score": pd.array(rng.integers(92, 121, rows), dtype="Int64"),
            "University Rating": pd.array(rng.integers(1, 6, rows), dtype="Int64"),
            "SOP": rng.integers(2, 11, rows) / 2,
            "LOR ": rng.integers(2, 11, rows) / 2,
            "CGPA": cgpa,
            "Research": pd.array(rng.integers(0, 2, rows).astype(bool), dtype="boolean"),
        }
    )
    signal = (cgpa - 6.8) / 3.1 * 0.4 + (gre - 290) / 50 * 0.2 + 0.3
    frame[TARGET_COLUMN] = (signal + rng.normal(0, 0.05, rows)).clip(0.05, 0.99).round(4)
    return frame


def leak_train_records_into_test(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Contaminate the test side with records copied straight out of training."""
    leaked_x = pd.concat([x_test, x_train.head(LEAKED_RECORDS)])
    leaked_y = pd.concat([y_test, y_train.head(LEAKED_RECORDS)])
    return x_train, leaked_x, y_train, leaked_y


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


class TestValidateTrainTestSplitAccepts:
    """A sound split: what the checks let through."""

    def test_accepts_the_split_the_pipeline_produces(self) -> None:
        splits = split_train_test(*split_features_and_label(realistic_features()))

        report = validate_train_test_split(*splits)

        assert report["errors"] == []

    def test_reports_every_check_it_ran(self) -> None:
        splits = split_train_test(*split_features_and_label(realistic_features()))

        report = validate_train_test_split(*splits)

        assert report["checks"]
        assert {check["status"] for check in report["checks"]} <= {"passed", "failed"}
        assert all(check["name"] for check in report["checks"])

    def test_reports_no_shared_records_between_the_two_sides(self) -> None:
        splits = split_train_test(*split_features_and_label(realistic_features()))

        report = validate_train_test_split(*splits)

        assert report["leakage"]["shared_index_records"] == 0
        assert report["leakage"]["shared_record_count"] == 0


class TestValidateTrainTestSplitRejects:
    """Leakage: the failure that invalidates every metric downstream."""

    def test_rejects_training_records_present_in_the_test_set(self) -> None:
        splits = split_train_test(*split_features_and_label(realistic_features()))

        with pytest.raises(TrainTestSplitError, match="Record overlap"):
            validate_train_test_split(*leak_train_records_into_test(*splits))

    def test_the_error_names_the_contaminated_share(self) -> None:
        splits = split_train_test(*split_features_and_label(realistic_features()))

        with pytest.raises(TrainTestSplitError) as failure:
            validate_train_test_split(*leak_train_records_into_test(*splits))

        assert "%" in str(failure.value)


class TestValidateTrainTestSplitWarns:
    """Distribution problems: reported, never fatal."""

    def test_a_drifted_split_warns_instead_of_failing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Drift is a signal to investigate, not proof that the split is unusable."""
        x_train, x_test, y_train, y_test = split_train_test(
            *split_features_and_label(realistic_features())
        )
        drifted = x_test.assign(CGPA=x_test["CGPA"] - 2.0)

        with caplog.at_level(logging.WARNING):
            report = validate_train_test_split(x_train, drifted, y_train, y_test)

        assert report["errors"] == []
        assert report["warnings"]
        assert "drift" in caplog.text.lower()
        drifted_columns = [c["name"] for c in report["drift"]["columns"] if c["drifted"]]
        assert "CGPA" in drifted_columns

    def test_measures_drift_on_every_column_including_the_label(self) -> None:
        """A drift report that skips a column says nothing about that column."""
        splits = split_train_test(*split_features_and_label(realistic_features()))

        report = validate_train_test_split(*splits)

        measured = {column["name"] for column in report["drift"]["columns"]}
        assert measured == {*NUMERIC_COLUMNS, *BINARY_COLUMNS, TARGET_COLUMN}


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


class TestCrossValidateModel:
    """Cross-validation over the training split."""

    def test_reports_mean_and_spread_for_every_metric(self) -> None:
        predictors, label = split_features_and_label(realistic_features())
        x_train, _, y_train, _ = split_train_test(predictors, label)

        result = cross_validate_model(x_train, y_train)

        assert tuple(result["metrics"]) == METRIC_NAMES
        for metric in METRIC_NAMES:
            assert {"mean", "std"} <= set(result["metrics"][metric])
            assert np.isfinite(result["metrics"][metric]["mean"])
            assert result["metrics"][metric]["std"] >= 0

    def test_records_how_the_folds_were_built(self) -> None:
        predictors, label = split_features_and_label(realistic_features())
        x_train, _, y_train, _ = split_train_test(predictors, label)

        result = cross_validate_model(x_train, y_train)

        assert result["folds"] == CV_FOLDS
        assert result["strategy"] == "StratifiedKFold"
        assert result["random_state"] == RANDOM_STATE

    def test_is_reproducible(self) -> None:
        predictors, label = split_features_and_label(realistic_features())
        x_train, _, y_train, _ = split_train_test(predictors, label)

        first = cross_validate_model(x_train, y_train)
        second = cross_validate_model(x_train, y_train)

        assert first["metrics"] == second["metrics"]

    def test_reduces_the_folds_to_what_the_data_supports(self) -> None:
        """Ten folds over a handful of records is not a measurement, it is noise."""
        predictors, label = split_features_and_label(realistic_features(SMALL_TRAINING_ROWS))
        x_train, _, y_train, _ = split_train_test(predictors, label)

        result = cross_validate_model(x_train, y_train)

        assert result["folds"] < CV_FOLDS
        assert result["folds"] >= MINIMUM_FOLDS

    def test_rejects_a_training_split_too_small_to_cross_validate(self) -> None:
        predictors, label = split_features_and_label(realistic_features(VALIDATION_ROWS))
        x_train, _, y_train, _ = split_train_test(predictors, label)

        with pytest.raises(ValueError, match="too small"):
            cross_validate_model(x_train.head(3), y_train.head(3))

    def test_fails_loudly_when_a_fold_cannot_be_scored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fold that silently scores NaN would be averaged into a clean-looking report."""

        class FailingRegressor(Ridge):
            def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
                raise RuntimeError("this fold cannot be fitted")

        monkeypatch.setattr(
            train_pipeline,
            "build_model",
            lambda: Pipeline([("model", FailingRegressor())]),
        )
        predictors, label = split_features_and_label(realistic_features())
        x_train, _, y_train, _ = split_train_test(predictors, label)

        with pytest.raises(RuntimeError, match="cannot be fitted"):
            cross_validate_model(x_train, y_train)

    def test_scores_a_learnable_signal_better_than_the_mean(self) -> None:
        """The baseline is measured, never copied from a notebook constant."""
        predictors, label = split_features_and_label(realistic_features())
        x_train, _, y_train, _ = split_train_test(predictors, label)

        result = cross_validate_model(x_train, y_train)

        assert result["baseline"]["mae"]["mean"] > result["metrics"]["mae"]["mean"]


class TestDiagnoseFit:
    """Reading underfitting, overfitting and test representativeness off the numbers."""

    def sound_numbers(self) -> tuple[dict[str, float], dict[str, Any], dict[str, float]]:
        train = {"mae": 0.044, "rmse": 0.061, "r2": 0.816}
        cross_validation = {
            "metrics": {
                "mae": {"mean": 0.046, "std": 0.007},
                "rmse": {"mean": 0.062, "std": 0.009},
                "r2": {"mean": 0.790, "std": 0.060},
            },
            "baseline": {"mae": {"mean": 0.114, "std": 0.011}},
        }
        test = {"mae": 0.047, "rmse": 0.072, "r2": 0.750}
        return train, cross_validation, test

    def test_clears_a_model_that_generalizes(self) -> None:
        diagnosis = diagnose_fit(*self.sound_numbers())

        assert diagnosis["overfitting"] is False
        assert diagnosis["underfitting"] is False
        assert diagnosis["test_is_representative"] is True
        assert diagnosis["actions"] == []

    def test_flags_a_model_that_memorized_the_training_split(self) -> None:
        train, cross_validation, test = self.sound_numbers()
        train["mae"] = 0.005

        diagnosis = diagnose_fit(train, cross_validation, test)

        assert diagnosis["overfitting"] is True
        assert any("regulariz" in action.lower() for action in diagnosis["actions"])

    def test_flags_a_model_that_barely_beats_the_mean(self) -> None:
        train, cross_validation, test = self.sound_numbers()
        cross_validation["metrics"]["mae"]["mean"] = 0.112

        diagnosis = diagnose_fit(train, cross_validation, test)

        assert diagnosis["underfitting"] is True
        assert diagnosis["actions"]

    def test_flags_a_test_split_outside_the_cross_validated_range(self) -> None:
        train, cross_validation, test = self.sound_numbers()
        test["mae"] = 0.090

        diagnosis = diagnose_fit(train, cross_validation, test)

        assert diagnosis["test_is_representative"] is False
        assert any("test" in action.lower() for action in diagnosis["actions"])

    def test_reports_a_perfect_training_fit_as_overfitting(self) -> None:
        """Zero training error against positive fold error is memorization, not skill."""
        train, cross_validation, test = self.sound_numbers()
        train["mae"] = 0.0

        diagnosis = diagnose_fit(train, cross_validation, test)

        assert diagnosis["overfitting"] is True
        assert diagnosis["generalization_gap_ratio"] == float("inf")

    def test_does_not_report_a_flawless_model_as_underfitting(self) -> None:
        """A model with no cross-validated error beats the mean by definition."""
        train, cross_validation, test = self.sound_numbers()
        train["mae"] = 0.0
        cross_validation["metrics"]["mae"] = {"mean": 0.0, "std": 0.0}
        test["mae"] = 0.0

        diagnosis = diagnose_fit(train, cross_validation, test)

        assert diagnosis["underfitting"] is False
        assert diagnosis["baseline_improvement"] == float("inf")

    def test_quantifies_every_gap_it_judges(self) -> None:
        """A verdict without its number is an opinion."""
        diagnosis = diagnose_fit(*self.sound_numbers())

        assert diagnosis["generalization_gap"] == pytest.approx(0.002, abs=1e-9)
        assert diagnosis["baseline_improvement"] > 1
        assert "test_distance_in_std" in diagnosis


class TestCollectMetrics:
    """The evaluation report that gets stored next to the model."""

    def test_reports_the_three_sets_of_numbers(self) -> None:
        predictors, label = split_features_and_label(synthetic_features())
        splits = split_train_test(predictors, label)
        model = train_model(splits[0], splits[2])

        report = collect_metrics(model, *splits)

        assert tuple(report["metrics"]) == ("train", "cross_validation", "test")
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

    def test_compares_train_cross_validation_and_test(self) -> None:
        predictors, label = split_features_and_label(realistic_features())
        splits = split_train_test(predictors, label)
        model = train_model(splits[0], splits[2])

        report = collect_metrics(model, *splits)

        assert tuple(report["metrics"]) == ("train", "cross_validation", "test")
        assert report["diagnosis"]["overfitting"] is False


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
        # The stored artifact must be the very model that was returned and evaluated.
        candidate = synthetic_features()[list(model.feature_names_in_)]
        np.testing.assert_allclose(
            joblib.load(model_path).predict(candidate), model.predict(candidate)
        )

    def test_a_leaky_feature_table_aborts_before_anything_is_stored(self, tmp_path: Path) -> None:
        """Duplicated records scatter copies of one candidate across both sides."""
        duplicated = pd.concat([realistic_features(VALIDATION_ROWS // 2)] * 2)
        source = write_features(tmp_path / "features.parquet", duplicated)
        model_path = tmp_path / "model.joblib"
        metrics_path = tmp_path / "metrics.json"

        with pytest.raises(TrainTestSplitError):
            run_pipeline(source, model_path, metrics_path)

        assert not model_path.exists()
        assert not metrics_path.exists()

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
