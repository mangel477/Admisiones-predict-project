"""Unit tests for the admissions inference pipeline, on a dummy model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pipelines.inference_pipeline.inference_pipeline import (
    PREDICTION_COLUMN,
    InferenceError,
    find_project_root,
    load_candidates,
    load_model,
    main,
    parse_args,
    predict,
    run_pipeline,
    save_predictions,
    select_model_features,
)

MODEL_COLUMNS = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    "LOR ",
    "CGPA",
    "Research",
]
CANDIDATE_ROWS = 20
TWO_ROWS = 2


def stored_candidates(rows: int = CANDIDATE_ROWS) -> pd.DataFrame:
    """Build candidates exactly as the feature store writes them: typed and complete."""
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "GRE Score": pd.array(rng.integers(290, 341, rows), dtype="Int64"),
            "TOEFL Score": pd.array(rng.integers(92, 121, rows), dtype="Int64"),
            "University Rating": pd.array(rng.integers(1, 6, rows), dtype="Int64"),
            "SOP": rng.integers(2, 11, rows) / 2,
            "LOR ": rng.integers(2, 11, rows) / 2,
            "CGPA": rng.uniform(6.8, 9.9, rows).round(2),
            "Research": pd.array(rng.integers(0, 2, rows).astype(bool), dtype="boolean"),
        }
    )


def dummy_model() -> Pipeline:
    """Fit a throwaway model shaped like the real artifact: preprocessing plus regressor.

    Deliberately trivial — it predicts a known function of CGPA. These tests are about
    the pipeline around the model, not the model, and a predictable estimator makes
    every assertion exact. The preprocessing step is what lets it take the nullable
    dtypes the feature store writes, exactly as the real artifact does.
    """
    candidates = stored_candidates()
    preprocessor = ColumnTransformer(
        transformers=[("numeric", SimpleImputer(strategy="median"), MODEL_COLUMNS)]
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=0.0))])
    model.fit(candidates[MODEL_COLUMNS], candidates["CGPA"] / 10)
    return model


def write_model(path: Path) -> Path:
    """Persist the dummy model where the pipeline expects to find it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dummy_model(), path)
    return path


def write_store(path: Path, frame: pd.DataFrame) -> Path:
    """Persist a batch the way the feature pipeline leaves it in the store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


class TestLoadModel:
    """Recovering the trained model from the project storage."""

    def test_loads_the_stored_pipeline(self, tmp_path: Path) -> None:
        model = load_model(write_model(tmp_path / "model.joblib"))

        assert isinstance(model, Pipeline)

    def test_recovers_the_preprocessing_fitted_during_training(self, tmp_path: Path) -> None:
        """The artifact carries its preprocessing: nothing is re-fitted here."""
        model = load_model(write_model(tmp_path / "model.joblib"))

        imputer = model.named_steps["preprocessor"].named_transformers_["numeric"]
        assert imputer.statistics_ is not None

    def test_recovers_the_columns_the_model_was_fitted_on(self, tmp_path: Path) -> None:
        model = load_model(write_model(tmp_path / "model.joblib"))

        assert list(model.feature_names_in_) == MODEL_COLUMNS

    def test_reports_a_missing_model(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="training pipeline"):
            load_model(tmp_path / "absent.joblib")


class TestLoadCandidates:
    """Reading the batch the feature store already validated."""

    def test_reads_the_stored_batch(self, tmp_path: Path) -> None:
        source = write_store(tmp_path / "candidatos.parquet", stored_candidates())

        loaded = load_candidates(source)

        assert len(loaded) == CANDIDATE_ROWS

    def test_preserves_the_dtypes_the_store_wrote(self, tmp_path: Path) -> None:
        source = write_store(tmp_path / "candidatos.parquet", stored_candidates())

        loaded = load_candidates(source)

        assert loaded["GRE Score"].dtype == "Int64"
        assert loaded["Research"].dtype == "boolean"

    def test_points_at_the_feature_pipeline_when_the_batch_is_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="feature pipeline"):
            load_candidates(tmp_path / "absent.parquet")

    def test_rejects_a_batch_without_records(self, tmp_path: Path) -> None:
        source = write_store(tmp_path / "candidatos.parquet", stored_candidates().head(0))

        with pytest.raises(InferenceError, match="no candidates"):
            load_candidates(source)


class TestSelectModelFeatures:
    """Handing the model the columns it was fitted on, in its own order."""

    def test_returns_the_columns_in_the_order_of_the_artifact(self) -> None:
        shuffled = stored_candidates()[list(reversed(MODEL_COLUMNS))]

        selected = select_model_features(shuffled, MODEL_COLUMNS)

        assert list(selected.columns) == MODEL_COLUMNS

    def test_leaves_the_values_untouched(self) -> None:
        """Every transformation already happened: upstream or inside the artifact."""
        candidates = stored_candidates()

        selected = select_model_features(candidates, MODEL_COLUMNS)

        assert selected["CGPA"].tolist() == candidates["CGPA"].tolist()
        assert selected["Research"].dtype == "boolean"

    def test_reports_a_store_that_does_not_match_the_artifact(self) -> None:
        outdated = stored_candidates().drop(columns=["CGPA"])

        with pytest.raises(InferenceError, match="CGPA"):
            select_model_features(outdated, MODEL_COLUMNS)


class TestPredict:
    """Turning stored candidates into admission chances."""

    def test_returns_one_prediction_per_candidate(self) -> None:
        model = dummy_model()

        predictions = predict(model, select_model_features(stored_candidates(), MODEL_COLUMNS))

        assert len(predictions) == CANDIDATE_ROWS

    def test_keeps_every_prediction_inside_the_unit_interval(self) -> None:
        """A linear model extrapolates past 1.0, and a chance above 100% is nonsense."""
        model = dummy_model()
        extreme = stored_candidates(TWO_ROWS)
        extreme.loc[:, "CGPA"] = [40.0, -20.0]

        predictions = predict(model, select_model_features(extreme, MODEL_COLUMNS))

        assert predictions.max() <= 1.0
        assert predictions.min() >= 0.0

    def test_leaves_predictions_inside_the_range_untouched(self) -> None:
        model = dummy_model()
        features = select_model_features(stored_candidates(), MODEL_COLUMNS)

        assert predict(model, features) == pytest.approx(model.predict(features))


class TestSavePredictions:
    """Storing the scored batch."""

    def test_writes_each_candidate_next_to_its_chance(self, tmp_path: Path) -> None:
        candidates = stored_candidates()
        predictions = np.linspace(0.1, 0.9, CANDIDATE_ROWS)
        destination = tmp_path / "out" / "predicciones.csv"

        save_predictions(candidates, predictions, destination)

        stored = pd.read_csv(destination)
        assert len(stored) == CANDIDATE_ROWS
        assert set(MODEL_COLUMNS) <= set(stored.columns)
        assert stored[PREDICTION_COLUMN].to_numpy() == pytest.approx(predictions)


class TestRunPipeline:
    """The autonomous entry point."""

    def test_scores_the_stored_batch_end_to_end(self, tmp_path: Path) -> None:
        model_path = write_model(tmp_path / "model.joblib")
        source = write_store(tmp_path / "candidatos.parquet", stored_candidates())
        destination = tmp_path / "predicciones.csv"

        scored = run_pipeline(model_path, source, destination)

        assert len(scored) == CANDIDATE_ROWS
        assert destination.exists()
        assert scored[PREDICTION_COLUMN].between(0.0, 1.0).all()

    def test_a_store_out_of_step_with_the_model_stores_nothing(self, tmp_path: Path) -> None:
        model_path = write_model(tmp_path / "model.joblib")
        source = write_store(
            tmp_path / "candidatos.parquet", stored_candidates().drop(columns=["CGPA"])
        )
        destination = tmp_path / "predicciones.csv"

        with pytest.raises(InferenceError):
            run_pipeline(model_path, source, destination)

        assert not destination.exists()

    def test_main_runs_end_to_end(self, tmp_path: Path) -> None:
        model_path = write_model(tmp_path / "model.joblib")
        source = write_store(tmp_path / "candidatos.parquet", stored_candidates())
        destination = tmp_path / "predicciones.csv"

        main(
            [
                "--model-path",
                str(model_path),
                "--candidates-path",
                str(source),
                "--output-path",
                str(destination),
                "--log-level",
                "ERROR",
            ]
        )

        assert len(pd.read_csv(destination)) == CANDIDATE_ROWS


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

    def test_defaults_read_the_store_and_the_stored_model(self) -> None:
        args = parse_args([])

        assert args.model_path.suffix == ".joblib"
        assert args.candidates_path.suffix == ".parquet"
        assert args.output_path.suffix == ".csv"
