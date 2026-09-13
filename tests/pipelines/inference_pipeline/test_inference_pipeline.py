"""Unit tests for the admissions inference pipeline, on a dummy model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pipelines.inference_pipeline.inference_pipeline import (
    PREDICTION_COLUMN,
    InferenceInputError,
    find_project_root,
    load_model,
    load_new_data,
    main,
    parse_args,
    predict,
    prepare_features,
    run_pipeline,
    save_predictions,
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
CANDIDATE_ROWS = 40
TWO_ROWS = 2


def synthetic_candidates(rows: int = CANDIDATE_ROWS) -> pd.DataFrame:
    """Build candidates shaped like the ones the model was fitted on."""
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "GRE Score": rng.integers(290, 341, rows).astype(float),
            "TOEFL Score": rng.integers(92, 121, rows).astype(float),
            "University Rating": rng.integers(1, 6, rows).astype(float),
            "SOP": rng.integers(2, 11, rows) / 2,
            "LOR ": rng.integers(2, 11, rows) / 2,
            "CGPA": rng.uniform(6.8, 9.9, rows).round(2),
            "Research": rng.integers(0, 2, rows).astype(float),
        }
    )


def dummy_model() -> Pipeline:
    """Fit a throwaway model whose predictions are a known function of CGPA.

    Deliberately trivial: these tests are about the pipeline around the model, not
    about the model. A predictable estimator makes every assertion exact.
    """
    candidates = synthetic_candidates()
    target = candidates["CGPA"] / 10
    model = Pipeline(steps=[("model", Ridge(alpha=0.0))])
    model.fit(candidates[MODEL_COLUMNS], target)
    return model


def write_model(path: Path) -> Path:
    """Persist the dummy model where the pipeline expects to find it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dummy_model(), path)
    return path


def write_candidates(path: Path, frame: pd.DataFrame) -> Path:
    """Persist a batch of new candidates as the CSV the pipeline reads."""
    frame.to_csv(path, index=False)
    return path


class TestLoadModel:
    """Recovering the trained model from the project storage."""

    def test_loads_the_stored_pipeline(self, tmp_path: Path) -> None:
        path = write_model(tmp_path / "model.joblib")

        model = load_model(path)

        assert isinstance(model, Pipeline)

    def test_recovers_the_columns_the_model_was_fitted_on(self, tmp_path: Path) -> None:
        """The input contract is read from the artifact, never hardcoded."""
        path = write_model(tmp_path / "model.joblib")

        model = load_model(path)

        assert list(model.feature_names_in_) == MODEL_COLUMNS

    def test_reports_a_missing_model(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="training pipeline"):
            load_model(tmp_path / "absent.joblib")


class TestLoadNewData:
    """Reading the batch of candidates to score."""

    def test_reads_the_candidates(self, tmp_path: Path) -> None:
        source = write_candidates(tmp_path / "new.csv", synthetic_candidates())

        loaded = load_new_data(source)

        assert len(loaded) == CANDIDATE_ROWS

    def test_reports_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"absent\.csv"):
            load_new_data(tmp_path / "absent.csv")

    def test_rejects_a_file_without_records(self, tmp_path: Path) -> None:
        source = write_candidates(tmp_path / "new.csv", synthetic_candidates().head(0))

        with pytest.raises(InferenceInputError, match="no records"):
            load_new_data(source)


class TestPrepareFeatures:
    """The same transformations the training data went through."""

    def test_accepts_the_canonical_columns(self) -> None:
        prepared = prepare_features(synthetic_candidates(), MODEL_COLUMNS)

        assert list(prepared.columns) == MODEL_COLUMNS

    def test_recognizes_column_names_regardless_of_case_and_spacing(self) -> None:
        renamed = synthetic_candidates().rename(
            columns={"GRE Score": "gre score", "LOR ": "lor", "CGPA": " CGPA "}
        )

        prepared = prepare_features(renamed, MODEL_COLUMNS)

        assert list(prepared.columns) == MODEL_COLUMNS

    def test_orders_the_columns_the_way_the_model_expects(self) -> None:
        shuffled = synthetic_candidates()[list(reversed(MODEL_COLUMNS))]

        prepared = prepare_features(shuffled, MODEL_COLUMNS)

        assert list(prepared.columns) == MODEL_COLUMNS

    def test_casts_the_dtypes_the_training_data_carried(self) -> None:
        prepared = prepare_features(synthetic_candidates(), MODEL_COLUMNS)

        assert prepared["GRE Score"].dtype == "Int64"
        assert prepared["Research"].dtype == "boolean"

    def test_reports_a_missing_column(self) -> None:
        incomplete = synthetic_candidates().drop(columns=["CGPA"])

        with pytest.raises(InferenceInputError, match="CGPA"):
            prepare_features(incomplete, MODEL_COLUMNS)

    def test_accepts_whole_numbers_written_as_decimals(self) -> None:
        """A CSV read by pandas hands back 324.0, not 324."""
        candidates = synthetic_candidates(TWO_ROWS)
        candidates.loc[:, "GRE Score"] = [324.0, 310.0]

        prepared = prepare_features(candidates, MODEL_COLUMNS)

        assert prepared["GRE Score"].tolist() == [324, 310]

    def test_refuses_to_round_a_fractional_score(self) -> None:
        """Rounding would score a different candidate than the one in the file."""
        candidates = synthetic_candidates(TWO_ROWS)
        candidates.loc[0, "GRE Score"] = 320.5

        with pytest.raises(InferenceInputError, match=r"320\.5"):
            prepare_features(candidates, MODEL_COLUMNS)

    def test_refuses_to_round_a_fractional_rating(self) -> None:
        candidates = synthetic_candidates(TWO_ROWS)
        candidates.loc[0, "University Rating"] = 3.7

        with pytest.raises(InferenceInputError, match="University Rating"):
            prepare_features(candidates, MODEL_COLUMNS)

    def test_reports_headers_that_collapse_to_the_same_column(self) -> None:
        """Two spellings of one column leave the batch ambiguous, not mergeable."""
        candidates = synthetic_candidates(TWO_ROWS)
        candidates[" research "] = 0

        with pytest.raises(InferenceInputError, match="Research"):
            prepare_features(candidates, MODEL_COLUMNS)

    def test_adapts_to_an_artifact_with_a_different_contract(self) -> None:
        """The contract comes from the artifact, so a narrower one must still work."""
        narrower = ["GRE Score", "CGPA"]

        prepared = prepare_features(synthetic_candidates(), narrower)

        assert list(prepared.columns) == narrower
        assert prepared["GRE Score"].dtype == "Int64"

    def test_reports_a_non_numeric_value(self) -> None:
        broken = synthetic_candidates().astype({"CGPA": "object"})
        broken.loc[0, "CGPA"] = "nueve"

        with pytest.raises(InferenceInputError, match="CGPA"):
            prepare_features(broken, MODEL_COLUMNS)


class TestResearchConversion:
    """The column whose unrecognized values would otherwise be guessed."""

    @pytest.mark.parametrize("spelling", [1, "1", "yes", "Si", "TRUE", "verdadero"])
    def test_understands_the_affirmative_spellings(self, spelling: object) -> None:
        candidates = synthetic_candidates().astype({"Research": "object"})
        candidates.loc[:, "Research"] = spelling

        prepared = prepare_features(candidates, MODEL_COLUMNS)

        assert prepared["Research"].all()

    @pytest.mark.parametrize("spelling", [0, "0", "no", "False", "falso"])
    def test_understands_the_negative_spellings(self, spelling: object) -> None:
        candidates = synthetic_candidates().astype({"Research": "object"})
        candidates.loc[:, "Research"] = spelling

        prepared = prepare_features(candidates, MODEL_COLUMNS)

        assert not prepared["Research"].any()

    def test_refuses_to_guess_an_unrecognized_value(self) -> None:
        """The model would silently score 'banana' as the majority class."""
        candidates = synthetic_candidates().astype({"Research": "object"})
        candidates.loc[0, "Research"] = "banana"

        with pytest.raises(InferenceInputError, match="banana"):
            prepare_features(candidates, MODEL_COLUMNS)

    def test_refuses_to_guess_a_missing_value(self) -> None:
        candidates = synthetic_candidates().astype({"Research": "object"})
        candidates.loc[0, "Research"] = None

        with pytest.raises(InferenceInputError, match="Research"):
            prepare_features(candidates, MODEL_COLUMNS)


class TestPredict:
    """Turning prepared candidates into admission chances."""

    def test_returns_one_prediction_per_candidate(self) -> None:
        model = dummy_model()
        prepared = prepare_features(synthetic_candidates(), MODEL_COLUMNS)

        predictions = predict(model, prepared)

        assert len(predictions) == CANDIDATE_ROWS

    def test_keeps_every_prediction_inside_the_unit_interval(self) -> None:
        """A linear model extrapolates past 1.0, and a chance above 100% is nonsense."""
        model = dummy_model()
        extreme = synthetic_candidates(TWO_ROWS)
        extreme.loc[:, "CGPA"] = [20.0, -5.0]

        predictions = predict(model, prepare_features(extreme, MODEL_COLUMNS))

        assert predictions.max() <= 1.0
        assert predictions.min() >= 0.0

    def test_leaves_predictions_inside_the_range_untouched(self) -> None:
        model = dummy_model()
        prepared = prepare_features(synthetic_candidates(), MODEL_COLUMNS)

        predictions = predict(model, prepared)
        raw = model.predict(prepared)

        assert predictions == pytest.approx(raw)


class TestSavePredictions:
    """Storing the scored batch."""

    def test_writes_the_candidates_next_to_their_prediction(self, tmp_path: Path) -> None:
        candidates = synthetic_candidates()
        predictions = np.linspace(0.1, 0.9, CANDIDATE_ROWS)
        destination = tmp_path / "out" / "predicciones.csv"

        save_predictions(candidates, predictions, destination)

        stored = pd.read_csv(destination)
        assert len(stored) == CANDIDATE_ROWS
        assert PREDICTION_COLUMN in stored.columns
        assert stored[PREDICTION_COLUMN].to_numpy() == pytest.approx(predictions)

    def test_keeps_the_original_columns(self, tmp_path: Path) -> None:
        candidates = synthetic_candidates()
        destination = tmp_path / "predicciones.csv"

        save_predictions(candidates, np.zeros(CANDIDATE_ROWS), destination)

        stored = pd.read_csv(destination)
        assert set(MODEL_COLUMNS) <= set(stored.columns)


class TestRunPipeline:
    """The autonomous entry point."""

    def test_scores_a_batch_end_to_end(self, tmp_path: Path) -> None:
        model_path = write_model(tmp_path / "model.joblib")
        source = write_candidates(tmp_path / "new.csv", synthetic_candidates())
        destination = tmp_path / "predicciones.csv"

        scored = run_pipeline(model_path, source, destination)

        assert len(scored) == CANDIDATE_ROWS
        assert destination.exists()
        assert scored[PREDICTION_COLUMN].between(0.0, 1.0).all()

    def test_invalid_input_stores_nothing(self, tmp_path: Path) -> None:
        model_path = write_model(tmp_path / "model.joblib")
        broken = synthetic_candidates().astype({"Research": "object"})
        broken.loc[0, "Research"] = "banana"
        source = write_candidates(tmp_path / "new.csv", broken)
        destination = tmp_path / "predicciones.csv"

        with pytest.raises(InferenceInputError):
            run_pipeline(model_path, source, destination)

        assert not destination.exists()

    def test_main_runs_end_to_end(self, tmp_path: Path) -> None:
        model_path = write_model(tmp_path / "model.joblib")
        source = write_candidates(tmp_path / "new.csv", synthetic_candidates())
        destination = tmp_path / "predicciones.csv"

        main(
            [
                "--model-path",
                str(model_path),
                "--input-path",
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

    def test_defaults_point_at_the_stored_model(self) -> None:
        args = parse_args(["--input-path", "candidatos.csv"])

        assert args.model_path.suffix == ".joblib"
        assert args.output_path.suffix == ".csv"
