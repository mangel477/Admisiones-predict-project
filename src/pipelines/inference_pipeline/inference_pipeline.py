"""Inference pipeline for the graduate admissions prediction project.

The stages run in this order: load the model stored by the training pipeline, read a
batch of new candidates, put those candidates through the same transformations the
training data went through, score them, and store the predictions.

The preprocessing is not reimplemented here, and that is the point. Imputation,
scaling and encoding were fitted on the training split and serialized inside the
artifact, so loading it recovers them with the values they learned. What does have to
happen here is everything upstream of the model: new candidates arrive through a
different door than the training source, so their columns and their dtypes have to be
brought to the shape the model was fitted on before it ever sees them.

That last part is not a formality. The encoder was fitted with
``handle_unknown="use_encoded_value"``, so a value it does not recognize becomes NaN
and the imputer then fills it with the majority class: a candidate who answered "No"
to research experience would be scored as if they had answered "Yes", silently. The
conversion below refuses to guess instead.

Run it standalone with::

    python src/pipelines/inference_pipeline/inference_pipeline.py --input-path new.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RESEARCH_COLUMN = "Research"
PREDICTION_COLUMN = "Predicted Chance of Admit"

# Columns typed as whole numbers in the feature layer. The rest stay floating point.
INTEGER_COLUMNS = ["GRE Score", "TOEFL Score", "University Rating"]

# Spellings accepted for the research flag. Anything outside these two sets is refused
# rather than guessed, because the model cannot tell an unknown value from a negative
# one and would score both as the majority class.
RESEARCH_TRUE = frozenset({"1", "1.0", "true", "t", "yes", "y", "si", "sí", "verdadero"})
RESEARCH_FALSE = frozenset({"0", "0.0", "false", "f", "no", "n", "falso"})

# A chance of admission outside [0, 1] is not a probability. The model is a linear
# regression, so it extrapolates past both ends for candidates at the extremes.
PREDICTION_BOUNDS = (0.0, 1.0)
MAX_REPORTED_EXAMPLES = 5

MODEL_RELATIVE_PATH = Path("src/model/modelo-seleccion-admisiones.joblib")
PREDICTIONS_RELATIVE_PATH = Path("src/model/predicciones.csv")


class InferenceInputError(Exception):
    """Raised when the batch of new candidates does not match what the model expects."""


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until the directory holding ``pyproject.toml`` is found."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found above {current}")


def load_model(path: Path) -> Any:
    """Load the trained pipeline, preprocessing included.

    What comes back is not a bare regressor: it is the whole pipeline, carrying the
    medians, means and standard deviations fitted on the training split. There is
    nothing to re-fit and nothing to keep in sync on the side.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"No model at {path}. Run the training pipeline first: "
            "python src/pipelines/training_pipeline/train_pipeline.py"
        )
    logger.info("Loading model from %s", path)
    model = joblib.load(path)
    logger.info("Model expects the columns: %s", list(model.feature_names_in_))
    return model


def load_new_data(path: Path) -> pd.DataFrame:
    """Read the batch of candidates to score."""
    if not path.is_file():
        raise FileNotFoundError(f"No input file at {path}")
    logger.info("Reading new candidates from %s", path)
    candidates = pd.read_csv(path)
    if candidates.empty:
        raise InferenceInputError(f"The input file {path} holds no records to score")
    logger.info("Read %d candidate(s)", len(candidates))
    return candidates


def _rename_to_canonical(frame: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Match incoming headers to the model's columns, ignoring case and spacing.

    The label the model was fitted on carries a trailing space (``"LOR "``), which no
    one types by hand. Matching on the stripped lowercase form spares every caller
    from reproducing that accident.
    """
    canonical = {name.strip().lower(): name for name in expected}
    renamed = {
        column: canonical[column.strip().lower()]
        for column in frame.columns
        if column.strip().lower() in canonical
    }
    return frame.rename(columns=renamed)


def _convert_research(values: pd.Series) -> pd.Series:
    """Turn the research flag into the booleans the model was fitted on.

    Unrecognized values are refused rather than passed through: the encoder maps them
    to NaN and the imputer fills that with the majority class, so "No", "0" read as
    text, and an outright typo would all be scored as "Yes".
    """
    spelled = values.astype("string").str.strip().str.lower()
    converted = spelled.map(
        lambda value: (
            True if value in RESEARCH_TRUE else (False if value in RESEARCH_FALSE else pd.NA)
        )
    )
    unrecognized = values[converted.isna()]
    if not unrecognized.empty:
        examples = ", ".join(repr(value) for value in unrecognized.head(MAX_REPORTED_EXAMPLES))
        raise InferenceInputError(
            f"Column {RESEARCH_COLUMN!r} holds {len(unrecognized)} value(s) that cannot be "
            f"read as yes or no: {examples}. The model would score every one of them as "
            f"the majority class, so they are refused instead of guessed."
        )
    return converted.astype("boolean")


def _convert_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce the numeric columns, refusing values that are not numbers."""
    converted = frame.copy()
    for column in columns:
        numeric = pd.to_numeric(converted[column], errors="coerce")
        broken = converted[column][numeric.isna() & converted[column].notna()]
        if not broken.empty:
            examples = ", ".join(repr(value) for value in broken.head(MAX_REPORTED_EXAMPLES))
            raise InferenceInputError(
                f"Column {column!r} holds {len(broken)} value(s) that are not numbers: {examples}"
            )
        converted[column] = numeric
    return converted


def prepare_features(candidates: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Bring a batch of new candidates to the shape the model was fitted on.

    The expected columns come from the artifact itself, so the contract cannot drift
    away from the model that has to honour it.
    """
    renamed = _rename_to_canonical(candidates, expected_columns)
    missing = [column for column in expected_columns if column not in renamed.columns]
    if missing:
        raise InferenceInputError(f"The batch is missing the column(s) the model needs: {missing}")

    numeric_columns = [column for column in expected_columns if column != RESEARCH_COLUMN]
    prepared = _convert_numeric(renamed[expected_columns], numeric_columns)
    prepared[INTEGER_COLUMNS] = prepared[INTEGER_COLUMNS].round().astype("Int64")
    prepared[RESEARCH_COLUMN] = _convert_research(renamed[RESEARCH_COLUMN])
    logger.info("Prepared %d candidate(s) for scoring", len(prepared))
    return prepared[expected_columns]


def predict(model: Any, prepared: pd.DataFrame) -> np.ndarray:
    """Score the prepared candidates, keeping every chance a real probability."""
    raw = np.asarray(model.predict(prepared), dtype="float64")
    lower, upper = PREDICTION_BOUNDS
    clipped = np.clip(raw, lower, upper)
    out_of_range = int((raw != clipped).sum())
    if out_of_range:
        logger.warning(
            "%d prediction(s) fell outside [%s, %s] and were clipped: a linear model "
            "extrapolates past both ends for candidates at the extremes",
            out_of_range,
            lower,
            upper,
        )
    logger.info("Scored %d candidate(s)", len(clipped))
    return clipped


def save_predictions(candidates: pd.DataFrame, predictions: np.ndarray, path: Path) -> pd.DataFrame:
    """Store every candidate next to the chance the model gave it."""
    scored = candidates.copy()
    scored[PREDICTION_COLUMN] = predictions
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False)
    logger.info("Predictions stored in %s", path)
    return scored


def run_pipeline(model_path: Path, input_path: Path, output_path: Path) -> pd.DataFrame:
    """Run the whole inference pipeline: load, read, prepare, score, persist."""
    model = load_model(model_path)
    candidates = load_new_data(input_path)
    prepared = prepare_features(candidates, list(model.feature_names_in_))
    predictions = predict(model, prepared)
    return save_predictions(candidates, predictions, output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments of the standalone entry point."""
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="CSV file holding the candidates to score.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / MODEL_RELATIVE_PATH,
        help="Trained model to load.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / PREDICTIONS_RELATIVE_PATH,
        help="File where the predictions are stored.",
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
    run_pipeline(args.model_path, args.input_path, args.output_path)


if __name__ == "__main__":
    main()
