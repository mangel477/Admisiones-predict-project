"""Inference pipeline for the graduate admissions prediction project.

The stages run in this order: load the model stored by the training pipeline, read the
batch of undecided candidates from the feature store, score them, and store the
predictions.

There is no data preparation here, and that is the whole point of the architecture.
The transformations that do not depend on the model — typing, reading the research flag
however it was written, refusing what cannot be read — already happened in the feature
pipeline, at the door where candidates enter the system. The ones that do depend on it —
imputation, scaling, encoding — were fitted on the training split and serialized inside
the artifact, so loading it recovers them with the values they learned.

Both sides of the split therefore reach the model through the same code that prepared
the training data, which is what makes the predictions comparable to the metrics the
training pipeline reported. What is left to do here is read, score, and write.

Run it standalone with::

    python src/pipelines/inference_pipeline/inference_pipeline.py
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

PREDICTION_COLUMN = "Predicted Chance of Admit"

# A chance of admission outside [0, 1] is not a probability. The model is a linear
# regression, so it extrapolates past both ends for candidates at the extremes.
PREDICTION_BOUNDS = (0.0, 1.0)

MODEL_RELATIVE_PATH = Path("src/model/modelo-seleccion-admisiones.joblib")
CANDIDATES_RELATIVE_PATH = Path("data/04_feature/candidatos_sin_etiqueta.parquet")
PREDICTIONS_RELATIVE_PATH = Path("src/model/predicciones.csv")


class InferenceError(Exception):
    """Raised when the stored batch and the stored model do not describe the same problem."""


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
    nothing to re-fit here and nothing to keep in sync on the side.
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


def load_candidates(path: Path) -> pd.DataFrame:
    """Read the undecided candidates the feature pipeline left in the store.

    Nothing is checked about their content. The feature store is the door where data
    enters the system and the only place that judges it; repeating those rules here
    would give them two owners and, sooner or later, two different answers.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"No candidates at {path}. Take a batch into the store with the feature "
            "pipeline first: "
            "python src/pipelines/feature_pipeline/feature_pipeline.py --candidates-path <csv>"
        )
    logger.info("Reading candidates from %s", path)
    candidates = pd.read_parquet(path, engine="pyarrow")
    if candidates.empty:
        raise InferenceError(f"The stored batch at {path} holds no candidates to score")
    logger.info("Read %d candidate(s)", len(candidates))
    return candidates


def select_model_features(candidates: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Hand the model the columns it was fitted on, in its own order.

    This is a contract check between two artifacts, not a second validation of the data:
    a model retrained on a different set of columns would otherwise fail deep inside
    scikit-learn, with a message that says nothing about which pipeline to run again.
    """
    missing = [column for column in expected_columns if column not in candidates.columns]
    if missing:
        raise InferenceError(
            f"The stored batch does not carry the column(s) this model was fitted on: "
            f"{missing}. The feature store and the model are out of step — regenerate "
            f"the batch, or train a model on the columns the store holds."
        )
    return candidates[expected_columns]


def predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Score the candidates, keeping every chance a real probability."""
    raw = np.asarray(model.predict(features), dtype="float64")
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
    """Store every candidate next to the chance the model gave it.

    The predictions are placed by position. Handing pandas a labelled series would let
    it align by index instead, and an index that does not match the batch writes a
    column of silent NaNs rather than failing — the batch would look scored when it is
    not. The count is checked for the same reason: this function promises one chance
    per candidate, so it says when it cannot keep that promise.
    """
    values = np.asarray(predictions, dtype="float64")
    if len(values) != len(candidates):
        raise InferenceError(
            f"Expected one prediction per candidate, got {len(values)} prediction(s) "
            f"for {len(candidates)} candidate(s)"
        )
    scored = candidates.copy()
    scored[PREDICTION_COLUMN] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(path, index=False)
    logger.info("Predictions stored in %s", path)
    return scored


def run_pipeline(model_path: Path, candidates_path: Path, output_path: Path) -> pd.DataFrame:
    """Run the whole inference pipeline: load, read, score, persist."""
    model = load_model(model_path)
    candidates = load_candidates(candidates_path)
    features = select_model_features(candidates, list(model.feature_names_in_))
    return save_predictions(candidates, predict(model, features), output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments of the standalone entry point."""
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-path",
        type=Path,
        default=project_root / MODEL_RELATIVE_PATH,
        help="Trained model to load.",
    )
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=project_root / CANDIDATES_RELATIVE_PATH,
        help="Parquet of undecided candidates to read from the feature store.",
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
    run_pipeline(args.model_path, args.candidates_path, args.output_path)


if __name__ == "__main__":
    main()
