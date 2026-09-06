"""Feature pipeline for the graduate admissions prediction project.

Reads the immutable raw dataset, applies the model-independent transformations
defined in the exploration and feature-engineering notebooks, and stores the
resulting reusable feature table in the feature layer.

Only model-independent transformations belong here (typing and deduplication).
Imputation, scaling and encoding are model-dependent: they are parameterized by
the training data and stay inside the scikit-learn pipeline of the training
stage, so that they are fitted on the training split alone.

Run it standalone with::

    python src/pipelines/feature_pipeline/feature_pipeline.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOR_COLUMN = "LOR "
TARGET_COLUMN = "Chance of Admit "
INTEGER_COLUMNS = ["GRE Score", "TOEFL Score", "University Rating"]
BOOLEAN_COLUMN = "Research"
FEATURE_COLUMNS = [
    "GRE Score",
    "TOEFL Score",
    "University Rating",
    "SOP",
    LOR_COLUMN,
    "CGPA",
    BOOLEAN_COLUMN,
]
EXPECTED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]

RAW_RELATIVE_PATH = Path("data/01_raw/Admission_Predict.csv")
FEATURE_RELATIVE_PATH = Path("data/04_feature/admisiones_features.parquet")


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until the directory holding ``pyproject.toml`` is found."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found above {current}")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the raw admissions CSV and check that the expected schema is present."""
    logger.info("Reading raw data from %s", path)
    raw = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Raw file {path} is missing expected columns: {missing}")
    logger.debug("Raw shape: %s", raw.shape)
    return raw


def cast_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Type the raw columns: discrete scores as ``Int64`` and research as ``boolean``.

    Nullable dtypes are used on purpose so that the missing values present in the
    raw file survive the cast instead of forcing the columns back to ``float``.
    """
    typed = frame.copy()
    typed[INTEGER_COLUMNS] = typed[INTEGER_COLUMNS].astype("Int64")
    typed[BOOLEAN_COLUMN] = (
        pd.to_numeric(typed[BOOLEAN_COLUMN], errors="coerce")
        .map({0.0: False, 1.0: True})
        .astype("boolean")
    )
    return typed


def drop_exact_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that repeat another row on every column, keeping the first one."""
    deduplicated = frame.drop_duplicates()
    logger.info("Exact duplicates removed: %d", len(frame) - len(deduplicated))
    return deduplicated


def flag_masked_duplicates(frame: pd.DataFrame) -> np.ndarray:
    """Flag rows that repeat an earlier row on every jointly observed column.

    A missing value hides an otherwise identical record from ``drop_duplicates``.
    Two rows are considered compatible when they share at least one observed
    column and agree on all of them; the later row is the one flagged.
    """
    values = frame.astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
    observed = ~np.isnan(values)
    is_duplicate = np.zeros(len(values), dtype=bool)
    for index in range(len(values) - 1):
        shared = observed[index] & observed[index + 1 :]
        matches = np.where(shared, values[index] == values[index + 1 :], True).all(
            axis=1
        ) & shared.any(axis=1)
        is_duplicate[np.nonzero(matches)[0] + index + 1] = True
    return is_duplicate


def drop_masked_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the duplicates that missing values keep hidden from an exact comparison."""
    deduplicated = frame[~flag_masked_duplicates(frame)]
    logger.info("Masked duplicates removed: %d", len(frame) - len(deduplicated))
    return deduplicated


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the model-independent transformations and return the feature table."""
    features = drop_masked_duplicates(drop_exact_duplicates(cast_dtypes(raw)))
    features = features[EXPECTED_COLUMNS].reset_index(drop=True)
    remaining_nulls = int(features.isna().sum().sum())
    if remaining_nulls:
        logger.warning("Feature table still holds %d missing values", remaining_nulls)
    logger.info("Feature table shape: %s", features.shape)
    return features


def save_features(features: pd.DataFrame, path: Path) -> Path:
    """Write the feature table to the feature layer as parquet, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Features stored in %s", path)
    return path


def run_pipeline(raw_path: Path, output_path: Path) -> pd.DataFrame:
    """Run the whole feature pipeline: read the raw layer, transform, persist."""
    features = build_features(load_raw_data(raw_path))
    save_features(features, output_path)
    return features


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments of the standalone entry point."""
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=project_root / RAW_RELATIVE_PATH,
        help="CSV file of the raw layer to read.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / FEATURE_RELATIVE_PATH,
        help="Parquet file of the feature layer to write.",
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
    run_pipeline(args.raw_path, args.output_path)


if __name__ == "__main__":
    main()
