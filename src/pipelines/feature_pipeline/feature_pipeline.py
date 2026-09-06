"""Feature pipeline for the graduate admissions prediction project.

The stages run in this order: extract the immutable raw dataset, validate it
against the contract of the source, apply the model-independent transformations
defined in the exploration and feature-engineering notebooks, and store the
resulting reusable feature table in the feature layer.

Validation sits right after extraction, at the boundary where data enters the
system, so a broken source fails where its cause is. Because the gate runs
before any transformation, a failure aborts the run and nothing is persisted.

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
import pandera.pandas as pa
from pandera.errors import SchemaErrors

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

# Validation contract of the raw layer.
#
# The bounds are the ones documented in ``data/01_raw/Informacion.txt``, not the
# ones observed in the sample. An observed range is a property of the sample, not
# a rule of the domain: checking new data against it belongs to drift monitoring
# at serving time, and the Streamlit demo already does that.
#
# Date formats are not part of the contract: this dataset holds no temporal column.
# Row uniqueness is not part of it either: duplicated records are legitimate input,
# removing them is precisely what this pipeline does afterwards.
DOMAIN_RANGES: dict[str, tuple[float, float]] = {
    "GRE Score": (0, 340),
    "TOEFL Score": (0, 120),
    "SOP": (0, 5),
    LOR_COLUMN: (0, 5),
    "CGPA": (0, 10),
    TARGET_COLUMN: (0, 1),
}
VALID_UNIVERSITY_RATINGS = [1.0, 2.0, 3.0, 4.0, 5.0]
VALID_RESEARCH_VALUES = [0.0, 1.0]
MAX_NULL_RATIO = 0.10
MAX_REPORTED_EXAMPLES = 3


class FeatureValidationError(Exception):
    """Raised when the source data breaks the validation contract of the raw layer."""


def _tolerated_nulls() -> pa.Check:
    """Check that a column stays under the tolerated ratio of missing values.

    ``ignore_na`` must be disabled: by default pandera hands checks a series with
    the missing values already dropped, which would make this ratio always zero.
    """
    return pa.Check(
        lambda column: bool(column.isna().mean() <= MAX_NULL_RATIO),
        ignore_na=False,
        error=f"more than {MAX_NULL_RATIO:.0%} of the values are missing",
    )


def _bounded_column(name: str, *, nullable: bool = True) -> pa.Column:
    """Declare a numeric column bounded by its documented domain range."""
    minimum, maximum = DOMAIN_RANGES[name]
    return pa.Column(
        float,
        checks=[pa.Check.ge(minimum), pa.Check.le(maximum), _tolerated_nulls()],
        nullable=nullable,
    )


def _categorical_column(valid_values: list[float]) -> pa.Column:
    """Declare a discrete column restricted to a closed set of valid categories."""
    return pa.Column(
        float,
        checks=[pa.Check.isin(valid_values), _tolerated_nulls()],
        nullable=True,
    )


RAW_SCHEMA = pa.DataFrameSchema(
    columns={
        "GRE Score": _bounded_column("GRE Score"),
        "TOEFL Score": _bounded_column("TOEFL Score"),
        "University Rating": _categorical_column(VALID_UNIVERSITY_RATINGS),
        "SOP": _bounded_column("SOP"),
        LOR_COLUMN: _bounded_column(LOR_COLUMN),
        "CGPA": _bounded_column("CGPA"),
        BOOLEAN_COLUMN: _categorical_column(VALID_RESEARCH_VALUES),
        # The label is the one column that may not be missing: a record without it
        # cannot train anything.
        TARGET_COLUMN: _bounded_column(TARGET_COLUMN, nullable=False),
    },
    checks=[pa.Check(lambda frame: not frame.empty, error="the raw table holds no rows")],
    strict=True,
    ordered=True,
    name="raw admissions data",
)


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until the directory holding ``pyproject.toml`` is found."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"No pyproject.toml found above {current}")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Read the raw admissions CSV, without judging its content.

    Extraction never rejects anything: every rule about the source lives in
    :data:`RAW_SCHEMA`, so there is a single place to read the contract from.
    """
    logger.info("Reading raw data from %s", path)
    raw = pd.read_csv(path)
    logger.debug("Raw shape: %s", raw.shape)
    return raw


def _describe_failures(error: SchemaErrors) -> str:
    """Turn the failure cases collected by pandera into a message a person can act on."""
    lines = ["The raw data does not satisfy the validation contract:"]
    for (column, check), cases in error.failure_cases.groupby(["column", "check"], dropna=False):
        subject = column if isinstance(column, str) else "the table"
        examples = ", ".join(
            str(case) for case in cases["failure_case"].head(MAX_REPORTED_EXAMPLES)
        )
        lines.append(f"  - {subject}: {check} — {len(cases)} case(s), e.g. {examples}")
    return "\n".join(lines)


def validate_raw_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Check the extracted data against the contract of the raw layer.

    Validation is lazy so that a single run reports every broken rule instead of
    dying on the first one. Any failure raises :class:`FeatureValidationError`
    before a single transformation runs, so nothing reaches the feature layer.
    """
    logger.info("Validating raw data against the schema of the source")
    try:
        validated = RAW_SCHEMA.validate(raw, lazy=True)
    except SchemaErrors as error:
        raise FeatureValidationError(_describe_failures(error)) from error
    logger.info("Raw data is valid: %d rows", len(validated))
    return pd.DataFrame(validated)


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
    """Run the whole feature pipeline: extract, validate, transform, persist."""
    raw = validate_raw_data(load_raw_data(raw_path))
    features = build_features(raw)
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
