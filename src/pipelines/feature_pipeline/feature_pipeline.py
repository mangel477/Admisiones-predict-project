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
CANDIDATES_RELATIVE_PATH = Path("data/04_feature/candidatos_sin_etiqueta.parquet")

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

# Spellings accepted for the research flag of an undecided candidate. Batches reach the
# store from outside the project, so the flag arrives written however someone typed it.
RESEARCH_TRUE = frozenset({"1", "1.0", "true", "t", "yes", "y", "si", "sí", "verdadero"})
RESEARCH_FALSE = frozenset({"0", "0.0", "false", "f", "no", "n", "falso"})
MAX_NULL_RATIO = 0.10
MAX_REPORTED_EXAMPLES = 3

# Granularity of the instrument behind each scale. A value inside the documented
# range can still be impossible: the letter scales are scored in half points and
# the exam scores are whole numbers, so 3.7 is not a valid SOP no matter what.
SCALE_STEPS: dict[str, float] = {
    "GRE Score": 1.0,
    "TOEFL Score": 1.0,
    "SOP": 0.5,
    LOR_COLUMN: 0.5,
}


class FeatureValidationError(Exception):
    """Raised when the data breaks one of the validation contracts of the pipeline."""


def _on_scale_grid(step: float) -> pa.Check:
    """Check that every value of a column falls on the grid of its scale."""
    return pa.Check(
        lambda column: (column / step) % 1 == 0,
        error=f"values must fall on the grid of {step} of their scale",
    )


def _every_record_observes_a_predictor(frame: pd.DataFrame) -> bool:
    """Integrity between fields: no record may arrive with every predictor missing.

    Such a record shares no observed column with any other, so it survives both
    deduplication stages and drags its gaps into the feature layer.
    """
    present = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if not present:
        return True
    return bool(frame[present].notna().any(axis=1).all())


def _holds_no_contradictory_records(frame: pd.DataFrame) -> bool:
    """Integrity between records: identical predictors must carry identical labels.

    Records whose predictors are all observed and equal describe the same candidate
    profile, so two different admission chances cannot both be true.
    """
    present = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if len(present) < len(FEATURE_COLUMNS) or TARGET_COLUMN not in frame.columns:
        return True
    complete = frame.dropna(subset=[*present, TARGET_COLUMN])
    if complete.empty:
        return True
    labels_per_profile = complete.groupby(present)[TARGET_COLUMN].transform("nunique")
    return bool(labels_per_profile.eq(1).all())


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
    """Declare a numeric column bounded by its documented domain range and its grid."""
    minimum, maximum = DOMAIN_RANGES[name]
    checks = [pa.Check.ge(minimum), pa.Check.le(maximum), _tolerated_nulls()]
    if name in SCALE_STEPS:
        checks.append(_on_scale_grid(SCALE_STEPS[name]))
    return pa.Column(float, checks=checks, nullable=nullable)


def _categorical_column(valid_values: list[float]) -> pa.Column:
    """Declare a discrete column restricted to a closed set of valid categories."""
    return pa.Column(
        float,
        checks=[pa.Check.isin(valid_values), _tolerated_nulls()],
        nullable=True,
    )


def _typed_feature(dtype: str, name: str) -> pa.Column:
    """Declare a feature column: the dtype the model expects, bounded and complete."""
    minimum, maximum = DOMAIN_RANGES[name]
    return pa.Column(
        dtype,
        checks=[pa.Check.ge(minimum), pa.Check.le(maximum)],
        nullable=False,
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
    checks=[
        pa.Check(lambda frame: not frame.empty, error="the raw table holds no rows"),
        pa.Check(
            _every_record_observes_a_predictor,
            error="a record arrived without a single observed predictor",
        ),
        pa.Check(
            _holds_no_contradictory_records,
            error="contradictory records: identical predictors carry different labels",
        ),
    ],
    strict=True,
    ordered=True,
    name="raw admissions data",
)

FEATURE_SCHEMA = pa.DataFrameSchema(
    columns={
        "GRE Score": _typed_feature("Int64", "GRE Score"),
        "TOEFL Score": _typed_feature("Int64", "TOEFL Score"),
        "University Rating": pa.Column(
            "Int64",
            checks=[pa.Check.isin([int(rating) for rating in VALID_UNIVERSITY_RATINGS])],
            nullable=False,
        ),
        "SOP": _typed_feature("float64", "SOP"),
        LOR_COLUMN: _typed_feature("float64", LOR_COLUMN),
        "CGPA": _typed_feature("float64", "CGPA"),
        BOOLEAN_COLUMN: pa.Column("boolean", nullable=False),
        TARGET_COLUMN: _typed_feature("float64", TARGET_COLUMN),
    },
    checks=[pa.Check(lambda frame: not frame.empty, error="the feature table holds no rows")],
    unique=EXPECTED_COLUMNS,
    strict=True,
    ordered=True,
    name="admissions feature table",
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
    """Turn the failure cases collected by pandera into a message a person can act on.

    Rules that judge a whole column or the whole table report their verdict as a
    boolean rather than an offending value, so for those only the rule is named.
    """
    lines = ["The data does not satisfy the validation contract:"]
    for (column, check), cases in error.failure_cases.groupby(["column", "check"], dropna=False):
        subject = column if isinstance(column, str) else "the table"
        offenders = [
            case for case in cases["failure_case"] if not isinstance(case, bool | np.bool_)
        ]
        if not offenders:
            lines.append(f"  - {subject}: {check}")
            continue
        examples = ", ".join(str(case) for case in offenders[:MAX_REPORTED_EXAMPLES])
        lines.append(f"  - {subject}: {check} — {len(offenders)} case(s), e.g. {examples}")
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


def _as_raw_scale(frame: pd.DataFrame) -> pd.DataFrame:
    """Bring a typed table back to the plain float representation of the raw layer."""
    values = frame.astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
    return pd.DataFrame(values, columns=list(frame.columns))


def _records_absent_from(features: pd.DataFrame, raw: pd.DataFrame) -> int:
    """Count feature records that cannot be traced back to a record of the source."""
    sources = _as_raw_scale(raw[list(features.columns)]).drop_duplicates()
    traced = _as_raw_scale(features).merge(sources, how="left", indicator=True)
    return int((traced["_merge"] != "both").sum())


def validate_feature_table(features: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Check the transformed table before it reaches the feature layer.

    Two contracts meet here. :data:`FEATURE_SCHEMA` states what the feature layer
    promises to whoever reads it — the dtypes the model was fitted on, complete and
    unique records — and the training pipeline reuses that same schema as its own
    entry gate. The remaining checks are integrity between datasets: transforming
    may drop records, never invent or alter them.
    """
    logger.info("Validating the feature table against the contract of the feature layer")
    failures: list[str] = []
    try:
        validated = FEATURE_SCHEMA.validate(features, lazy=True)
    except SchemaErrors as error:
        raise FeatureValidationError(_describe_failures(error)) from error

    if len(features) > len(raw):
        failures.append(
            f"  - the feature table holds more records than the raw layer: "
            f"{len(features)} against {len(raw)}"
        )
    elif (orphans := _records_absent_from(features, raw)) > 0:
        failures.append(f"  - {orphans} record(s) are not present in the raw layer")

    if failures:
        raise FeatureValidationError(
            "\n".join(["The feature table breaks its integrity with the raw layer:", *failures])
        )
    logger.info("Feature table is valid: %d rows", len(validated))
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


# The historical source tolerates gaps because deduplication resolves them: a record
# missing a value turns out to be a partial copy of a complete one. A candidate batch has
# no such mechanism, and the model would quietly fill the hole with the training median —
# a missing CGPA moves the predicted chance by almost eleven percentage points, with
# nothing in the output to say the number was partly invented. So this door demands
# complete records.
CANDIDATE_SCHEMA = pa.DataFrameSchema(
    columns={
        name: pa.Column(column.dtype, checks=column.checks, nullable=False)
        for name, column in RAW_SCHEMA.columns.items()
        if name != TARGET_COLUMN
    },
    checks=[
        pa.Check(lambda frame: not frame.empty, error="the candidate batch holds no rows"),
        pa.Check(
            _every_record_observes_a_predictor,
            error="a candidate arrived without a single observed predictor",
        ),
    ],
    strict=True,
    ordered=True,
    name="undecided admissions candidates",
)


# What the candidate batch promises to whoever reads it. Identical to the historical
# contract except for one invariant: rows are not unique here, because two applicants
# with the same scores are two people and each one needs their own prediction.
CANDIDATE_FEATURE_SCHEMA = pa.DataFrameSchema(
    columns={
        name: column for name, column in FEATURE_SCHEMA.columns.items() if name != TARGET_COLUMN
    },
    checks=[pa.Check(lambda frame: not frame.empty, error="the candidate table holds no rows")],
    strict=True,
    ordered=True,
    name="undecided candidate features",
)


def _rename_to_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Match incoming headers to the expected columns, ignoring case and spacing.

    Candidates reach the feature store from outside the project, written by people who
    do not know that ``"LOR "`` carries a trailing space. Two headers that collapse to
    the same column leave the batch ambiguous, so they are refused rather than merged.
    """
    canonical = {name.strip().lower(): name for name in FEATURE_COLUMNS}
    renamed = {
        column: canonical[column.strip().lower()]
        for column in frame.columns
        if column.strip().lower() in canonical
    }
    collisions: dict[str, list[str]] = {}
    for original, target in renamed.items():
        collisions.setdefault(target, []).append(original)
    ambiguous = {target: sources for target, sources in collisions.items() if len(sources) > 1}
    if ambiguous:
        raise FeatureValidationError(
            "The batch carries more than one header for the same column, so which one "
            f"should reach the model is undecidable: {ambiguous}"
        )
    return frame.rename(columns=renamed)


def _normalize_research(values: pd.Series) -> pd.Series:
    """Read the research flag however it was written, refusing what cannot be read.

    Unrecognized values must not survive this door. Downstream, the encoder maps them
    to NaN and the imputer fills that with the majority class, so "No", "0" read as
    text, and an outright typo would all end up scored as "Yes".
    """
    spelled = values.astype("string").str.strip().str.lower()
    converted = spelled.map(
        lambda value: 1.0 if value in RESEARCH_TRUE else (0.0 if value in RESEARCH_FALSE else pd.NA)
    )
    unrecognized = values[converted.isna() & values.notna()]
    if not unrecognized.empty:
        examples = ", ".join(repr(value) for value in unrecognized.head(MAX_REPORTED_EXAMPLES))
        raise FeatureValidationError(
            f"Column {BOOLEAN_COLUMN!r} holds {len(unrecognized)} value(s) that cannot be "
            f"read as yes or no: {examples}. Every one of them would be scored as the "
            f"majority class, so they are refused instead of guessed."
        )
    return converted.astype("float64")


def _as_float_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Bring already-numeric columns to float, the representation the contract states.

    A clean candidate batch carries no gaps, so pandas types its whole numbers as
    ``int64`` and the schema would refuse valid data over a detail of type inference.
    Columns that are not numeric at all are left untouched: the schema has to see them
    and say so itself.
    """
    numeric = frame.copy()
    for column in numeric.columns:
        if pd.api.types.is_numeric_dtype(numeric[column]):
            numeric[column] = numeric[column].astype("float64")
    return numeric


def validate_candidate_data(candidates: pd.DataFrame) -> pd.DataFrame:
    """Check a batch of undecided candidates against the contract of the source.

    Same rules as the historical source, minus the two that need a label: there is no
    target to bound, and records cannot contradict each other on an answer nobody has
    given yet. Duplicates are allowed on purpose — two applicants with identical scores
    are two people, and each one needs their own prediction.
    """
    logger.info("Validating a batch of undecided candidates")
    try:
        validated = CANDIDATE_SCHEMA.validate(_as_float_columns(candidates), lazy=True)
    except SchemaErrors as error:
        raise FeatureValidationError(_describe_failures(error)) from error
    logger.info("Candidate batch is valid: %d candidate(s)", len(validated))
    return pd.DataFrame(validated)


def build_candidate_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Apply the model-independent transformations to a batch of new candidates.

    What leaves here carries the dtypes and the ranges the model was fitted on, and no
    gaps, exactly like the historical feature table. It differs in one invariant, on
    purpose: rows are not unique. Deduplication protects training from counting one
    record twice; here every applicant is a person waiting for their own answer.
    """
    renamed = _rename_to_canonical(candidates)
    if BOOLEAN_COLUMN in renamed.columns:
        renamed[BOOLEAN_COLUMN] = _normalize_research(renamed[BOOLEAN_COLUMN])
    features = cast_dtypes(validate_candidate_data(renamed))[FEATURE_COLUMNS]
    features = features.reset_index(drop=True)
    try:
        CANDIDATE_FEATURE_SCHEMA.validate(features, lazy=True)
    except SchemaErrors as error:
        raise FeatureValidationError(_describe_failures(error)) from error
    logger.info("Candidate feature table shape: %s", features.shape)
    return features


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
    features = validate_feature_table(build_features(raw), raw)
    save_features(features, output_path)
    return features


def run_candidate_pipeline(candidates_path: Path, output_path: Path) -> pd.DataFrame:
    """Take a batch of undecided candidates into the feature store.

    The second door of the store. Records arrive without a label because nobody has
    decided them yet, and leave holding the dtypes, ranges and completeness the model
    expects, so whoever reads them downstream does not have to check the data again.
    The one invariant that differs from the historical table is uniqueness: repeated
    applicants are kept.
    """
    features = build_candidate_features(load_raw_data(candidates_path))
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
        "--candidates-path",
        type=Path,
        default=None,
        help="CSV of undecided candidates. When given, that batch is processed instead "
        "of the historical source.",
    )
    parser.add_argument(
        "--candidates-output-path",
        type=Path,
        default=project_root / CANDIDATES_RELATIVE_PATH,
        help="Parquet file where the undecided candidates are stored.",
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
    if args.candidates_path is not None:
        run_candidate_pipeline(args.candidates_path, args.candidates_output_path)
        return
    run_pipeline(args.raw_path, args.output_path)


if __name__ == "__main__":
    main()
