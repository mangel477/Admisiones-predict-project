"""Unit tests for the model-independent admissions feature pipeline."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import (
    FEATURE_COLUMNS,
    FEATURE_RELATIVE_PATH,
    LOR_COLUMN,
    RAW_RELATIVE_PATH,
    TARGET_COLUMN,
    build_features,
    cast_dtypes,
    drop_exact_duplicates,
    drop_masked_duplicates,
    find_project_root,
    load_raw_data,
    main,
    parse_args,
    run_pipeline,
    save_features,
)

COMPLETE_ROW: dict[str, float | None] = {
    "GRE Score": 337.0,
    "TOEFL Score": 118.0,
    "University Rating": 4.0,
    "SOP": 4.5,
    LOR_COLUMN: 4.5,
    "CGPA": 9.65,
    "Research": 1.0,
    TARGET_COLUMN: 0.92,
}
OTHER_ROW: dict[str, float | None] = {
    "GRE Score": 324.0,
    "TOEFL Score": 107.0,
    "University Rating": 3.0,
    "SOP": 4.0,
    LOR_COLUMN: 3.5,
    "CGPA": 8.87,
    "Research": 0.0,
    TARGET_COLUMN: 0.76,
}

ONE_ROW = 1
TWO_ROWS = 2
GRE_OF_COMPLETE_ROW = 337
CGPA_OF_COMPLETE_ROW = 9.65


def raw_frame(*rows: dict[str, float | None]) -> pd.DataFrame:
    """Build a raw-shaped dataframe, mimicking how ``read_csv`` types the source file."""
    return pd.DataFrame(list(rows), columns=list(COMPLETE_ROW)).astype("float64")


def write_raw_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Persist a raw-shaped dataframe as the pipeline expects to find it on disk."""
    frame.to_csv(path, index=False)
    return path


class TestLoadRawData:
    """Reading the immutable raw layer."""

    def test_reads_the_expected_columns(self, tmp_path: Path) -> None:
        source = write_raw_csv(tmp_path / "raw.csv", raw_frame(COMPLETE_ROW, OTHER_ROW))

        loaded = load_raw_data(source)

        assert list(loaded.columns) == list(COMPLETE_ROW)
        assert len(loaded) == TWO_ROWS

    def test_preserves_the_trailing_spaces_of_the_source_header(self, tmp_path: Path) -> None:
        source = write_raw_csv(tmp_path / "raw.csv", raw_frame(COMPLETE_ROW))

        loaded = load_raw_data(source)

        assert LOR_COLUMN in loaded.columns
        assert TARGET_COLUMN in loaded.columns

    def test_rejects_a_source_missing_expected_columns(self, tmp_path: Path) -> None:
        incomplete = raw_frame(COMPLETE_ROW).drop(columns=["CGPA"])
        source = write_raw_csv(tmp_path / "raw.csv", incomplete)

        with pytest.raises(ValueError, match="CGPA"):
            load_raw_data(source)


class TestCastDtypes:
    """Model-independent typing of the raw layer."""

    def test_casts_discrete_scores_to_nullable_integers(self) -> None:
        typed = cast_dtypes(raw_frame(COMPLETE_ROW))

        assert typed["GRE Score"].dtype == "Int64"
        assert typed["TOEFL Score"].dtype == "Int64"
        assert typed["University Rating"].dtype == "Int64"

    def test_casts_research_to_nullable_boolean(self) -> None:
        typed = cast_dtypes(raw_frame(COMPLETE_ROW, OTHER_ROW))

        assert typed["Research"].dtype == "boolean"
        assert bool(typed["Research"].iloc[0]) is True
        assert bool(typed["Research"].iloc[1]) is False

    def test_keeps_missing_values_as_na(self) -> None:
        incomplete = {**COMPLETE_ROW, "GRE Score": None, "Research": None}

        typed = cast_dtypes(raw_frame(incomplete))

        assert typed["GRE Score"].isna().all()
        assert typed["Research"].isna().all()

    def test_leaves_continuous_columns_untouched(self) -> None:
        typed = cast_dtypes(raw_frame(COMPLETE_ROW))

        assert typed["CGPA"].dtype == "float64"
        assert typed[TARGET_COLUMN].dtype == "float64"


class TestDropExactDuplicates:
    """Removal of byte-identical records."""

    def test_keeps_a_single_copy_of_repeated_rows(self) -> None:
        frame = cast_dtypes(raw_frame(COMPLETE_ROW, COMPLETE_ROW, OTHER_ROW))

        deduplicated = drop_exact_duplicates(frame)

        assert len(deduplicated) == TWO_ROWS

    def test_leaves_distinct_rows_alone(self) -> None:
        frame = cast_dtypes(raw_frame(COMPLETE_ROW, OTHER_ROW))

        assert len(drop_exact_duplicates(frame)) == TWO_ROWS


class TestDropMaskedDuplicates:
    """Removal of duplicates hidden behind missing values."""

    def test_removes_a_partial_copy_of_a_complete_row(self) -> None:
        partial_copy = {**COMPLETE_ROW, "GRE Score": None}
        frame = cast_dtypes(raw_frame(COMPLETE_ROW, partial_copy))

        deduplicated = drop_masked_duplicates(frame)

        assert len(deduplicated) == ONE_ROW
        assert deduplicated["GRE Score"].iloc[0] == GRE_OF_COMPLETE_ROW

    def test_keeps_the_first_occurrence(self) -> None:
        partial_copy = {**COMPLETE_ROW, "TOEFL Score": None}
        frame = cast_dtypes(raw_frame(partial_copy, COMPLETE_ROW))

        deduplicated = drop_masked_duplicates(frame)

        assert len(deduplicated) == ONE_ROW
        assert deduplicated["TOEFL Score"].isna().all()

    def test_keeps_an_incomplete_row_that_differs_on_an_observed_column(self) -> None:
        distinct = {**OTHER_ROW, "SOP": None}
        frame = cast_dtypes(raw_frame(COMPLETE_ROW, distinct))

        assert len(drop_masked_duplicates(frame)) == TWO_ROWS

    def test_keeps_a_row_with_no_column_in_common(self) -> None:
        empty_row: dict[str, float | None] = dict.fromkeys(COMPLETE_ROW)
        frame = cast_dtypes(raw_frame(COMPLETE_ROW, empty_row))

        assert len(drop_masked_duplicates(frame)) == TWO_ROWS


class TestBuildFeatures:
    """The end-to-end model-independent transformation."""

    def test_returns_features_and_label_in_a_stable_order(self) -> None:
        features = build_features(raw_frame(COMPLETE_ROW, OTHER_ROW))

        assert list(features.columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]

    def test_applies_both_deduplication_stages(self) -> None:
        partial_copy = {**COMPLETE_ROW, "CGPA": None}
        features = build_features(raw_frame(COMPLETE_ROW, COMPLETE_ROW, partial_copy, OTHER_ROW))

        assert len(features) == TWO_ROWS

    def test_does_not_scale_or_encode_the_values(self) -> None:
        """Scaling, imputing and encoding are model-dependent: they belong to training."""
        features = build_features(raw_frame(COMPLETE_ROW, OTHER_ROW))

        assert features["CGPA"].iloc[0] == CGPA_OF_COMPLETE_ROW
        assert features["GRE Score"].iloc[0] == GRE_OF_COMPLETE_ROW
        assert features["Research"].dtype == "boolean"

    def test_does_not_impute_the_missing_values_it_cannot_deduplicate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Imputation is model-dependent; a surviving gap is reported, never filled."""
        incomplete = {**OTHER_ROW, "SOP": None}

        with caplog.at_level(logging.WARNING):
            features = build_features(raw_frame(COMPLETE_ROW, incomplete))

        assert features["SOP"].isna().sum() == ONE_ROW
        assert "missing values" in caplog.text

    def test_resets_the_index_after_dropping_rows(self) -> None:
        features = build_features(raw_frame(COMPLETE_ROW, COMPLETE_ROW, OTHER_ROW))

        assert list(features.index) == list(range(len(features)))


class TestSaveFeatures:
    """Persistence into the feature layer."""

    def test_creates_the_parent_directory(self, tmp_path: Path) -> None:
        destination = tmp_path / "04_feature" / "features.parquet"

        save_features(build_features(raw_frame(COMPLETE_ROW)), destination)

        assert destination.exists()

    def test_round_trips_the_nullable_dtypes(self, tmp_path: Path) -> None:
        destination = tmp_path / "features.parquet"
        features = build_features(raw_frame(COMPLETE_ROW, OTHER_ROW))

        save_features(features, destination)
        reloaded = pd.read_parquet(destination)

        assert reloaded["GRE Score"].dtype == "Int64"
        assert reloaded["Research"].dtype == "boolean"
        assert list(reloaded.columns) == list(features.columns)


class TestRunPipeline:
    """The orchestration of the three stages."""

    def test_writes_the_feature_table_from_the_raw_file(self, tmp_path: Path) -> None:
        source = write_raw_csv(
            tmp_path / "raw.csv", raw_frame(COMPLETE_ROW, COMPLETE_ROW, OTHER_ROW)
        )
        destination = tmp_path / "features.parquet"

        features = run_pipeline(source, destination)

        assert destination.exists()
        assert len(features) == TWO_ROWS
        assert list(pd.read_parquet(destination).columns) == [*FEATURE_COLUMNS, TARGET_COLUMN]


class TestProjectRoot:
    """Path resolution, so the script runs from any working directory."""

    def test_finds_the_directory_holding_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        nested = tmp_path / "src" / "pipelines"
        nested.mkdir(parents=True)

        assert find_project_root(nested) == tmp_path.resolve()

    def test_fails_when_there_is_no_project_above(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match=r"pyproject\.toml"):
            find_project_root(tmp_path)


class TestCommandLine:
    """The autonomous entry point."""

    def test_defaults_point_at_the_raw_and_feature_layers(self) -> None:
        args = parse_args([])

        assert args.raw_path == find_project_root() / RAW_RELATIVE_PATH
        assert args.output_path == find_project_root() / FEATURE_RELATIVE_PATH

    def test_accepts_explicit_paths(self, tmp_path: Path) -> None:
        args = parse_args(["--raw-path", str(tmp_path / "in.csv"), "--output-path", "out.parquet"])

        assert args.raw_path == tmp_path / "in.csv"
        assert args.output_path == Path("out.parquet")

    def test_main_runs_the_pipeline_end_to_end(self, tmp_path: Path) -> None:
        source = write_raw_csv(tmp_path / "raw.csv", raw_frame(COMPLETE_ROW, OTHER_ROW))
        destination = tmp_path / "features.parquet"

        main(["--raw-path", str(source), "--output-path", str(destination), "--log-level", "ERROR"])

        assert len(pd.read_parquet(destination)) == TWO_ROWS
