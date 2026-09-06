"""Unit tests for the model-independent admissions feature pipeline."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from pipelines.feature_pipeline.feature_pipeline import (
    FEATURE_COLUMNS,
    FEATURE_RELATIVE_PATH,
    LOR_COLUMN,
    MAX_NULL_RATIO,
    RAW_RELATIVE_PATH,
    TARGET_COLUMN,
    FeatureValidationError,
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
    validate_feature_table,
    validate_raw_data,
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

# Row counts that put a single missing value on either side of the tolerated ratio.
ROWS_UNDER_NULL_THRESHOLD = round(2 / MAX_NULL_RATIO)
ROWS_OVER_NULL_THRESHOLD = round(1 / MAX_NULL_RATIO) - 1


def raw_frame(*rows: dict[str, float | None]) -> pd.DataFrame:
    """Build a raw-shaped dataframe, mimicking how ``read_csv`` types the source file."""
    return pd.DataFrame(list(rows), columns=list(COMPLETE_ROW)).astype("float64")


def write_raw_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Persist a raw-shaped dataframe as the pipeline expects to find it on disk."""
    frame.to_csv(path, index=False)
    return path


def rows_with_one_null(column: str, total_rows: int) -> pd.DataFrame:
    """Build ``total_rows`` valid rows where a single one misses ``column``."""
    incomplete: dict[str, float | None] = {**COMPLETE_ROW, column: None}
    return raw_frame(incomplete, *([COMPLETE_ROW] * (total_rows - 1)))


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

    def test_only_extracts_and_leaves_judgement_to_the_validation_step(
        self, tmp_path: Path
    ) -> None:
        """Extraction never rejects: the schema owns every rule about the source."""
        incomplete = raw_frame(COMPLETE_ROW).drop(columns=["CGPA"])
        source = write_raw_csv(tmp_path / "raw.csv", incomplete)

        assert "CGPA" not in load_raw_data(source).columns


class TestValidateRawDataAccepts:
    """Valid source data: what the contract deliberately lets through."""

    def test_accepts_a_frame_that_meets_the_contract(self) -> None:
        valid = raw_frame(COMPLETE_ROW, OTHER_ROW)

        validated = validate_raw_data(valid)

        assert len(validated) == TWO_ROWS

    def test_accepts_duplicate_rows(self) -> None:
        """Duplicates are legitimate input: removing them is the pipeline's job."""
        validate_raw_data(raw_frame(COMPLETE_ROW, COMPLETE_ROW, OTHER_ROW))

    def test_accepts_missing_values_below_the_threshold(self) -> None:
        below_threshold = rows_with_one_null("GRE Score", ROWS_UNDER_NULL_THRESHOLD)

        validate_raw_data(below_threshold)

    def test_accepts_the_extreme_values_of_each_documented_range(self) -> None:
        extremes: dict[str, float | None] = {
            **COMPLETE_ROW,
            "GRE Score": 340.0,
            "TOEFL Score": 120.0,
            "University Rating": 5.0,
            "SOP": 5.0,
            LOR_COLUMN: 5.0,
            "CGPA": 10.0,
            TARGET_COLUMN: 1.0,
        }

        validate_raw_data(raw_frame(extremes))


class TestValidateRawDataRejects:
    """Invalid source data: one case per rule of the contract."""

    def test_rejects_a_missing_column(self) -> None:
        incomplete = raw_frame(COMPLETE_ROW).drop(columns=["CGPA"])

        with pytest.raises(FeatureValidationError, match="CGPA"):
            validate_raw_data(incomplete)

    def test_rejects_an_unexpected_column(self) -> None:
        with_extra = raw_frame(COMPLETE_ROW).assign(**{"Serial No.": 1.0})

        with pytest.raises(FeatureValidationError, match=r"Serial No\."):
            validate_raw_data(with_extra)

    def test_rejects_columns_out_of_order(self) -> None:
        reordered = raw_frame(COMPLETE_ROW)[[*reversed(list(COMPLETE_ROW))]]

        with pytest.raises(FeatureValidationError):
            validate_raw_data(reordered)

    def test_rejects_a_non_numeric_column(self) -> None:
        text_in_cgpa = raw_frame(COMPLETE_ROW).astype({"CGPA": "object"})
        text_in_cgpa.loc[0, "CGPA"] = "nueve"

        with pytest.raises(FeatureValidationError, match="CGPA"):
            validate_raw_data(text_in_cgpa)

    def test_rejects_a_score_above_its_documented_maximum(self) -> None:
        impossible = {**COMPLETE_ROW, "GRE Score": 400.0}

        with pytest.raises(FeatureValidationError, match="GRE Score"):
            validate_raw_data(raw_frame(impossible))

    def test_rejects_a_negative_score(self) -> None:
        impossible = {**COMPLETE_ROW, "TOEFL Score": -1.0}

        with pytest.raises(FeatureValidationError, match="TOEFL Score"):
            validate_raw_data(raw_frame(impossible))

    def test_rejects_a_probability_outside_the_unit_interval(self) -> None:
        impossible = {**COMPLETE_ROW, TARGET_COLUMN: 1.5}

        with pytest.raises(FeatureValidationError, match="Chance of Admit"):
            validate_raw_data(raw_frame(impossible))

    def test_rejects_an_invalid_university_rating(self) -> None:
        off_scale = {**COMPLETE_ROW, "University Rating": 7.0}

        with pytest.raises(FeatureValidationError, match="University Rating"):
            validate_raw_data(raw_frame(off_scale))

    def test_rejects_an_invalid_research_value(self) -> None:
        not_binary = {**COMPLETE_ROW, "Research": 2.0}

        with pytest.raises(FeatureValidationError, match="Research"):
            validate_raw_data(raw_frame(not_binary))

    def test_rejects_too_many_missing_values_in_a_column(self) -> None:
        above_threshold = rows_with_one_null("SOP", ROWS_OVER_NULL_THRESHOLD)

        with pytest.raises(FeatureValidationError, match="SOP"):
            validate_raw_data(above_threshold)

    def test_rejects_a_missing_label(self) -> None:
        """A record without its label is not data, it is a hole."""
        unlabelled: dict[str, float | None] = {**COMPLETE_ROW, TARGET_COLUMN: None}

        with pytest.raises(FeatureValidationError, match="Chance of Admit"):
            validate_raw_data(raw_frame(unlabelled))

    def test_rejects_an_empty_table(self) -> None:
        with pytest.raises(FeatureValidationError):
            validate_raw_data(raw_frame())


class TestIntegrityBetweenFields:
    """Coherence inside a single record."""

    def test_accepts_a_record_with_a_single_observed_predictor(self) -> None:
        barely_observed: dict[str, float | None] = {
            **dict.fromkeys(FEATURE_COLUMNS),
            "CGPA": 8.5,
            TARGET_COLUMN: 0.7,
        }
        # Padded with complete records so the sparse one does not trip the null ratio.
        padding = [COMPLETE_ROW] * (ROWS_UNDER_NULL_THRESHOLD - 1)

        validate_raw_data(raw_frame(barely_observed, *padding))

    def test_rejects_a_record_with_no_observed_predictor(self) -> None:
        """Such a record cannot be compared to any other, so it survives deduplication."""
        only_the_label: dict[str, float | None] = {
            **dict.fromkeys(FEATURE_COLUMNS),
            TARGET_COLUMN: 0.7,
        }
        padding = [COMPLETE_ROW] * (ROWS_UNDER_NULL_THRESHOLD - 1)

        with pytest.raises(FeatureValidationError, match="predictor"):
            validate_raw_data(raw_frame(only_the_label, *padding))

    def test_survives_a_frame_without_any_predictor_column(self) -> None:
        """The missing columns are the schema's business; this rule must not crash."""
        label_only = raw_frame(COMPLETE_ROW)[[TARGET_COLUMN]]

        with pytest.raises(FeatureValidationError, match="GRE Score"):
            validate_raw_data(label_only)


class TestScaleFormat:
    """Values must fall on the grid of the instrument that measured them."""

    def test_accepts_half_points_in_the_letter_scales(self) -> None:
        on_grid = {**COMPLETE_ROW, "SOP": 3.5, LOR_COLUMN: 2.5}

        validate_raw_data(raw_frame(on_grid))

    def test_rejects_a_value_off_the_half_point_grid(self) -> None:
        off_grid = {**COMPLETE_ROW, "SOP": 3.7}

        with pytest.raises(FeatureValidationError, match="SOP"):
            validate_raw_data(raw_frame(off_grid))

    def test_rejects_a_fractional_score(self) -> None:
        fractional = {**COMPLETE_ROW, "GRE Score": 337.5}

        with pytest.raises(FeatureValidationError, match="GRE Score"):
            validate_raw_data(raw_frame(fractional))


class TestIntegrityBetweenRecords:
    """Coherence across records of the same table."""

    def test_accepts_identical_records_that_agree_on_the_label(self) -> None:
        validate_raw_data(raw_frame(COMPLETE_ROW, COMPLETE_ROW))

    def test_rejects_identical_predictors_with_different_labels(self) -> None:
        """The same candidate profile cannot hold two different admission chances."""
        contradiction = {**COMPLETE_ROW, TARGET_COLUMN: 0.31}

        with pytest.raises(FeatureValidationError, match="contradictory"):
            validate_raw_data(raw_frame(COMPLETE_ROW, contradiction))


class TestValidationErrorMessage:
    """The error a person actually reads when the source is broken."""

    def test_reports_every_broken_rule_at_once(self) -> None:
        doubly_broken = {**COMPLETE_ROW, "GRE Score": 400.0, "University Rating": 7.0}

        with pytest.raises(FeatureValidationError) as failure:
            validate_raw_data(raw_frame(doubly_broken))

        assert "GRE Score" in str(failure.value)
        assert "University Rating" in str(failure.value)

    def test_does_not_report_a_verdict_as_if_it_were_an_offending_value(self) -> None:
        """Whole-table rules answer with a boolean; printing it reads as noise."""
        only_the_label: dict[str, float | None] = {
            **dict.fromkeys(FEATURE_COLUMNS),
            TARGET_COLUMN: 0.7,
        }
        padding = [COMPLETE_ROW] * (ROWS_UNDER_NULL_THRESHOLD - 1)

        with pytest.raises(FeatureValidationError) as failure:
            validate_raw_data(raw_frame(only_the_label, *padding))

        assert "predictor" in str(failure.value)
        assert "False" not in str(failure.value)

    def test_reports_the_offending_value(self) -> None:
        impossible = {**COMPLETE_ROW, "GRE Score": 400.0}

        with pytest.raises(FeatureValidationError) as failure:
            validate_raw_data(raw_frame(impossible))

        assert "400" in str(failure.value)


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

    def test_invalid_source_data_aborts_before_writing_anything(self, tmp_path: Path) -> None:
        impossible = {**COMPLETE_ROW, "GRE Score": 400.0}
        source = write_raw_csv(tmp_path / "raw.csv", raw_frame(impossible))
        destination = tmp_path / "features.parquet"

        with pytest.raises(FeatureValidationError, match="GRE Score"):
            run_pipeline(source, destination)

        assert not destination.exists()

    def test_a_previous_feature_table_survives_a_failed_run(self, tmp_path: Path) -> None:
        """A broken source must not destroy the last good feature table either."""
        destination = tmp_path / "features.parquet"
        run_pipeline(write_raw_csv(tmp_path / "good.csv", raw_frame(COMPLETE_ROW)), destination)
        impossible = {**COMPLETE_ROW, "CGPA": 42.0}
        broken = write_raw_csv(tmp_path / "broken.csv", raw_frame(impossible))

        with pytest.raises(FeatureValidationError):
            run_pipeline(broken, destination)

        assert len(pd.read_parquet(destination)) == ONE_ROW


class TestValidateFeatureTableContract:
    """The contract the feature layer offers to the training pipeline and the model.

    These properties are false in the raw layer by construction, so the entry gate
    cannot check them: they are the postcondition of the transformation.
    """

    def test_accepts_the_table_the_pipeline_produces(self) -> None:
        raw = raw_frame(COMPLETE_ROW, COMPLETE_ROW, OTHER_ROW)

        validated = validate_feature_table(build_features(raw), raw)

        assert len(validated) == TWO_ROWS

    def test_rejects_a_table_typed_as_the_raw_layer(self) -> None:
        raw = raw_frame(COMPLETE_ROW, OTHER_ROW)

        with pytest.raises(FeatureValidationError, match="GRE Score"):
            validate_feature_table(raw, raw)

    def test_rejects_duplicate_rows(self) -> None:
        raw = raw_frame(COMPLETE_ROW, COMPLETE_ROW)
        repeated = cast_dtypes(raw)

        with pytest.raises(FeatureValidationError):
            validate_feature_table(repeated, raw)

    def test_rejects_missing_values(self) -> None:
        incomplete = {**OTHER_ROW, "SOP": None}
        raw = raw_frame(COMPLETE_ROW, incomplete)

        with pytest.raises(FeatureValidationError, match="SOP"):
            validate_feature_table(build_features(raw), raw)

    def test_rejects_an_unexpected_column(self) -> None:
        raw = raw_frame(COMPLETE_ROW, OTHER_ROW)
        with_extra = build_features(raw).assign(**{"Serial No.": 1})

        with pytest.raises(FeatureValidationError, match=r"Serial No\."):
            validate_feature_table(with_extra, raw)


class TestIntegrityBetweenDatasets:
    """The feature table must be derivable from the source it claims to come from."""

    def test_rejects_more_records_than_the_source_holds(self) -> None:
        raw = raw_frame(COMPLETE_ROW)
        inflated = cast_dtypes(raw_frame(COMPLETE_ROW, OTHER_ROW))

        with pytest.raises(FeatureValidationError, match="more records"):
            validate_feature_table(inflated, raw)

    def test_rejects_a_record_absent_from_the_source(self) -> None:
        """Transforming may drop records; it may never invent one."""
        raw = raw_frame(COMPLETE_ROW, OTHER_ROW)
        invented = cast_dtypes(raw_frame({**COMPLETE_ROW, "CGPA": 7.11}))

        with pytest.raises(FeatureValidationError, match="not present in the raw"):
            validate_feature_table(invented, raw)

    def test_accepts_a_table_that_only_dropped_records(self) -> None:
        raw = raw_frame(COMPLETE_ROW, COMPLETE_ROW, OTHER_ROW)

        validate_feature_table(build_features(raw), raw)


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
