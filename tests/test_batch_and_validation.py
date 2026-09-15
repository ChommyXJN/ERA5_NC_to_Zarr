from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BATCH = load("batch_convert", "5_batch_convert.py")
VALIDATE = load("validate_zarr", "4_validate_zarr.py")
EXTRACT = load("extract_single_day", "1_extract_single_day.py")
PIPELINE = VALIDATE.load_pipeline()


class ExtractionInputModeTests(unittest.TestCase):
    def test_real_monthly_fixtures_extract_the_full_daily_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "day"
            EXTRACT.run(
                Namespace(
                    source=DATA / "monthly",
                    date=date(2025, 1, 1),
                    input_mode="monthly",
                    output=output_root,
                    overwrite=False,
                    skip_static=False,
                )
            )
            self.assertTrue(
                BATCH.daily_tree_complete(
                    output_root, date(2025, 1, 1), PIPELINE, converted=False
                )
            )
            expected_files = sorted((DATA / "raw").rglob("*.nc"))
            self.assertEqual(len(expected_files), 40)
            for expected_path in expected_files:
                actual_path = output_root / expected_path.relative_to(DATA / "raw")
                self.assertTrue(actual_path.is_file(), actual_path)
                with xr.open_dataset(
                    actual_path, engine="netcdf4"
                ) as actual, xr.open_dataset(
                    expected_path, engine="netcdf4"
                ) as expected:
                    self.assertEqual(set(actual.data_vars), set(expected.data_vars))
                    for name in expected.variables:
                        actual_values = actual[name].values
                        expected_values = expected[name].values
                        message = str(expected_path.relative_to(DATA / "raw"))
                        if np.issubdtype(expected_values.dtype, np.number):
                            np.testing.assert_allclose(
                                actual_values,
                                expected_values,
                                rtol=0,
                                atol=0,
                                equal_nan=True,
                                err_msg=message,
                            )
                        else:
                            np.testing.assert_array_equal(
                                actual_values, expected_values, err_msg=message
                            )

    def test_static_monthly_fixture_is_reused_for_a_later_month(self) -> None:
        relative = Path("static") / "lsm" / "2026"
        source = EXTRACT.source_file(
            DATA / "monthly", relative, date(2026, 7, 31), "monthly"
        )
        self.assertEqual(source.name, "lsm_20251.nc")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "2026.07.31.nc"
            EXTRACT.extract_file(
                source,
                output,
                date(2026, 7, 31),
                overwrite=False,
                static=True,
            )
            with xr.open_dataset(output, engine="netcdf4") as actual, xr.open_dataset(
                source, engine="netcdf4"
            ) as original:
                np.testing.assert_allclose(
                    actual["lsm"].isel(valid_time=0),
                    original["lsm"].isel(valid_time=0),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )

    def test_auto_prefers_exact_daily_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sfc" / "tp" / "2025"
            source.mkdir(parents=True)
            daily = source / "2025.01.01.nc"
            monthly = source / "tp_20251.nc"
            daily.touch()
            monthly.touch()
            relative = Path("sfc") / "tp" / "2025"
            self.assertEqual(
                EXTRACT.source_file(root, relative, date(2025, 1, 1), "auto"),
                daily,
            )
            self.assertEqual(
                EXTRACT.source_file(root, relative, date(2025, 1, 1), "monthly"),
                monthly,
            )

    def test_daily_mode_rejects_monthly_only_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sfc" / "tp" / "2025"
            source.mkdir(parents=True)
            (source / "tp_20251.nc").touch()
            with self.assertRaises(FileNotFoundError):
                EXTRACT.source_file(
                    root,
                    Path("sfc") / "tp" / "2025",
                    date(2025, 1, 1),
                    "daily",
                )


class BatchDateTests(unittest.TestCase):
    def test_real_converted_fixture_covers_all_116_channels(self) -> None:
        days, files, times = PIPELINE.discover(
            DATA / "converted", date(2025, 1, 1)
        )
        self.assertEqual(days, [date(2025, 1, 1)])
        self.assertEqual(len(times), 4)
        self.assertEqual(len(PIPELINE.DYNAMIC_CHANNELS), 116)
        self.assertTrue(
            BATCH.daily_tree_complete(
                DATA / "raw", date(2025, 1, 1), PIPELINE, converted=False
            )
        )
        self.assertTrue(
            BATCH.daily_tree_complete(
                DATA / "converted", date(2025, 1, 1), PIPELINE, converted=True
            )
        )
        self.assertEqual(set(files), set(PIPELINE.NORMALIZED_INPUT_VARIABLES))

    def test_date_range_is_inclusive(self) -> None:
        self.assertEqual(
            BATCH.dates_inclusive(date(2025, 1, 30), date(2025, 2, 2)),
            [
                date(2025, 1, 30),
                date(2025, 1, 31),
                date(2025, 2, 1),
                date(2025, 2, 2),
            ],
        )

    def test_reversed_date_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "--end"):
            BATCH.dates_inclusive(date(2025, 2, 1), date(2025, 1, 31))

    def test_range_label_uses_months_for_complete_calendar_range(self) -> None:
        self.assertEqual(
            BATCH.range_label(date(2025, 1, 1), date(2026, 7, 31)),
            "202501-202607",
        )

    def test_range_label_uses_days_for_partial_calendar_range(self) -> None:
        self.assertEqual(
            BATCH.range_label(date(2025, 1, 2), date(2025, 1, 31)),
            "20250102-20250131",
        )

    def test_date_partition_is_selected_for_daily_or_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition = root / "2025.01.01"
            partition.mkdir()
            self.assertEqual(
                BATCH.source_for_day(root, date(2025, 1, 1), "auto"), partition
            )
            self.assertEqual(
                BATCH.source_for_day(root, date(2025, 1, 1), "daily"), partition
            )
            self.assertEqual(
                BATCH.source_for_day(root, date(2025, 1, 1), "monthly"), root
            )

    def test_completion_marker_requires_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "done.json"
            expected = {"stage": "extract", "date": "2025-01-01"}
            BATCH.write_marker(marker, expected)
            self.assertTrue(BATCH.marker_matches(marker, expected))
            self.assertFalse(
                BATCH.marker_matches(
                    marker, {"stage": "extract", "date": "2025-01-02"}
                )
            )


class ValidationSamplingTests(unittest.TestCase):
    def test_sampling_includes_first_and_last(self) -> None:
        self.assertEqual(VALIDATE.sample_indices(10, 3), [0, 4, 9])

    def test_sampling_deduplicates_short_ranges(self) -> None:
        self.assertEqual(VALIDATE.sample_indices(2, 10), [0, 1])

    def test_negative_sample_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample-count"):
            VALIDATE.sample_indices(10, -1)

    def test_raw_tp_file_supports_date_partitioned_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "2025.01.01" / "sfc" / "tp" / "2025"
            target.mkdir(parents=True)
            expected = target / "2025.01.01.nc"
            expected.touch()
            self.assertEqual(
                VALIDATE.raw_tp_file(root, np.datetime64("2025-01-01T06")),
                expected,
            )

    def test_metadata_error_identifies_tp_channel(self) -> None:
        pipeline = VALIDATE.load_pipeline()
        attributes = pipeline.root_attributes(
            "era5.20250101.c116.p25.h6.v2",
            "20260915",
            radiation_seconds=pipeline.DEFAULT_RADIATION_SECONDS,
        )
        attributes["channel_metadata"] = dict(attributes["channel_metadata"])
        attributes["channel_metadata"]["tp"] = {
            "variable": "tp",
            "units": "1",
            "preprocess": [{"operation": "log1p"}],
        }

        class FakeGroup:
            attrs = attributes

        with self.assertRaisesRegex(ValueError, "differs for=tp"):
            VALIDATE.validate_root_metadata(FakeGroup(), pipeline)


if __name__ == "__main__":
    unittest.main()
