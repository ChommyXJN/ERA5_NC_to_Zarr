from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    def test_auto_prefers_exact_daily_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sfc" / "t2m" / "2025"
            source.mkdir(parents=True)
            daily = source / "2025.01.01.nc"
            monthly = source / "t2m_20251.nc"
            daily.touch()
            monthly.touch()
            relative = Path("sfc") / "t2m" / "2025"
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
            source = root / "sfc" / "t2m" / "2025"
            source.mkdir(parents=True)
            (source / "t2m_20251.nc").touch()
            with self.assertRaises(FileNotFoundError):
                EXTRACT.source_file(
                    root,
                    Path("sfc") / "t2m" / "2025",
                    date(2025, 1, 1),
                    "daily",
                )


class BatchDateTests(unittest.TestCase):
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

    def test_metadata_error_identifies_changed_channel(self) -> None:
        pipeline = VALIDATE.load_pipeline()
        attributes = pipeline.root_attributes(
            "era5.20250101.c116.p25.h6.v2",
            "20260915",
            radiation_seconds=pipeline.DEFAULT_RADIATION_SECONDS,
        )
        attributes["channel_metadata"] = dict(attributes["channel_metadata"])
        attributes["channel_metadata"]["t2m"] = {
            "variable": "t2m",
            "units": "invalid",
            "preprocess": [],
        }

        class FakeGroup:
            attrs = attributes

        with self.assertRaisesRegex(ValueError, "differs for=t2m"):
            VALIDATE.validate_root_metadata(FakeGroup(), pipeline)


if __name__ == "__main__":
    unittest.main()
