"""Integration checks against one externally configured ERA5 data flow.

Copy ``integration_paths.example.json`` to ``integration_paths.json``, update
the paths, then run this module explicitly. No ERA5 data is stored in Git.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from argparse import Namespace
from datetime import date
from pathlib import Path

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("integration_paths.json")
CONFIG_PATH = Path(os.environ.get("ERA5_TEST_CONFIG", DEFAULT_CONFIG))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXTRACT = load("integration_extract", "1_extract_single_day.py")
BATCH = load("integration_batch", "5_batch_convert.py")
VALIDATE = load("integration_validate", "4_validate_zarr.py")
PIPELINE = VALIDATE.load_pipeline()


class ExternalDataFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not CONFIG_PATH.is_file():
            raise unittest.SkipTest(
                "copy tests/integration_paths.example.json to "
                "tests/integration_paths.json and set the external data paths"
            )
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
        required = ("source", "extracted_day", "unit_converted_day", "zarr", "date")
        missing = [name for name in required if not config.get(name)]
        if missing:
            raise ValueError(f"{CONFIG_PATH}: missing values: {', '.join(missing)}")
        cls.day = date.fromisoformat(config["date"])
        cls.paths = {
            name: Path(config[name]).expanduser().resolve()
            for name in required[:-1]
        }
        absent = [
            f"{name}={path}" for name, path in cls.paths.items() if not path.exists()
        ]
        if absent:
            raise FileNotFoundError(
                "configured integration paths do not exist: " + "; ".join(absent)
            )

    def test_source_contains_every_input_for_the_configured_day(self) -> None:
        source = BATCH.source_for_day(self.paths["source"], self.day, "auto")
        variables = tuple(PIPELINE.DIRECT_VARIABLES) + tuple(PIPELINE.STATIC)
        for variable in variables:
            relative = (
                Path(PIPELINE.GROUPS[variable])
                / PIPELINE.SOURCE_DIRECTORIES.get(variable, variable)
                / str(self.day.year)
            )
            with self.subTest(variable=variable):
                self.assertTrue(
                    EXTRACT.source_file(source, relative, self.day, "auto").is_file()
                )

    def test_daily_stages_cover_the_same_complete_day(self) -> None:
        extracted = self.paths["extracted_day"]
        converted = self.paths["unit_converted_day"]
        self.assertTrue(
            BATCH.daily_tree_complete(
                extracted, self.day, PIPELINE, converted=False
            )
        )
        self.assertTrue(
            BATCH.daily_tree_complete(
                converted, self.day, PIPELINE, converted=True
            )
        )
        days, files, times = PIPELINE.discover(converted, self.day)
        self.assertEqual(days, [self.day])
        self.assertEqual(set(files), set(PIPELINE.NORMALIZED_INPUT_VARIABLES))
        expected_times = np.array(
            [
                np.datetime64(self.day) + np.timedelta64(hour, "h")
                for hour in (0, 6, 12, 18)
            ]
        )
        np.testing.assert_array_equal(times.astype("datetime64[h]"), expected_times)

    def test_zarr_contains_and_validates_the_same_day(self) -> None:
        path = self.paths["zarr"]
        VALIDATE.run(
            Namespace(
                zarr=path,
                sample_count=3,
                channels=list(VALIDATE.DEFAULT_CHANNELS),
                full_scan=False,
                allow_partial=False,
            )
        )
        group = zarr.open_group(str(path), mode="r", use_consolidated=False)
        times = VALIDATE.decoded_times(group).astype("datetime64[h]")
        selected = times[times.astype("datetime64[D]") == np.datetime64(self.day)]
        expected_times = np.array(
            [
                np.datetime64(self.day) + np.timedelta64(hour, "h")
                for hour in (0, 6, 12, 18)
            ]
        )
        np.testing.assert_array_equal(selected, expected_times)
        channels = [str(value) for value in group["channel"][:].tolist()]
        self.assertEqual(channels, list(PIPELINE.DYNAMIC_CHANNELS))


if __name__ == "__main__":
    unittest.main()
