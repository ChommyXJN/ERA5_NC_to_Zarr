from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
SCRIPT = ROOT / "2_convert_units_single_day.py"
SPEC = importlib.util.spec_from_file_location("convert_units_single_day", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def raw_path(group: str, name: str) -> Path:
    return DATA / "raw" / group / name / "2025" / "2025.01.01.nc"


def converted_path(group: str, name: str) -> Path:
    return (
        DATA
        / "converted"
        / group
        / name
        / "2025"
        / "2025.01.01.unit_converted.nc"
    )


class RealUnitConversionFixtureTests(unittest.TestCase):
    def assert_matches_real_converted_fixture(
        self, group: str, logical_name: str
    ) -> xr.Dataset:
        actual = MODULE.transformed_dataset(raw_path(group, logical_name), logical_name)
        expected = xr.open_dataset(converted_path(group, logical_name), engine="netcdf4")
        self.addCleanup(actual.close)
        self.addCleanup(expected.close)
        actual_name = MODULE.find_data_variable(actual, logical_name)
        expected_name = MODULE.find_data_variable(expected, logical_name)
        np.testing.assert_allclose(
            actual[actual_name].values,
            expected[expected_name].values,
            rtol=1e-6,
            atol=0,
            equal_nan=True,
        )
        return actual

    def test_real_tp_metres_are_converted_before_log1p(self) -> None:
        converted = self.assert_matches_real_converted_fixture("sfc", "tp")
        raw = xr.open_dataset(raw_path("sfc", "tp"), engine="netcdf4")
        self.addCleanup(raw.close)
        expected = np.log1p(
            np.maximum(raw["tp"].values.astype("f4") * np.float32(1000), 0)
        )
        np.testing.assert_allclose(converted["tp"].values, expected, rtol=1e-6)
        self.assertEqual(converted["tp"].attrs["units"], "1")
        self.assertEqual(converted["tp"].attrs["original_units"], "m")
        self.assertEqual(converted["tp"].attrs["intermediate_units"], "mm")

    def test_real_q_is_converted_from_kgkg_to_gkg(self) -> None:
        converted = self.assert_matches_real_converted_fixture("pl", "q")
        self.assertEqual(converted["q"].attrs["units"], "g/kg")
        self.assertEqual(converted["q"].attrs["unit_conversion"], "value * 1000.0")

    def test_real_ssrd_is_divided_by_21600_seconds(self) -> None:
        converted = self.assert_matches_real_converted_fixture("cldrad", "ssrd")
        self.assertEqual(converted["ssrd"].attrs["units"], "W m-2")
        self.assertEqual(
            converted["ssrd"].attrs["accumulation_window_seconds"], 21600.0
        )

    def test_real_wind_components_reproduce_ws10m(self) -> None:
        actual = MODULE.derive_wind_speed(
            raw_path("sfc", "u10m"),
            raw_path("sfc", "v10m"),
            "u10m",
            "v10m",
            "ws10m",
            "10 metre",
        )
        expected = xr.open_dataset(converted_path("sfc", "ws10m"), engine="netcdf4")
        self.addCleanup(actual.close)
        self.addCleanup(expected.close)
        np.testing.assert_allclose(
            actual["ws10m"].values,
            expected["ws10m"].values,
            rtol=1e-6,
            atol=0,
            equal_nan=True,
        )

    def test_real_tp_fixture_with_wrong_units_is_rejected(self) -> None:
        with xr.open_dataset(raw_path("sfc", "tp"), engine="netcdf4") as opened:
            altered = opened.load()
        altered["tp"].attrs = dict(altered["tp"].attrs)
        altered["tp"].attrs["units"] = "mm"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tp_wrong_units.nc"
            altered.to_netcdf(source, engine="netcdf4")
            with self.assertRaisesRegex(ValueError, "expected raw tp units in metres"):
                MODULE.transformed_dataset(source, "tp")
        altered.close()


if __name__ == "__main__":
    unittest.main()
