from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT = Path(__file__).resolve().parents[1] / "2_convert_units_single_day.py"
SPEC = importlib.util.spec_from_file_location("convert_units_single_day", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TotalPrecipitationConversionTests(unittest.TestCase):
    def test_metres_are_converted_to_millimetres_before_log1p(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tp.nc"
            xr.Dataset(
                {
                    "tp": xr.DataArray(
                        np.array([-0.001, 0.0, 0.001, 0.010], dtype="float32"),
                        dims=("sample",),
                        attrs={"units": "m"},
                    )
                }
            ).to_netcdf(source, engine="netcdf4")

            converted = MODULE.transformed_dataset(source, "tp")
            try:
                expected = np.log1p(
                    np.array([0.0, 0.0, 1.0, 10.0], dtype="float32")
                )
                np.testing.assert_allclose(converted["tp"].values, expected, rtol=1e-6)
                self.assertEqual(converted["tp"].attrs["units"], "1")
                self.assertEqual(converted["tp"].attrs["original_units"], "m")
                self.assertEqual(converted["tp"].attrs["intermediate_units"], "mm")
                self.assertEqual(
                    converted.attrs["processing_rule"],
                    "log1p(max(tp * 1000.0, 0.0))",
                )
            finally:
                converted.close()

    def test_rejects_tp_that_is_not_in_metres(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tp.nc"
            xr.Dataset(
                {
                    "tp": xr.DataArray(
                        np.array([1.0], dtype="float32"),
                        dims=("sample",),
                        attrs={"units": "mm"},
                    )
                }
            ).to_netcdf(source, engine="netcdf4")

            with self.assertRaisesRegex(ValueError, "expected raw tp units in metres"):
                MODULE.transformed_dataset(source, "tp")


if __name__ == "__main__":
    unittest.main()
