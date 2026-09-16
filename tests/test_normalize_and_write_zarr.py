from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "3_normalize_and_write_zarr.py"
SPEC = importlib.util.spec_from_file_location("normalize_and_write_zarr", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LatitudeConventionTests(unittest.TestCase):
    def test_target_latitude_preserves_descending_era5_order(self) -> None:
        latitude = MODULE.TARGET_LAT
        self.assertEqual(latitude.dtype, np.dtype("float32"))
        self.assertEqual(latitude.shape, (721,))
        self.assertEqual(float(latitude[0]), 90.0)
        self.assertEqual(float(latitude[-1]), -90.0)
        self.assertTrue(np.all(np.diff(latitude) == np.float32(-0.25)))

    def test_regridder_preserves_descending_output_alignment(self) -> None:
        source_latitude = MODULE.TARGET_LAT.copy()
        source_values = np.broadcast_to(
            source_latitude[:, None],
            (len(source_latitude), len(MODULE.TARGET_LON)),
        ).copy()
        field = xr.DataArray(
            source_values,
            dims=("lat", "lon"),
            coords={"lat": source_latitude, "lon": MODULE.TARGET_LON},
        )
        result = MODULE.Regridder().apply(field)
        np.testing.assert_array_equal(result[:, 0], MODULE.TARGET_LAT)
        np.testing.assert_array_equal(result[:, -1], MODULE.TARGET_LAT)

    def test_content_version_is_v2(self) -> None:
        self.assertEqual(MODULE.CONTENT_VERSION, "v2")

if __name__ == "__main__":
    unittest.main()
