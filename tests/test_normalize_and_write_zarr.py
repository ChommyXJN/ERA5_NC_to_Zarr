from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT = Path(__file__).resolve().parents[1] / "3_normalize_and_write_zarr.py"
SPEC = importlib.util.spec_from_file_location("normalize_and_write_zarr", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LatitudeConventionTests(unittest.TestCase):
    def test_target_latitude_is_strictly_increasing(self) -> None:
        latitude = MODULE.TARGET_LAT

        self.assertEqual(latitude.dtype, np.dtype("float32"))
        self.assertEqual(latitude.shape, (721,))
        self.assertEqual(float(latitude[0]), -90.0)
        self.assertEqual(float(latitude[-1]), 90.0)
        self.assertTrue(np.all(np.diff(latitude) == np.float32(0.25)))
        MODULE.validate_coordinate_grid(latitude, MODULE.TARGET_LON)

    def test_descending_latitude_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            MODULE.validate_coordinate_grid(
                MODULE.TARGET_LAT[::-1], MODULE.TARGET_LON
            )

    def test_regridder_reorders_descending_source_data(self) -> None:
        source_latitude = MODULE.TARGET_LAT[::-1].copy()
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

        self.assertEqual(result.shape, (721, 1440))
        np.testing.assert_array_equal(result[:, 0], MODULE.TARGET_LAT)
        np.testing.assert_array_equal(result[:, -1], MODULE.TARGET_LAT)

    def test_content_version_marks_new_latitude_convention(self) -> None:
        self.assertEqual(MODULE.CONTENT_VERSION, "v3")


if __name__ == "__main__":
    unittest.main()
