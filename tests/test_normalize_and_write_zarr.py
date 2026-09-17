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

    def test_content_version_is_v3(self) -> None:
        self.assertEqual(MODULE.CONTENT_VERSION, "v3")
        self.assertEqual(MODULE.SCHEMA_VERSION, "2.0")

    def test_fractional_masks_are_clipped_and_complementary(self) -> None:
        source = np.array([[-0.1, 0.25, 0.5, 0.75, 1.1]], dtype="f4")
        land, sea = MODULE.derive_land_sea_masks(source)
        np.testing.assert_array_equal(
            land, np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype="f4")
        )
        np.testing.assert_array_equal(sea, np.float32(1.0) - land)
        np.testing.assert_array_equal(land + sea, np.ones_like(land))

    def test_v3_paths_place_statistics_at_root_and_masks_in_group(self) -> None:
        self.assertIn("mean", MODULE.EXPECTED_CHILDREN)
        self.assertIn("std", MODULE.EXPECTED_CHILDREN)
        self.assertIn("mask/mask_channel", MODULE.EXPECTED_CHILDREN)
        self.assertIn("mask/land_mask", MODULE.EXPECTED_CHILDREN)
        self.assertIn("mask/sea_mask", MODULE.EXPECTED_CHILDREN)
        self.assertFalse(any(path.startswith("auxiliary") for path in MODULE.EXPECTED_CHILDREN))

if __name__ == "__main__":
    unittest.main()
