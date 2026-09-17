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

    def test_masks_are_thresholded_binary_and_complementary(self) -> None:
        source = np.array([[-0.1, 0.25, 0.5, 0.75, 1.1]], dtype="f4")
        land, sea = MODULE.derive_land_sea_masks(source)
        np.testing.assert_array_equal(
            land, np.array([[0, 0, 0, 1, 1]], dtype="u1")
        )
        self.assertEqual(land.dtype, np.dtype("u1"))
        np.testing.assert_array_equal(sea, np.uint8(1) - land)
        np.testing.assert_array_equal(land + sea, np.ones_like(land))

    def test_v3_paths_place_statistics_and_single_mask_array_at_root(self) -> None:
        self.assertIn("mean", MODULE.EXPECTED_CHILDREN)
        self.assertIn("std", MODULE.EXPECTED_CHILDREN)
        self.assertIn("mask", MODULE.EXPECTED_CHILDREN)
        self.assertIn("mask_channel", MODULE.EXPECTED_CHILDREN)
        self.assertFalse(any(path.startswith("auxiliary") for path in MODULE.EXPECTED_CHILDREN))

    def test_channel_info_is_complete_and_key_aligned(self) -> None:
        mean = np.arange(MODULE.CHANNEL_COUNT, dtype="f4")
        std = np.arange(1, MODULE.CHANNEL_COUNT + 1, dtype="f4")
        attributes = MODULE.channel_attributes(mean=mean, std=std)
        info = attributes["channel_info"]
        self.assertEqual(set(MODULE.DYNAMIC_CHANNELS), set(info))
        required = {
            "long_name", "source_name", "source_units", "units", "level_type",
            "variable_type", "preprocessing",
        }
        for channel in MODULE.DYNAMIC_CHANNELS:
            self.assertTrue(required.issubset(info[channel]), channel)
        self.assertEqual(info["q500"]["level"], 500)
        self.assertEqual(info["q500"]["level_units"], "hPa")
        self.assertNotIn("level", info["msl"])
        t2m_index = MODULE.DYNAMIC_CHANNELS.index("t2m")
        self.assertEqual(info["t2m"]["scale_factor"], float(std[t2m_index]))
        self.assertEqual(info["t2m"]["add_offset"], float(mean[t2m_index]))

    def test_data_and_time_metadata_follow_schema(self) -> None:
        self.assertEqual(MODULE.DATA_ATTRIBUTES["data_representation"], "normalized")
        self.assertEqual(MODULE.DATA_ATTRIBUTES["normalization_method"], "channel_dependent")
        self.assertIn("_FillValue", MODULE.DATA_ATTRIBUTES)
        self.assertEqual(MODULE.TIME_ATTRIBUTES["timezone"], "UTC")
        self.assertNotIn("channel_count", MODULE.root_attributes("id", "revision"))

    def test_denormalization_only_reverses_channel_zscore(self) -> None:
        values = np.array([0.0, 1.0], dtype="f4")
        np.testing.assert_array_equal(
            MODULE.denormalize_values(values, "t2m", np.float32(10), np.float32(2)),
            np.array([10.0, 12.0], dtype="f4"),
        )
        np.testing.assert_array_equal(
            MODULE.denormalize_values(values, "tp", np.float32(0), np.float32(1)),
            values,
        )

    def test_complete_day_validation_accepts_nanosecond_times(self) -> None:
        times = np.array(
            [
                "2025-01-01T00:00:00.000000000",
                "2025-01-01T06:00:00.000000000",
                "2025-01-01T12:00:00.000000000",
                "2025-01-01T18:00:00.000000000",
            ],
            dtype="datetime64[ns]",
        )
        MODULE.validate_time_coverage(times, allow_partial=False)

if __name__ == "__main__":
    unittest.main()
