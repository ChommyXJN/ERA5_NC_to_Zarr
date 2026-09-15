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
DATA = Path(__file__).resolve().parent / "data"


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


class RealFullChannelReferenceTests(unittest.TestCase):
    def test_real_reference_files_have_all_channels_and_expected_normalization(self) -> None:
        reference = DATA / "reference"
        with xr.open_dataset(
            reference / "era5.20250101.c116.p25.h6.raw_truth.nc",
            engine="netcdf4",
        ) as raw, xr.open_dataset(
            reference / "era5.20250101.c116.p25.h6.unit_converted.nc",
            engine="netcdf4",
        ) as converted, xr.open_dataset(
            reference / "era5.20250101.c116.p25.h6.normalized.nc",
            engine="netcdf4",
        ) as normalized, xr.open_dataset(
            ROOT / "mean.nc", engine="netcdf4"
        ) as mean_dataset, xr.open_dataset(
            ROOT / "std.nc", engine="netcdf4"
        ) as std_dataset:
            channels = [str(value) for value in converted.channel.values.tolist()]
            self.assertEqual(channels, list(MODULE.DYNAMIC_CHANNELS))
            self.assertEqual(len(channels), 116)
            self.assertEqual(
                [str(value) for value in raw.channel.values.tolist()], channels
            )
            self.assertEqual(
                [str(value) for value in normalized.channel.values.tolist()], channels
            )

            converted_values = np.asarray(converted["data"].values, dtype="f4")
            normalized_values = np.asarray(normalized["data"].values, dtype="f4")
            mean = np.asarray(mean_dataset["mean"].values, dtype="f4")
            std = np.asarray(std_dataset["std"].values, dtype="f4")
            expected = (converted_values - mean[None, :, None, None]) / std[
                None, :, None, None
            ]
            tp_index = channels.index("tp")
            expected[:, tp_index] = converted_values[:, tp_index]
            np.testing.assert_allclose(
                normalized_values,
                expected,
                rtol=2e-6,
                atol=2e-6,
                equal_nan=True,
            )

            raw_tp = np.asarray(raw["data"][:, tp_index], dtype="f4")
            expected_tp = np.log1p(
                np.maximum(raw_tp * np.float32(1000.0), np.float32(0.0))
            )
            np.testing.assert_allclose(
                converted_values[:, tp_index],
                expected_tp,
                rtol=1e-6,
                atol=0,
                equal_nan=True,
            )


if __name__ == "__main__":
    unittest.main()
