"""Optional integration checks against the user's full E: drive samples.

Run explicitly with ``ERA5_RUN_FULL_INTEGRATION=1``. These tests are skipped in
portable CI because the external files are not part of the repository.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


ROOT = Path(__file__).resolve().parents[1]
RAW_DAY = Path(os.environ.get("ERA5_RAW_DAY", r"E:\era5_2025.01.01_nc"))
CONVERTED_DAY = Path(
    os.environ.get(
        "ERA5_CONVERTED_DAY", r"E:\era5_2025.01.01_unit_converted_nc"
    )
)
RAW_ARCHIVE = Path(
    os.environ.get("ERA5_RAW_ARCHIVE", r"E:\era5_2025.01-2026.07_nc")
)
TEST_ZARR = Path(
    os.environ.get(
        "ERA5_TEST_ZARR",
        r"E:\era5_testsample\era5.20250101.c116.p25.h6.v2.zarr",
    )
)
RUN_FULL = os.environ.get("ERA5_RUN_FULL_INTEGRATION") == "1"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONVERT = load("integration_convert", "2_convert_units_single_day.py")
VALIDATE = load("integration_validate", "4_validate_zarr.py")


@unittest.skipUnless(
    RUN_FULL,
    "set ERA5_RUN_FULL_INTEGRATION=1 to use the external full-size ERA5 samples",
)
class FullRealDataIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for path in (RAW_DAY, CONVERTED_DAY, RAW_ARCHIVE, TEST_ZARR):
            if not path.exists():
                raise unittest.SkipTest(f"external integration path is missing: {path}")

    def test_full_daily_tp_conversion_matches_real_converted_file(self) -> None:
        raw_path = RAW_DAY / "sfc" / "tp" / "2025" / "2025.01.01.nc"
        expected_path = (
            CONVERTED_DAY
            / "sfc"
            / "tp"
            / "2025"
            / "2025.01.01.unit_converted.nc"
        )
        actual = CONVERT.transformed_dataset(raw_path, "tp")
        try:
            with xr.open_dataset(expected_path, engine="netcdf4") as expected:
                np.testing.assert_allclose(
                    actual["tp"].values,
                    expected["tp"].values,
                    rtol=1e-6,
                    atol=0,
                    equal_nan=True,
                )
        finally:
            actual.close()

    def test_zarr_tp_samples_recompute_from_raw_archive(self) -> None:
        pipeline = VALIDATE.load_pipeline()
        group = zarr.open_group(str(TEST_ZARR), mode="r", use_consolidated=False)
        times = VALIDATE.decoded_times(group)
        channels = [str(value) for value in group["channel"][:].tolist()]
        VALIDATE.validate_tp_against_raw(
            group,
            channels,
            times,
            [0, len(times) - 1],
            RAW_ARCHIVE,
            pipeline,
        )

    def test_zarr_metadata_matches_current_strict_schema(self) -> None:
        pipeline = VALIDATE.load_pipeline()
        group = zarr.open_group(str(TEST_ZARR), mode="r", use_consolidated=False)
        VALIDATE.validate_root_metadata(group, pipeline)


if __name__ == "__main__":
    unittest.main()
