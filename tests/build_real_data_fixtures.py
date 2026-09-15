#!/usr/bin/env python3
"""Build small, repository-safe test fixtures from the real ERA5 samples.

This is a maintainer utility, not a test.  It reads the user-provided full-size
sample trees and writes tiny coordinate subsets below ``tests/data``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_RAW_DAY = Path(r"E:\era5_2025.01.01_nc")
DEFAULT_CONVERTED_DAY = Path(r"E:\era5_2025.01.01_unit_converted_nc")
DEFAULT_MONTHLY = Path(r"E:\era5_2025.01-2026.07_nc")
DEFAULT_TESTSAMPLE = Path(r"E:\era5_testsample")


def load_extract_module():
    path = PROJECT_ROOT / "1_extract_single_day.py"
    spec = importlib.util.spec_from_file_location("era5_extract", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXTRACT = load_extract_module()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-day", type=Path, default=DEFAULT_RAW_DAY)
    parser.add_argument("--converted-day", type=Path, default=DEFAULT_CONVERTED_DAY)
    parser.add_argument("--monthly", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument("--testsample", type=Path, default=DEFAULT_TESTSAMPLE)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def subset(source: Path, time_count: int = 4, levels: list[int] | None = None) -> xr.Dataset:
    with EXTRACT.actual_netcdf(source) as actual:
        with xr.open_dataset(actual, engine="netcdf4", cache=False) as opened:
            indexers: dict[str, object] = {}
            selected_spatial: dict[str, list[int]] = {}
            for name in ("valid_time", "time"):
                if name in opened.dims:
                    indexers[name] = slice(0, time_count)
                    break
            for name in ("latitude", "lat"):
                if name in opened.dims:
                    size = opened.sizes[name]
                    selected_spatial[name] = sorted({0, size // 2, size - 1})
                    indexers[name] = selected_spatial[name]
                    break
            for name in ("longitude", "lon"):
                if name in opened.dims:
                    size = opened.sizes[name]
                    selected_spatial[name] = sorted({0, 1, size // 2, size - 1})
                    indexers[name] = selected_spatial[name]
                    break
            selected = opened.isel(indexers)
            if levels is not None:
                level_name = next(
                    name for name in ("pressure_level", "level") if name in selected.coords
                )
                selected = selected.sel({level_name: levels})
            result = selected.load()
            result.attrs = dict(opened.attrs)
    result.attrs.update(
        {
            "test_fixture": "cropped from real ERA5 sample data",
            "test_fixture_spatial_indices": ";".join(
                f"{name}={values}" for name, values in selected_spatial.items()
            ),
        }
    )
    return result


def write_fixture(
    source: Path,
    destination: Path,
    overwrite: bool,
    time_count: int = 4,
    levels: list[int] | None = None,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"fixture exists; use --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    dataset = subset(source, time_count=time_count, levels=levels)
    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            **({"dtype": "float32"} if np.issubdtype(value.dtype, np.floating) else {}),
        }
        for name, value in dataset.data_vars.items()
    }
    try:
        dataset.to_netcdf(
            temporary,
            engine="netcdf4",
            format="NETCDF4",
            encoding=encoding,
        )
        os.replace(temporary, destination)
    finally:
        dataset.close()
        if temporary.exists():
            temporary.unlink()
    print(f"[fixture] {destination.relative_to(destination.parents[4])}")


def run(args: argparse.Namespace) -> None:
    raw = args.raw_day.resolve()
    converted = args.converted_day.resolve()
    monthly = args.monthly.resolve()
    testsample = args.testsample.resolve()
    output = args.output.resolve()
    specifications: list[tuple[Path, Path, int, list[int] | None]] = []
    for daily_source in sorted(raw.rglob("*.nc")):
        relative = daily_source.relative_to(raw)
        monthly_directory = monthly / relative.parent
        candidates = sorted(monthly_directory.glob("*_20251.nc"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected one January 2025 source in {monthly_directory}, "
                f"found {len(candidates)}"
            )
        source = candidates[0]
        specifications.append(
            (source, output / "monthly" / relative.parent / source.name, 8, None)
        )
    for stage, tree in (("raw", raw), ("converted", converted)):
        for source in sorted(tree.rglob("*.nc")):
            specifications.append(
                (source, output / stage / source.relative_to(tree), 4, None)
            )
    for stage in ("raw_truth", "unit_converted", "normalized"):
        filename = f"era5.20250101.c116.p25.h6.{stage}.nc"
        specifications.append(
            (
                testsample / filename,
                output / "reference" / filename,
                4,
                None,
            )
        )
    for source, destination, time_count, levels in specifications:
        write_fixture(
            source,
            destination,
            args.overwrite,
            time_count=time_count,
            levels=levels,
        )
    print(f"[DONE] real-data fixtures written below {output}")


if __name__ == "__main__":
    run(parse_args())
