#!/usr/bin/env python3
"""Apply only the agreed unit/preprocessing rules to a daily ERA5 NC tree.

Rules:
* q at every pressure level: kg/kg * 1000 -> g/kg
* ssr/ssrd/fdir/ttr: J/m2 / 21600 s -> W/m2
* tp: m * 1000 -> mm, then log1p(max(tp_mm, 0)), with no normalization
* ws10m: derived as hypot(u10m, v10m)
* ws100m: derived as hypot(u100m, v100m)

All unlisted files are copied without changing their data values.  No
normalization, regridding, dtype reduction, or other physical conversion is
performed.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr


RADIATION = frozenset({"ssr", "ssrd", "fdir", "ttr"})
RADIATION_SECONDS = 21600.0
TP_METRES_TO_MILLIMETRES = 1000.0


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def default_source(day: date) -> Path:
    return Path(f"E:\\era5_{day:%Y.%m.%d}_nc")


def default_output(day: date) -> Path:
    return Path(f"E:\\era5_{day:%Y.%m.%d}_unit_converted_nc")


def find_data_variable(dataset: xr.Dataset, logical_name: str) -> str:
    aliases = {
        "u10m": ("u10m", "u10"),
        "v10m": ("v10m", "v10"),
        "u100m": ("u100m", "u100"),
        "v100m": ("v100m", "v100"),
    }
    candidates = aliases.get(logical_name, (logical_name,))
    found = [name for name in candidates if name in dataset.data_vars]
    if len(found) != 1:
        raise ValueError(
            f"expected one variable for {logical_name}, found {found or list(dataset.data_vars)}"
        )
    return found[0]


def transformed_dataset(source: Path, logical_name: str) -> xr.Dataset:
    with xr.open_dataset(source, engine="netcdf4") as opened:
        variable_name = find_data_variable(opened, logical_name)
        result = opened.load()

    variable = result[variable_name]
    original_units = str(variable.attrs.get("units", "unknown"))
    values = variable.astype("float32")

    if logical_name == "q":
        values = values * np.float32(1000.0)
        values.attrs = dict(variable.attrs)
        values.attrs.update(
            {
                "units": "g/kg",
                "original_units": original_units,
                "unit_conversion": "value * 1000.0",
            }
        )
    elif logical_name in RADIATION:
        values = values / np.float32(RADIATION_SECONDS)
        values.attrs = dict(variable.attrs)
        values.attrs.update(
            {
                "units": "W m-2",
                "original_units": original_units,
                "unit_conversion": "value / 21600.0",
                "accumulation_window_seconds": RADIATION_SECONDS,
            }
        )
    elif logical_name == "tp":
        # ERA5 total precipitation is stored as metres of water equivalent.
        # Convert to millimetres before applying the nonlinear transform.
        if original_units.strip().lower() not in {
            "m", "metre", "metres", "meter", "meters"
        }:
            raise ValueError(
                f"{source}: expected raw tp units in metres, got {original_units!r}"
            )
        values = values * np.float32(TP_METRES_TO_MILLIMETRES)
        values = np.log1p(np.maximum(values, np.float32(0.0)))
        values.attrs = dict(variable.attrs)
        values.attrs.update(
            {
                "units": "1",
                "original_units": original_units,
                "intermediate_units": "mm",
                "unit_conversion": "value * 1000.0 (m to mm)",
                "transformation": "log1p(max(tp * 1000.0, 0.0))",
                "normalization_applied": "false",
            }
        )
    else:
        raise ValueError(f"no transformation configured for {logical_name}")

    result[variable_name] = values
    result.attrs = dict(result.attrs)
    result.attrs.update(
        {
            "processing_stage": "unit_conversion_only",
            "processing_rule": values.attrs.get(
                "transformation", values.attrs.get("unit_conversion")
            ),
        }
    )
    return result


def float_encoding(dataset: xr.Dataset, transformed_name: str) -> dict[str, dict]:
    encoding: dict[str, dict] = {}
    for name in dataset.data_vars:
        item: dict = {"zlib": True, "complevel": 4, "shuffle": True}
        if name == transformed_name:
            item["dtype"] = "float32"
        encoding[name] = item
    return encoding


def write_dataset(dataset: xr.Dataset, destination: Path, transformed_name: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        dataset.to_netcdf(
            temporary,
            engine="netcdf4",
            format="NETCDF4",
            encoding=float_encoding(dataset, transformed_name),
        )
        os.replace(temporary, destination)
    finally:
        dataset.close()
        if temporary.exists():
            temporary.unlink()


def derive_wind_speed(
    u_path: Path,
    v_path: Path,
    u_logical_name: str,
    v_logical_name: str,
    output_name: str,
    height_description: str,
) -> xr.Dataset:
    with xr.open_dataset(u_path, engine="netcdf4") as u_opened:
        u_name = find_data_variable(u_opened, u_logical_name)
        u = u_opened[u_name].load().astype("float32")
        coordinates = {name: coordinate.load() for name, coordinate in u_opened.coords.items()}
        global_attrs = dict(u_opened.attrs)
    with xr.open_dataset(v_path, engine="netcdf4") as v_opened:
        v_name = find_data_variable(v_opened, v_logical_name)
        v = v_opened[v_name].load().astype("float32")

    u_aligned, v_aligned = xr.align(u, v, join="exact")
    ws = np.hypot(u_aligned, v_aligned).astype("float32")
    ws.name = output_name
    ws.attrs = {
        "long_name": f"{height_description} wind speed",
        "standard_name": "wind_speed",
        "units": str(u.attrs.get("units", "m s-1")),
        "derivation": f"hypot({u_name}, {v_name})",
        "source_u_file": u_path.name,
        "source_v_file": v_path.name,
    }
    global_attrs.update(
        {
            "processing_stage": "derived_variable_only",
            "processing_rule": f"{output_name} = hypot({u_name}, {v_name})",
        }
    )
    return xr.Dataset({output_name: ws}, coords=coordinates, attrs=global_attrs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_date, default=date(2025, 1, 1))
    parser.add_argument("--source", type=Path, help="daily NC tree from script 1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    day: date = args.date
    source_root = (args.source or default_source(day)).resolve()
    output_root = (args.output or default_output(day)).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output must not be the source directory or one of its children")

    inputs = sorted(source_root.rglob("*.nc"))
    if not inputs:
        raise ValueError(f"no NetCDF files found under {source_root}")
    converted_filename = f"{day:%Y.%m.%d}.unit_converted.nc"
    transform_by_directory = {
        ("pl", "q"): "q",
        **{("cldrad", name): name for name in RADIATION},
        ("sfc", "tp"): "tp",
    }

    print(f"source: {source_root}")
    print(f"output: {output_root}")
    for index, source in enumerate(inputs, start=1):
        relative = source.relative_to(source_root)
        if len(relative.parts) < 4:
            raise ValueError(f"expected group/variable/year/file layout: {source}")
        logical_name = transform_by_directory.get(tuple(relative.parts[:2]))
        destination = output_root / relative.parent / converted_filename
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {destination}")
        print(
            f"[{index:02d}/{len(inputs):02d}] {relative.parts[0]}/{relative.parts[1]}: "
            f"{'transform ' + logical_name if logical_name else 'copy unchanged'}"
        )
        if logical_name:
            dataset = transformed_dataset(source, logical_name)
            transformed_name = find_data_variable(dataset, logical_name)
            write_dataset(dataset, destination, transformed_name)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            if temporary.exists():
                temporary.unlink()
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()

    wind_speeds = (
        ("u10m", "v10m", "ws10m", "10 metre"),
        ("u100m", "v100m", "ws100m", "100 metre"),
    )
    for u_logical, v_logical, ws_name, description in wind_speeds:
        u_path = next(
            (output_root / "sfc" / u_logical / str(day.year)).glob("*.nc"), None
        )
        v_path = next(
            (output_root / "sfc" / v_logical / str(day.year)).glob("*.nc"), None
        )
        if u_path is None or v_path is None:
            raise FileNotFoundError(
                f"cannot derive {ws_name}: converted {u_logical}/{v_logical} files are missing"
            )
        ws_path = output_root / "sfc" / ws_name / str(day.year) / converted_filename
        if ws_path.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {ws_path}")
        print(f"[derive] sfc/{ws_name}: hypot({u_logical}, {v_logical})")
        dataset = derive_wind_speed(
            u_path,
            v_path,
            u_logical,
            v_logical,
            ws_name,
            description,
        )
        write_dataset(dataset, ws_path, ws_name)

    print(f"[DONE] wrote {len(inputs) + len(wind_speeds)} NetCDF files to {output_root}")
    return output_root


if __name__ == "__main__":
    run(parse_args())
