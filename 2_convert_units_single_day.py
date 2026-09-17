#!/usr/bin/env python3
"""Select one day and apply the agreed unit/preprocessing rules to ERA5 NC.

The input may be a monthly archive, a shared daily tree, or an already
isolated daily tree.  Selection and conversion happen in one pass, so batch
processing does not need to materialize an intermediate raw daily copy.

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
import importlib.util
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr


RADIATION = frozenset({"ssr", "ssrd", "fdir", "ttr"})
RADIATION_SECONDS = 21600.0
TP_METRES_TO_MILLIMETRES = 1000.0
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_SCRIPT = SCRIPT_DIR / "1_extract_single_day.py"


def load_extractor():
    """Reuse archive discovery and disguised-ZIP handling from script 1."""
    spec = importlib.util.spec_from_file_location("era5_daily_selector", EXTRACT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(EXTRACT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def transform_loaded_dataset(
    result: xr.Dataset, source: Path, logical_name: str
) -> xr.Dataset:
    variable_name = find_data_variable(result, logical_name)
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


def transformed_dataset(source: Path, logical_name: str) -> xr.Dataset:
    """Backward-compatible transform for an already isolated daily file."""
    with xr.open_dataset(source, engine="netcdf4") as opened:
        result = opened.load()
    return transform_loaded_dataset(result, source, logical_name)


def load_selected_day(
    source: Path,
    day: date,
    static: bool,
    extractor,
) -> xr.Dataset:
    """Load only the requested day from a daily or monthly source file."""
    with extractor.actual_netcdf(source) as actual:
        with xr.open_dataset(actual, engine="netcdf4") as opened:
            time_name = next(
                (name for name in ("valid_time", "time") if name in opened.coords),
                None,
            )
            if time_name is None:
                raise ValueError(f"no valid_time/time coordinate in {source}")
            start = np.datetime64(day.isoformat(), "ns")
            stop = np.datetime64((day + timedelta(days=1)).isoformat(), "ns")
            values = np.asarray(opened[time_name].values).astype("datetime64[ns]")
            indices = np.flatnonzero((values >= start) & (values < stop))
            if static and indices.size == 0 and values.size:
                indices = np.array([0], dtype=int)
            if indices.size == 0:
                raise ValueError(f"{source} contains no records for {day.isoformat()}")
            expected = np.arange(indices[0], indices[-1] + 1)
            if not np.array_equal(indices, expected):
                raise ValueError(
                    f"non-contiguous records for {day.isoformat()} in {source}"
                )
            result = opened.isel(
                {time_name: slice(indices[0], indices[-1] + 1)}
            ).load()
            result.attrs = dict(opened.attrs)
    result.attrs.update(
        {
            "extraction_date": day.isoformat(),
            "extraction_source_file": source.name,
        }
    )
    return result


def float_encoding(
    dataset: xr.Dataset, transformed_name: str | None
) -> dict[str, dict]:
    encoding: dict[str, dict] = {}
    for name in dataset.data_vars:
        item: dict = {"zlib": True, "complevel": 4, "shuffle": True}
        if name == transformed_name:
            item["dtype"] = "float32"
        encoding[name] = item
    return encoding


def write_dataset(
    dataset: xr.Dataset, destination: Path, transformed_name: str | None
) -> None:
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
    parser.add_argument(
        "--source", type=Path,
        help="monthly, shared-daily, date-partitioned, or isolated daily NC root",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "monthly", "daily"),
        default="auto",
        help="source filename layout (default: detect daily first, then monthly)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="omit static fields (batch mode stores them only once)",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    day: date = args.date
    source_root = (args.source or default_source(day)).resolve()
    output_root = (args.output or default_output(day)).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output must not be the source directory or one of its children")

    extractor = load_extractor()
    variable_dirs = sorted(
        path
        for group in source_root.iterdir()
        if group.is_dir()
        for path in group.iterdir()
        if path.is_dir()
    )
    if args.skip_static:
        variable_dirs = [
            path
            for path in variable_dirs
            if path.relative_to(source_root).parts[0] != "static"
        ]
    if not variable_dirs:
        raise ValueError(f"no group/variable directories found under {source_root}")
    converted_filename = f"{day:%Y.%m.%d}.unit_converted.nc"
    transform_by_directory = {
        ("pl", "q"): "q",
        **{("cldrad", name): name for name in RADIATION},
        ("sfc", "tp"): "tp",
    }

    print(f"source: {source_root}")
    print(f"output: {output_root}")
    for index, variable_dir in enumerate(variable_dirs, start=1):
        relative = variable_dir.relative_to(source_root)
        source = extractor.source_file(
            source_root,
            relative / str(day.year),
            day,
            args.input_mode,
        )
        logical_name = transform_by_directory.get(tuple(relative.parts))
        destination = output_root / relative / str(day.year) / converted_filename
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {destination}")
        print(
            f"[{index:02d}/{len(variable_dirs):02d}] {relative}: "
            f"{source.name} -> "
            f"{'transform ' + logical_name if logical_name else 'select unchanged'}"
        )
        dataset = load_selected_day(
            source,
            day,
            static=relative.parts[0] == "static",
            extractor=extractor,
        )
        if logical_name:
            dataset = transform_loaded_dataset(dataset, source, logical_name)
            transformed_name = find_data_variable(dataset, logical_name)
            write_dataset(dataset, destination, transformed_name)
        else:
            write_dataset(dataset, destination, None)

    wind_speeds = (
        ("u10m", "v10m", "ws10m", "10 metre"),
        ("u100m", "v100m", "ws100m", "100 metre"),
    )
    for u_logical, v_logical, ws_name, description in wind_speeds:
        u_path = (
            output_root / "sfc" / u_logical / str(day.year) / converted_filename
        )
        v_path = (
            output_root / "sfc" / v_logical / str(day.year) / converted_filename
        )
        if not u_path.is_file() or not v_path.is_file():
            raise FileNotFoundError(
                f"cannot derive {ws_name}: converted {u_path} or {v_path} is missing"
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

    print(
        f"[DONE] wrote {len(variable_dirs) + len(wind_speeds)} "
        f"NetCDF files to {output_root}"
    )
    return output_root


if __name__ == "__main__":
    run(parse_args())
