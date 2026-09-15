#!/usr/bin/env python3
"""Extract one day from monthly or daily ERA5 NetCDF files, preserving tree layout.

The source archive contains a few ZIP files whose names still end in ``.nc``;
those files are detected and unpacked transparently.  Every output is a real
NetCDF file with the same relative ``group/variable/year`` directory layout.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr


DEFAULT_SOURCE = Path(r"E:\era5_2025.01-2026.07_nc")


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def default_output(day: date) -> Path:
    return Path(f"E:\\era5_{day:%Y.%m.%d}_nc")


def source_file(
    root: Path,
    relative_dir: Path,
    day: date,
    input_mode: str = "auto",
) -> Path:
    directory = root / relative_dir
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    daily_names = (f"{day:%Y.%m.%d}.nc", f"{day:%Y%m%d}.nc")
    daily = [directory / name for name in daily_names if (directory / name).is_file()]
    monthly = sorted(directory.glob(f"*_{day.year}{day.month}.nc"))
    candidates = daily if input_mode == "daily" else monthly
    if input_mode == "auto":
        candidates = daily or monthly
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one {input_mode} source file for {day.isoformat()} "
            f"in {directory}, found {len(candidates)}"
        )
    return candidates[0]


# Retained for callers written before daily-input support was added.
monthly_file = source_file


@contextmanager
def actual_netcdf(path: Path) -> Iterator[Path]:
    """Yield a readable NetCDF path, unpacking a ZIP-disguised .nc if needed."""
    with path.open("rb") as stream:
        signature = stream.read(4)
    if signature != b"PK\x03\x04":
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="era5_extract_") as temporary:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"{path} must contain exactly one file")
            archive.extract(members[0], temporary)
        yield Path(temporary) / members[0]


def output_encoding(dataset: xr.Dataset) -> dict[str, dict]:
    """Use bounded compression without carrying incompatible source encodings."""
    encoding: dict[str, dict] = {}
    for name, variable in dataset.data_vars.items():
        item: dict = {"zlib": True, "complevel": 4, "shuffle": True}
        if np.issubdtype(variable.dtype, np.floating):
            item["dtype"] = variable.dtype
        encoding[name] = item
    return encoding


def extract_file(source: Path, destination: Path, day: date, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        with actual_netcdf(source) as actual:
            with xr.open_dataset(actual, engine="netcdf4") as dataset:
                time_name = next(
                    (name for name in ("valid_time", "time") if name in dataset.coords), None
                )
                if time_name is None:
                    raise ValueError(f"no valid_time/time coordinate in {source}")
                start = np.datetime64(day.isoformat(), "ns")
                stop = np.datetime64((day + timedelta(days=1)).isoformat(), "ns")
                values = np.asarray(dataset[time_name].values).astype("datetime64[ns]")
                indices = np.flatnonzero((values >= start) & (values < stop))
                if indices.size == 0:
                    raise ValueError(f"{source} contains no records for {day.isoformat()}")
                expected = np.arange(indices[0], indices[-1] + 1)
                if not np.array_equal(indices, expected):
                    raise ValueError(f"non-contiguous records for {day.isoformat()} in {source}")
                selected = dataset.isel({time_name: slice(indices[0], indices[-1] + 1)})
                selected.attrs = dict(dataset.attrs)
                selected.attrs.update(
                    {
                        "extraction_date": day.isoformat(),
                        "extraction_source_file": source.name,
                    }
                )
                selected.to_netcdf(
                    temporary,
                    engine="netcdf4",
                    format="NETCDF4",
                    encoding=output_encoding(selected),
                )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--date", type=parse_date, default=date(2025, 1, 1))
    parser.add_argument(
        "--input-mode",
        choices=("auto", "monthly", "daily"),
        default="auto",
        help="source filename layout (default: detect daily first, then monthly)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=r"output root (default: E:\era5_YYYY.MM.DD_nc)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    source_root = args.source.resolve()
    day: date = args.date
    output_root = (args.output or default_output(day)).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output must not be the source directory or one of its children")

    variable_dirs = sorted(
        path
        for group in source_root.iterdir()
        if group.is_dir()
        for path in group.iterdir()
        if path.is_dir()
    )
    if not variable_dirs:
        raise ValueError(f"no group/variable directories found under {source_root}")

    filename = f"{day:%Y.%m.%d}.nc"
    print(f"source: {source_root}")
    print(f"date:   {day.isoformat()}")
    print(f"output: {output_root}")
    for index, variable_dir in enumerate(variable_dirs, start=1):
        relative = variable_dir.relative_to(source_root)
        source = source_file(
            source_root,
            relative / str(day.year),
            day,
            args.input_mode,
        )
        destination = output_root / relative / str(day.year) / filename
        print(f"[{index:02d}/{len(variable_dirs):02d}] {relative}: {source.name}")
        extract_file(source, destination, day, args.overwrite)

    print(f"[DONE] wrote {len(variable_dirs)} daily NetCDF files to {output_root}")
    return output_root


if __name__ == "__main__":
    run(parse_args())
