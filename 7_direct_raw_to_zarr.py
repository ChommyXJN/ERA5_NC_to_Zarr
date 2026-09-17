#!/usr/bin/env python3
"""Convert a continuous raw ERA5 NC archive directly to the final Zarr v3.

This is the high-throughput production path.  It applies the same selection,
unit conversion, derived winds, regridding, normalization, float16 cast,
compression, layout, and metadata rules as scripts 1 -> 2 -> 3, without
materializing extracted or unit-converted daily NetCDF files.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import zarr


SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_SCRIPT = SCRIPT_DIR / "1_extract_single_day.py"
PIPELINE_SCRIPT = SCRIPT_DIR / "3_normalize_and_write_zarr.py"
_WORKER_CONFIG: dict | None = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def dates_inclusive(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("--end must be on or after --start")
    return [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    ]


def partitioned_day_root(source: Path, day: date) -> Path | None:
    for name in (f"{day:%Y.%m.%d}", day.isoformat(), f"{day:%Y%m%d}"):
        candidate = source / name
        if candidate.is_dir():
            return candidate
    return None


def source_root_for_day(source: Path, day: date, input_mode: str) -> Path:
    if input_mode == "monthly":
        return source
    return partitioned_day_root(source, day) or source


def expected_times(days: list[date]) -> np.ndarray:
    values = [
        np.datetime64(day.isoformat(), "h") + np.timedelta64(hour, "h")
        for day in days
        for hour in (0, 6, 12, 18)
    ]
    return np.asarray(values, dtype="datetime64[h]")


def source_path(
    source: Path,
    day: date,
    variable: str,
    input_mode: str,
    extractor,
    pipeline,
) -> Path:
    root = source_root_for_day(source, day, input_mode)
    relative = (
        Path(pipeline.GROUPS[variable])
        / pipeline.SOURCE_DIRECTORIES.get(variable, variable)
        / str(day.year)
    )
    return extractor.source_file(root, relative, day, input_mode)


def load_raw_day(
    path: Path,
    variable: str,
    day: date,
    extractor,
    pipeline,
    static: bool = False,
) -> xr.Dataset:
    with extractor.actual_netcdf(path) as actual:
        with xr.open_dataset(actual, engine="netcdf4", cache=False) as opened:
            rename = {
                old: new
                for old, new in {
                    "valid_time": "time",
                    "latitude": "lat",
                    "longitude": "lon",
                    "pressure_level": "level",
                }.items()
                if old in opened
            }
            candidates = pipeline.SOURCE_VARIABLES.get(variable, (variable,))
            source_name = next(
                (name for name in candidates if name in opened.data_vars), None
            )
            if source_name is None:
                raise ValueError(f"{path} does not contain {variable}")
            if source_name != variable:
                rename[source_name] = variable
            dataset = opened.rename(rename)
            if "time" not in dataset.coords:
                raise ValueError(f"{path} has no valid_time/time coordinate")
            start = np.datetime64(day.isoformat(), "ns")
            stop = np.datetime64((day + timedelta(days=1)).isoformat(), "ns")
            values = np.asarray(dataset.time.values).astype("datetime64[ns]")
            indices = np.flatnonzero((values >= start) & (values < stop))
            if static and indices.size == 0 and values.size:
                indices = np.array([0], dtype=int)
            if indices.size == 0:
                raise ValueError(f"{path} contains no records for {day.isoformat()}")
            expected = np.arange(indices[0], indices[-1] + 1)
            if not np.array_equal(indices, expected):
                raise ValueError(f"non-contiguous records for {day} in {path}")
            result = dataset.isel(
                time=slice(indices[0], indices[-1] + 1)
            ).load()
    return result


def convert_in_memory(dataset: xr.Dataset, variable: str, path: Path) -> xr.Dataset:
    values = dataset[variable].astype("float32")
    if variable == "q":
        values = values * np.float32(1000.0)
    elif variable in {"ssr", "ssrd", "fdir", "ttr"}:
        values = values / np.float32(21600.0)
    elif variable == "tp":
        units = str(dataset[variable].attrs.get("units", "")).strip().lower()
        if units not in {"m", "metre", "metres", "meter", "meters"}:
            raise ValueError(f"{path}: expected raw tp units in metres, got {units!r}")
        values = np.log1p(
            np.maximum(values * np.float32(1000.0), np.float32(0.0))
        )
    dataset[variable] = values
    return dataset


def open_converted_day(
    source: Path,
    day: date,
    input_mode: str,
    extractor,
    pipeline,
) -> dict[str, xr.Dataset]:
    datasets: dict[str, xr.Dataset] = {}
    reference_time: np.ndarray | None = None
    for variable in pipeline.DIRECT_VARIABLES:
        path = source_path(
            source, day, variable, input_mode, extractor, pipeline
        )
        dataset = load_raw_day(path, variable, day, extractor, pipeline)
        current_time = np.asarray(dataset.time.values).astype("datetime64[h]")
        if reference_time is None:
            reference_time = current_time
        elif not np.array_equal(reference_time, current_time):
            raise ValueError(f"time coordinate mismatch for {path}")
        datasets[variable] = convert_in_memory(dataset, variable, path)

    expected = np.asarray(
        [np.datetime64(day.isoformat(), "h") + np.timedelta64(h, "h")
         for h in (0, 6, 12, 18)],
        dtype="datetime64[h]",
    )
    if reference_time is None or not np.array_equal(reference_time, expected):
        raise ValueError(
            f"{day.isoformat()} must contain exactly 00, 06, 12, 18 UTC"
        )

    for u_name, v_name, output_name in (
        ("u10m", "v10m", "ws10m"),
        ("u100m", "v100m", "ws100m"),
    ):
        u, v = xr.align(
            datasets[u_name][u_name].astype("float32"),
            datasets[v_name][v_name].astype("float32"),
            join="exact",
        )
        speed = np.hypot(u, v).astype("float32")
        speed.name = output_name
        datasets[output_name] = xr.Dataset({output_name: speed})
    return datasets


def close_datasets(datasets: dict[str, xr.Dataset]) -> None:
    for dataset in datasets.values():
        dataset.close()


def initialize_worker(
    store_path: str,
    source_path_text: str,
    input_mode: str,
    mean_values: list[float],
    std_values: list[float],
) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = {
        "store_path": store_path,
        "source": Path(source_path_text),
        "input_mode": input_mode,
        "mean": np.asarray(mean_values, dtype="f4"),
        "std": np.asarray(std_values, dtype="f4"),
        "extractor": load_module("era5_direct_extractor", EXTRACT_SCRIPT),
        "pipeline": load_module("era5_direct_pipeline", PIPELINE_SCRIPT),
    }


def write_one_day(day_text: str, day_index: int) -> str:
    if _WORKER_CONFIG is None:
        raise RuntimeError("direct conversion worker is not initialized")
    config = _WORKER_CONFIG
    day = date.fromisoformat(day_text)
    source = config["source"]
    input_mode = config["input_mode"]
    mean = config["mean"]
    std = config["std"]
    extractor = config["extractor"]
    pipeline = config["pipeline"]
    datasets = open_converted_day(
        source, day, input_mode, extractor, pipeline
    )
    try:
        buffer = np.empty(
            (4, pipeline.CHANNEL_COUNT, 721, 1440), dtype="f2"
        )
        regridder = pipeline.Regridder()
        for offset, channel_index, channel, values in pipeline.iter_block(
            datasets, pipeline.DYNAMIC_CHANNELS, 0, 4, regridder
        ):
            normalized = pipeline.normalize_values(
                values, channel, mean[channel_index], std[channel_index]
            )
            buffer[offset, channel_index] = normalized.astype("f2")
        data = zarr.open_group(
            config["store_path"], mode="r+", use_consolidated=False
        )["data"]
        start = day_index * 4
        for offset in range(4):
            data[start + offset] = buffer[offset]
    finally:
        close_datasets(datasets)
    return day_text


def load_statistics(path: Path, statistic: str, pipeline) -> np.ndarray:
    values = pipeline.read_statistic_file(path, statistic)
    missing = [name for name in pipeline.DYNAMIC_CHANNELS if name not in values]
    extra = sorted(set(values) - set(pipeline.DYNAMIC_CHANNELS))
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(extra))
        raise ValueError(f"{path}: channel mismatch: {'; '.join(details)}")
    result = np.asarray(
        [values[name] for name in pipeline.DYNAMIC_CHANNELS], dtype="f4"
    )
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: non-finite {statistic}")
    if statistic == "std" and np.any(result <= 0):
        raise ValueError(f"{path}: standard deviations must be positive")
    return result


def add_raw_masks(
    group: zarr.Group,
    source: Path,
    day: date,
    input_mode: str,
    extractor,
    pipeline,
) -> None:
    path = source_path(
        source, day, "lsm", input_mode, extractor, pipeline
    )
    dataset = load_raw_day(
        path, "lsm", day, extractor, pipeline, static=True
    )
    try:
        field = dataset["lsm"]
        values = pipeline.Regridder().apply(
            field, 0 if "time" in field.dims else None
        )
    finally:
        dataset.close()
    land, sea = pipeline.derive_land_sea_masks(values)
    masks = np.stack((land, sea), axis=0)
    group.create_array(
        "mask",
        data=masks,
        chunks=(1, 721, 1440),
        compressors=pipeline.codecs(),
        fill_value=np.uint8(0),
        dimension_names=("mask_channel", "lat", "lon"),
        attributes=pipeline.MASK_ATTRIBUTES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--input-mode",
        choices=("auto", "monthly", "daily"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--mean", type=Path, default=SCRIPT_DIR / "mean.nc")
    parser.add_argument("--std", type=Path, default=SCRIPT_DIR / "std.nc")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--channel-chunk", type=int, default=116)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path | None:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    pipeline = load_module("era5_direct_main_pipeline", PIPELINE_SCRIPT)
    extractor = load_module("era5_direct_main_extractor", EXTRACT_SCRIPT)
    if not 1 <= args.channel_chunk <= pipeline.CHANNEL_COUNT:
        raise ValueError("--channel-chunk must be between 1 and 116")
    days = dates_inclusive(args.start, args.end)
    times = expected_times(days)
    pipeline.validate_time_coverage(times, False)
    mean = load_statistics(args.mean.resolve(), "mean", pipeline)
    std = load_statistics(args.std.resolve(), "std", pipeline)
    label = pipeline.dataset_label(times, False)
    dataset_id = (
        f"era5.{label}.c{pipeline.CHANNEL_COUNT}.p25.h6."
        f"{pipeline.CONTENT_VERSION}"
    )
    final_path = output / f"{dataset_id}.zarr"
    print(f"source:  {source}")
    print(f"range:   {args.start} .. {args.end} ({len(days)} days)")
    print(f"target:  {final_path}")
    print(f"shape:   ({len(times)}, {pipeline.CHANNEL_COUNT}, 721, 1440)")
    print(f"workers: {args.workers}")
    if args.dry_run:
        for variable in pipeline.DIRECT_VARIABLES + ("lsm",):
            source_path(
                source, days[0], variable, args.input_mode, extractor, pipeline
            )
            source_path(
                source, days[-1], variable, args.input_mode, extractor, pipeline
            )
        print("[DRY RUN] boundary files and statistics are available")
        return None
    if final_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {final_path}")
    output.mkdir(parents=True, exist_ok=True)
    staging = output / f".{dataset_id}.{uuid.uuid4().hex}.tmp"
    started = time.perf_counter()
    try:
        group, _ = pipeline.create_store(
            staging,
            times,
            dataset_id,
            mean,
            std,
            args.channel_chunk,
            pipeline.DEFAULT_RADIATION_SECONDS,
        )
        add_raw_masks(
            group, source, days[0], args.input_mode, extractor, pipeline
        )
        del group
        worker_args = [
            (day.isoformat(), index) for index, day in enumerate(days)
        ]
        completed = 0
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(
                str(staging),
                str(source),
                args.input_mode,
                mean.tolist(),
                std.tolist(),
            ),
        ) as executor:
            futures = [executor.submit(write_one_day, *item) for item in worker_args]
            for future in as_completed(futures):
                day_text = future.result()
                completed += 1
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed else 0.0
                eta = (len(days) - completed) / rate if rate else 0.0
                print(
                    f"[direct] {completed}/{len(days)} "
                    f"({completed / len(days):.2%}) day={day_text} "
                    f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
                    flush=True,
                )
        pipeline.consolidate_metadata(staging)
        pipeline.validate_output(
            staging,
            times,
            args.channel_chunk,
            pipeline.DEFAULT_RADIATION_SECONDS,
        )
        if final_path.exists():
            shutil.rmtree(final_path)
        os.replace(staging, final_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[DONE] {final_path}")
    return final_path


if __name__ == "__main__":
    run(parse_args())
