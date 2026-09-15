#!/usr/bin/env python3
"""Independently validate a Zarr produced by 3_normalize_and_write_zarr.py."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr
import zarr


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = SCRIPT_DIR / "3_normalize_and_write_zarr.py"
TIME_ORIGIN = np.datetime64("1979-01-01T00", "h")
DEFAULT_CHANNELS = ("z500", "t2m", "tp", "swh")


def load_pipeline():
    spec = importlib.util.spec_from_file_location("era5_zarr_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decoded_times(group: zarr.Group) -> np.ndarray:
    units = str(group["time"].attrs.get("units", ""))
    if units != "hours since 1979-01-01":
        raise ValueError(f"unexpected time units: {units!r}")
    offsets = np.asarray(group["time"][:], dtype="i8")
    return TIME_ORIGIN + offsets.astype("timedelta64[h]")


def sample_indices(total: int, count: int) -> list[int]:
    if count < 0:
        raise ValueError("--sample-count must be non-negative")
    if total < 1 or count == 0:
        return []
    return sorted(set(np.linspace(0, total - 1, min(total, count), dtype=int).tolist()))


def validate_root_metadata(group: zarr.Group, pipeline) -> None:
    """Validate exact product metadata while reporting useful field-level errors."""

    actual = dict(group.attrs)
    dataset_id = str(actual.get("dataset_id", ""))
    revision = str(actual.get("data_revision", ""))
    expected = pipeline.root_attributes(
        dataset_id,
        revision,
        radiation_seconds=pipeline.DEFAULT_RADIATION_SECONDS,
    )
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    differing = sorted(
        key for key in set(actual) & set(expected) if actual[key] != expected[key]
    )
    if not (missing or extra or differing):
        return
    details: list[str] = []
    if missing:
        details.append("missing=" + ",".join(missing))
    if extra:
        details.append("unexpected=" + ",".join(extra))
    if "channel_metadata" in differing:
        actual_channels = actual.get("channel_metadata", {})
        expected_channels = expected["channel_metadata"]
        changed = [
            name
            for name in pipeline.DYNAMIC_CHANNELS
            if actual_channels.get(name) != expected_channels.get(name)
        ]
        details.append("channel_metadata differs for=" + ",".join(changed))
        differing.remove("channel_metadata")
    if differing:
        details.append("different=" + ",".join(differing))
    raise ValueError("root metadata does not match the v2 schema: " + "; ".join(details))


def tp_directory(path: Path) -> Path:
    nested = path / "sfc" / "tp"
    return nested if nested.is_dir() else path


def raw_tp_file(root: Path, timestamp: np.datetime64) -> Path:
    text = str(timestamp.astype("datetime64[h]"))
    year, month = int(text[:4]), int(text[5:7])
    day_iso = text[:10]
    day_dotted = day_iso.replace("-", ".")
    roots = [root]
    roots.extend(
        candidate
        for name in (day_dotted, day_iso, day_iso.replace("-", ""))
        for candidate in (root / name,)
        if candidate.is_dir()
    )
    for candidate_root in roots:
        directory = tp_directory(candidate_root) / str(year)
        daily = sorted(directory.glob(f"{day_dotted}*.nc"))
        monthly = sorted(directory.glob(f"*_{year}{month}.nc"))
        candidates = daily or monthly
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise FileNotFoundError(
                f"multiple raw TP files for {day_iso} in {directory}"
            )
    raise FileNotFoundError(f"no raw TP file for {day_iso} below {root}")


@contextmanager
def actual_netcdf(path: Path) -> Iterator[Path]:
    with path.open("rb") as stream:
        zipped = stream.read(4) == b"PK\x03\x04"
    if not zipped:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="era5_validate_tp_") as temporary:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"{path} must contain exactly one archived file")
            archive.extract(members[0], temporary)
        yield Path(temporary) / members[0]


def validate_tp_against_raw(
    group: zarr.Group,
    channels: list[str],
    times: np.ndarray,
    indices: list[int],
    raw_root: Path,
    pipeline,
) -> None:
    """Recompute raw metres -> millimetres -> log1p -> regrid for samples."""

    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    tp_index = channels.index("tp")
    regridder = pipeline.Regridder()
    for position, time_index in enumerate(indices, start=1):
        timestamp = times[time_index]
        path = raw_tp_file(raw_root, timestamp)
        with actual_netcdf(path) as actual_path:
            with xr.open_dataset(actual_path, engine="netcdf4", cache=False) as dataset:
                time_name = next(
                    (name for name in ("valid_time", "time") if name in dataset.coords),
                    None,
                )
                lat_name = next(
                    (name for name in ("latitude", "lat") if name in dataset.coords),
                    None,
                )
                lon_name = next(
                    (name for name in ("longitude", "lon") if name in dataset.coords),
                    None,
                )
                if time_name is None or lat_name is None or lon_name is None or "tp" not in dataset:
                    raise ValueError(f"{path}: missing tp/time/latitude/longitude")
                units = str(dataset["tp"].attrs.get("units", "")).strip()
                if units != "m":
                    raise ValueError(f"{path}: raw TP units must be 'm', got {units!r}")
                source_times = np.asarray(dataset[time_name].values).astype("datetime64[h]")
                matches = np.flatnonzero(source_times == timestamp.astype("datetime64[h]"))
                if len(matches) != 1:
                    raise ValueError(
                        f"{path}: expected one record at {timestamp}, found {len(matches)}"
                    )
                field = dataset["tp"].isel({time_name: int(matches[0])}).rename(
                    {lat_name: "lat", lon_name: "lon"}
                )
                converted = np.log1p(
                    np.maximum(field.astype("f4") * np.float32(1000.0), np.float32(0.0))
                )
                expected = regridder.apply(converted).astype("f2")
        observed = np.asarray(group["data"][time_index, tp_index], dtype="f2")
        equal = np.array_equal(observed, expected, equal_nan=True)
        if not equal:
            close = np.isclose(observed, expected, rtol=0, atol=0, equal_nan=True)
            difference = np.abs(observed.astype("f4") - expected.astype("f4"))
            raise ValueError(
                f"TP raw-value verification failed at {timestamp}: "
                f"mismatches={int(np.count_nonzero(~close))}, "
                f"max_abs_difference={float(np.nanmax(difference)):.7g}"
            )
        print(
            f"[tp-raw] {position}/{len(indices)} time={timestamp} PASS "
            "(m * 1000 -> clip_min(0) -> log1p)",
            flush=True,
        )


def inspect_values(
    group: zarr.Group,
    channels: list[str],
    selected_channels: list[str],
    indices: list[int],
) -> None:
    data = group["data"]
    selected = [(channels.index(name), name) for name in selected_channels]
    tp_index = channels.index("tp")
    for position, time_index in enumerate(indices, start=1):
        started = time.perf_counter()
        values = np.asarray(data[time_index])
        if np.isinf(values).any():
            raise ValueError(f"infinite data found at time index {time_index}")
        for channel_index, channel in selected:
            field = np.asarray(values[channel_index], dtype="f4")
            finite = np.isfinite(field)
            if not finite.any():
                raise ValueError(
                    f"sampled channel {channel} is entirely NaN at time index {time_index}"
                )
            print(
                f"[sample] time={time_index} channel={channel} "
                f"finite={finite.mean():.2%} "
                f"range={float(np.nanmin(field)):.7g}..{float(np.nanmax(field)):.7g}"
            )
        tp = np.asarray(values[tp_index], dtype="f4")
        if not np.isfinite(tp).any():
            raise ValueError(f"TP is entirely NaN at time index {time_index}")
        if np.nanmin(tp) < 0:
            raise ValueError(f"TP contains negative values at time index {time_index}")
        print(
            f"[sample-progress] {position}/{len(indices)} "
            f"read_time={time.perf_counter() - started:.2f}s",
            flush=True,
        )


def full_scan(group: zarr.Group, channels: list[str]) -> None:
    data = group["data"]
    finite_counts = np.zeros(len(channels), dtype="i8")
    minimum = np.full(len(channels), np.inf, dtype="f8")
    maximum = np.full(len(channels), -np.inf, dtype="f8")
    started = time.perf_counter()
    for index in range(data.shape[0]):
        values = np.asarray(data[index])
        if np.isinf(values).any():
            raise ValueError(f"infinite data found at time index {index}")
        for channel_index in range(len(channels)):
            field = np.asarray(values[channel_index], dtype="f4")
            finite = np.isfinite(field)
            count = int(finite.sum())
            finite_counts[channel_index] += count
            if count:
                minimum[channel_index] = min(
                    minimum[channel_index], float(np.nanmin(field))
                )
                maximum[channel_index] = max(
                    maximum[channel_index], float(np.nanmax(field))
                )
        elapsed = time.perf_counter() - started
        rate = (index + 1) / elapsed if elapsed else 0.0
        eta = (data.shape[0] - index - 1) / rate if rate else 0.0
        print(
            f"[full-scan] {index + 1}/{data.shape[0]} "
            f"({(index + 1) / data.shape[0]:.2%}) "
            f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
            flush=True,
        )
    missing = [channels[index] for index in np.flatnonzero(finite_counts == 0)]
    if missing:
        raise ValueError(f"channels with no finite values: {missing}")
    tp_index = channels.index("tp")
    if minimum[tp_index] < 0:
        raise ValueError("TP contains negative values")
    print("[full-scan] all channels contain finite values and no infinities")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=list(DEFAULT_CHANNELS),
        help="channels reported during sampled data reads",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="read every dynamic time step; substantially slower than sampling",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow intentionally incomplete test time coverage",
    )
    parser.add_argument(
        "--raw-tp-source",
        type=Path,
        help="optional raw archive root or sfc/tp directory for exact sampled TP checks",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    path = args.zarr.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    pipeline = load_pipeline()
    group = zarr.open_group(str(path), mode="r", use_consolidated=False)
    dataset_id = str(group.attrs.get("dataset_id", ""))
    expected_name = f"{dataset_id}.zarr"
    if not dataset_id or path.name != expected_name:
        raise ValueError(
            f"Zarr directory name must match dataset_id: expected {expected_name!r}, "
            f"got {path.name!r}"
        )
    times = decoded_times(group)
    pipeline.validate_time_coverage(times, args.allow_partial)
    validate_root_metadata(group, pipeline)
    channel_chunk = int(group["data"].chunks[1])
    pipeline.validate_output(
        path,
        times,
        channel_chunk,
        pipeline.DEFAULT_RADIATION_SECONDS,
    )
    consolidated = zarr.open_group(str(path), mode="r", use_consolidated=True)
    channels = [str(value) for value in consolidated["channel"][:].tolist()]
    unknown = sorted(set(args.channels) - set(channels))
    if unknown:
        raise ValueError(f"unknown sampled channels: {unknown}")
    indices = sample_indices(len(times), args.sample_count)
    if args.full_scan:
        full_scan(consolidated, channels)
    else:
        inspect_values(
            consolidated,
            channels,
            list(args.channels),
            indices,
        )
    if args.raw_tp_source is not None:
        validate_tp_against_raw(
            consolidated,
            channels,
            times,
            indices,
            args.raw_tp_source.resolve(),
            pipeline,
        )
    print(f"[PASS] {path}")
    print(f"time: {times[0]} .. {times[-1]} ({len(times)} steps)")
    print(f"shape: {consolidated['data'].shape}")
    print("latitude: 90 .. -90, strictly decreasing by -0.25 degrees")
    print("metadata: non-consolidated and consolidated reads passed")


if __name__ == "__main__":
    run(parse_args())
