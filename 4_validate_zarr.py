#!/usr/bin/env python3
"""Independently validate a Zarr produced by 3_normalize_and_write_zarr.py."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import zarr


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = SCRIPT_DIR / "3_normalize_and_write_zarr.py"
DEFAULT_CHANNELS = ("z500", "t2m", "q500", "swh")


def load_pipeline():
    spec = importlib.util.spec_from_file_location("era5_zarr_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decoded_times(group: zarr.Group) -> np.ndarray:
    values = np.asarray(group["time"][:])
    if not np.issubdtype(values.dtype, np.datetime64):
        raise ValueError(f"time dtype must be datetime64, got {values.dtype}")
    values = values.astype("datetime64[ns]")
    if np.isnat(values).any():
        raise ValueError("time contains NaT")
    return values


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
    if differing:
        details.append("different=" + ",".join(differing))
    raise ValueError("root metadata does not match the v3 schema: " + "; ".join(details))


def validate_channel_metadata(group: zarr.Group, pipeline) -> None:
    """Validate /channel attributes without relying on JSON object order."""

    channel = group["channel"]
    labels = [str(value) for value in channel[:].tolist()]
    attributes = dict(channel.attrs)
    info = attributes.get("channel_info")
    if not isinstance(info, dict):
        raise ValueError("/channel channel_info must be an object")
    missing = sorted(set(labels) - set(info))
    extra = sorted(set(info) - set(labels))
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("unexpected=" + ",".join(extra))
        raise ValueError("channel_info keys differ from /channel: " + "; ".join(details))
    expected = pipeline.channel_attributes(
        radiation_seconds=pipeline.DEFAULT_RADIATION_SECONDS
    )
    changed = [
        name
        for name in labels
        if info.get(name) != expected["channel_info"].get(name)
    ]
    non_info_actual = {key: value for key, value in attributes.items() if key != "channel_info"}
    non_info_expected = {key: value for key, value in expected.items() if key != "channel_info"}
    if changed or non_info_actual != non_info_expected:
        details = []
        if changed:
            details.append("channel_info differs for=" + ",".join(changed))
        if non_info_actual != non_info_expected:
            details.append("/channel scalar attributes differ")
        raise ValueError("/channel metadata does not match the v3 schema: " + "; ".join(details))


def inspect_values(
    group: zarr.Group,
    channels: list[str],
    selected_channels: list[str],
    indices: list[int],
) -> None:
    data = group["data"]
    selected = [(channels.index(name), name) for name in selected_channels]
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
        print(
            f"[sample-progress] {position}/{len(indices)} "
            f"read_time={time.perf_counter() - started:.2f}s",
            flush=True,
        )


def full_scan(group: zarr.Group, channels: list[str]) -> None:
    data = group["data"]
    finite_counts = np.zeros(len(channels), dtype="i8")
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
    validate_channel_metadata(group, pipeline)
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
    print(f"[PASS] {path}")
    print(f"time: {times[0]} .. {times[-1]} ({len(times)} steps)")
    print(f"shape: {consolidated['data'].shape}")
    print("latitude: 90 .. -90, strictly decreasing by -0.25 degrees")
    print("masks: uint8 land_mask and sea_mask are binary and complementary")
    print("metadata: non-consolidated and consolidated reads passed")


if __name__ == "__main__":
    run(parse_args())
