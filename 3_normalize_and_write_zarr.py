#!/usr/bin/env python3
"""Normalize converted ERA5 NC data and write a self-contained Zarr v3 dataset.

This file intentionally owns the complete second-stage contract: channel
schema, source layout, normalization/denormalization, Zarr metadata, writing,
consolidation, and final validation.

Final processing rules
----------------------
* Input files must already contain converted/preprocessed physical values.
* C78 channels use externally supplied climatological mean/std.
* TP is converted from m to mm, then log1p(max(tp_mm, 0)), and is not
  z-score normalized.
* The other 38 channels use supplied dataset-specific mean/std.  They can be
  computed explicitly with --compute-missing-additional-stats if absent.
* No raw-field unit conversion or statistics-unit correction is performed here.
  Input values and supplied mean/std must already use the same final basis.
* Output data are float16; statistics are float32 with float64 accumulation.
* Channel coordinates are string labels, so xarray .sel(channel="z500") works.
* Latitude is stored strictly increasingly from -90 to 90 at 0.25 degrees.

The output is written to a staging directory and published only after full
validation.  Existing output is never touched unless --overwrite is provided.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Complete channel/source schema for this Zarr product
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"
CONTENT_VERSION = "v3"
CHANNEL_COUNT = 116
DEFAULT_RADIATION_SECONDS = 21600.0

TARGET_LAT = np.linspace(-90.0, 90.0, 721, dtype=np.float32)
TARGET_LON = np.arange(0.0, 360.0, 0.25, dtype=np.float32)

BASE_LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
W_LEVELS = BASE_LEVELS + (10,)
PRESSURE_VARIABLES = ("z", "t", "u", "v", "q")
SURFACE = (
    "d2m", "t2m", "msl", "sp", "skt", "u10m", "v10m", "u100m", "v100m",
    "sst", "tcw", "tp", "sd",
)
DERIVED = ("ws10m", "ws100m")
CLOUD_RADIATION = ("lcc", "mcc", "hcc", "tcc", "ssr", "ssrd", "fdir", "ttr")
SOIL = ("swvl1", "swvl2", "stl1", "stl2")
WAVE = ("wmb", "mwd", "cdww", "mwp", "swh")
STATIC = ("lsm", "slor", "sdor", "z_sfc")
RADIATION = frozenset(("ssr", "ssrd", "fdir", "ttr"))

DYNAMIC_CHANNELS = (
    tuple(f"{variable}{level}" for variable in PRESSURE_VARIABLES for level in BASE_LEVELS)
    + SURFACE
    + tuple(f"{variable}10" for variable in PRESSURE_VARIABLES)
    + tuple(f"w{level}" for level in W_LEVELS)
    + DERIVED + CLOUD_RADIATION + SOIL + WAVE
)
REFERENCE_CHANNELS = tuple(
    f"{variable}{level}"
    for variable in PRESSURE_VARIABLES
    for level in (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
) + (
    "t2m", "d2m", "sst", "u10m", "v10m", "u100m", "v100m", "msl",
    "ssr", "ssrd", "fdir", "ttr", "tp",
)
ADDITIONAL_CHANNELS = tuple(
    name for name in DYNAMIC_CHANNELS if name not in REFERENCE_CHANNELS
)
DIRECT_VARIABLES = tuple(dict.fromkeys(
    PRESSURE_VARIABLES + ("w",) + SURFACE + CLOUD_RADIATION + SOIL + WAVE
))
NORMALIZED_INPUT_VARIABLES = tuple(dict.fromkeys(DIRECT_VARIABLES + DERIVED))

SOURCE_VARIABLES = {
    "u10m": ("u10m", "u10"),
    "v10m": ("v10m", "v10"),
    "u100m": ("u100m", "u100"),
    "v100m": ("v100m", "v100"),
    "swvl1": ("swvl1", "vsw1"),
    "swvl2": ("swvl2", "vsw2"),
    "stl1": ("stl1", "sot1"),
    "stl2": ("stl2", "sot2"),
    "z_sfc": ("z_sfc", "z"),
}
SOURCE_DIRECTORIES = {
    "swvl1": "vsw1", "swvl2": "vsw2", "stl1": "sot1", "stl2": "sot2",
}
GROUPS = {
    **{name: "pl" for name in PRESSURE_VARIABLES + ("w",)},
    **{name: "sfc" for name in SURFACE + DERIVED},
    **{name: "cldrad" for name in CLOUD_RADIATION},
    **{name: "soil" for name in SOIL},
    **{name: "wave" for name in WAVE},
    **{name: "static" for name in STATIC},
}

STATIC_METADATA = {
    "lsm": ("land_sea_mask", "Land-sea mask stored as land fraction", "1"),
    "slor": ("slope_of_sub_gridscale_orography", "Slope of sub-gridscale orography", "1"),
    "sdor": ("standard_deviation_of_orography", "Standard deviation of orography", "m"),
    "z_sfc": ("surface_geopotential", "Surface geopotential (not geopotential height)", "m2 s-2"),
}
VARIABLE_METADATA = {
    "z": ("m2 s-2", "Geopotential"), "t": ("K", "Temperature"),
    "u": ("m s-1", "Eastward wind"), "v": ("m s-1", "Northward wind"),
    "q": ("g kg-1", "Specific humidity"), "w": ("Pa s-1", "Vertical velocity"),
    "d2m": ("K", "2 metre dewpoint temperature"), "t2m": ("K", "2 metre temperature"),
    "msl": ("Pa", "Mean sea level pressure"), "sp": ("Pa", "Surface pressure"),
    "skt": ("K", "Skin temperature"), "u10m": ("m s-1", "10 metre eastward wind"),
    "v10m": ("m s-1", "10 metre northward wind"),
    "u100m": ("m s-1", "100 metre eastward wind"),
    "v100m": ("m s-1", "100 metre northward wind"),
    "sst": ("K", "Sea surface temperature"), "tcw": ("kg m-2", "Total column water"),
    "tp": ("1", "Log-transformed total precipitation"),
    "sd": ("m", "Snow depth water equivalent"),
    "ws10m": ("m s-1", "10 metre wind speed"),
    "ws100m": ("m s-1", "100 metre wind speed"),
    "lcc": ("1", "Low cloud cover"), "mcc": ("1", "Medium cloud cover"),
    "hcc": ("1", "High cloud cover"), "tcc": ("1", "Total cloud cover"),
    "ssr": ("W m-2", "Surface net short-wave radiation"),
    "ssrd": ("W m-2", "Surface short-wave radiation downwards"),
    "fdir": ("W m-2", "Surface direct short-wave radiation"),
    "ttr": ("W m-2", "Top net long-wave radiation"),
    "swvl1": ("m3 m-3", "Volumetric soil water layer 1"),
    "swvl2": ("m3 m-3", "Volumetric soil water layer 2"),
    "stl1": ("K", "Soil temperature level 1"),
    "stl2": ("K", "Soil temperature level 2"),
    "wmb": ("m", "Model bathymetry"), "mwd": ("degree", "Mean wave direction"),
    "cdww": ("1", "Coefficient of drag with waves"),
    "mwp": ("s", "Mean wave period"),
    "swh": ("m", "Significant height of combined wind waves and swell"),
}


def split_channel(channel: str) -> tuple[str, int | None]:
    for variable in (*PRESSURE_VARIABLES, "w"):
        suffix = channel.removeprefix(variable)
        if channel.startswith(variable) and suffix.isdigit():
            return variable, int(suffix)
    return channel, None


def preprocessing_metadata(
    channel: str,
    variable: str,
    *,
    radiation_seconds: float = DEFAULT_RADIATION_SECONDS,
) -> list[dict[str, Any]]:
    zscore = {"operation": "zscore", "mean": "/auxiliary/mean", "std": "/auxiliary/std"}
    if variable == "q":
        return [{
            "operation": "multiply", "factor": 1000.0,
            "input_units": "kg kg-1", "output_units": "g kg-1",
        }, zscore]
    if channel in RADIATION:
        seconds = float(radiation_seconds)
        return [{
            "operation": "divide", "divisor": seconds,
            "accumulation_window_seconds": int(seconds),
            "input_units": "J m-2", "output_units": "W m-2",
        }, zscore]
    if channel == "tp":
        return [
            {
                "operation": "multiply", "factor": 1000.0,
                "input_units": "m", "output_units": "mm",
            },
            {"operation": "clip_min", "minimum": 0.0},
            {"operation": "log1p"},
        ]
    if channel == "ws10m":
        return [{"operation": "hypot", "inputs": ["u10m", "v10m"]}, zscore]
    if channel == "ws100m":
        return [{"operation": "hypot", "inputs": ["u100m", "v100m"]}, zscore]
    return [zscore]


def build_channel_metadata(
    channels: Sequence[str] = DYNAMIC_CHANNELS,
    *,
    radiation_seconds: float = DEFAULT_RADIATION_SECONDS,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for channel in channels:
        variable, level = split_channel(str(channel))
        units, long_name = VARIABLE_METADATA[variable]
        result[str(channel)] = {
            "variable": variable, "level": level, "units": units,
            "long_name": long_name,
            "preprocess": preprocessing_metadata(
                str(channel), variable, radiation_seconds=radiation_seconds
            ),
        }
    return result


if len(DYNAMIC_CHANNELS) != CHANNEL_COUNT or len(set(DYNAMIC_CHANNELS)) != CHANNEL_COUNT:
    raise RuntimeError("canonical ERA5 channel definition must contain 116 unique labels")

# ---------------------------------------------------------------------------
# Zarr v3 attributes and metadata serialization
# ---------------------------------------------------------------------------

EXPECTED_CHILDREN = (
    "data", "channel", "lat", "lon", "time", "auxiliary",
    "auxiliary/mean", "auxiliary/std", "auxiliary/land_sea_mask",
    "auxiliary/slope_of_sub_gridscale_orography",
    "auxiliary/standard_deviation_of_orography", "auxiliary/surface_geopotential",
)
DATA_ATTRIBUTES = {
    "long_name": "Preprocessed and normalized ERA5 fields",
    "data_representation": "channel-dependent; see root channel_metadata",
    "normalization_mean": "/auxiliary/mean", "normalization_std": "/auxiliary/std",
}
CHANNEL_ATTRIBUTES = {
    "long_name": "weather variable channel name",
    "description": "String channel labels aligned positionally with data[:, channel, :, :]",
    "channel_count": CHANNEL_COUNT,
}
LAT_ATTRIBUTES = {
    "standard_name": "latitude", "long_name": "latitude",
    "units": "degrees_north", "axis": "Y",
}
LON_ATTRIBUTES = {
    "standard_name": "longitude", "long_name": "longitude",
    "units": "degrees_east", "axis": "X",
}
TIME_ATTRIBUTES = {
    "standard_name": "time", "long_name": "time", "axis": "T",
    "units": "hours since 1979-01-01", "calendar": "proleptic_gregorian",
}
MEAN_ATTRIBUTES = {"alignment": "index-aligned with /channel", "long_name": "Channel normalization mean"}
STD_ATTRIBUTES = {"alignment": "index-aligned with /channel", "long_name": "Channel normalization standard deviation"}
AUXILIARY_ATTRIBUTES = {
    "description": "Normalization statistics and static auxiliary fields",
    "static_variables": [value[0] for value in STATIC_METADATA.values()],
    "normalization_mean": "mean", "normalization_std": "std",
    "statistics_alignment": "mean[i] and std[i] correspond to channel[i]",
}


def root_attributes(
    dataset_id: str,
    data_revision: str,
    *,
    channels: Sequence[str] = DYNAMIC_CHANNELS,
    radiation_seconds: float = DEFAULT_RADIATION_SECONDS,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id, "schema_version": SCHEMA_VERSION,
        "content_version": CONTENT_VERSION, "data_revision": data_revision,
        "geospatial_lat_range": [-90.0, 90.0],
        "geospatial_lon_range": [0.0, 359.75],
        "channel_metadata": build_channel_metadata(
            channels, radiation_seconds=radiation_seconds
        ),
    }


def static_attributes(source_name: str) -> dict[str, Any]:
    _, long_name, units = STATIC_METADATA[source_name]
    return {
        "source_name": source_name, "long_name": long_name,
        "units": units, "coordinates": "lat lon",
    }


def consolidate_metadata(store: Path) -> None:
    root_path = store / "zarr.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    metadata: dict[str, dict[str, Any]] = {}
    for child_path in store.rglob("zarr.json"):
        if child_path == root_path:
            continue
        child = json.loads(child_path.read_text(encoding="utf-8"))
        child.pop("consolidated_metadata", None)
        metadata[child_path.parent.relative_to(store).as_posix()] = child
    if set(metadata) != set(EXPECTED_CHILDREN):
        raise RuntimeError(
            f"consolidated child content differs: expected {EXPECTED_CHILDREN}, got {tuple(metadata)}"
        )
    consolidated = {
        "kind": "inline", "must_understand": False,
        "metadata": {relative: metadata[relative] for relative in EXPECTED_CHILDREN},
    }
    root = {
        "attributes": root.get("attributes", {}),
        "zarr_format": root.get("zarr_format"),
        "consolidated_metadata": consolidated,
        "node_type": root.get("node_type"),
    }
    temporary = root_path.with_suffix(root_path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(root, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, root_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    for relative in EXPECTED_CHILDREN:
        child = json.loads((store / Path(relative) / "zarr.json").read_text(encoding="utf-8"))
        child.pop("consolidated_metadata", None)
        if consolidated["metadata"][relative] != child:
            raise RuntimeError(f"consolidated metadata differs for {relative}")

# ---------------------------------------------------------------------------
# Forward/reverse normalization shared by validation workflows
# ---------------------------------------------------------------------------


def normalize_values(
    values: np.ndarray, channel: str, mean: np.float32, std: np.float32
) -> np.ndarray:
    result = np.asarray(values, dtype="f4")
    if channel == "tp":
        return result
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"invalid standard deviation for {channel}: {std}")
    return (result - np.float32(mean)) / np.float32(std)


def denormalize_values(
    values: np.ndarray, channel: str, mean: np.float32, std: np.float32
) -> np.ndarray:
    """Reverse z-score; TP is unchanged because it is not z-score normalized."""

    result = np.asarray(values, dtype="f4")
    if channel == "tp":
        return result
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"invalid standard deviation for {channel}: {std}")
    return result * np.float32(std) + np.float32(mean)


@dataclass
class OpenSource:
    dataset: xr.Dataset
    original: xr.Dataset
    temporary: tempfile.TemporaryDirectory[str] | None

    def __enter__(self) -> xr.Dataset:
        return self.dataset

    def __exit__(self, *_: object) -> None:
        self.dataset.close()
        self.original.close()
        if self.temporary is not None:
            self.temporary.cleanup()


class Progress:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = total
        self.completed = 0
        self.started = time.perf_counter()
        self.last = 0.0
        self.show(True)

    def advance(self, amount: int = 1) -> None:
        self.completed += amount
        self.show(self.completed >= self.total)

    def show(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last < 1.0:
            return
        elapsed = now - self.started
        fraction = self.completed / self.total if self.total else 1.0
        rate = self.completed / elapsed if elapsed and self.completed else 0.0
        eta = (self.total - self.completed) / rate if rate else 0.0
        width = 30
        filled = round(width * fraction)
        print(
            f"[{self.label}] [{'#' * filled}{'-' * (width-filled)}] "
            f"{fraction:6.2%} {self.completed}/{self.total} "
            f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
            flush=True,
        )
        self.last = now


class Regridder:
    """Separable linear interpolation to the p25 grid; identity when grids match."""

    def __init__(self) -> None:
        self.cache: dict[tuple[bytes, bytes], tuple[np.ndarray, ...]] = {}

    @staticmethod
    def weights(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, ...]:
        upper = np.searchsorted(source, target, side="right")
        upper = np.clip(upper, 1, len(source) - 1)
        lower = upper - 1
        weight_upper = ((target - source[lower]) / (source[upper] - source[lower])).astype("f4")
        return lower, upper, 1.0 - weight_upper, weight_upper

    def apply(self, data: xr.DataArray, time_index: int | None = None) -> np.ndarray:
        if time_index is not None:
            data = data.isel(time=time_index)
        data = data.squeeze(drop=True).transpose("lat", "lon")
        source_lat = np.asarray(data.lat.values, dtype="f8")
        source_lon = np.mod(np.asarray(data.lon.values, dtype="f8"), 360.0)
        if (
            np.array_equal(source_lat.astype("f4"), TARGET_LAT)
            and np.array_equal(source_lon.astype("f4"), TARGET_LON)
        ):
            return np.asarray(data.values, dtype="f4")
        lat_order = np.argsort(source_lat)
        lon_order = np.argsort(source_lon)
        source_lat = source_lat[lat_order]
        source_lon = source_lon[lon_order]
        values = np.asarray(data.values, dtype="f4")[lat_order][:, lon_order]
        source_lon = np.concatenate((source_lon, source_lon[:1] + 360.0))
        values = np.concatenate((values, values[:, :1]), axis=1)
        key = (source_lat.tobytes(), source_lon.tobytes())
        weights = self.cache.get(key)
        if weights is None:
            weights = self.weights(source_lat, TARGET_LAT) + self.weights(source_lon, TARGET_LON)
            self.cache[key] = weights
        lat0, lat1, lat_w0, lat_w1, lon0, lon1, lon_w0, lon_w1 = weights
        along_lat = values[lat0] * lat_w0[:, None] + values[lat1] * lat_w1[:, None]
        result = along_lat[:, lon0] * lon_w0[None, :] + along_lat[:, lon1] * lon_w1[None, :]
        return result.astype("f4", copy=False)


def parse_day(path: Path) -> date:
    """Parse the date from one ``*.unit_converted.nc`` filename."""

    name = path.name
    for pattern in ("%Y.%m.%d.unit_converted.nc", "%Y%m%d.unit_converted.nc"):
        try:
            return datetime.strptime(name, pattern).date()
        except ValueError:
            pass
    raise ValueError(f"cannot parse converted day from {name}")


def parse_day_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def index_files(root: Path, variable: str) -> dict[date, Path]:
    directory = SOURCE_DIRECTORIES.get(variable, variable)
    paths = sorted(
        (root / GROUPS[variable] / directory).glob("*/*.unit_converted.nc")
    )
    if not paths:
        raise ValueError(
            f"no converted daily files found for {variable}; this stage accepts only "
            "*.unit_converted.nc produced by 2_convert_units_single_day.py"
        )
    result = {parse_day(path): path for path in paths}
    if len(result) != len(paths):
        raise ValueError(f"duplicate converted daily files for {variable}")
    return result


def validate_converted_units(dataset: xr.Dataset, variable: str, path: Path) -> None:
    """Reject raw or ambiguously preprocessed inputs before normalization."""

    units = str(dataset[variable].attrs.get("units", "")).strip()
    expected: set[str] | None = None
    if variable == "q":
        expected = {"g/kg", "g kg-1", "g kg**-1"}
    elif variable in RADIATION:
        expected = {"W m-2", "W m**-2", "W/m2", "W m^-2"}
    elif variable == "tp":
        expected = {"1", "dimensionless"}
    elif variable in DERIVED:
        expected = {"m s-1", "m s**-1", "m/s", "m s^-1"}
    if expected is not None and units not in expected:
        raise ValueError(
            f"{path}: {variable} units {units!r} are not converted; "
            "run 2_convert_units_single_day.py first"
        )


def open_source(path: Path, variable: str) -> OpenSource:
    temporary = None
    actual = path
    with path.open("rb") as stream:
        zipped = stream.read(4) == b"PK\x03\x04"
    if zipped:
        temporary = tempfile.TemporaryDirectory(prefix=f"era5_{variable}_")
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                temporary.cleanup()
                raise ValueError(f"{path} must contain exactly one archived file")
            archive.extract(members[0], temporary.name)
            actual = Path(temporary.name) / members[0]
    original = xr.open_dataset(actual, engine="netcdf4", cache=False)
    rename = {
        old: new
        for old, new in {
            "valid_time": "time", "latitude": "lat", "longitude": "lon",
            "pressure_level": "level",
        }.items()
        if old in original
    }
    candidates = SOURCE_VARIABLES.get(variable, (variable,))
    source_name = next((name for name in candidates if name in original), None)
    if source_name is None:
        original.close()
        if temporary is not None:
            temporary.cleanup()
        raise ValueError(f"{path} does not contain {variable}")
    if source_name != variable:
        rename[source_name] = variable
    dataset = original.rename(rename)
    try:
        validate_converted_units(dataset, variable, path)
    except BaseException:
        original.close()
        if temporary is not None:
            temporary.cleanup()
        raise
    return OpenSource(dataset, original, temporary)


def discover(
    root: Path, selected_day: date | None
) -> tuple[list[date], dict[str, dict[date, Path]], np.ndarray]:
    files = {
        variable: index_files(root, variable)
        for variable in NORMALIZED_INPUT_VARIABLES
    }
    if selected_day is not None:
        missing = [
            variable for variable, values in files.items()
            if selected_day not in values
        ]
        if missing:
            raise ValueError(
                f"day {selected_day.isoformat()} missing for: {', '.join(missing)}"
            )
        files = {
            variable: {selected_day: values[selected_day]}
            for variable, values in files.items()
        }
    days = sorted(files[PRESSURE_VARIABLES[0]])
    result_times = []
    previous = None
    required_levels = {**{name: set(BASE_LEVELS + (10,)) for name in PRESSURE_VARIABLES}, "w": set(W_LEVELS)}
    for day in days:
        reference = None
        for variable, daily in files.items():
            if set(daily) != set(days):
                raise ValueError(f"day coverage differs for {variable}")
            with open_source(daily[day], variable) as dataset:
                current = np.asarray(dataset.time.values).astype("datetime64[h]")
                if reference is None:
                    reference = current
                elif not np.array_equal(reference, current):
                    raise ValueError(f"time coordinate mismatch: {daily[day]}")
                if variable in required_levels:
                    available = set(np.asarray(dataset.level.values).astype(int).tolist())
                    if not required_levels[variable].issubset(available):
                        raise ValueError(f"pressure levels missing in {daily[day]}")
        assert reference is not None
        if np.any(np.diff(reference) != np.timedelta64(6, "h")):
            raise ValueError(f"non-six-hour interval on {day.isoformat()}")
        if previous is not None and reference[0] - previous != np.timedelta64(6, "h"):
            raise ValueError(f"time gap before {day.isoformat()}")
        previous = reference[-1]
        result_times.append(reference)
    return days, files, np.concatenate(result_times)


def validate_time_coverage(
    times: np.ndarray, allow_partial: bool
) -> None:
    if len(np.unique(times)) != len(times) or np.any(np.diff(times) != np.timedelta64(6, "h")):
        raise ValueError("time coordinate must be a unique continuous six-hour sequence")
    if allow_partial:
        return
    first_hour = int(str(times[0]).split("T")[1])
    last_hour = int(str(times[-1]).split("T")[1])
    if first_hour != 0 or last_hour != 18 or len(times) % 4 != 0:
        raise ValueError(
            "input must contain complete UTC days (00, 06, 12, 18); "
            "use --allow-partial only for an intentionally partial test"
        )


def dataset_label(times: np.ndarray, allow_partial: bool) -> str:
    start_day = str(times[0].astype("datetime64[D]")).replace("-", "")
    end_day = str(times[-1].astype("datetime64[D]")).replace("-", "")
    if allow_partial or start_day == end_day:
        return start_day if start_day == end_day else f"{start_day}-{end_day}"
    start_month, end_month = start_day[:6], end_day[:6]
    end_year, end_month_number = int(end_day[:4]), int(end_day[4:6])
    last_calendar_day = calendar.monthrange(end_year, end_month_number)[1]
    if start_day[6:] == "01" and int(end_day[6:]) == last_calendar_day:
        return start_month if start_month == end_month else f"{start_month}-{end_month}"
    return f"{start_day}-{end_day}"


def open_day(
    files: dict[str, dict[date, Path]],
    day: date,
    stack: ExitStack,
) -> dict[str, xr.Dataset]:
    return {
        variable: stack.enter_context(open_source(daily[day], variable))
        for variable, daily in files.items()
    }


def iter_block(
    datasets: dict[str, xr.Dataset],
    channels: tuple[str, ...],
    start: int,
    stop: int,
    regridder: Regridder,
):
    grouped: dict[str, list[tuple[int, str]]] = {}
    for index, channel in enumerate(channels):
        match = re.fullmatch(r"([a-z]+)(\d+)", channel)
        variable = match.group(1) if match and match.group(1) in datasets else channel
        grouped.setdefault(variable, []).append((index, channel))
    for variable, entries in grouped.items():
        slab = datasets[variable][variable].isel(time=slice(start, stop)).load()
        for index, channel in entries:
            match = re.fullmatch(r"([a-z]+)(\d+)", channel)
            field = slab.sel(level=int(match.group(2))) if "level" in slab.dims else slab
            for offset in range(stop - start):
                yield offset, index, channel, regridder.apply(field, offset)
        del slab


def required_files_for_channels(
    all_files: dict[str, dict[date, Path]], channels: tuple[str, ...]
) -> dict[str, dict[date, Path]]:
    required = set()
    for channel in channels:
        match = re.fullmatch(r"([a-z]+)(\d+)", channel)
        variable = match.group(1) if match and match.group(1) in all_files else channel
        required.add(variable)
    return {name: all_files[name] for name in all_files if name in required}


def read_statistic_file(path: Path, statistic: str) -> dict[str, np.float32]:
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = xr.open_dataset(path, engine="netcdf4")
    try:
        if "channel" not in dataset.coords:
            raise ValueError(f"{path}: channel must be a string coordinate")
        variable = statistic if statistic in dataset.data_vars else None
        if variable is None and len(dataset.data_vars) == 1:
            variable = next(iter(dataset.data_vars))
        if variable is None:
            raise ValueError(f"{path}: cannot identify {statistic} variable")
        channels = [str(value) for value in dataset.channel.values.tolist()]
        values = np.asarray(dataset[variable].values, dtype="f4").reshape(-1)
    finally:
        dataset.close()
    if len(channels) != len(values) or len(channels) != len(set(channels)):
        raise ValueError(f"{path}: invalid channel coordinate")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: contains non-finite {statistic}")
    return dict(zip(channels, values, strict=True))


def compute_selected_statistics(
    days: list[date],
    all_files: dict[str, dict[date, Path]],
    selected_channels: tuple[str, ...],
    times: np.ndarray,
    time_block: int,
) -> tuple[dict[str, np.float32], dict[str, np.float32]]:
    read_channels = tuple(selected_channels)
    files = required_files_for_channels(all_files, read_channels)
    selected_index = {name: index for index, name in enumerate(selected_channels)}
    sums = np.zeros(len(selected_channels), dtype="f8")
    sums_squared = np.zeros(len(selected_channels), dtype="f8")
    counts = np.zeros(len(selected_channels), dtype="i8")
    progress = Progress("additional-stats", len(times) * len(read_channels))
    regridder = Regridder()
    for day in days:
        with ExitStack() as stack:
            datasets = open_day(files, day, stack)
            size = datasets[next(iter(datasets))].sizes["time"]
            for start in range(0, size, time_block):
                stop = min(start + time_block, size)
                for _, _, channel, values in iter_block(
                    datasets, read_channels, start, stop, regridder
                ):
                    if channel in selected_index:
                        finite = np.isfinite(values)
                        chosen = values[finite].astype("f8")
                        index = selected_index[channel]
                        sums[index] += chosen.sum()
                        sums_squared[index] += np.square(chosen).sum()
                        counts[index] += chosen.size
                    progress.advance()
    if np.any(counts == 0):
        missing = [selected_channels[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"no finite values for computed channels: {missing}")
    means = sums / counts
    variances = np.maximum(sums_squared / counts - np.square(means), 0.0)
    stds = np.sqrt(variances)
    stds[stds == 0] = 1.0
    return (
        {name: np.float32(means[i]) for i, name in enumerate(selected_channels)},
        {name: np.float32(stds[i]) for i, name in enumerate(selected_channels)},
    )


def write_statistics_nc(
    path: Path,
    channels: tuple[str, ...],
    means: dict[str, np.float32],
    stds: dict[str, np.float32],
    period: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = xr.Dataset(
        data_vars={
            "mean": (("channel",), np.asarray([means[name] for name in channels], dtype="f4")),
            "std": (("channel",), np.asarray([stds[name] for name in channels], dtype="f4")),
        },
        coords={"channel": (("channel",), np.asarray(channels, dtype=str))},
        attrs={
            "statistics_period": period,
            "method": "nan-aware population mean/std; float64 accumulation",
            "preprocessing": "same as output dataset",
        },
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataset.to_netcdf(temporary, engine="netcdf4", format="NETCDF4")
    dataset.close()
    os.replace(temporary, path)


def assemble_statistics(
    args: argparse.Namespace,
    days: list[date],
    files: dict[str, dict[date, Path]],
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    supplied_mean = read_statistic_file(args.mean, "mean")
    supplied_std = read_statistic_file(args.std, "std")
    if set(supplied_mean) != set(supplied_std):
        raise ValueError("mean and std files have different channels")
    missing_reference = sorted(set(REFERENCE_CHANNELS) - set(supplied_mean))
    if missing_reference:
        raise ValueError(f"external C78 statistics missing: {missing_reference}")
    if any(supplied_std[name] <= 0 for name in supplied_std):
        raise ValueError("all supplied std values must be positive")

    final_mean: dict[str, np.float32] = {}
    final_std: dict[str, np.float32] = {}
    for channel in REFERENCE_CHANNELS:
        final_mean[channel] = np.float32(supplied_mean[channel])
        final_std[channel] = np.float32(supplied_std[channel])
    # TP stays in C78 but is intentionally not z-score normalized.
    final_mean["tp"] = np.float32(0.0)
    final_std["tp"] = np.float32(1.0)

    supplied_additional = [name for name in ADDITIONAL_CHANNELS if name in supplied_mean]
    for channel in supplied_additional:
        final_mean[channel] = np.float32(supplied_mean[channel])
        final_std[channel] = np.float32(supplied_std[channel])
    missing_additional = tuple(name for name in ADDITIONAL_CHANNELS if name not in final_mean)
    if missing_additional:
        if not args.compute_missing_additional_stats:
            raise ValueError(
                "statistics are missing additional channels: " + ", ".join(missing_additional)
                + "; provide them or use --compute-missing-additional-stats"
            )
        computed_mean, computed_std = compute_selected_statistics(
            days, files, missing_additional, times, args.time_block
        )
        final_mean.update(computed_mean)
        final_std.update(computed_std)
        if args.computed_stats_output:
            period = f"{times[0].astype('datetime64[D]')}/{times[-1].astype('datetime64[D]')}"
            write_statistics_nc(
                args.computed_stats_output, missing_additional, computed_mean, computed_std, period
            )

    mean = np.asarray([final_mean[name] for name in DYNAMIC_CHANNELS], dtype="f4")
    std = np.asarray([final_std[name] for name in DYNAMIC_CHANNELS], dtype="f4")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("assembled statistics are invalid")
    return mean, std


def codecs() -> list:
    return [BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")]


def create_store(
    path: Path,
    times: np.ndarray,
    dataset_id: str,
    mean: np.ndarray,
    std: np.ndarray,
    channel_chunk: int,
    radiation_seconds: float,
) -> tuple[zarr.Group, zarr.Array]:
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update(root_attributes(
        dataset_id,
        time.strftime("%Y%m%d"),
        radiation_seconds=radiation_seconds,
    ))
    data = group.create_array(
        "data",
        shape=(len(times), CHANNEL_COUNT, 721, 1440),
        chunks=(1, channel_chunk, 721, 1440),
        dtype="f2",
        fill_value=np.nan,
        compressors=codecs(),
        dimension_names=("time", "channel", "lat", "lon"),
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        attributes=DATA_ATTRIBUTES,
    )
    group.create_array(
        "channel", data=np.asarray(DYNAMIC_CHANNELS, dtype=str), chunks=(CHANNEL_COUNT,),
        fill_value="",
        dimension_names=("channel",),
        attributes=CHANNEL_ATTRIBUTES,
    )
    group.create_array(
        "lat", data=TARGET_LAT, chunks=(721,), fill_value=np.nan,
        dimension_names=("lat",),
        attributes=LAT_ATTRIBUTES,
    )
    group.create_array(
        "lon", data=TARGET_LON, chunks=(1440,), fill_value=np.nan,
        dimension_names=("lon",),
        attributes=LON_ATTRIBUTES,
    )
    encoded_time = (
        times.astype("datetime64[h]") - np.datetime64("1979-01-01T00", "h")
    ).astype("i8")
    time_array = group.create_array(
        "time", data=encoded_time, chunks=encoded_time.shape, fill_value=0,
        dimension_names=("time",)
    )
    time_array.attrs.update(TIME_ATTRIBUTES)
    auxiliary = group.create_group("auxiliary")
    auxiliary.attrs.update(AUXILIARY_ATTRIBUTES)
    auxiliary.create_array(
        "mean", data=mean, chunks=(CHANNEL_COUNT,), compressors=codecs(),
        fill_value=np.nan,
        dimension_names=("channel",),
        attributes=MEAN_ATTRIBUTES,
    )
    auxiliary.create_array(
        "std", data=std, chunks=(CHANNEL_COUNT,), compressors=codecs(),
        fill_value=np.nan,
        dimension_names=("channel",),
        attributes=STD_ATTRIBUTES,
    )
    return group, data


def add_static_fields(
    group: zarr.Group,
    input_root: Path,
    selected_day: date | None,
) -> None:
    auxiliary = group["auxiliary"]
    regridder = Regridder()
    for source_name in STATIC:
        daily = index_files(input_root, source_name)
        day = selected_day if selected_day is not None else min(daily)
        if day not in daily:
            raise ValueError(f"static source missing for {day}: {source_name}")
        with open_source(daily[day], source_name) as dataset:
            field = dataset[source_name]
            values = regridder.apply(field, 0 if "time" in field.dims else None)
        array_name, _, _ = STATIC_METADATA[source_name]
        auxiliary.create_array(
            array_name, data=values.astype("f4"), chunks=(721, 1440),
            compressors=codecs(), fill_value=np.nan, dimension_names=("lat", "lon"),
            attributes=static_attributes(source_name),
        )


def write_dynamic(
    store: zarr.Array,
    days: list[date],
    files: dict[str, dict[date, Path]],
    mean: np.ndarray,
    std: np.ndarray,
    total_steps: int,
    time_block: int,
) -> None:
    progress = Progress("write", total_steps * (CHANNEL_COUNT + 1))
    regridder = Regridder()
    global_time = 0
    for day in days:
        with ExitStack() as stack:
            datasets = open_day(files, day, stack)
            size = datasets[next(iter(datasets))].sizes["time"]
            for start in range(0, size, time_block):
                stop = min(start + time_block, size)
                buffer = np.empty(
                    (stop - start, CHANNEL_COUNT, 721, 1440), dtype="f2"
                )
                for offset, channel_index, channel, values in iter_block(
                    datasets, DYNAMIC_CHANNELS, start, stop, regridder
                ):
                    normalized = normalize_values(
                        values, channel, mean[channel_index], std[channel_index]
                    )
                    buffer[offset, channel_index] = normalized.astype("f2")
                    progress.advance()
                for offset in range(stop - start):
                    store[global_time] = buffer[offset]
                    global_time += 1
                    progress.advance()
                del buffer
    if global_time != total_steps:
        raise RuntimeError(f"wrote {global_time} time steps, expected {total_steps}")


def validate_coordinate_grid(latitude: np.ndarray, longitude: np.ndarray) -> None:
    """Require the canonical p25 grid, including ascending latitude order."""

    latitude = np.asarray(latitude, dtype="f4")
    longitude = np.asarray(longitude, dtype="f4")
    if latitude.shape != (721,):
        raise ValueError(f"latitude shape mismatch: {latitude.shape}")
    if not np.all(np.isfinite(latitude)):
        raise ValueError("latitude contains non-finite values")
    latitude_steps = np.diff(latitude)
    if not np.all(latitude_steps > 0):
        raise ValueError("latitude must be strictly increasing from -90 to 90")
    if not np.all(latitude_steps == np.float32(0.25)):
        raise ValueError("latitude step must be exactly +0.25 degrees")
    if not np.array_equal(latitude, TARGET_LAT):
        raise ValueError("latitude coordinate does not match -90 to 90")
    if not np.array_equal(longitude, TARGET_LON):
        raise ValueError("longitude coordinate grid mismatch")


def validate_output(
    path: Path,
    times: np.ndarray,
    channel_chunk: int,
    radiation_seconds: float,
) -> None:
    expected_members = {"data", "time", "channel", "lat", "lon", "auxiliary"}
    for consolidated in (False, True):
        group = zarr.open_group(
            str(path), mode="r", use_consolidated=consolidated
        )
        if set(group.keys()) != expected_members:
            raise ValueError(f"unexpected root members: {list(group.keys())}")
        if dict(group.attrs) != root_attributes(
            str(group.attrs["dataset_id"]),
            str(group.attrs["data_revision"]),
            radiation_seconds=radiation_seconds,
        ):
            raise ValueError(f"root metadata does not match the {CONTENT_VERSION} schema")
        data = group["data"]
        if data.shape != (len(times), CHANNEL_COUNT, 721, 1440):
            raise ValueError("data shape mismatch")
        if data.chunks != (1, channel_chunk, 721, 1440) or data.dtype != np.dtype("f2"):
            raise ValueError("data chunk or dtype mismatch")
        if dict(data.attrs) != DATA_ATTRIBUTES:
            raise ValueError("data attributes mismatch")
        if [str(value) for value in group["channel"][:].tolist()] != list(DYNAMIC_CHANNELS):
            raise ValueError("channel coordinate mismatch")
        if dict(group["channel"].attrs) != CHANNEL_ATTRIBUTES:
            raise ValueError("channel attributes mismatch")
        validate_coordinate_grid(group["lat"][:], group["lon"][:])
        if dict(group["lat"].attrs) != LAT_ATTRIBUTES or dict(group["lon"].attrs) != LON_ATTRIBUTES:
            raise ValueError("coordinate attributes mismatch")
        if dict(group["time"].attrs) != TIME_ATTRIBUTES:
            raise ValueError("time attributes mismatch")
        auxiliary = group["auxiliary"]
        if dict(auxiliary.attrs) != AUXILIARY_ATTRIBUTES:
            raise ValueError("auxiliary attributes mismatch")
        if auxiliary["mean"].shape != (CHANNEL_COUNT,) or auxiliary["std"].shape != (CHANNEL_COUNT,):
            raise ValueError("statistics shape mismatch")
        if dict(auxiliary["mean"].attrs) != MEAN_ATTRIBUTES or dict(auxiliary["std"].attrs) != STD_ATTRIBUTES:
            raise ValueError("statistics attributes mismatch")
        if not np.isfinite(auxiliary["mean"][:]).all() or not np.all(auxiliary["std"][:] > 0):
            raise ValueError("statistics values invalid")
        for source_name, (array_name, _, _) in STATIC_METADATA.items():
            if auxiliary[array_name].shape != (721, 1440):
                raise ValueError(f"static array invalid: {source_name}")
            if dict(auxiliary[array_name].attrs) != static_attributes(source_name):
                raise ValueError(f"static metadata invalid: {source_name}")

        dataset = xr.open_zarr(str(path), consolidated=consolidated)
        try:
            if dataset["data"].sel(channel="z500").shape != (len(times), 721, 1440):
                raise ValueError("xarray channel label selection failed")
        finally:
            dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="root of daily NC tree already produced by 2_convert_units_single_day.py",
    )
    parser.add_argument("--output", type=Path, required=True, help="output parent directory")
    parser.add_argument(
        "--mean", type=Path, default=SCRIPT_DIR / "mean.nc",
        help="mean NetCDF (default: mean.nc beside this script)",
    )
    parser.add_argument(
        "--std", type=Path, default=SCRIPT_DIR / "std.nc",
        help="std NetCDF (default: std.nc beside this script)",
    )
    parser.add_argument(
        "--date", type=parse_day_arg,
        help="optional YYYY-MM-DD selection (default: all complete days found)",
    )
    parser.add_argument("--allow-partial", action="store_true", help="allow an incomplete test sample")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--time-block", type=int, default=4, help="source steps buffered together")
    parser.add_argument(
        "--channel-chunk",
        type=int,
        default=CHANNEL_COUNT,
        help="channels per data chunk",
    )
    parser.add_argument(
        "--compute-missing-additional-stats", action="store_true",
        help="compute only missing non-C78 statistics from the input period",
    )
    parser.add_argument(
        "--computed-stats-output", type=Path,
        help="optional NC file for newly computed additional statistics",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path | None:
    if args.time_block < 1:
        raise ValueError("--time-block must be positive")
    if not 1 <= args.channel_chunk <= CHANNEL_COUNT:
        raise ValueError(
            f"--channel-chunk must be between 1 and {CHANNEL_COUNT}"
        )
    if len(DYNAMIC_CHANNELS) != CHANNEL_COUNT or len(REFERENCE_CHANNELS) != 78 or len(ADDITIONAL_CHANNELS) != 38:
        raise AssertionError("internal channel configuration is invalid")

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_root}")
    days, files, times = discover(input_root, args.date)
    validate_time_coverage(times, args.allow_partial)
    label = dataset_label(times, args.allow_partial)
    dataset_id = f"era5.{label}.c{CHANNEL_COUNT}.p25.h6.{CONTENT_VERSION}"
    final_path = output_root / f"{dataset_id}.zarr"
    print(f"source: {len(days)} day(s), {len(times)} six-hour steps")
    print(f"target: {final_path}")
    print(f"shape: ({len(times)}, {CHANNEL_COUNT}, 721, 1440)")
    print(f"chunk: (1, {args.channel_chunk}, 721, 1440)")
    print(
        "radiation: input and mean/std must already use W m-2 with divisor "
        f"{DEFAULT_RADIATION_SECONDS:g} s"
    )
    print(
        f"write buffer: {args.time_block * CHANNEL_COUNT * 721 * 1440 * 2 / 2**20:.0f} MiB "
        "plus source/codec memory"
    )
    mean, std = assemble_statistics(args, days, files, times)
    if args.dry_run:
        print("[DRY RUN] source, time, levels, channels and statistics are valid")
        return None
    if final_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {final_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / f".{dataset_id}.{uuid.uuid4().hex}.tmp"
    try:
        group, data = create_store(
            staging, times, dataset_id, mean, std,
            args.channel_chunk, DEFAULT_RADIATION_SECONDS,
        )
        add_static_fields(group, input_root, args.date)
        write_dynamic(
            data, days, files, mean, std, len(times), args.time_block,
        )
        consolidate_metadata(staging)
        validate_output(
            staging, times, args.channel_chunk, DEFAULT_RADIATION_SECONDS
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
