#!/usr/bin/env python3
"""Run the ERA5 extraction and conversion pipeline for an inclusive date range.

Each day is extracted into an isolated working directory, while all converted
daily files are accumulated in one range-specific tree.  Script 3 is invoked
once after every day is ready, producing one Zarr for the complete range.
Completion markers make interrupted runs resumable without trusting partial
daily output.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACT_SCRIPT = SCRIPT_DIR / "1_extract_single_day.py"
CONVERT_SCRIPT = SCRIPT_DIR / "2_convert_units_single_day.py"
ZARR_SCRIPT = SCRIPT_DIR / "3_normalize_and_write_zarr.py"
VALIDATE_SCRIPT = SCRIPT_DIR / "4_validate_zarr.py"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("era5_batch_pipeline", ZARR_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(ZARR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def display_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def script_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partitioned_day_root(source: Path, day: date) -> Path | None:
    for name in (f"{day:%Y.%m.%d}", day.isoformat(), f"{day:%Y%m%d}"):
        candidate = source / name
        if candidate.is_dir():
            return candidate
    return None


def source_for_day(source: Path, day: date, input_mode: str) -> Path:
    if input_mode == "monthly":
        return source
    partitioned = partitioned_day_root(source, day)
    return partitioned or source


def expected_daily_paths(
    root: Path,
    day: date,
    variables: Sequence[str],
    pipeline,
    converted: bool,
) -> list[Path]:
    suffix = ".unit_converted.nc" if converted else ".nc"
    filename = f"{day:%Y.%m.%d}{suffix}"
    return [
        root
        / pipeline.GROUPS[variable]
        / pipeline.SOURCE_DIRECTORIES.get(variable, variable)
        / str(day.year)
        / filename
        for variable in variables
    ]


def daily_tree_complete(
    root: Path,
    day: date,
    pipeline,
    converted: bool,
    include_static: bool = True,
) -> bool:
    dynamic = (
        pipeline.NORMALIZED_INPUT_VARIABLES
        if converted
        else pipeline.DIRECT_VARIABLES
    )
    required = tuple(dynamic) + (tuple(pipeline.STATIC) if include_static else ())
    return all(
        path.is_file()
        for path in expected_daily_paths(root, day, required, pipeline, converted)
    )


def remove_daily_extraction(path: Path, extracted_root: Path) -> None:
    resolved = path.resolve()
    safe_root = extracted_root.resolve()
    if resolved.parent != safe_root:
        raise ValueError(f"refusing to remove extraction outside {safe_root}: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)


def range_label(start: date, end: date) -> str:
    start_day = f"{start:%Y%m%d}"
    end_day = f"{end:%Y%m%d}"
    if start == end:
        return start_day
    if start.day == 1 and end.day == calendar.monthrange(end.year, end.month)[1]:
        start_month = f"{start:%Y%m}"
        end_month = f"{end:%Y%m}"
        return start_month if start_month == end_month else f"{start_month}-{end_month}"
    return f"{start_day}-{end_day}"


def run_command(command: list[str], plan: bool) -> None:
    print(f"[command] {display_command(command)}", flush=True)
    if not plan:
        subprocess.run(command, check=True)


def marker_matches(marker: Path, expected: dict[str, str]) -> bool:
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == expected
    except (OSError, json.JSONDecodeError):
        return False


def write_marker(marker: Path, payload: dict[str, str]) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True,
        help="monthly, shared-daily, or date-partitioned raw NC root",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "monthly", "daily"),
        default="auto",
        help="raw source layout; daily also supports date-partitioned roots",
    )
    parser.add_argument("--work", type=Path, required=True, help="batch working root")
    parser.add_argument("--output", type=Path, required=True, help="final Zarr parent")
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--mean", type=Path, default=SCRIPT_DIR / "mean.nc")
    parser.add_argument("--std", type=Path, default=SCRIPT_DIR / "std.nc")
    parser.add_argument("--time-block", type=int, default=4)
    parser.add_argument("--channel-chunk", type=int, default=116)
    parser.add_argument(
        "--force-days",
        action="store_true",
        help="rerun extraction and unit conversion even when completion markers exist",
    )
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="retain isolated raw daily copies after successful unit conversion",
    )
    parser.add_argument(
        "--overwrite-zarr",
        action="store_true",
        help="allow script 3 to replace an existing final Zarr",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="prepare all daily files and run script 3 with --dry-run",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=3,
        help="number of evenly spaced final Zarr samples (default: 3)",
    )
    parser.add_argument(
        "--skip-final-validation",
        action="store_true",
        help="skip script 4 after Zarr publication (not recommended)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print commands and paths without reading or writing data",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    days = dates_inclusive(args.start, args.end)
    source = args.source.resolve()
    work = args.work.resolve()
    output = args.output.resolve()
    mean = args.mean.resolve()
    std = args.std.resolve()
    if args.time_block < 1:
        raise ValueError("--time-block must be positive")
    if not 1 <= args.channel_chunk <= 116:
        raise ValueError("--channel-chunk must be between 1 and 116")
    if args.validation_samples < 0:
        raise ValueError("--validation-samples must be non-negative")
    if not args.plan:
        if not source.is_dir():
            raise FileNotFoundError(source)
        if not mean.is_file():
            raise FileNotFoundError(mean)
        if not std.is_file():
            raise FileNotFoundError(std)
    for script in (EXTRACT_SCRIPT, CONVERT_SCRIPT, ZARR_SCRIPT, VALIDATE_SCRIPT):
        if not script.is_file():
            raise FileNotFoundError(script)
    pipeline = load_pipeline()
    extract_digest = script_digest(EXTRACT_SCRIPT)
    convert_digest = script_digest(CONVERT_SCRIPT)

    batch_name = f"{args.start:%Y%m%d}_{args.end:%Y%m%d}"
    batch_root = work / batch_name
    extracted_root = batch_root / "extracted"
    converted_root = batch_root / "unit_converted"
    state_root = batch_root / "state"
    print(f"range:     {args.start} .. {args.end} ({len(days)} days)")
    print(f"source:    {source}")
    print(f"work:      {batch_root}")
    print(f"converted: {converted_root}")
    print(f"output:    {output}")

    started = time.perf_counter()
    for index, day in enumerate(days, start=1):
        day_text = day.isoformat()
        daily_extracted = extracted_root / f"{day:%Y.%m.%d}"
        daily_source = source_for_day(source, day, args.input_mode)
        include_static = index == 1
        needs_extraction = daily_source == source
        conversion_source = daily_extracted if needs_extraction else daily_source
        extract_marker = state_root / f"{day_text}.extract.json"
        convert_marker = state_root / f"{day_text}.convert.json"
        extract_state = {
            "stage": "extract",
            "date": day_text,
            "source": str(daily_source),
            "output": str(daily_extracted),
            "input_mode": args.input_mode,
            "include_static": str(include_static),
            "script_sha256": extract_digest,
        }
        convert_state = {
            "stage": "unit_conversion",
            "date": day_text,
            "source": str(conversion_source),
            "output": str(converted_root),
            "script_sha256": convert_digest,
            "include_static": str(include_static),
        }
        print(f"[day {index}/{len(days)}] {day_text}", flush=True)

        convert_complete = (
            marker_matches(convert_marker, convert_state)
            and daily_tree_complete(
                converted_root,
                day,
                pipeline,
                converted=True,
                include_static=include_static,
            )
        )
        if convert_complete and not args.force_days and not args.plan:
            print("[resume] converted daily files are complete", flush=True)
        else:
            if needs_extraction:
                extract_complete = (
                    marker_matches(extract_marker, extract_state)
                    and daily_tree_complete(
                        daily_extracted,
                        day,
                        pipeline,
                        converted=False,
                        include_static=include_static,
                    )
                )
                if extract_complete and not args.force_days and not args.plan:
                    print("[resume] extraction already complete", flush=True)
                else:
                    command = [
                        sys.executable,
                        str(EXTRACT_SCRIPT),
                        "--source",
                        str(daily_source),
                        "--date",
                        day_text,
                        "--input-mode",
                        args.input_mode,
                        "--output",
                        str(daily_extracted),
                        "--overwrite",
                    ]
                    if not include_static:
                        command.append("--skip-static")
                    run_command(command, args.plan)
                    if not args.plan:
                        write_marker(extract_marker, extract_state)
            else:
                print(f"[daily-input] using isolated source {daily_source}", flush=True)
            command = [
                sys.executable,
                str(CONVERT_SCRIPT),
                "--date",
                day_text,
                "--source",
                str(conversion_source),
                "--output",
                str(converted_root),
                "--overwrite",
            ]
            if not include_static:
                command.append("--skip-static")
            run_command(command, args.plan)
            if not args.plan:
                write_marker(convert_marker, convert_state)
                if needs_extraction and not args.keep_extracted:
                    remove_daily_extraction(daily_extracted, extracted_root)
                    print("[cleanup] removed isolated raw daily copy", flush=True)

        elapsed = time.perf_counter() - started
        rate = index / elapsed if elapsed else 0.0
        eta = (len(days) - index) / rate if rate else 0.0
        print(
            f"[batch] {index}/{len(days)} ({index / len(days):.2%}) "
            f"elapsed={elapsed / 60:.1f}m ETA={eta / 60:.1f}m",
            flush=True,
        )

    zarr_command = [
        sys.executable,
        str(ZARR_SCRIPT),
        "--input",
        str(converted_root),
        "--output",
        str(output),
        "--mean",
        str(mean),
        "--std",
        str(std),
        "--time-block",
        str(args.time_block),
        "--channel-chunk",
        str(args.channel_chunk),
    ]
    if args.validate_only:
        zarr_command.append("--dry-run")
    if args.overwrite_zarr:
        zarr_command.append("--overwrite")
    run_command(zarr_command, args.plan)
    final_zarr = output / (
        f"era5.{range_label(args.start, args.end)}.c{pipeline.CHANNEL_COUNT}."
        f"p25.h6.{pipeline.CONTENT_VERSION}.zarr"
    )
    if not args.validate_only and not args.skip_final_validation:
        validate_command = [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--zarr",
            str(final_zarr),
            "--sample-count",
            str(args.validation_samples),
        ]
        run_command(validate_command, args.plan)
    if args.plan:
        print("[PLAN] no files were changed")
    elif args.validate_only:
        print("[DONE] daily conversion complete; Zarr input validation passed")
    else:
        print(f"[DONE] batch conversion, Zarr publication and validation: {final_zarr}")


if __name__ == "__main__":
    run(parse_args())
