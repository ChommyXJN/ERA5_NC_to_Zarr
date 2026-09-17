#!/usr/bin/env python3
"""Materialize and validate one-day normalization/denormalization round trip."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import uuid
from contextlib import ExitStack
from datetime import date
from pathlib import Path

import numpy as np
import zarr


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = SCRIPT_DIR / "3_normalize_and_write_zarr.py"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("era5_zarr_pipeline", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_day(value: str) -> date:
    return date.fromisoformat(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="unit-converted NC root")
    parser.add_argument("--normalized", type=Path, required=True, help="normalized Zarr")
    parser.add_argument("--output", type=Path, required=True, help="validation output directory")
    parser.add_argument("--date", type=parse_day, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def create_field_store(
    path: Path,
    normalized: zarr.Group,
    role: str,
    pipeline,
) -> tuple[zarr.Group, zarr.Array]:
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.attrs.update(
        {
            "validation_role": role,
            "comparison_basis": (
                "unit-converted, preprocessed and target-grid float32 values"
            ),
            "source_dataset_id": str(normalized.attrs["dataset_id"]),
        }
    )
    source_data = normalized["data"]
    data = group.create_array(
        "data",
        shape=source_data.shape,
        chunks=(1, 1, 721, 1440),
        dtype="f4",
        fill_value=np.nan,
        compressors=pipeline.codecs(),
        dimension_names=("time", "channel", "lat", "lon"),
        attributes={
            "long_name": role,
            "data_representation": "preprocessed" if role == "original_preprocessed" else "denormalized",
            "_FillValue": "AAAAAAAA+H8=",
        },
    )
    for name in ("time", "channel", "lat", "lon"):
        source = normalized[name]
        values = np.asarray(source[:])
        fill_value = "" if values.dtype.kind in "US" else source.fill_value
        group.create_array(
            name,
            data=values,
            chunks=source.chunks,
            fill_value=fill_value,
            dimension_names=(name,),
            attributes=dict(source.attrs),
        )
    return group, data


def new_stats(channel: str) -> dict:
    return {
        "channel": channel,
        "total_count": 0,
        "finite_count": 0,
        "nan_count": 0,
        "nan_mask_mismatch_count": 0,
        "normalized_bit_mismatch_count": 0,
        "normalized_quantization_count": 0,
        "normalized_quantization_absolute_error_sum": 0.0,
        "normalized_quantization_squared_error_sum": 0.0,
        "normalized_quantization_max_absolute_error": 0.0,
        "inverse_formula_bit_mismatch_count": 0,
        "reconstructed_float32_exact_count": 0,
        "source_absolute_sum": 0.0,
        "source_squared_sum": 0.0,
        "signed_error_sum": 0.0,
        "absolute_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
        "max_error_location": None,
    }


def update_stats(
    stats: dict,
    source: np.ndarray,
    stored: np.ndarray,
    normalized_float32: np.ndarray,
    expected_normalized: np.ndarray,
    restored: np.ndarray,
    expected_restored: np.ndarray,
    time_index: int,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> None:
    source_nan = np.isnan(source)
    stored_nan = np.isnan(stored)
    expected_nan = np.isnan(expected_normalized)
    restored_nan = np.isnan(restored)
    stats["total_count"] += int(source.size)
    stats["nan_count"] += int(source_nan.sum())
    stats["nan_mask_mismatch_count"] += int(
        np.count_nonzero(source_nan != stored_nan)
        + np.count_nonzero(expected_nan != stored_nan)
        + np.count_nonzero(source_nan != restored_nan)
    )

    comparable_normalized = ~(expected_nan | stored_nan)
    normalized_bits_equal = (
        expected_normalized.view("u2") == stored.view("u2")
    )
    stats["normalized_bit_mismatch_count"] += int(
        np.count_nonzero(comparable_normalized & ~normalized_bits_equal)
    )
    normalized_finite = np.isfinite(normalized_float32) & np.isfinite(stored)
    normalized_error = np.abs(
        stored[normalized_finite].astype("f8")
        - normalized_float32[normalized_finite].astype("f8")
    )
    stats["normalized_quantization_count"] += int(normalized_error.size)
    stats["normalized_quantization_absolute_error_sum"] += float(
        normalized_error.sum(dtype="f8")
    )
    stats["normalized_quantization_squared_error_sum"] += float(
        np.square(normalized_error).sum(dtype="f8")
    )
    stats["normalized_quantization_max_absolute_error"] = max(
        stats["normalized_quantization_max_absolute_error"],
        float(normalized_error.max(initial=0.0)),
    )

    comparable_inverse = ~(np.isnan(expected_restored) | restored_nan)
    inverse_bits_equal = expected_restored.view("u4") == restored.view("u4")
    stats["inverse_formula_bit_mismatch_count"] += int(
        np.count_nonzero(comparable_inverse & ~inverse_bits_equal)
    )

    finite = np.isfinite(source) & np.isfinite(restored)
    count = int(finite.sum())
    stats["finite_count"] += count
    if not count:
        return
    source_finite = source[finite]
    restored_finite = restored[finite]
    exact = source_finite.view("u4") == restored_finite.view("u4")
    stats["reconstructed_float32_exact_count"] += int(exact.sum())
    error = np.abs(restored_finite.astype("f8") - source_finite.astype("f8"))
    signed_error = restored_finite.astype("f8") - source_finite.astype("f8")
    stats["source_absolute_sum"] += float(
        np.abs(source_finite.astype("f8")).sum(dtype="f8")
    )
    stats["source_squared_sum"] += float(
        np.square(source_finite.astype("f8")).sum(dtype="f8")
    )
    stats["signed_error_sum"] += float(signed_error.sum(dtype="f8"))
    stats["absolute_error_sum"] += float(error.sum(dtype="f8"))
    stats["squared_error_sum"] += float(np.square(error).sum(dtype="f8"))
    relative = error / np.maximum(np.abs(source_finite.astype("f8")), 1e-12)
    stats["max_relative_error"] = max(
        stats["max_relative_error"], float(relative.max(initial=0.0))
    )
    local_max = float(error.max(initial=0.0))
    if local_max > stats["max_absolute_error"]:
        error_grid = np.full(source.shape, -1.0, dtype="f8")
        error_grid[finite] = np.abs(restored[finite].astype("f8") - source[finite].astype("f8"))
        lat_index, lon_index = np.unravel_index(int(np.argmax(error_grid)), source.shape)
        stats["max_absolute_error"] = local_max
        stats["max_error_location"] = {
            "time_index": time_index,
            "lat_index": int(lat_index),
            "lon_index": int(lon_index),
            "lat": float(latitudes[lat_index]),
            "lon": float(longitudes[lon_index]),
            "source": float(source[lat_index, lon_index]),
            "restored": float(restored[lat_index, lon_index]),
        }


def finalize_stats(stats: dict) -> dict:
    result = dict(stats)
    finite = result["finite_count"]
    total = result["total_count"]
    result["normalized_bit_exact_count"] = (
        total - result["nan_count"] - result["normalized_bit_mismatch_count"]
    )
    denominator = max(total - result["nan_count"], 1)
    result["normalized_bit_exact_percent"] = (
        100.0 * result["normalized_bit_exact_count"] / denominator
    )
    result["inverse_formula_bit_exact_percent"] = (
        100.0 * (denominator - result["inverse_formula_bit_mismatch_count"])
        / denominator
    )
    result["reconstructed_float32_exact_percent"] = (
        100.0 * result["reconstructed_float32_exact_count"] / max(finite, 1)
    )
    quantization_count = result["normalized_quantization_count"]
    result["normalized_quantization_mean_absolute_error"] = (
        result.pop("normalized_quantization_absolute_error_sum")
        / max(quantization_count, 1)
    )
    result["normalized_quantization_rmse"] = math.sqrt(
        result.pop("normalized_quantization_squared_error_sum")
        / max(quantization_count, 1)
    )
    result["mean_absolute_error"] = result.pop("absolute_error_sum") / max(finite, 1)
    result["rmse"] = math.sqrt(result.pop("squared_error_sum") / max(finite, 1))
    mean_absolute_source = result.pop("source_absolute_sum") / max(finite, 1)
    rms_source = math.sqrt(result.pop("source_squared_sum") / max(finite, 1))
    result["mean_error_bias"] = result.pop("signed_error_sum") / max(finite, 1)
    result["mean_absolute_source"] = mean_absolute_source
    result["rms_source"] = rms_source
    result["nmae_percent"] = (
        100.0 * result["mean_absolute_error"] / max(mean_absolute_source, 1e-30)
    )
    result["nrmse_percent"] = (
        100.0 * result["rmse"] / max(rms_source, 1e-30)
    )
    result["bias_percent"] = (
        100.0 * result["mean_error_bias"] / max(mean_absolute_source, 1e-30)
    )
    return result


def write_reports(output: Path, report: dict) -> None:
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = report["channels"]
    fields = [
        "channel", "units", "total_count", "finite_count", "nan_count",
        "nan_mask_mismatch_count", "normalized_bit_mismatch_count",
        "normalized_bit_exact_percent", "inverse_formula_bit_mismatch_count",
        "inverse_formula_bit_exact_percent", "reconstructed_float32_exact_count",
        "reconstructed_float32_exact_percent", "max_absolute_error",
        "mean_absolute_error", "rmse", "mean_error_bias",
        "mean_absolute_source", "rms_source", "nmae_percent",
        "nrmse_percent", "bias_percent", "max_relative_error",
        "normalized_quantization_max_absolute_error",
        "normalized_quantization_mean_absolute_error",
        "normalized_quantization_rmse", "max_error_location",
    ]
    with (output / "validation_by_channel.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["max_error_location"] = json.dumps(
                item["max_error_location"], ensure_ascii=False
            )
            writer.writerow(item)
    summary = report["summary"]
    markdown = f"""# ERA5 单日归一化往返校验报告

- 日期：{report['date']}
- 比较元素总数：{summary['total_count']:,}
- 有限元素数：{summary['finite_count']:,}
- NaN 位置不一致数：{summary['nan_mask_mismatch_count']:,}
- 归一化 float16 bit 不一致数：{summary['normalized_bit_mismatch_count']:,}
- 归一化 float16 bit 一致率：{summary['normalized_bit_exact_percent']:.12f}%
- 反归一化公式 float32 bit 不一致数：{summary['inverse_formula_bit_mismatch_count']:,}
- 反归一化公式 float32 bit 一致率：{summary['inverse_formula_bit_exact_percent']:.12f}%
- 归一化空间最大 float16 量化误差：{summary['normalized_quantization_max_absolute_error']:.12g}
- 归一化空间平均 float16 量化误差：{summary['normalized_quantization_mean_absolute_error']:.12g}
- 归一化空间 float16 量化 RMSE：{summary['normalized_quantization_rmse']:.12g}
- 与归一化前 float32 完全一致率：{summary['reconstructed_float32_exact_percent']:.12f}%
- 最大绝对误差：{summary['max_absolute_error']:.12g}
- 平均绝对误差：{summary['mean_absolute_error']:.12g}
- RMSE：{summary['rmse']:.12g}

注意：最后三项聚合了不同物理单位，只用于定位实现问题。科学精度应查看
`validation_by_channel.csv` 中按 Channel 和单位分别统计的结果。float32 完全一致率
不作为通过条件，因为归一化结果保存为有损的 float16。

## 判定标准

1. 所有 time × channel × lat × lon 元素均参与，不抽样。
2. NaN 空间位置必须完全一致。
3. 单位转换后数据重新归一化并量化为 float16 后，必须与 Zarr 数据逐 bit 一致。
4. Zarr 数据按当前规则反归一化后，必须与相同 float16 输入直接计算的结果逐 bit 一致。
5. 反归一化数据与归一化前 float32 的误差属于 float16 量化误差，报告误差统计但不要求逐 bit 相等。
6. TP 的比较基准是 log1p(mm)，反归一化不执行 expm1 或单位反转换。

## 百分比指标定义

- `NMAE% = MAE / mean(abs(X)) × 100%`
- `NRMSE% = RMSE / RMS(X) × 100% = ||X' - X||₂ / ||X||₂ × 100%`
- `Bias% = mean(X' - X) / mean(abs(X)) × 100%`

这些指标在 `validation_by_channel.csv` 中按 Channel 提供。普通 MAPE 在变量接近0
或穿越0时会失真，因此不作为主要指标。

最终判定：**{report['status']}**
"""
    (output / "validation_report.md").write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    pipeline = load_pipeline()
    input_root = args.input.resolve()
    normalized_path = args.normalized.resolve()
    output = args.output.resolve()
    if not input_root.is_dir() or not normalized_path.is_dir():
        raise FileNotFoundError("input or normalized Zarr directory does not exist")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {output}")

    normalized = zarr.open_group(str(normalized_path), mode="r", use_consolidated=True)
    days, files, source_times = pipeline.discover(input_root, args.date)
    if days != [args.date]:
        raise ValueError("validation must contain exactly the requested day")
    stored_times = np.asarray(normalized["time"][:]).astype("datetime64[h]")
    if not np.array_equal(source_times.astype("datetime64[h]"), stored_times):
        raise ValueError("source and normalized time coordinates differ")
    channels = [str(value) for value in normalized["channel"][:].tolist()]
    if channels != list(pipeline.DYNAMIC_CHANNELS):
        raise ValueError("normalized channel coordinate differs from the pipeline")
    mean = np.asarray(normalized["mean"][:], dtype="f4")
    std = np.asarray(normalized["std"][:], dtype="f4")
    latitudes = np.asarray(normalized["lat"][:], dtype="f4")
    longitudes = np.asarray(normalized["lon"][:], dtype="f4")

    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        staging.mkdir(parents=True)
        normalized_output_parent = staging / "02_normalized"
        normalized_output_parent.mkdir()
        shutil.copytree(
            normalized_path, normalized_output_parent / normalized_path.name
        )
        original_group, original_data = create_field_store(
            staging / "01_original_preprocessed.zarr",
            normalized,
            "original_preprocessed",
            pipeline,
        )
        restored_group, restored_data = create_field_store(
            staging / "03_denormalized.zarr",
            normalized,
            "denormalized",
            pipeline,
        )
        stats = [new_stats(channel) for channel in channels]
        regridder = pipeline.Regridder()
        global_time = 0
        for day in days:
            with ExitStack() as stack:
                datasets = pipeline.open_day(files, day, stack)
                size = datasets[next(iter(datasets))].sizes["time"]
                for time_index in range(size):
                    for _, channel_index, channel, values in pipeline.iter_block(
                        datasets,
                        pipeline.DYNAMIC_CHANNELS,
                        time_index,
                        time_index + 1,
                        regridder,
                    ):
                        source = np.asarray(values, dtype="f4")
                        stored = np.asarray(
                            normalized["data"][global_time, channel_index], dtype="f2"
                        )
                        normalized_float32 = pipeline.normalize_values(
                            source, channel, mean[channel_index], std[channel_index]
                        )
                        expected_normalized = normalized_float32.astype("f2")
                        restored = pipeline.denormalize_values(
                            stored, channel, mean[channel_index], std[channel_index]
                        ).astype("f4")
                        expected_restored = pipeline.denormalize_values(
                            expected_normalized,
                            channel,
                            mean[channel_index],
                            std[channel_index],
                        ).astype("f4")
                        original_data[global_time, channel_index] = source
                        restored_data[global_time, channel_index] = restored
                        update_stats(
                            stats[channel_index], source, stored, normalized_float32,
                            expected_normalized,
                            restored, expected_restored, global_time,
                            latitudes, longitudes,
                        )
                    print(f"[roundtrip] {global_time + 1}/{len(source_times)}", flush=True)
                    global_time += 1
        if global_time != len(source_times):
            raise RuntimeError("written time count differs")
        zarr.consolidate_metadata(str(staging / "01_original_preprocessed.zarr"))
        zarr.consolidate_metadata(str(staging / "03_denormalized.zarr"))
        finalized = [finalize_stats(item) for item in stats]
        channel_info = dict(normalized["channel"].attrs["channel_info"])
        for item in finalized:
            item["units"] = channel_info[item["channel"]]["units"]
        totals = new_stats("ALL")
        for item in stats:
            for key in (
                "total_count", "finite_count", "nan_count", "nan_mask_mismatch_count",
                "normalized_bit_mismatch_count", "inverse_formula_bit_mismatch_count",
                "reconstructed_float32_exact_count", "absolute_error_sum", "squared_error_sum",
                "source_absolute_sum", "source_squared_sum", "signed_error_sum",
                "normalized_quantization_count",
                "normalized_quantization_absolute_error_sum",
                "normalized_quantization_squared_error_sum",
            ):
                totals[key] += item[key]
            totals["normalized_quantization_max_absolute_error"] = max(
                totals["normalized_quantization_max_absolute_error"],
                item["normalized_quantization_max_absolute_error"],
            )
            totals["max_relative_error"] = max(totals["max_relative_error"], item["max_relative_error"])
            if item["max_absolute_error"] > totals["max_absolute_error"]:
                totals["max_absolute_error"] = item["max_absolute_error"]
                totals["max_error_location"] = {
                    "channel": item["channel"], **(item["max_error_location"] or {})
                }
        summary = finalize_stats(totals)
        passed = (
            summary["nan_mask_mismatch_count"] == 0
            and summary["normalized_bit_mismatch_count"] == 0
            and summary["inverse_formula_bit_mismatch_count"] == 0
        )
        report = {
            "date": args.date.isoformat(),
            "status": "PASS" if passed else "FAIL",
            "comparison_basis": "unit-converted, preprocessed and target-grid float32",
            "normalized_dtype": "float16",
            "original_and_denormalized_dtype": "float32",
            "summary": summary,
            "channels": finalized,
        }
        write_reports(staging, report)
        if not passed:
            raise ValueError("round-trip validation failed; inspect staging report")
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"[PASS] {output}")
    return output


if __name__ == "__main__":
    run(parse_args())
