# ERA5 两阶段转换流程

当前流程明确拆成两个可独立运行的阶段：

```text
原始单日 NC
  -> 2_convert_units_single_day.py
  -> 单位转换后的单日 NC
  -> 3_normalize_and_write_zarr.py
  -> 归一化后的 Zarr v3
```

## 模块职责

- `2_convert_units_single_day.py`：只负责单位/数值预处理和派生风速，不归一化、不重网格、不写 Zarr。
- `3_normalize_and_write_zarr.py`：完整的第二阶段，只读取 `*.unit_converted.nc`，集中包含116个 channel 的 schema、源路径映射、正/反归一化函数、重网格、Zarr v3 属性、inline consolidated metadata、数据写入及最终校验。

## 最小交付清单

```text
2_convert_units_single_day.py
3_normalize_and_write_zarr.py
mean.nc
std.nc
ERA5_converter_requirements.txt
ERA5_RELEASE_CONVERTER_README.md
```

`mean.nc/std.nc` 必须与3号脚本放在同一目录；也可以在运行时通过 `--mean` 和 `--std` 显式指定其他位置。四个辐射统计量已经永久订正为 `/21600` 口径，不能再次除以6。

## 环境安装

要求 Python 3.12 或更高版本：

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r .\ERA5_converter_requirements.txt
```

## 第一阶段：单位转换

```powershell
python .\2_convert_units_single_day.py `
  --date 2025-01-01 `
  --source "E:\era5_2025.01.01_nc" `
  --output "E:\era5_2025.01.01_unit_converted_nc" `
  --overwrite
```

该阶段实施：

- `q`: `kg/kg * 1000 -> g/kg`
- `ssr/ssrd/fdir/ttr`: `J/m² / 21600 s -> W/m²`
- `tp`: `log1p(max(tp, 0))`，不做 z-score
- `ws10m/ws100m`: 分别由对应 U/V 分量通过 `hypot` 派生
- 其他文件原值复制，文件名统一为 `YYYY.MM.DD.unit_converted.nc`

## 第二阶段：归一化并写 Zarr

先进行只读校验：

```powershell
python .\3_normalize_and_write_zarr.py `
  --input "E:\era5_2025.01.01_unit_converted_nc" `
  --output "E:\era5_release_output" `
  --date 2025-01-01 `
  --dry-run
```

正式写入：

```powershell
python .\3_normalize_and_write_zarr.py `
  --input "E:\era5_2025.01.01_unit_converted_nc" `
  --output "E:\era5_release_output" `
  --date 2025-01-01
```

预期文件名：`era5.20250101.c116.p25.h6.v2.zarr`。

第二阶段不会再次执行 q、辐射、tp 或风速变换。它会校验 q、辐射、tp、ws 的单位/处理状态。项目中的 `mean.nc/std.nc` 已永久订正为 `/21600` 口径；第二阶段不会再缩放统计值，输入数据和统计量必须事先保持一致。

## Zarr 输出约定

- 网格：纬度 `90 -> -90` 共 721 点；经度 `0 -> 359.75` 共 1440 点
- 时间：6 小时间隔，完整日为 `00/06/12/18 UTC`
- `data`: `float16`，默认 chunk `(1, 116, 721, 1440)`
- `mean/std`: `float32`，位于 `auxiliary/mean` 和 `auxiliary/std`
- `channel`: 116 个字符串，可通过 `ds.data.sel(channel="z500")` 选择
- 根元数据包含有序的 `channel_metadata` 和 12 个子节点的 inline consolidated metadata
- 不写入 mean/std 来源绝对路径

## 测试

```powershell
python -m unittest discover -s .\tests -v
```

若要处理多日，先将每一天的单位转换结果写入同一个、保持相同子目录布局的输入根目录，再让第二阶段一次读取全部完整日。`--allow-partial` 只用于有意构造的不完整测试样本。
