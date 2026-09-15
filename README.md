# ERA5 NC to Zarr

将原始 ERA5 NetCDF（`.nc`）数据转换为适合分析与机器学习使用的 Zarr v3 数据集。

本项目主要实现：

- ERA5 变量的单位转换与数值预处理；
- 使用预先计算的均值和标准差进行归一化；
- 将数据重网格到 `0.25°` 全球经纬度网格；
- 将动态变量、静态变量及统计量写入 Zarr v3；
- 校验时间、变量、层级、元数据和最终输出结构。

## 处理流程

```text
月度 ERA5 NetCDF（可选）
  │
  ├─ 1_extract_single_day.py
  ▼
单日原始 NetCDF
  │
  ├─ 2_convert_units_single_day.py
  ▼
单位转换后的 NetCDF
  │
  ├─ 3_normalize_and_write_zarr.py
  ▼
归一化后的 Zarr v3 数据集
```

`1_extract_single_day.py` 是可选预处理工具。如果已经拥有按日组织的 NetCDF 数据，可以直接从第二步开始。

## 主要转换规则

| 变量 | 处理方式 |
| --- | --- |
| `q` | `kg/kg × 1000 → g/kg` |
| `ssr`、`ssrd`、`fdir`、`ttr` | `J/m² ÷ 21600 s → W/m²` |
| `tp` | `m × 1000 → mm`，再执行 `log1p(max(tp_mm, 0))`，不做 z-score |
| `ws10m` | 由 `u10m`、`v10m` 计算风速 |
| `ws100m` | 由 `u100m`、`v100m` 计算风速 |
| 其他动态变量 | 使用 `mean.nc` 和 `std.nc` 归一化 |

## 环境要求

- Python 3.12 或更高版本
- NumPy
- Xarray
- netCDF4
- Zarr 3

在 PowerShell 中创建虚拟环境并安装依赖：

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r .\ERA5_converter_requirements.txt
```

## 快速开始

### 1. 从月度数据抽取单日数据（可选）

```powershell
python .\1_extract_single_day.py `
  --source "E:\era5_monthly_nc" `
  --date 2025-01-01 `
  --output "E:\era5_2025.01.01_nc"
```

### 2. 单位转换与派生变量计算

```powershell
python .\2_convert_units_single_day.py `
  --date 2025-01-01 `
  --source "E:\era5_2025.01.01_nc" `
  --output "E:\era5_2025.01.01_unit_converted_nc" `
  --overwrite
```

### 3. 归一化并写入 Zarr

建议先进行只读校验：

```powershell
python .\3_normalize_and_write_zarr.py `
  --input "E:\era5_2025.01.01_unit_converted_nc" `
  --output "E:\era5_release_output" `
  --date 2025-01-01 `
  --dry-run
```

校验通过后正式写入：

```powershell
python .\3_normalize_and_write_zarr.py `
  --input "E:\era5_2025.01.01_unit_converted_nc" `
  --output "E:\era5_release_output" `
  --date 2025-01-01
```

## 输出约定

默认输出文件名类似：

```text
era5.20250101.c116.p25.h6.v2.zarr
```

- 动态变量：116 个 channel；
- 空间网格：`721 × 1440`，分辨率 `0.25°`；
- 纬度：`90° → -90°`；
- 经度：`0° → 359.75°`；
- 时间间隔：6 小时；
- 动态数据类型：`float16`；
- 均值和标准差：`float32`，位于 `auxiliary/mean` 和 `auxiliary/std`；
- 输出格式：Zarr v3，包含 consolidated metadata。

## 项目文件

```text
1_extract_single_day.py            # 从月度数据抽取指定日期
2_convert_units_single_day.py      # 单位转换和派生风速
3_normalize_and_write_zarr.py      # 归一化、重网格、写入和校验 Zarr
mean.nc                            # 归一化均值
std.nc                             # 归一化标准差
ERA5_converter_requirements.txt   # Python 依赖
ERA5_RELEASE_CONVERTER_README.md  # 完整处理规则与发布说明
```

更多实现细节、输入约束和多日处理说明，请参阅 [ERA5_RELEASE_CONVERTER_README.md](ERA5_RELEASE_CONVERTER_README.md)。
