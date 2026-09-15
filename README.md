# ERA5 NC to Zarr

将 ERA5 NetCDF（`.nc`）转换为适合分析和机器学习使用的 Zarr v3 数据集。

本项目完成：

- ERA5 变量单位转换与数值预处理；
- 派生 10 m 和 100 m 风速；
- 使用 mean/std 进行归一化；
- 重网格到全球 `0.25°` 网格；
- 写入包含完整通道描述和 consolidated metadata 的 Zarr v3；
- 在发布前校验时间、层级、通道、坐标、统计量和元数据。

脚本2和脚本3的完整逐步说明见
[ERA5_RELEASE_CONVERTER_README.md](ERA5_RELEASE_CONVERTER_README.md)。该文件是两阶段转换规则的详细技术文档；本 README 只保留项目入口和常用命令。

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
单日 *.unit_converted.nc
  │
  ├─ 3_normalize_and_write_zarr.py
  ▼
归一化后的 Zarr v3 数据集
```

`1_extract_single_day.py` 是可选工具；如果已经有按日组织的原始 NetCDF，可以直接从脚本2开始。

## 关键处理规则

| 变量 | 脚本2执行的处理 | 脚本3执行的处理 |
| --- | --- | --- |
| `q` | `kg/kg × 1000 → g/kg` | z-score |
| `ssr/ssrd/fdir/ttr` | `J/m² ÷ 21600 s → W/m²` | z-score |
| `tp` | `m × 1000 → mm`，再执行 `log1p(max(tp_mm, 0))` | 保持不变，不做 z-score |
| `ws10m/ws100m` | 分别由对应 U/V 分量通过 `hypot` 派生 | z-score |
| 其他动态变量 | 原值复制 | z-score |

脚本3不会重复执行脚本2的单位转换。它会先检查关键输入单位，再进行重网格和归一化。

## 环境安装

要求 Python 3.12 或更高版本：

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r .\ERA5_converter_requirements.txt
```

## 快速开始

### 1. 抽取单日数据（可选）

```powershell
python .\1_extract_single_day.py `
  --source "E:\era5_monthly_nc" `
  --date 2025-01-01 `
  --output "E:\era5_2025.01.01_nc"
```

### 2. 单位转换与派生变量

```powershell
python .\2_convert_units_single_day.py `
  --date 2025-01-01 `
  --source "E:\era5_2025.01.01_nc" `
  --output "E:\era5_2025.01.01_unit_converted_nc" `
  --overwrite
```

### 3. 只读检查 Zarr 输入

```powershell
python .\3_normalize_and_write_zarr.py `
  --input "E:\era5_2025.01.01_unit_converted_nc" `
  --output "E:\era5_release_output" `
  --date 2025-01-01 `
  --dry-run
```

### 4. 正式写入 Zarr

```powershell
python .\3_normalize_and_write_zarr.py `
  --input "E:\era5_2025.01.01_unit_converted_nc" `
  --output "E:\era5_release_output" `
  --date 2025-01-01
```

## 输出约定

默认单日输出名称：

```text
era5.20250101.c116.p25.h6.v3.zarr
```

- 动态变量：116 个 channel；
- 数据维度：`(time, channel, lat, lon)`；
- 网格：`721 × 1440`，分辨率 `0.25°`；
- 纬度：`-90° → 90°`，严格递增，步长 `+0.25°`；
- 经度：`0° → 359.75°`；
- 时间间隔：6小时；
- 动态数据：`float16`；
- mean/std 和静态场：`float32`；
- 格式：Zarr v3，包含116通道元数据和 inline consolidated metadata。

## 测试

```powershell
python -m unittest discover -s .\tests -v
```

## 主要文件

```text
1_extract_single_day.py              # 可选：从月度数据抽取指定日期
2_convert_units_single_day.py        # 单位转换、TP预处理、派生风速
3_normalize_and_write_zarr.py        # 校验、统计量、重网格、归一化、Zarr发布
mean.nc                              # 归一化均值
std.nc                               # 归一化标准差
ERA5_converter_requirements.txt     # Python依赖
ERA5_RELEASE_CONVERTER_README.md    # 脚本2/3详细技术说明
```
