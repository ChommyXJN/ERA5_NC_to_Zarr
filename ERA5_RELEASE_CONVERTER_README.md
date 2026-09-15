# ERA5 两阶段转换技术说明

本文档详细说明 `2_convert_units_single_day.py` 和
`3_normalize_and_write_zarr.py` 的实际执行步骤、输入约束和输出规范。

## 总体流程

```text
单日原始 NC
  ↓
2_convert_units_single_day.py
  ↓
单日 *.unit_converted.nc
  ↓
3_normalize_and_write_zarr.py
  ↓
归一化、重网格后的 Zarr v3
  ↓
合并元数据、完整校验、正式发布
```

职责边界：

- 脚本2负责物理单位转换、TP数值预处理和派生风速，不归一化、不重网格、不写 Zarr。
- 脚本3只读取脚本2生成的 `*.unit_converted.nc`，负责输入校验、统计量、重网格、归一化、Zarr写入、元数据和最终校验。
- 脚本3的 `channel_metadata.preprocess` 记录从原始 ERA5 到最终产品的完整处理链，但不会重复执行脚本2已经完成的单位转换。

## 环境与必需文件

要求 Python 3.12 或更高版本：

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r .\ERA5_converter_requirements.txt
```

最小交付清单：

```text
2_convert_units_single_day.py
3_normalize_and_write_zarr.py
mean.nc
std.nc
ERA5_converter_requirements.txt
ERA5_RELEASE_CONVERTER_README.md
```

`mean.nc/std.nc` 默认与脚本3放在同一目录，也可以通过 `--mean` 和
`--std` 指定。辐射统计量必须已经使用 `/21600` 后的 `W/m²` 口径，脚本3不会再次缩放统计值。

## 阶段一：单位转换与数值预处理

运行示例：

```powershell
python .\2_convert_units_single_day.py `
  --date 2025-01-01 `
  --source "E:\era5_2025.01.01_nc" `
  --output "E:\era5_2025.01.01_unit_converted_nc" `
  --overwrite
```

### 1. 搜索和组织输入文件

脚本递归查找输入目录中的 `.nc` 文件，要求文件位于类似
`group/variable/year/file.nc` 的目录结构。输出保留变量目录结构，并统一使用：

```text
YYYY.MM.DD.unit_converted.nc
```

### 2. 比湿 q

对 q 文件中的全部气压层执行：

```text
kg/kg × 1000 → g/kg
```

转换后写入 `units`、`original_units`、`unit_conversion` 和全局
`processing_rule`。

### 3. 六小时累计辐射

对 `ssr`、`ssrd`、`fdir`、`ttr` 执行：

```text
J/m² ÷ 21600 s → W/m²
```

其中 `21600` 秒为6小时累计窗口，并写入
`accumulation_window_seconds=21600`。

### 4. 总降水 TP

脚本先检查原始单位必须为米，然后严格按顺序执行：

```text
m × 1000 → mm
clip_min(0)
log1p
```

等价公式：

```python
tp = np.log1p(np.maximum(tp_m * 1000.0, 0.0))
```

TP 输出单位为 `1`，记录中间单位 `mm`，并明确
`normalization_applied=false`。

### 5. 派生风速

```text
ws10m  = hypot(u10m, v10m)
ws100m = hypot(u100m, v100m)
```

U/V 坐标必须完全对齐，否则拒绝派生。

### 6. 其他变量

未列入上述规则的 NetCDF 原值复制，不做单位转换、归一化、重网格或精度降低。

### 7. NetCDF写入安全

发生转换或派生的变量以 `float32` 写入 NetCDF4，并使用 zlib level 4 和 shuffle。每个目标先写临时文件，成功后通过原子重命名发布；已有输出只有指定 `--overwrite` 才能替换。

## 阶段二：归一化并写入 Zarr

推荐先执行只读检查：

```powershell
python .\3_normalize_and_write_zarr.py `
  --input "E:\era5_2025.01.01_unit_converted_nc" `
  --output "E:\era5_release_output" `
  --date 2025-01-01 `
  --dry-run
```

检查通过后正式写入：

```powershell
python .\3_normalize_and_write_zarr.py `
  --input "E:\era5_2025.01.01_unit_converted_nc" `
  --output "E:\era5_release_output" `
  --date 2025-01-01
```

### 1. 固定116通道 schema

| 通道类别 | 数量 |
| --- | ---: |
| z、t、u、v、q 的13个标准气压层 | 65 |
| 地表变量 | 13 |
| z、t、u、v、q 的10 hPa层 | 5 |
| w 的14个气压层 | 14 |
| ws10m、ws100m | 2 |
| 云量和辐射 | 8 |
| 土壤变量 | 4 |
| 海浪变量 | 5 |
| 合计 | 116 |

通道名称和顺序是固定产品协议，最终 Zarr 的 `channel` 坐标必须完全一致。

### 2. 输入发现与一致性检查

脚本只接受 `*.unit_converted.nc`，并检查：

- 每个必需变量和日期是否存在；
- 日期、时间和变量之间是否重复或缺失；
- 完整日期是否具有 `00/06/12/18 UTC` 四个时间步；
- 多日数据是否连续保持6小时间隔；
- z/t/u/v/q/w 的必需气压层是否齐全；
- 所有变量的时间坐标是否一致。

`--allow-partial` 只应用于有意构造的不完整测试样本。

### 3. 坐标和变量名统一

读取时统一：

```text
valid_time     → time
latitude       → lat
longitude      → lon
pressure_level → level
```

同时兼容 `u10/u10m`、`v10/v10m`、`u100/u100m`、`v100/v100m` 等源名称。

### 4. 单位状态检查

在归一化前要求：

| 变量 | 接受的转换后单位 |
| --- | --- |
| q | `g/kg` 等等价写法 |
| ssr、ssrd、fdir、ttr | `W m-2` 等等价写法 |
| tp | `1` 或 `dimensionless` |
| ws10m、ws100m | `m s-1` 等等价写法 |

不满足时会拒绝继续，并提示先运行脚本2。

### 5. mean/std 装配

- 78个参考通道必须由外部 `mean.nc/std.nc` 提供。
- 38个附加通道可以由统计文件提供。
- 缺少附加统计量时，可以通过 `--compute-missing-additional-stats` 从输入期计算。
- 计算使用忽略 NaN 的总体 mean/std 和 float64 累加，最终存为 float32。
- 所有标准差必须有限且大于0。
- TP 不做 z-score，强制使用 `mean=0`、`std=1`。

可通过 `--computed-stats-output` 保存新计算的附加统计量。

### 6. 重网格

源纬度先排序，源经度归一化到 `[0, 360)` 并处理周期边界，再进行可分离线性插值。目标网格为：

```text
lat: 90 → -90，721点，严格递减，步长 -0.25°
lon: 0 → 359.75，1440点，步长 +0.25°
```

当源网格已经完全一致时直接复用数值。动态变量与所有静态空间场使用相同网格约定。

### 7. 归一化和精度

除 TP 外的动态通道执行：

```python
normalized = (value - mean) / std
```

TP 直接保留阶段一的 `m → mm → clip_min → log1p` 结果。动态数据最终写为
`float16`；mean/std 与静态空间场写为 `float32`。

### 8. Zarr结构与压缩

输出内容版本为 `v2`，单日名称示例：

```text
era5.20250101.c116.p25.h6.v2.zarr
```

主要节点：

```text
data                                  (time, channel, lat, lon) float16
time                                  (time)
channel                               (channel)
lat                                   (lat)
lon                                   (lon)
auxiliary/mean                        (channel) float32
auxiliary/std                         (channel) float32
auxiliary/land_sea_mask               (lat, lon) float32
auxiliary/slope_of_sub_gridscale_orography
auxiliary/standard_deviation_of_orography
auxiliary/surface_geopotential
```

`data` 默认 chunk 为 `(1, 116, 721, 1440)`，可以用 `--channel-chunk`
调整通道分块。压缩采用 Blosc Zstandard、level 5、bitshuffle。

### 9. 分块读取和进度

输入默认按4个时间步缓存，可通过 `--time-block` 调整。脚本在统计量计算和动态数据写入时输出完成比例、已用时间和 ETA。

### 10. 元数据

根 `zarr.json` 包含：

- `dataset_id`、`schema_version`、`content_version=v2`、`data_revision`；
- 经纬度覆盖范围；
- 按 channel 顺序保存的116项 `channel_metadata`；
- 每个通道的 variable、level、units、long_name 和完整 preprocess；
- 12个子节点的 inline consolidated metadata。

TP 的完整预处理描述为 `m → mm → clip_min(0) → log1p`，且不包含 z-score。

### 11. 发布前验证

脚本同时以 consolidated 和 non-consolidated 模式验证：

- 根节点、数组形状、dtype 和 chunks；
- 116个通道名称及顺序；
- 纬度严格 `90 → -90` 且步长严格为 `-0.25°`；
- 经度、时间编码和坐标属性；
- mean/std 的形状、有限性及正标准差；
- 静态场形状和属性；
- xarray 能否执行 `data.sel(channel="z500")`。

### 12. staging与正式发布

所有内容先写入隐藏 staging 目录。只有数据写入、consolidated metadata 和最终校验全部成功后，才原子重命名为正式输出；失败时清理 staging。已有输出只有指定 `--overwrite` 才会被替换。

## 多日处理

先将每一天的脚本2输出写入保持相同子目录布局的输入根目录，再让脚本3一次发现并处理全部完整日期。不传 `--date` 时，脚本3默认处理找到的完整连续时间范围。

## 测试

```powershell
python -m unittest discover -s .\tests -v
```
