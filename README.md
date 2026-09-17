# ERA5 NC to Zarr

将 ERA5 NetCDF（`.nc`）转换为适合分析和机器学习使用的 Zarr v3
数据集。本项目固定输出116个动态通道，内容版本为 `v3`，纬度保持 ERA5
原始的北到南顺序 `90° → -90°`。

> `Zarr v3` 指存储格式；文件名中的 `.v3.zarr` 和根属性
> `content_version=v3` 指本项目的数据内容版本，两者含义不同。

## 处理流程

```text
多年/月度或逐日 ERA5 NetCDF
  │
  ├─ 2_convert_units_single_day.py（直接选日并转换）
  ▼
单日 *.unit_converted.nc 目录树
  │
  ├─ 3_normalize_and_write_zarr.py
  ▼
归一化、重网格后的 Zarr v3
  │
  ├─ 4_validate_zarr.py
  ▼
结构、元数据和通用数值健康校验
```

5_batch_convert.py 可按日期范围自动串联脚本2至4；脚本1保留为独立抽样和调试工具。

追求吞吐量且不需要保留逐日单位转换NC时，可使用
`7_direct_raw_to_zarr.py`。它在内存中完成与脚本1→2→3相同的日期选择、单位转换、
派生、归一化和float16写入，并允许多个日期并行直写互不重叠的Zarr时间块。

职责边界：

- 脚本1按需导出指定日期，保留原始变量、单位和目录结构，不进入批量主链路。
- 脚本2直接从月度、共享逐日或单日目录选择日期，执行物理单位转换、TP预处理和派生风速，不归一化、不重网格。
- 脚本3校验转换状态，进行重网格、归一化、Zarr写入和发布前校验。
- 脚本4独立复核最终Zarr的结构、元数据和抽样数值。
- 脚本5负责批量调度、断点续跑、工作区清理和最终校验。
- 脚本7是高速生产入口，不生成中间NC；脚本1至6仍作为可审计参考链路。

## 环境和必需文件

要求 Python 3.12 或更高版本：

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r .\ERA5_converter_requirements.txt
```

正式流程文件：

```text
1_extract_single_day.py
2_convert_units_single_day.py
3_normalize_and_write_zarr.py
4_validate_zarr.py
5_batch_convert.py
mean.nc
std.nc
ERA5_converter_requirements.txt
README.md
```

`mean.nc/std.nc` 默认与脚本3放在同一目录，也可以通过 `--mean` 和
`--std` 指定。辐射统计量必须使用 `/21600` 后的 `W/m²` 口径，脚本3不会再次
缩放统计值。

## 原始输入目录

脚本1、脚本2和脚本5支持三种布局。

多年/月度库：

```text
source/group/variable/year/*_YYYYM.nc
```

共享逐日目录树：

```text
source/group/variable/year/YYYY.MM.DD.nc
```

按日期分区的逐日目录树（每个日期目录与单日输入完全相同）：

```text
source/YYYY.MM.DD/group/variable/year/YYYY.MM.DD.nc
```

相关命令使用：

```text
--input-mode monthly
--input-mode daily
--input-mode auto
```

`auto` 优先使用日期分区或精确的当天文件，否则选择对应月文件。项目还会自动
识别实际为ZIP文件但扩展名仍为 `.nc` 的源文件。

## 脚本1：抽取单日原始数据

```powershell
python .\1_extract_single_day.py `
  --source "E:\era5_monthly_nc" `
  --input-mode monthly `
  --date 2025-01-01 `
  --output "E:\era5_2025.01.01_nc"
```

输出保持 `group/variable/year/file.nc` 布局，只保留当天的 `00、06、12、18 UTC`
记录，不执行单位转换、重网格或精度降低。静态场在原始库中只需保存一份；当目标
日期没有同月静态文件时，脚本1会读取该变量唯一的静态源文件。批处理通过
`--skip-static` 只在首日暂存静态场。

## 脚本2：单位转换与数值预处理

```powershell
python .\2_convert_units_single_day.py `
  --date 2025-01-01 `
  --source "E:\era5_monthly_nc" `
  --input-mode auto `
  --output "E:\era5_2025.01.01_unit_converted_nc" `
  --overwrite
```

脚本2在内存中只选择指定日期的记录并直接写出单位转换文件，不生成单日原始NC
副本；它也继续兼容脚本1生成的独立单日目录。

输出文件统一命名为：

```text
YYYY.MM.DD.unit_converted.nc
```

### 比湿 q

对所有气压层执行：

```text
kg/kg × 1000 → g/kg
```

### 六小时累计辐射

对 `ssr`、`ssrd`、`fdir`、`ttr` 执行：

```text
J/m² ÷ 21600 s → W/m²
```

其中 `21600` 秒为6小时累计窗口。

### 总降水 TP

原始单位必须为米，严格按以下顺序处理：

```text
m × 1000 → mm
clip_min(0)
log1p
```

等价公式：

```python
tp = np.log1p(np.maximum(tp_m * 1000.0, 0.0))
```

TP输出单位为 `1`，记录原始单位 `m`、中间单位 `mm` 和完整转换描述，并明确
不执行z-score。

### 派生风速

```text
ws10m  = hypot(u10m, v10m)
ws100m = hypot(u100m, v100m)
```

U/V坐标必须完全对齐。其他未列出的NetCDF原值复制。发生转换或派生的变量使用
float32和NetCDF4 zlib level 4；目标先写临时文件，成功后原子替换。

## 脚本3：归一化并写入Zarr

推荐先进行只读检查：

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

### 固定116通道

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

通道名称及顺序是固定产品协议，最终 `channel` 坐标必须完全一致。

### 输入检查

脚本3只接受脚本2生成的 `*.unit_converted.nc`，并检查：

- 每个必需变量和日期是否存在；
- 日期、时间和变量之间是否重复或缺失；
- 完整日期是否具有 `00/06/12/18 UTC` 四个时次；
- 多日数据是否保持连续6小时间隔；
- z/t/u/v/q/w 的必需气压层是否齐全；
- 所有变量的时间坐标是否一致；
- q、辐射、TP和派生风速是否已经具有转换后的单位。

读取时统一坐标名：

```text
valid_time     → time
latitude       → lat
longitude      → lon
pressure_level → level
```

### mean/std和归一化

- 78个参考通道必须由外部 `mean.nc/std.nc` 提供。
- 38个附加通道可以由统计文件提供。
- 缺少附加统计量时，可使用 `--compute-missing-additional-stats` 从输入期计算。
- 统计计算忽略NaN，使用float64累加，最终保存为float32。
- 所有标准差必须有限且大于0。
- TP强制使用 `mean=0、std=1`，不做z-score。
- 除TP外，动态通道执行 `(value - mean) / std`。
- 反归一化仅执行 `value * std + mean`；TP原样返回，不执行 `expm1`，所有通道
  都不在反归一化阶段进行单位反转换。

### 重网格和纬度约定

源纬度先排序，经度归一化到 `[0, 360)` 并处理周期边界，然后进行可分离线性
插值。目标网格为：

```text
lat: 90 → -90，721点，严格递减，步长 -0.25°
lon: 0 → 359.75，1440点，步长 +0.25°
```

当源网格完全一致时直接复用数值；动态变量和静态空间场采用相同约定。

### Zarr结构、精度和压缩

单日输出示例：

```text
era5.20250101.c116.p25.h6.v3.zarr
```

主要节点：

```text
data                                  (time, channel, lat, lon) float16
time                                  (time) datetime64[ns]
channel                               (channel) string
lat                                   (lat) float32
lon                                   (lon) float32
mean                                  (channel) float32
std                                   (channel) float32
mask                                  (mask_channel, lat, lon) uint8
mask_channel                          (mask_channel) string
```

`mask[0]` 为 `land_mask = (lsm > 0.5)`，`mask[1]` 为
`sea_mask = 1 - land_mask`。二者均为 `uint8` 二值数组；`mask_channel`
按顺序保存 `land_mask`、`sea_mask`，以后增加掩码时沿该维扩展。其他静态场、
const和纬度权重暂不发布到最终Zarr。

`data` 默认chunk为 `(1, 116, 721, 1440)`，可通过 `--channel-chunk` 调整。
压缩使用Blosc Zstandard level 5和bitshuffle。输入默认按4个时间步缓存，可通过
`--time-block` 调整。

根 `zarr.json` 包含：

- `dataset_id`、`schema_version=2.0`、`content_version=v3`、`data_revision`；
- 经纬度覆盖范围；
- `/channel` 属性中的 `channel_info` 保存116项通道语义，主数据顺序仅由
  `/channel[:]` 决定；两者的键集合必须完全一致；
- 每个通道保存source、units、level类型、variable类型和完整preprocessing；
- 每个通道同时保存 `scale_factor=std` 和 `add_offset=mean`，用于
  `preprocessed_value = normalized_value * scale_factor + add_offset`；根目录
  `/std`、`/mean` 数组继续保留，校验器要求两种表示逐Channel完全一致；
- 9个根数组的inline consolidated metadata。

所有内容先写入隐藏staging目录。数据、元数据和发布前校验全部成功后才原子发布；
已有输出只有指定 `--overwrite` 才会替换。

## 脚本4：独立校验最终Zarr

```powershell
python .\4_validate_zarr.py `
  --zarr "E:\era5_release_output\era5.202501.c116.p25.h6.v3.zarr" `
  --sample-count 3
```

默认校验：

- 时间唯一、连续、每6小时一次且覆盖完整UTC日；
- consolidated和non-consolidated两种读取方式；
- 根属性、116通道元数据和11个元数据节点；
- Zarr目录名与根属性 `dataset_id` 一致；
- data的shape、dtype和chunks；
- 纬度严格 `90→-90`、步长 `-0.25°`；
- 经度、时间、channel、根目录mean/std和mask注册表；
- land/sea比例范围、有限性和严格互补关系；
- 均匀抽取首、中、末等时间步读取 `z500/t2m/q500/swh`；
- 抽样通道包含有限值，全部抽样数据不含无穷值。

`--channels` 可以指定需要报告的任意通道。`--full-scan` 会读取每个时间步和全部
116通道，其耗时和读取量接近完整扫描。

## 脚本5：批量处理完整日期范围

先打印执行计划，不读写数据：

```powershell
python .\5_batch_convert.py `
  --source "E:\era5_monthly_nc" `
  --input-mode auto `
  --work "E:\era5_batch_work" `
  --output "E:\era5_release_output" `
  --start 2025-01-01 `
  --end 2025-01-31 `
  --plan
```

确认后删除 `--plan`。处理过程为：

```text
首日：脚本2直接选择当天动态数据和唯一静态场并完成单位转换
后续每天：脚本2直接选择当天动态数据并完成单位转换
全部日期：脚本3生成一个完整Zarr → 脚本4独立校验
```

月度库、共享逐日树和日期分区输入都直接交给脚本2。月度文件只读取所需日期，
不再在工作区形成单日原始副本。

工作目录：

```text
work/YYYYMMDD_YYYYMMDD/
├─ unit_converted/
│  └─ group/variable/year/YYYY.MM.DD.unit_converted.nc
└─ state/
   └─ YYYY-MM-DD.convert.json
```

断点续跑规则：

- 每个阶段成功后才原子写入完成标记；
- 同时检查标记内容、脚本SHA-256和当天全部必需文件；
- 文件缺失、标记不匹配或脚本变化时自动重做对应日期；
- `--force-days` 强制重做每日单位转换；
- `--overwrite-zarr` 允许脚本3替换同名最终Zarr；
- `--validate-only` 只准备日文件并运行脚本3的 `--dry-run`；
- `--skip-final-validation` 完全跳过脚本4，不推荐；
- `--time-block` 和 `--channel-chunk` 会传递给脚本3。

批量流程不会修改或删除 `--source` 中的用户原始数据。全部逐日单位转换文件必须
保留到脚本3完成，因此仍需为
`unit_converted` 预留足够空间。4个静态输入场只在批次首日保存一次；当前v3产物
仅使用其中的 `lsm` 生成land/sea比例mask，其他静态场暂不发布。

## 脚本7：原始NC直接并行写入Zarr

脚本7适用于只需要最终Zarr、不需要保留逐日中间NC的生产任务：

```powershell
python .\7_direct_raw_to_zarr.py `
  --source "E:\era5_2025.01-2026.07_nc" `
  --input-mode auto `
  --output "E:\era5_release_output" `
  --start 2025-01-01 `
  --end 2026-07-31 `
  --workers 8
```

建议先用 `--dry-run` 检查首尾日期文件和mean/std，再正式运行。每个worker在内存中
直接完成一天4个时次的单位转换、派生、重网格、归一化和float16转换，并写入独立
的Zarr时间块，因此不会创建 `extracted` 或 `unit_converted` 目录。单worker约需
1 GiB输出缓冲区，实际内存还包括当天原始变量；增加worker前应同时考虑内存、源
存储并发读取和目标存储写入能力。

脚本7复用脚本3的通道协议、统计量、元数据、压缩和发布前结构检查。它不执行脚本4
的独立抽样校验；需要时可在产物完成后单独运行脚本4。

## 测试

```powershell
python -m unittest discover -s .\tests -v
```

仓库不包含 ERA5 测试数据。默认命令只运行不依赖外部数据的路径选择、日期范围、
元数据错误定位、抽样索引、纬度顺序和内容版本等轻量测试；外部数据集成测试在
没有本地配置时显示为 skipped。

### 使用同一批数据进行集成测试

复制路径配置示例并按本机实际位置修改：

```powershell
Copy-Item .\tests\integration_paths.example.json .\tests\integration_paths.json
```

配置文件包含五项：

```json
{
  "date": "2025-01-01",
  "source": "E:\\era5_2025.01-2026.07_nc",
  "extracted_day": "E:\\era5_2025.01.01_nc",
  "unit_converted_day": "E:\\era5_2025.01.01_unit_converted_nc",
  "zarr": "E:\\era5_release_output\\era5.20250101.c116.p25.h6.v3.zarr"
}
```

`source` 可以是两年或其他跨度的原下载数据，但其余三个阶段必须包含 `date`
指定的同一天。测试会从源归档检查全部必需输入，确认单日切取和单位转换目录均
完整覆盖该日的4个时次，再确认Zarr包含同一天、全部116通道并通过脚本4校验。
这样四个路径代表同一批数据从原始输入到最终Zarr的完整流程。

运行集成测试：

```powershell
python -m unittest .\tests\test_full_data_integration.py -v
```

`tests/integration_paths.json` 已加入 `.gitignore`。也可以把配置放在任意位置，并用
`ERA5_TEST_CONFIG` 指向它。
