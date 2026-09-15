# Real ERA5 test fixtures

These NetCDF files are small coordinate subsets of the user's real ERA5 data.
They contain real values and metadata, but only the following samples:

- times: 2025-01-01 00/06/12/18 UTC (all monthly fixtures also include the
  four 2025-01-02 time steps so date slicing can be tested);
- latitude indices: 0, 360, 720 (`90`, `0`, `-90` degrees);
- longitude indices: 0, 1, 720, 1439;
- all source pressure levels needed by the 116-channel schema.

Source trees:

```text
E:\era5_2025.01-2026.07_nc
E:\era5_2025.01.01_nc
E:\era5_2025.01.01_unit_converted_nc
```

Included data:

```text
monthly:   all 40 source variables for 2025-01-01 and 2025-01-02
raw:       all 40 source variable files for 2025-01-01
converted: all 42 files, including ws10m and ws100m, for 2025-01-01
reference: raw_truth, unit_converted, and normalized stacked c116 NetCDF files
```

The three reference files retain all 116 channels and all four daily time
steps. This allows portable tests to verify channel order, TP preprocessing,
and normalization for every channel while keeping only a few spatial points.

Regenerate the fixtures from the full local samples with:

```powershell
python .\tests\build_real_data_fixtures.py --overwrite
```

The fixture builder never modifies the source trees. The cropped files are
small enough for normal Git storage and let CI verify transformations using
real ERA5 values without requiring the full external archive.
