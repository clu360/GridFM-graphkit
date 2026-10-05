# Stage K PowerWorld Weather Export

PowerWorld is required to decode the binary `.pww` weather values. The notebook
can render the overlays after a CSV/Parquet table exists at:

```text
data/stage_k_powerworld_weather_export.parquet
```

## Recommended PowerWorld Route

1. Open PowerWorld Simulator.
2. Load this auxiliary file:

```text
powerworld_export_stage_k_weather.aux
```

3. The script loads the one-day Texas geographic slice from:

```text
data/raw/2023-06-25_Texas_Heat_Dome_Texas.pww
```

4. It writes the raw PowerWorld CSV to:

```text
data/stage_k_powerworld_weather_input_raw.csv
```

5. Convert the raw CSV to the notebook schema:

```powershell
python experiments/test/wildfire_tests/stage_k_case_study/convert_powerworld_weather_export.py `
  --input experiments/test/wildfire_tests/stage_k_case_study/data/stage_k_powerworld_weather_input_raw.csv `
  --output experiments/test/wildfire_tests/stage_k_case_study/data/stage_k_powerworld_weather_export.parquet
```

The notebook will automatically prefer this real export over the modeled
preview weather table.

## Target Schema

The final notebook table should contain:

```text
lon, lat, timestamp, temperature, dew_point, wind_100m
```

PowerWorld fields used by the AUX export:

```text
WEATHERSTATION_TEMPF
WEATHERSTATION_DEWPOINTF
WEATHERSTATION_WINDSPEED100MS
```

The notebook aligns the selected one-day weather block to load scenarios
`0..23`.
