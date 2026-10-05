"""Normalize a PowerWorld timestep weather CSV export for the Stage K notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


VARIABLE_ALIASES = {
    "temperature": ["weatherstation_tempf", "tempf", "temperature", "temperature_f", "temp_f"],
    "dew_point": ["weatherstation_dewpointf", "dewpointf", "dew_point", "dewpoint", "dew_point_f"],
    "wind_100m": [
        "weatherstation_windspeed100ms",
        "windspeed100ms",
        "wind_100m",
        "wind_speed_100m",
        "ws100",
    ],
}


def _clean_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    cleaned = {_clean_name(col): col for col in columns}
    for alias in aliases:
        if _clean_name(alias) in cleaned:
            return cleaned[_clean_name(alias)]
    for clean, original in cleaned.items():
        if any(_clean_name(alias) in clean for alias in aliases):
            return original
    return None


def _parse_lat_lon_from_station(value: object) -> tuple[float | None, float | None]:
    text = str(value)
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*[,;/ ]\s*([+-]?\d+(?:\.\d+)?)", text)
    if not match:
        return None, None
    a = float(match.group(1))
    b = float(match.group(2))
    if -90 <= a <= 90 and -180 <= b <= 180:
        return a, b
    if -90 <= b <= 90 and -180 <= a <= 180:
        return b, a
    return None, None


def normalize_powerworld_weather(input_csv: Path, output_parquet: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_csv)
    columns = list(raw.columns)
    rename = {}

    lon_col = _find_column(columns, ["lon", "longitude", "x"])
    lat_col = _find_column(columns, ["lat", "latitude", "y"])
    time_col = _find_column(columns, ["timestamp", "datetime", "date_time", "time", "date", "hour"])

    if lon_col:
        rename[lon_col] = "lon"
    if lat_col:
        rename[lat_col] = "lat"
    if time_col:
        rename[time_col] = "timestamp"

    for target, aliases in VARIABLE_ALIASES.items():
        col = _find_column(columns, aliases)
        if col:
            rename[col] = target

    out = raw.rename(columns=rename).copy()

    if "lon" not in out or "lat" not in out:
        station_col = _find_column(columns, ["weatherstation", "station", "name", "object"])
        if station_col is None:
            raise ValueError(
                "Could not find lon/lat columns or a station/object column with parseable coordinates. "
                "Export weather station latitude and longitude from PowerWorld if available."
            )
        parsed = raw[station_col].map(_parse_lat_lon_from_station)
        out["lat"] = [lat for lat, _ in parsed]
        out["lon"] = [lon for _, lon in parsed]

    required = {"lon", "lat", "temperature", "dew_point", "wind_100m"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Normalized weather export is missing required columns: {missing}")

    if "timestamp" not in out:
        scenario_col = _find_column(columns, ["scenario", "timestep", "time_step", "hour"])
        if scenario_col is None:
            out["timestamp"] = pd.Timestamp("2023-06-25") + pd.to_timedelta(0, unit="h")
        else:
            out["timestamp"] = pd.Timestamp("2023-06-25") + pd.to_timedelta(raw[scenario_col].astype(int), unit="h")

    out = out[["lon", "lat", "timestamp", "temperature", "dew_point", "wind_100m"]].copy()
    out = out.dropna(subset=["lon", "lat", "temperature", "dew_point", "wind_100m"])
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_parquet, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/stage_k_powerworld_weather_input_raw.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage_k_powerworld_weather_export.parquet"),
    )
    args = parser.parse_args()
    out = normalize_powerworld_weather(args.input, args.output)
    print(f"Wrote {args.output}")
    print(f"Rows: {len(out)}")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
