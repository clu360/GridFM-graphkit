"""Lightweight helpers for the Stage K modifiedTexas2k case-study notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


BUS_COLUMNS = [
    "BUS_I",
    "BUS_TYPE",
    "PD",
    "QD",
    "GS",
    "BS",
    "BUS_AREA",
    "VM",
    "VA",
    "BASE_KV",
    "ZONE",
    "VMAX",
    "VMIN",
    "LAM_P",
    "LAM_Q",
    "MU_VMAX",
    "MU_VMIN",
]

GEN_COLUMNS = [
    "GEN_BUS",
    "PG",
    "QG",
    "QMAX",
    "QMIN",
    "VG",
    "MBASE",
    "GEN_STATUS",
    "PMAX",
    "PMIN",
    "PC1",
    "PC2",
    "QC1MIN",
    "QC1MAX",
    "QC2MIN",
    "QC2MAX",
    "RAMP_AGC",
    "RAMP_10",
    "RAMP_30",
    "RAMP_Q",
    "APF",
    "MU_PMAX",
    "MU_PMIN",
    "MU_QMAX",
    "MU_QMIN",
]

BRANCH_COLUMNS = [
    "F_BUS",
    "T_BUS",
    "BR_R",
    "BR_X",
    "BR_B",
    "RATE_A",
    "RATE_B",
    "RATE_C",
    "TAP",
    "SHIFT",
    "BR_STATUS",
    "ANGMIN",
    "ANGMAX",
    "PF",
    "QF",
    "PT",
    "QT",
    "MU_SF",
    "MU_ST",
    "MU_ANGMIN",
    "MU_ANGMAX",
]

WEATHER_VARIABLE_ALIASES = {
    "temperature": [
        "temperature",
        "temp",
        "temp_f",
        "temperature_f",
        "temp_c",
        "temperature_c",
        "t2m",
        "air_temperature",
    ],
    "dew_point": [
        "dew_point",
        "dewpoint",
        "dew_point_f",
        "dewpoint_f",
        "dew_point_c",
        "dewpoint_c",
        "d2m",
    ],
    "wind_100m": [
        "wind_100m",
        "wind100m",
        "wind_speed_100m",
        "wind_speed",
        "windspeed",
        "ws100",
        "w100",
        "100m_wind",
    ],
}


@dataclass(frozen=True)
class StageKPaths:
    case_file: Path
    load_file: Path
    coordinate_file: Path
    weather_file: Path | None = None


def default_paths() -> StageKPaths:
    raw = Path(__file__).resolve().parent / "data" / "raw"
    return StageKPaths(
        case_file=raw / "modifiedTexas2k.m",
        load_file=raw / "modifiedTexas2k_loads_scenarios_0_23.parquet",
        coordinate_file=raw / "modifiedTexas2k_coordinates.csv",
        weather_file=raw / "2023-06-25_Texas_Heat_Dome_Texas.pww",
    )


def _parse_matpower_matrix(text: str, name: str, columns: list[str]) -> pd.DataFrame:
    pattern = rf"mpc\.{re.escape(name)}\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        raise ValueError(f"Could not find mpc.{name} matrix.")

    rows: list[list[float]] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip().rstrip(";")
        if not line or line.startswith("%"):
            continue
        rows.append([float(value) for value in re.split(r"\s+", line) if value])

    df = pd.DataFrame(rows)
    df.columns = columns[: df.shape[1]]
    return df


def _parse_matpower_strings(text: str, name: str) -> list[str]:
    pattern = rf"mpc\.{re.escape(name)}\s*=\s*\{{(.*?)\}};"
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return []
    return re.findall(r"'([^']*)'", match.group(1))


def load_case(case_file: str | Path) -> dict[str, pd.DataFrame | float | list[str]]:
    case_file = Path(case_file)
    text = case_file.read_text(errors="ignore")
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9.]+)\s*;", text)
    base_mva = float(base_match.group(1)) if base_match else np.nan

    bus = _parse_matpower_matrix(text, "bus", BUS_COLUMNS)
    gen = _parse_matpower_matrix(text, "gen", GEN_COLUMNS)
    branch = _parse_matpower_matrix(text, "branch", BRANCH_COLUMNS)
    gencost = _parse_matpower_matrix(
        text, "gencost", ["MODEL", "STARTUP", "SHUTDOWN", "NCOST", "C2", "C1", "C0"]
    )

    bus["row_id"] = np.arange(len(bus), dtype=int)
    branch["branch_id"] = np.arange(len(branch), dtype=int)
    gen["gen_id"] = np.arange(len(gen), dtype=int)

    bus_names = _parse_matpower_strings(text, "bus_name")
    if len(bus_names) == len(bus):
        bus["bus_name"] = bus_names

    genfuel = _parse_matpower_strings(text, "genfuel")
    if len(genfuel) == len(gen):
        gen["genfuel"] = genfuel

    gentype = _parse_matpower_strings(text, "gentype")
    if len(gentype) == len(gen):
        gen["gentype"] = gentype

    return {
        "base_mva": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "gencost": gencost,
    }


def load_stage_k_inputs(paths: StageKPaths | None = None) -> dict[str, object]:
    paths = paths or default_paths()
    case = load_case(paths.case_file)
    coords = pd.read_csv(paths.coordinate_file).rename(columns={"id": "row_id", "x": "lon", "y": "lat"})
    loads = pd.read_parquet(paths.load_file)

    bus = case["bus"].merge(coords, on="row_id", how="left", validate="one_to_one")
    case["bus"] = bus
    case["paths"] = paths
    case["coords"] = coords
    case["loads"] = loads
    return case


def attach_load_geography(case: dict[str, object], reference_scenario: int = 16) -> pd.DataFrame:
    bus = case["bus"].copy()
    loads = case["loads"].copy()
    load_rows = bus.loc[bus["PD"] != 0, ["row_id", "BUS_I", "bus_name", "PD", "QD", "BASE_KV", "lon", "lat"]]
    out = loads.merge(load_rows, left_on="bus_id", right_on="row_id", how="left", validate="many_to_one")

    ref = out.loc[out["scenario"] == reference_scenario, ["bus_id", "p_mw"]].rename(
        columns={"p_mw": "p_ref_mw"}
    )
    out = out.merge(ref, on="bus_id", how="left", validate="many_to_one")
    out["case_to_ref_scale"] = out["PD"] / out["p_ref_mw"]
    out["p_case_scaled_mw"] = out["p_mw"] * out["case_to_ref_scale"]
    out["q_case_scaled_mvar"] = out["q_mvar"] * out["case_to_ref_scale"]
    return out


def build_branch_geography(case: dict[str, object]) -> pd.DataFrame:
    bus = case["bus"]
    branch = case["branch"].copy()
    lookup = bus.set_index("BUS_I")[["row_id", "lon", "lat", "BASE_KV"]]
    branch = branch.join(lookup.add_prefix("from_"), on="F_BUS")
    branch = branch.join(lookup.add_prefix("to_"), on="T_BUS")

    mean_lat = np.deg2rad(bus["lat"].mean())
    dx = (branch["from_lon"] - branch["to_lon"]) * 111.32 * np.cos(mean_lat)
    dy = (branch["from_lat"] - branch["to_lat"]) * 110.57
    branch["geo_length_km"] = np.sqrt(dx * dx + dy * dy)
    branch["voltage_pair"] = (
        branch["from_BASE_KV"].round(1).astype(str) + " -> " + branch["to_BASE_KV"].round(1).astype(str)
    )
    return branch


def summarize_network(case: dict[str, object], branch_geo: pd.DataFrame) -> dict[str, object]:
    bus = case["bus"]
    gen = case["gen"]
    online = gen["GEN_STATUS"] > 0
    return {
        "base_mva": case["base_mva"],
        "buses": len(bus),
        "branches": len(branch_geo),
        "generators": len(gen),
        "online_generators": int(online.sum()),
        "load_buses": int((bus["PD"] != 0).sum()),
        "total_case_pd_mw": float(bus["PD"].sum()),
        "total_case_qd_mvar": float(bus["QD"].sum()),
        "online_pmax_mw": float(gen.loc[online, "PMAX"].sum()),
        "branch_rateA_median_mva": float(branch_geo["RATE_A"].median()),
        "branch_length_median_km": float(branch_geo["geo_length_km"].median()),
        "lon_bounds": (float(bus["lon"].min()), float(bus["lon"].max())),
        "lat_bounds": (float(bus["lat"].min()), float(bus["lat"].max())),
    }


def high_load_tables(load_geo: pd.DataFrame, top_n: int = 20, region_bins: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    bus_stats = (
        load_geo.groupby(["bus_id", "BUS_I", "bus_name", "lon", "lat"], dropna=False)
        .agg(
            avg_p_mw=("p_case_scaled_mw", "mean"),
            peak_p_mw=("p_case_scaled_mw", "max"),
            min_p_mw=("p_case_scaled_mw", "min"),
            peak_scenario=("p_case_scaled_mw", "idxmax"),
        )
        .reset_index()
    )
    idx_to_scenario = load_geo["scenario"].to_dict()
    bus_stats["peak_scenario"] = bus_stats["peak_scenario"].map(idx_to_scenario).astype(int)
    top_buses = bus_stats.sort_values("peak_p_mw", ascending=False).head(top_n)

    working = load_geo.copy()
    working["lon_bin"] = pd.cut(working["lon"], bins=region_bins)
    working["lat_bin"] = pd.cut(working["lat"], bins=region_bins)
    regions = (
        working.groupby(["lon_bin", "lat_bin"], observed=True)
        .agg(
            avg_p_mw=("p_case_scaled_mw", "mean"),
            peak_p_mw=("p_case_scaled_mw", "max"),
            total_peak_like_mw=("p_case_scaled_mw", "sum"),
            buses=("bus_id", "nunique"),
            lon_center=("lon", "mean"),
            lat_center=("lat", "mean"),
        )
        .reset_index()
        .sort_values("total_peak_like_mw", ascending=False)
    )
    return top_buses, regions


def plot_physical_grid(case: dict[str, object], branch_geo: pd.DataFrame, ax=None, title: str = "modifiedTexas2k physical grid"):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    bus = case["bus"]
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 9))

    segments = branch_geo[["from_lon", "from_lat", "to_lon", "to_lat"]].dropna().to_numpy().reshape(-1, 2, 2)
    lc = LineCollection(segments, colors="#425466", linewidths=0.35, alpha=0.28, zorder=1)
    ax.add_collection(lc)

    load_mask = bus["PD"] > 0
    gen_buses = set(case["gen"].loc[case["gen"]["GEN_STATUS"] > 0, "GEN_BUS"].astype(float))
    gen_mask = bus["BUS_I"].isin(gen_buses)

    ax.scatter(bus.loc[:, "lon"], bus.loc[:, "lat"], s=4, c="#9aa6b2", alpha=0.45, linewidths=0, label="bus", zorder=2)
    ax.scatter(
        bus.loc[load_mask, "lon"],
        bus.loc[load_mask, "lat"],
        s=8,
        c="#2a9d8f",
        alpha=0.75,
        linewidths=0,
        label="load bus",
        zorder=3,
    )
    ax.scatter(
        bus.loc[gen_mask, "lon"],
        bus.loc[gen_mask, "lat"],
        s=12,
        c="#e76f51",
        alpha=0.72,
        linewidths=0,
        label="online generator bus",
        zorder=4,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.15)
    return ax


def plot_load_profile(load_geo: pd.DataFrame, ax=None, scaled: bool = True):
    import matplotlib.pyplot as plt

    column = "p_case_scaled_mw" if scaled else "p_mw"
    hourly = load_geo.groupby("scenario")[column].sum().sort_index()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hourly.index, hourly.values / 1000.0, marker="o", color="#264653")
    ax.fill_between(hourly.index, hourly.values / 1000.0, alpha=0.16, color="#2a9d8f")
    ax.set_xlabel("Scenario / hour")
    ax.set_ylabel("Total active load (GW)")
    ax.set_title("24-hour total active load profile")
    ax.grid(alpha=0.2)
    return ax


def plot_load_snapshot(
    case: dict[str, object],
    branch_geo: pd.DataFrame,
    load_geo: pd.DataFrame,
    scenario: int = 16,
    ax=None,
    title: str | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 9))
    segments = branch_geo[["from_lon", "from_lat", "to_lon", "to_lat"]].dropna().to_numpy().reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segments, colors="#405261", linewidths=0.28, alpha=0.18, zorder=1))

    frame = load_geo.loc[load_geo["scenario"] == scenario].copy()
    sizes = 8 + 75 * np.sqrt(frame["p_case_scaled_mw"].clip(lower=0) / frame["p_case_scaled_mw"].max())
    scatter = ax.scatter(
        frame["lon"],
        frame["lat"],
        s=sizes,
        c=frame["p_case_scaled_mw"],
        cmap="inferno",
        alpha=0.72,
        linewidths=0,
        zorder=3,
    )
    ax.set_title(title or f"Load distribution, scenario {scenario}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    plt.colorbar(scatter, ax=ax, shrink=0.78, label="Active load (MW)")
    return ax


def plot_high_load_regions(case: dict[str, object], branch_geo: pd.DataFrame, regions: pd.DataFrame, ax=None, top_n: int = 12):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 9))
    segments = branch_geo[["from_lon", "from_lat", "to_lon", "to_lat"]].dropna().to_numpy().reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segments, colors="#405261", linewidths=0.28, alpha=0.15, zorder=1))

    plot_regions = regions.head(top_n)
    sizes = 80 + 520 * plot_regions["total_peak_like_mw"] / plot_regions["total_peak_like_mw"].max()
    scatter = ax.scatter(
        plot_regions["lon_center"],
        plot_regions["lat_center"],
        s=sizes,
        c=plot_regions["total_peak_like_mw"] / 1000.0,
        cmap="magma",
        alpha=0.68,
        edgecolors="#1d2935",
        linewidths=0.45,
        zorder=4,
    )
    ax.set_title("High-load geographic regions")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    plt.colorbar(scatter, ax=ax, shrink=0.78, label="Regional load intensity (GW-scenario)")
    return ax


def animate_load_24h(case: dict[str, object], branch_geo: pd.DataFrame, load_geo: pd.DataFrame, interval_ms: int = 550):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    scenarios = sorted(load_geo["scenario"].unique())
    vmax = load_geo["p_case_scaled_mw"].max()
    segments = branch_geo[["from_lon", "from_lat", "to_lon", "to_lat"]].dropna().to_numpy().reshape(-1, 2, 2)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(LineCollection(segments, colors="#405261", linewidths=0.25, alpha=0.15, zorder=1))
    first = load_geo.loc[load_geo["scenario"] == scenarios[0]]
    sizes = 8 + 72 * np.sqrt(first["p_case_scaled_mw"].clip(lower=0) / vmax)
    scatter = ax.scatter(
        first["lon"],
        first["lat"],
        s=sizes,
        c=first["p_case_scaled_mw"],
        cmap="inferno",
        vmin=0,
        vmax=vmax,
        alpha=0.72,
        linewidths=0,
        zorder=3,
    )
    title = ax.set_title("")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    fig.colorbar(scatter, ax=ax, shrink=0.78, label="Active load (MW)")

    def update(scenario):
        frame = load_geo.loc[load_geo["scenario"] == scenario]
        scatter.set_offsets(frame[["lon", "lat"]].to_numpy())
        scatter.set_array(frame["p_case_scaled_mw"].to_numpy())
        scatter.set_sizes(8 + 72 * np.sqrt(frame["p_case_scaled_mw"].clip(lower=0) / vmax))
        total_gw = frame["p_case_scaled_mw"].sum() / 1000.0
        title.set_text(f"modifiedTexas2k load animation - scenario {scenario:02d}, total {total_gw:.1f} GW")
        return scatter, title

    anim = FuncAnimation(fig, update, frames=scenarios, interval=interval_ms, blit=False)
    plt.close(fig)
    return anim


def inspect_weather_file(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {"exists": False, "note": "No weather file path was supplied."}
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path), "note": "Weather file was not found."}
    head = path.read_bytes()[:128]
    printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in head)
    likely_text = printable / max(len(head), 1) > 0.85
    header_preview = head[:80].decode("ascii", errors="backslashreplace")
    return {
        "exists": True,
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_mb": path.stat().st_size / (1024 * 1024),
        "likely_text": likely_text,
        "header_preview": header_preview,
        "note": (
            "This looks text-like and may be inspectable directly."
            if likely_text
            else "This looks binary. If it is a PowerWorld .pww file, use PowerWorld to export temperature, dew point, and 100m wind to CSV/Parquet before overlaying."
        ),
    }


def inspect_pww_coordinate_markers(path: str | Path, max_bytes: int | None = 2_000_000) -> pd.DataFrame:
    """Extract text coordinate markers visible in a PowerWorld weather file.

    This does not decode binary weather values. It only helps verify whether the
    file appears to contain a latitude/longitude grid over the region of
    interest before exporting a tabular one-day slice from PowerWorld.
    """
    path = Path(path)
    data = path.read_bytes() if max_bytes is None else path.read_bytes()[:max_bytes]
    strings = re.findall(rb"[+-]\d{2,3}\.\d{2}[+-]\d{3}\.\d{2}/", data)
    rows = []
    for raw in strings:
        text = raw.decode("ascii").rstrip("/")
        match = re.match(r"([+-]\d+\.\d+)([+-]\d+\.\d+)", text)
        if match:
            rows.append((float(match.group(1)), float(match.group(2))))
    out = pd.DataFrame(rows, columns=["lat", "lon"]).drop_duplicates()
    return out.sort_values(["lat", "lon"]).reset_index(drop=True)


def load_weather_overlay_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(
            "Weather overlays currently expect CSV or Parquet with lon/lat/value-like columns. "
            "For .pww, export a table first."
        )

    rename = {}
    lower = {c.lower(): c for c in df.columns}
    for target, options in {
        "lon": ["lon", "longitude", "x"],
        "lat": ["lat", "latitude", "y"],
        "weather_value": ["weather_value", "risk", "value", "p_env"],
        "timestamp": ["timestamp", "datetime", "date_time", "time", "date", "hour"],
    }.items():
        for option in options:
            if option in lower:
                rename[lower[option]] = target
                break
    for target, options in WEATHER_VARIABLE_ALIASES.items():
        for option in options:
            if option in lower:
                rename[lower[option]] = target
                break
    df = df.rename(columns=rename)
    weather_columns = {"weather_value", *WEATHER_VARIABLE_ALIASES.keys()} & set(df.columns)
    missing = {"lon", "lat"} - set(df.columns)
    if missing:
        raise ValueError(f"Weather overlay table is missing required columns: {sorted(missing)}")
    if not weather_columns:
        raise ValueError(
            "Weather overlay table needs at least one value column: weather_value, "
            "temperature, dew_point, or wind_100m."
        )
    return df


def select_one_day_weather(
    weather: pd.DataFrame,
    day: str | int | None = None,
    time_column: str | None = None,
    max_steps: int = 24,
) -> pd.DataFrame:
    """Select one 24-step weather snippet and align it to load scenarios.

    The input may be either already one day of weather points or a multi-day
    export. If a timestamp/date-like column is present, the first day is used by
    default, or `day` can select a specific date.
    """
    df = weather.copy()
    candidates = [time_column] if time_column else []
    candidates += ["timestamp", "datetime", "time", "date_time", "hour"]
    time_col = next((col for col in candidates if col and col in df.columns), None)

    if time_col is None:
        df["weather_step"] = 0
        df["weather_day"] = "single_export"
        df["scenario"] = 0
        return df

    parsed = pd.to_datetime(df[time_col], errors="coerce")
    if parsed.notna().any():
        df["_parsed_time"] = parsed
        df["weather_day"] = df["_parsed_time"].dt.date.astype(str)
        selected_day = str(day) if day is not None else sorted(df["weather_day"].dropna().unique())[0]
        day_df = df.loc[df["weather_day"] == selected_day].copy()
        ordered_steps = sorted(day_df["_parsed_time"].dropna().unique())[:max_steps]
        day_df = day_df.loc[day_df["_parsed_time"].isin(ordered_steps)].copy()
        step_map = {value: i for i, value in enumerate(ordered_steps)}
        day_df["weather_step"] = day_df["_parsed_time"].map(step_map).astype(int)
        day_df["scenario"] = day_df["weather_step"]
        return day_df.drop(columns=["_parsed_time"])

    unique_steps = sorted(df[time_col].dropna().unique())
    if day is None:
        selected_steps = unique_steps[:max_steps]
    else:
        start = int(day) * max_steps
        selected_steps = unique_steps[start : start + max_steps]
    out = df.loc[df[time_col].isin(selected_steps)].copy()
    out["weather_day"] = "selected_block"
    out["weather_step"] = out[time_col].map({value: i for i, value in enumerate(selected_steps)}).astype(int)
    out["scenario"] = out["weather_step"]
    return out


def plot_weather_overlay(
    case: dict[str, object],
    branch_geo: pd.DataFrame,
    weather: pd.DataFrame,
    ax=None,
    title: str = "Weather overlay on physical grid",
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 9))
    segments = branch_geo[["from_lon", "from_lat", "to_lon", "to_lat"]].dropna().to_numpy().reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segments, colors="#25313a", linewidths=0.28, alpha=0.18, zorder=1))
    scatter = ax.scatter(
        weather["lon"],
        weather["lat"],
        c=weather["weather_value"],
        s=35,
        cmap="coolwarm",
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(case["bus"]["lon"], case["bus"]["lat"], s=3, c="#111827", alpha=0.28, linewidths=0, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    plt.colorbar(scatter, ax=ax, shrink=0.78, label="Weather / hazard value")
    return ax


def plot_weather_variable_overlay(
    case: dict[str, object],
    branch_geo: pd.DataFrame,
    weather: pd.DataFrame,
    variable: str,
    scenario: int = 16,
    ax=None,
    cmap: str | None = None,
    label: str | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if variable not in weather.columns:
        raise ValueError(f"Weather table does not contain column {variable!r}.")
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 9))

    segments = branch_geo[["from_lon", "from_lat", "to_lon", "to_lat"]].dropna().to_numpy().reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segments, colors="#1f2933", linewidths=0.25, alpha=0.18, zorder=1))

    frame = weather.loc[weather["scenario"] == scenario] if "scenario" in weather else weather
    if len(frame) == 0:
        frame = weather

    default_cmaps = {
        "temperature": "inferno",
        "dew_point": "viridis",
        "wind_100m": "plasma",
        "weather_value": "coolwarm",
    }
    scatter = ax.scatter(
        frame["lon"],
        frame["lat"],
        c=frame[variable],
        s=34,
        cmap=cmap or default_cmaps.get(variable, "coolwarm"),
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(case["bus"]["lon"], case["bus"]["lat"], s=3, c="#111827", alpha=0.26, linewidths=0, zorder=3)
    title = label or variable.replace("_", " ").title()
    ax.set_title(f"{title} over modifiedTexas2k grid, scenario {scenario:02d}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    plt.colorbar(scatter, ax=ax, shrink=0.78, label=title)
    return ax


def plot_weather_variable_panel(
    case: dict[str, object],
    branch_geo: pd.DataFrame,
    weather: pd.DataFrame,
    scenario: int = 16,
    variables: tuple[str, ...] = ("temperature", "dew_point", "wind_100m"),
):
    import matplotlib.pyplot as plt

    available = [variable for variable in variables if variable in weather.columns]
    if not available:
        raise ValueError(f"None of the requested weather variables are present: {variables}")
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 6), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]
    labels = {
        "temperature": "Temperature",
        "dew_point": "Dew Point",
        "wind_100m": "100m Wind Speed",
    }
    for ax, variable in zip(axes, available):
        plot_weather_variable_overlay(
            case,
            branch_geo,
            weather,
            variable=variable,
            scenario=scenario,
            ax=ax,
            label=labels.get(variable),
        )
    return fig, axes


def plot_weather_and_load_overlay(
    case: dict[str, object],
    branch_geo: pd.DataFrame,
    weather: pd.DataFrame,
    load_geo: pd.DataFrame,
    scenario: int = 16,
    ax=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 9))
    segments = branch_geo[["from_lon", "from_lat", "to_lon", "to_lat"]].dropna().to_numpy().reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segments, colors="#1f2933", linewidths=0.25, alpha=0.16, zorder=1))

    w = weather.loc[weather.get("scenario", scenario) == scenario] if "scenario" in weather else weather
    if len(w) == 0:
        w = weather
    weather_scatter = ax.scatter(
        w["lon"],
        w["lat"],
        c=w["weather_value"],
        s=30,
        cmap="coolwarm",
        alpha=0.42,
        linewidths=0,
        zorder=2,
    )

    frame = load_geo.loc[load_geo["scenario"] == scenario]
    sizes = 8 + 64 * np.sqrt(frame["p_case_scaled_mw"].clip(lower=0) / frame["p_case_scaled_mw"].max())
    ax.scatter(
        frame["lon"],
        frame["lat"],
        s=sizes,
        c="#111827",
        alpha=0.52,
        linewidths=0,
        zorder=3,
    )
    ax.set_title(f"Weather and load overlay, scenario {scenario:02d}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    plt.colorbar(weather_scatter, ax=ax, shrink=0.78, label="Weather / hazard value")
    return ax


def simple_contingency_screen(branch_geo: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Rank lines for first-look contingency discussion using observable data only."""
    score = (
        branch_geo["geo_length_km"].fillna(0) / branch_geo["geo_length_km"].quantile(0.95)
        + branch_geo["RATE_A"].median() / branch_geo["RATE_A"].clip(lower=1)
    )
    out = branch_geo.assign(first_look_contingency_score=score)
    columns = [
        "branch_id",
        "F_BUS",
        "T_BUS",
        "RATE_A",
        "geo_length_km",
        "voltage_pair",
        "first_look_contingency_score",
    ]
    return out.sort_values("first_look_contingency_score", ascending=False)[columns].head(top_n)
