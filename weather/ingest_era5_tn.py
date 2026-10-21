#!/usr/bin/env python3
"""Ingest daily ERA5 meteorological covariates for 5 Tamil Nadu Regional Clusters (2019-2024).

Clusters:
  1. North Coast: Chennai (lat 13.08, lon 80.27)
  2. West / Industrial: Coimbatore (lat 11.02, lon 76.96)
  3. Central Plain: Trichy (lat 10.79, lon 78.70)
  4. South Central: Madurai (lat 9.93, lon 78.12)
  5. Deep South: Tirunelveli (lat 8.71, lon 77.76)

Aggregates at Sentinel-5P overpass window (07:00-09:00 UTC):
  blh, u10, v10, wind_speed, t2m, rh, tp
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_era5_tn")

CLUSTERS = {
    "chennai": {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    "coimbatore": {"name": "Coimbatore", "lat": 11.0168, "lon": 76.9558},
    "trichy": {"name": "Trichy", "lat": 10.7905, "lon": 78.7047},
    "madurai": {"name": "Madurai", "lat": 9.9252, "lon": 78.1198},
    "tirunelveli": {"name": "Tirunelveli", "lat": 8.7139, "lon": 77.7567},
}


def fetch_year(year: int, lat: float, lon: float) -> pd.DataFrame:
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,relative_humidity_2m,precipitation,boundary_layer_height,"
        f"wind_speed_10m,wind_direction_10m&windspeed_unit=ms"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "AirQ-DualClock/1.0"})
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            hourly = pd.DataFrame(data["hourly"])
            hourly["time"] = pd.to_datetime(hourly["time"])
            return hourly
        except Exception as e:
            log.warning("Attempt %d failed for year %d (%.2f, %.2f): %s", attempt, year, lat, lon, e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch weather for {year}")


def process_hourly(hourly: pd.DataFrame, cluster_id: str, cluster_name: str, lat: float, lon: float) -> pd.DataFrame:
    hourly["date"] = hourly["time"].dt.date
    daily_tp = hourly.groupby("date")["precipitation"].sum().rename("tp")

    # Sentinel-5P overpass window (07:00, 08:00, 09:00 UTC)
    overpass = hourly[hourly["time"].dt.hour.isin([7, 8, 9])].copy()
    rad = np.radians(overpass["wind_direction_10m"].values)
    s = overpass["wind_speed_10m"].values

    overpass["u10"] = -s * np.sin(rad)
    overpass["v10"] = -s * np.cos(rad)
    overpass["t2m"] = overpass["temperature_2m"] + 273.15
    overpass["rh"] = overpass["relative_humidity_2m"]
    overpass["blh"] = overpass["boundary_layer_height"]
    overpass["wind_speed"] = s

    daily_overpass = overpass.groupby("date").agg({
        "blh": "mean",
        "u10": "mean",
        "v10": "mean",
        "wind_speed": "mean",
        "t2m": "mean",
        "rh": "mean",
    })

    daily = pd.concat([daily_overpass, daily_tp], axis=1).reset_index()
    daily["cluster"] = cluster_id
    daily["cluster_name"] = cluster_name
    daily["cluster_lat"] = lat
    daily["cluster_lon"] = lon
    daily["date"] = pd.to_datetime(daily["date"])

    cols = ["cluster", "cluster_name", "cluster_lat", "cluster_lon", "date", "blh", "u10", "v10", "wind_speed", "tp", "t2m", "rh"]
    return daily[cols]


def main():
    years = list(range(2019, 2025))
    all_cluster_dfs = []

    # Check if Chennai weather exists already to save API calls
    chennai_existing = Path("weather/era5_daily.csv")
    if not chennai_existing.exists():
        chennai_existing = Path("era5_daily.csv")

    for cid, info in CLUSTERS.items():
        name, lat, lon = info["name"], info["lat"], info["lon"]
        log.info("Processing cluster: %s (%s, lat=%.4f, lon=%.4f)...", name, cid, lat, lon)

        # If chennai already exists on disk, load and adapt
        if cid == "chennai" and chennai_existing.exists():
            log.info("Loading existing Chennai weather from %s...", chennai_existing)
            df_ch = pd.read_csv(chennai_existing)
            df_ch["cluster"] = cid
            df_ch["cluster_name"] = name
            df_ch["cluster_lat"] = lat
            df_ch["cluster_lon"] = lon
            df_ch["date"] = pd.to_datetime(df_ch["date"])
            cols = ["cluster", "cluster_name", "cluster_lat", "cluster_lon", "date", "blh", "u10", "v10", "wind_speed", "tp", "t2m", "rh"]
            all_cluster_dfs.append(df_ch[cols])
            continue

        cluster_records = []
        for y in years:
            hourly = fetch_year(y, lat, lon)
            daily = process_hourly(hourly, cid, name, lat, lon)
            cluster_records.append(daily)
            time.sleep(0.5)

        cluster_df = pd.concat(cluster_records, ignore_index=True)
        all_cluster_dfs.append(cluster_df)
        log.info("Finished %s: %d daily rows.", name, len(cluster_df))

    final_df = pd.concat(all_cluster_dfs, ignore_index=True)
    out_dir = Path("weather")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "era5_tn_clusters.csv"
    final_df.to_csv(out_file, index=False)
    log.info("Saved all Tamil Nadu regional cluster weather (%d rows) to %s!", len(final_df), out_file)


if __name__ == "__main__":
    main()
