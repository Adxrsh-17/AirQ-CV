#!/usr/bin/env python3
"""Ingest daily ERA5 meteorological covariates for Chennai (2019-01-01 to 2024-12-31).

Sources hourly reanalysis from Open-Meteo (Copernicus ERA5 reanalysis archive).
Aggregates conditions around Sentinel-5P overpass (07:00-09:00 UTC):
  - blh: Boundary Layer Height (m)
  - u10: 10m zonal wind component (m/s)
  - v10: 10m meridional wind component (m/s)
  - wind_speed: 10m wind speed (m/s)
  - t2m: 2m temperature (K)
  - rh: 2m relative humidity (%)
  - tp: Total precipitation (daily sum in mm)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest_era5")

# Chennai industrial corridor center
LAT = 13.0827
LON = 80.2707


def fetch_year(year: int, lat: float = LAT, lon: float = LON) -> pd.DataFrame:
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
        f"hourly=temperature_2m,relative_humidity_2m,precipitation,boundary_layer_height,"
        f"wind_speed_10m,wind_direction_10m&windspeed_unit=ms"
    )
    log.info("Fetching %d (lat=%.4f, lon=%.4f)...", year, lat, lon)
    req = urllib.request.Request(url, headers={"User-Agent": "AirQ-DualClock/1.0"})
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            hourly = pd.DataFrame(data["hourly"])
            hourly["time"] = pd.to_datetime(hourly["time"])
            return hourly
        except Exception as e:
            log.warning("Attempt %d failed for %d: %s", attempt, year, e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch weather for {year}")


def process_weather(hourly: pd.DataFrame, city: str = "Chennai") -> pd.DataFrame:
    # 1. Daily total precipitation (24h sum)
    hourly["date"] = hourly["time"].dt.date
    daily_tp = hourly.groupby("date")["precipitation"].sum().rename("tp")

    # 2. Overpass window: 07:00, 08:00, 09:00 UTC (~12:30 - 14:30 IST)
    overpass = hourly[hourly["time"].dt.hour.isin([7, 8, 9])].copy()

    # Convert wind speed and direction to u10 and v10
    rad = np.radians(overpass["wind_direction_10m"].values)
    s = overpass["wind_speed_10m"].values
    overpass["u10"] = -s * np.sin(rad)
    overpass["v10"] = -s * np.cos(rad)

    # Convert Celsius to Kelvin
    overpass["t2m"] = overpass["temperature_2m"] + 273.15
    overpass["rh"] = overpass["relative_humidity_2m"]
    overpass["blh"] = overpass["boundary_layer_height"]
    overpass["wind_speed"] = s

    # Aggregate overpass mean
    daily_overpass = overpass.groupby("date").agg({
        "blh": "mean",
        "u10": "mean",
        "v10": "mean",
        "wind_speed": "mean",
        "t2m": "mean",
        "rh": "mean",
    })

    daily = pd.concat([daily_overpass, daily_tp], axis=1).reset_index()
    daily["city"] = city
    daily["date"] = pd.to_datetime(daily["date"])

    cols = ["city", "date", "blh", "u10", "v10", "wind_speed", "tp", "t2m", "rh"]
    return daily[cols]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=2019)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--out", default="era5_daily.csv")
    args = ap.parse_args(argv)

    all_hourly = []
    for y in range(args.start_year, args.end_year + 1):
        h = fetch_year(y)
        all_hourly.append(h)
        time.sleep(1)

    full_hourly = pd.concat(all_hourly, ignore_index=True)
    daily = process_weather(full_hourly, city="Chennai")

    out_p = Path(args.out)
    daily.to_csv(out_p, index=False)
    log.info("Saved %d daily weather records to %s", len(daily), out_p)
    log.info("Summary ranges:")
    log.info("  BLH (m):       min=%.1f, median=%.1f, max=%.1f", daily["blh"].min(), daily["blh"].median(), daily["blh"].max())
    log.info("  Wind (m/s):    min=%.2f, median=%.2f, max=%.2f", daily["wind_speed"].min(), daily["wind_speed"].median(), daily["wind_speed"].max())
    log.info("  T2M (K):       min=%.1f, median=%.1f, max=%.1f", daily["t2m"].min(), daily["t2m"].median(), daily["t2m"].max())
    log.info("  RH (%%):        min=%.1f, median=%.1f, max=%.1f", daily["rh"].min(), daily["rh"].median(), daily["rh"].max())
    log.info("  Precip (mm):   min=%.1f, median=%.1f, max=%.1f", daily["tp"].min(), daily["tp"].median(), daily["tp"].max())


if __name__ == "__main__":
    main()
