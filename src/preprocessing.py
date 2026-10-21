#!/usr/bin/env python3
"""Stage 1 of the v4 pipeline: raw inputs -> one clean window table, written once with a manifest.

Built around the issues found in the v3 run and the frozen 2023-24 test:
  * nodata (-9999 / NaN) is missing; 0 is a valid value and negatives are kept (SO2 has ~30 % negatives)
  * every unit is put on one regular 5-day window grid; off-grid dates are an error, not silently shifted
  * the forecast unit is the city mean (optionally 3x3 blocks), with valid-pixel fraction kept per window
  * lockdown windows (2020 and 2021 waves) are FLAGGED, never dropped; the feature stage decides how to use them
  * a processor-segment id per window, so the level feature can reset at retrieval-version changes
  * ERA5 (daily -> 5-day mean) and FIRMS fire counts (daily -> 5-day sum) are aligned to the same windows
  * the 2026 holdout is never read: any input dated on or after holdout.start raises HoldoutError
  * all development years are flagged exploratory (2023-25 were already scored once)

No climatology, scaling or other fitted quantity is computed here: everything fitted lives in
feature_extractor.py and is fitted per fold on training years only.

Usage
  python preprocessing.py --cube Chennai=cube_v3_chennai.npz --out data_v4
  python preprocessing.py --cube Chennai=cube_v3_chennai.npz --blocks Chennai \
         --era5 era5_daily.csv --firms firms_daily.csv --out data_v4
  python preprocessing.py --city-table tnaq_windows.csv --era5 era5_daily.csv --out data_v4

Inputs
  --cube UNIT=PATH     .npz with `values` (T, G, H, W) and `dates` (window starts); optional `gases`.
                       Repeat for each city. Same format as cube_v3_chennai.npz / train_kaggle_v3 --export-cube.
  --city-table CSV     long table already aggregated per city: city, window_start, gas, value, n_valid, n_total
  --era5 CSV           daily: city, date, <vars in config era5_vars>  (a missing var column is allowed)
  --firms CSV          daily: date, count  [, city]   (no city column -> the same regional count for every city)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("preprocessing")

DEFAULT_CONFIG = {
    "window_days": 5,
    "grid_start": "2019-01-01",
    "dev_end": "2025-12-31",
    "holdout": {"start": "2026-01-01", "end": "2026-09-30"},     # declared clean holdout: never read before the freeze
    "nodata": -9999.0,
    "gases": ["NO2", "CO", "SO2"],
    "lockdowns": [
        {"name": "covid_wave1", "start": "2020-03-25", "end": "2020-06-30"},
        {"name": "covid_wave2_tn", "start": "2021-05-10", "end": "2021-06-30"},
    ],
    # Retrieval-processor change dates per gas. UNVERIFIED defaults: check them against the S5P product
    # readme / PRF history before relying on them. They only control where the level feature resets.
    "processor_boundaries": {
        "NO2": ["2021-07-01", "2022-07-17"],
        "CO": ["2021-07-01"],
        "SO2": ["2021-07-01"],
    },
    "spent_years": [2023, 2024, 2025],
    "era5_vars": ["blh", "u10", "v10", "wind_speed", "tp", "t2m", "rh"],
    "min_days_per_window": 3,          # ERA5 window mean needs at least this many daily values
    "block_split": 3,                  # --blocks: split the grid into block_split x block_split blocks
    "zero_warn_frac": 0.01,            # warn if more than this fraction of valid pixels are exactly 0
}


class HoldoutError(RuntimeError):
    """Raised when an input contains dates inside or after the declared holdout."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_holdout(dates, cfg, what: str):
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(dates) and dates.max() >= pd.Timestamp(cfg["holdout"]["start"]):
        raise HoldoutError(f"{what} contains dates from {cfg['holdout']['start']} onward (max {dates.max().date()}). "
                           f"The holdout must not be read before the freeze: trim the input to <= {cfg['dev_end']}.")


def window_index(dates, cfg, strict: bool):
    """Window index on the regular grid. strict=True requires dates to be exact window starts (S5P windows)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    days = (d - pd.Timestamp(cfg["grid_start"])).days.values
    if strict and (days % cfg["window_days"]).any():
        bad = d[(days % cfg["window_days"]) != 0][0]
        raise ValueError(f"window start {bad.date()} is off the {cfg['window_days']}-day grid from {cfg['grid_start']}")
    if (days < 0).any():
        raise ValueError(f"date before grid_start {cfg['grid_start']}")
    return days // cfg["window_days"]


def window_grid(cfg) -> pd.DatetimeIndex:
    end = min(pd.Timestamp(cfg["dev_end"]), pd.Timestamp(cfg["holdout"]["start"]) - pd.Timedelta(days=1))
    return pd.date_range(cfg["grid_start"], end, freq=f"{cfg['window_days']}D")


# ------------------------------------------------------------------------------------------------------
# S5P
# ------------------------------------------------------------------------------------------------------
def load_cube(path: Path, cfg):
    z = np.load(path, allow_pickle=True)          # allow_pickle: some cube_v3 files store dates as objects
    values = z["values"].astype(np.float64)
    dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
    gases = [str(g) for g in z["gases"]] if "gases" in z.files else list(cfg["gases"])
    if values.ndim != 4 or values.shape[0] != len(dates) or values.shape[1] != len(gases):
        raise ValueError(f"{path}: values {values.shape} do not match {len(dates)} dates x {len(gases)} gases")
    guard_holdout(dates, cfg, str(path))
    bad = ~np.isfinite(values) | (values == cfg["nodata"])
    values[bad] = np.nan                           # only nodata/NaN become missing; 0 and negatives are kept
    return values, dates, gases


def pixel_stats(values, gases):
    out = {}
    for g, gas in enumerate(gases):
        v = values[:, g]
        ok = np.isfinite(v)
        n = int(ok.sum())
        out[gas] = {"valid_pixel_frac": round(float(ok.mean()), 4),
                    "zero_frac_of_valid": round(float((v[ok] == 0).sum() / max(n, 1)), 5),
                    "negative_frac_of_valid": round(float((v[ok] < 0).sum() / max(n, 1)), 4)}
    return out


def aggregate_cube(unit, values, dates, gases, cfg, blocks: bool):
    """City mean (and optional blocks) per window and gas over valid pixels only."""
    T, G, H, W = values.shape
    footprint = np.isfinite(values).any(axis=(0, 1))          # pixels that are ever valid
    units = [(unit, "city", footprint)]
    if blocks:
        k = cfg["block_split"]
        for bi, rows in enumerate(np.array_split(np.arange(H), k)):
            for bj, cols in enumerate(np.array_split(np.arange(W), k)):
                m = np.zeros((H, W), bool)
                m[np.ix_(rows, cols)] = True
                units.append((f"{unit}_b{bi}{bj}", "block", m & footprint))
    rows = []
    for name, utype, mask in units:
        n_total = int(mask.sum())
        if n_total == 0:
            continue
        for g, gas in enumerate(gases):
            v = values[:, g][:, mask]                          # (T, n_total)
            ok = np.isfinite(v)
            n_valid = ok.sum(1)
            with np.errstate(invalid="ignore"):
                mean = np.where(n_valid > 0, np.nansum(np.where(ok, v, 0), 1) / np.maximum(n_valid, 1), np.nan)
            rows.append(pd.DataFrame({"unit": name, "unit_type": utype, "parent": unit, "window_start": dates,
                                      "gas": gas, "value": mean, "n_valid": n_valid, "n_total": n_total}))
    return pd.concat(rows, ignore_index=True)


def load_city_table(path: Path, cfg):
    t = pd.read_csv(path, parse_dates=["window_start"])
    need = {"city", "window_start", "gas", "value", "n_valid", "n_total"}
    if need - set(t.columns):
        raise ValueError(f"{path}: missing columns {sorted(need - set(t.columns))}")
    guard_holdout(t["window_start"], cfg, str(path))
    t["value"] = t["value"].where(np.isfinite(t["value"]) & (t["value"] != cfg["nodata"]))
    t = t.rename(columns={"city": "unit"}).assign(unit_type="city", parent=lambda d: d["unit"])
    return t[["unit", "unit_type", "parent", "window_start", "gas", "value", "n_valid", "n_total"]]


# ------------------------------------------------------------------------------------------------------
# Drivers
# ------------------------------------------------------------------------------------------------------
def load_era5(path: Path, cfg, grid):
    d = pd.read_csv(path, parse_dates=["date"])
    guard_holdout(d["date"], cfg, str(path))
    if "wind_speed" in cfg["era5_vars"] and "wind_speed" not in d and {"u10", "v10"} <= set(d.columns):
        d["wind_speed"] = np.hypot(d["u10"], d["v10"])
    vars_ = [v for v in cfg["era5_vars"] if v in d.columns]
    missing = [v for v in cfg["era5_vars"] if v not in d.columns]
    if missing:
        log.warning("ERA5 file has no %s; those features will be absent", missing)
    d = d[d["date"] >= pd.Timestamp(cfg["grid_start"])]
    d["win"] = window_index(d["date"], cfg, strict=False)
    g = d.groupby(["city", "win"])
    mean = g[vars_].mean()
    cnt = g[vars_].count()
    mean = mean.where(cnt >= cfg["min_days_per_window"])
    mean = mean.reset_index()
    mean = mean[mean["win"] < len(grid)]
    mean["window_start"] = grid[mean["win"].values]
    return mean.drop(columns="win").rename(columns={"city": "parent", **{v: f"era5_{v}" for v in vars_}})


def load_firms(path: Path, cfg, grid):
    d = pd.read_csv(path, parse_dates=["date"])
    guard_holdout(d["date"], cfg, str(path))
    d = d[d["date"] >= pd.Timestamp(cfg["grid_start"])]
    d["win"] = window_index(d["date"], cfg, strict=False)
    keys = ["city", "win"] if "city" in d.columns else ["win"]
    s = d.groupby(keys)["count"].agg(["sum", "count"]).reset_index()
    s["firms_count"] = s["sum"].where(s["count"] >= cfg["min_days_per_window"])
    s = s[s["win"] < len(grid)]
    s["window_start"] = grid[s["win"].values]
    s = s.drop(columns=["win", "sum", "count"])
    return s.rename(columns={"city": "parent"})


# ------------------------------------------------------------------------------------------------------
# Flags known in advance (calendar, lockdowns, processor segment, split status)
# ------------------------------------------------------------------------------------------------------
def add_flags(t: pd.DataFrame, cfg):
    wd = cfg["window_days"]
    start = t["window_start"]
    end = start + pd.Timedelta(days=wd - 1)
    t["year"] = start.dt.year
    t["month"] = (start + pd.Timedelta(days=wd // 2)).dt.month
    ang = 2 * np.pi * ((start + pd.Timedelta(days=wd / 2)).dt.dayofyear) / 365.25
    t["doy_sin"], t["doy_cos"] = np.sin(ang), np.cos(ang)
    dow = np.stack([(start + pd.Timedelta(days=k)).dt.dayofweek.values for k in range(wd)], 1)
    t["weekend_frac"] = (dow >= 5).mean(1)
    t["lockdown"] = False
    t["lockdown_name"] = ""
    for lk in cfg["lockdowns"]:
        hit = (end >= pd.Timestamp(lk["start"])) & (start <= pd.Timestamp(lk["end"]))
        t.loc[hit, "lockdown"] = True
        t.loc[hit, "lockdown_name"] = lk["name"]
    t["segment"] = 0
    for gas, bounds in cfg["processor_boundaries"].items():
        sel = t["gas"] == gas
        t.loc[sel, "segment"] = np.searchsorted(pd.to_datetime(bounds).values, start[sel].values, side="right")
    t["spent"] = t["year"].isin(cfg["spent_years"])
    t["exploratory"] = True             # every development year: nothing here is a clean test
    t["valid_frac"] = t["n_valid"] / t["n_total"].clip(lower=1)
    return t


def build(args, cfg):
    grid = window_grid(cfg)
    inputs, parts, pix = {}, [], {}
    for spec in args.cube or []:
        unit, path = spec.split("=", 1)
        path = Path(path)
        values, dates, gases = load_cube(path, cfg)
        window_index(dates, cfg, strict=True)
        pix[unit] = pixel_stats(values, gases)
        for gas, s in pix[unit].items():
            if s["zero_frac_of_valid"] > cfg["zero_warn_frac"]:
                log.warning("%s %s: %.1f%% of valid pixels are exactly 0 - possible legacy zero-fill export",
                            unit, gas, 100 * s["zero_frac_of_valid"])
        parts.append(aggregate_cube(unit, values, dates, gases, cfg, blocks=unit in (args.blocks or [])))
        inputs[f"cube:{unit}"] = {"path": str(path), "sha256": sha256_file(path), "shape": list(values.shape)}
    if args.city_table:
        p = Path(args.city_table)
        parts.append(load_city_table(p, cfg))
        inputs["city_table"] = {"path": str(p), "sha256": sha256_file(p)}
    if not parts:
        raise SystemExit("give at least one --cube or --city-table")
    t = pd.concat(parts, ignore_index=True)

    # one regular grid for every unit and gas: missing windows become rows with NaN value
    full = pd.MultiIndex.from_product([t["unit"].unique(), grid, t["gas"].unique()],
                                      names=["unit", "window_start", "gas"])
    meta = t.drop_duplicates("unit").set_index("unit")[["unit_type", "parent"]]
    t = t.set_index(["unit", "window_start", "gas"]).reindex(full).reset_index()
    t["unit_type"] = t["unit"].map(meta["unit_type"])
    t["parent"] = t["unit"].map(meta["parent"])
    t["n_valid"] = t["n_valid"].fillna(0).astype(int)
    t["n_total"] = t.groupby(["unit"])["n_total"].transform("max").fillna(0).astype(int)

    if args.era5:
        p = Path(args.era5)
        t = t.merge(load_era5(p, cfg, grid), on=["parent", "window_start"], how="left")
        inputs["era5"] = {"path": str(p), "sha256": sha256_file(p)}
    if args.firms:
        p = Path(args.firms)
        f = load_firms(p, cfg, grid)
        t = t.merge(f, on=["parent", "window_start"] if "parent" in f else ["window_start"], how="left")
        inputs["firms"] = {"path": str(p), "sha256": sha256_file(p)}

    t = add_flags(t, cfg).sort_values(["unit", "gas", "window_start"]).reset_index(drop=True)
    return t, inputs, pix


def write_table(t: pd.DataFrame, out: Path) -> Path:
    try:
        path = out / "windows.parquet"
        t.to_parquet(path, index=False)
    except ImportError:
        path = out / "windows.csv.gz"
        log.warning("pyarrow not installed: writing %s instead of parquet", path.name)
        t.to_csv(path, index=False)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", action="append", help="UNIT=PATH to a cube .npz (repeatable)")
    ap.add_argument("--city-table", help="long CSV already aggregated per city")
    ap.add_argument("--blocks", action="append", help="also write 3x3 block units for this cube unit")
    ap.add_argument("--era5", help="daily ERA5 CSV: city, date, vars")
    ap.add_argument("--firms", help="daily FIRMS CSV: date, count [, city]")
    ap.add_argument("--config", help="JSON file overriding DEFAULT_CONFIG keys")
    ap.add_argument("--out", default="data_v4", help="output folder")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t, inputs, pix = build(args, cfg)
    path = write_table(t, out)
    cov = (t.assign(ok=np.isfinite(t["value"])).groupby(["unit", "gas", "year"])["ok"].mean().round(3)
           .unstack("year").to_dict(orient="index"))
    manifest = {
        "stage": "preprocessing", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": sha256_file(Path(__file__)), "config": cfg, "inputs": inputs,
        "table": {"path": path.name, "sha256": sha256_file(path), "rows": len(t), "columns": list(t.columns)},
        "date_range": [str(t["window_start"].min().date()), str(t["window_start"].max().date())],
        "units": sorted(t["unit"].unique().tolist()),
        "holdout": {**cfg["holdout"], "read": False},
        "pixel_stats": pix, "window_coverage_by_year": {f"{u}|{g}": v for (u, g), v in cov.items()},
        "notes": ["all rows are development data and flagged exploratory=True",
                  "lockdown windows are flagged, not dropped",
                  "processor_boundaries are unverified defaults unless the config says otherwise"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("wrote %s (%d rows, %d units) and manifest.json", path, len(t), t["unit"].nunique())


if __name__ == "__main__":
    try:
        main()
    except HoldoutError as e:
        sys.exit(f"HOLDOUT GUARD: {e}")
