#!/usr/bin/env python3
"""Stage 2 of the v4 pipeline: window table -> saved per-fold feature tables, so training never recomputes.

Everything fitted (monthly climatology, anomaly std, ERA5/FIRMS climatology and scaling) is fitted per fold on
that fold's TRAINING years only, with lockdown windows and low-coverage windows excluded. Validation-year values
are transformed with the training-year statistics and never influence them.

Causality. A row is (unit, gas, target window t, horizon h). Forecast-mode features only read windows
<= t - h, with these exceptions that are known in advance: target-window calendar (day of year, weekend share)
and the target window's own lockdown flag. Target-window weather is the perfect-prognosis upper bound; it is
written to a SEPARATE file (features_pp_*) so it can never slip into a forecast-mode model.

Features (forecast mode), all in standardised-anomaly units (z):
  own_lag{k}, own_mask{k}     own gas z at t-h-k (0 + mask when missing), k = 0..n_lags-1
  own_last_valid              most recent valid own z among those lags (damped-persistence input)
  own_level, own_level_mask   causal trailing mean of z over level_windows windows ending at t-h, skipping
                              lockdown windows and RESET at processor-segment boundaries
  x_<GAS>_lag0, x_<GAS>_mask0 the other gases' z at t-h
  era5_<v>_lag{k}, _mask{k}   ERA5 5-day mean anomaly (vs training-year monthly climatology), standardised
  firms_lag{k}, firms_mask{k} log1p FIRMS count anomaly, standardised
  doy_sin, doy_cos, weekend_frac   target-window calendar
Targets: y (z at t), y_phys, clim_phys, std_phys (for scoring in physical units), valid_frac.
Row flags: role (train / eval), train_ok (role=train, no lockdown, finite target, coverage ok), lockdown_target,
exploratory, spent.

Usage
  python feature_extractor.py --data data_v4 --out features_v4
  python feature_extractor.py --data data_v4 --out features_v4 --horizons 1 2 --modes forecast perfect_prog
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("feature_extractor")

DEFAULT_CONFIG = {
    "horizons": [1, 2],
    "n_lags": 4,
    "n_met_lags": 2,
    "level_windows": 6,            # 30 days
    "level_min_obs": 3,
    "clim_min_obs": 3,             # monthly climatology needs this many training windows, else unit-gas mean
    "min_valid_frac": 0.2,         # windows below this coverage are missing (inputs and targets)
    "folds": [{"name": f"dev_{y}", "train": list(range(2019, y)), "eval": [y]} for y in range(2021, 2026)],
    "final_fold": {"name": "final_2026", "train": list(range(2019, 2026)), "eval": [2026]},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(data_dir: Path):
    man = json.loads((data_dir / "manifest.json").read_text())
    path = data_dir / man["table"]["path"]
    if sha256_file(path) != man["table"]["sha256"]:
        raise RuntimeError(f"{path} does not match its manifest hash: it was edited after preprocessing")
    t = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, parse_dates=["window_start"])
    return t, man


class Panel:
    """Dense arrays per unit on the shared window grid: values (U, T, G), drivers (U, T, V), flags (T,)."""

    def __init__(self, t: pd.DataFrame):
        self.units = sorted(t["unit"].unique())
        self.gases = sorted(t["gas"].unique())
        self.dates = pd.DatetimeIndex(sorted(t["window_start"].unique()))
        U, T, G = len(self.units), len(self.dates), len(self.gases)
        ui = t["unit"].map({u: i for i, u in enumerate(self.units)}).values
        ti = pd.DatetimeIndex(t["window_start"]).map({d: i for i, d in enumerate(self.dates)}).values
        gi = t["gas"].map({g: i for i, g in enumerate(self.gases)}).values
        self.value = np.full((U, T, G), np.nan)
        self.vfrac = np.zeros((U, T, G))
        self.segment = np.zeros((U, T, G), int)
        self.value[ui, ti, gi] = t["value"].values
        self.vfrac[ui, ti, gi] = t["valid_frac"].values
        self.segment[ui, ti, gi] = t["segment"].values
        first = t.drop_duplicates(["unit", "window_start"])
        fu = first["unit"].map({u: i for i, u in enumerate(self.units)}).values
        ft = pd.DatetimeIndex(first["window_start"]).map({d: i for i, d in enumerate(self.dates)}).values
        self.met_names = [c for c in t.columns if c.startswith("era5_")]
        self.has_firms = "firms_count" in t.columns
        self.met = np.full((U, T, len(self.met_names)), np.nan)
        for k, c in enumerate(self.met_names):
            self.met[fu, ft, k] = first[c].values
        self.firms = np.full((U, T), np.nan)
        if self.has_firms:
            self.firms[fu, ft] = np.log1p(first["firms_count"].values)
        cal = first.drop_duplicates("window_start").set_index("window_start").reindex(self.dates)
        self.year = self.dates.year.values
        self.month = cal["month"].values.astype(int)
        self.lockdown = cal["lockdown"].values.astype(bool)
        self.cal = cal[["doy_sin", "doy_cos", "weekend_frac"]].values
        self.unit_type = first.drop_duplicates("unit").set_index("unit").reindex(self.units)["unit_type"].values
        self.spent = cal["spent"].values.astype(bool)


def monthly_clim(x, fit_mask, month, min_obs):
    """x (U, T, K); fit_mask (U, T, K) bool. Returns clim (U, T, K) from training windows, by calendar month."""
    U, T, K = x.shape
    xm = np.where(fit_mask, x, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        overall = np.nanmean(xm, axis=1)                               # (U, K)
        clim12 = np.empty((U, 12, K))
        for mo in range(1, 13):
            sel = xm[:, month == mo]
            n = np.isfinite(sel).sum(1)
            clim12[:, mo - 1] = np.where(n >= min_obs, np.nanmean(sel, axis=1), overall)
    return clim12[:, month - 1]                                        # (U, T, K)


def fit_transform(p: Panel, train_years, cfg):
    """Fit every statistic on training years (lockdown and low-coverage windows excluded); apply to all windows."""
    tr = np.isin(p.year, train_years) & ~p.lockdown                    # (T,)
    ok = np.isfinite(p.value) & (p.vfrac >= cfg["min_valid_frac"])
    value = np.where(ok, p.value, np.nan)
    fit = ok & tr[None, :, None]
    clim = monthly_clim(value, fit, p.month, cfg["clim_min_obs"])
    anom = value - clim
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        std = np.nanstd(np.where(fit, anom, np.nan), axis=1)           # (U, G)
    std = np.where(np.isfinite(std) & (std > 0), std, np.nan)
    z = anom / std[:, None, :]

    def drivers(x):
        if x.shape[-1] == 0:
            return x, {}
        f = np.isfinite(x) & tr[None, :, None]
        c = monthly_clim(x, f, p.month, cfg["clim_min_obs"])
        a = x - c
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = np.nanstd(np.where(f, a, np.nan), axis=1)
        s = np.where(np.isfinite(s) & (s > 0), s, 1.0)
        return a / s[:, None, :], {"std": s.tolist()}

    met_z, met_stats = drivers(p.met)
    firms_z, firms_stats = drivers(p.firms[..., None])
    stats = {"train_years": list(train_years), "n_train_windows": int(tr.sum()),
             "anomaly_std": {u: dict(zip(p.gases, map(float, std[i]))) for i, u in enumerate(p.units)},
             "met_std": met_stats, "firms_std": firms_stats}
    return {"value": value, "clim": clim, "std": std, "z": z, "met_z": met_z, "firms_z": firms_z[..., 0]}, stats


def causal_level(p: Panel, z, cfg):
    """level[u, s, g] = mean of z over windows s-L+1..s in the same processor segment, lockdown windows skipped."""
    L, mn = cfg["level_windows"], cfg["level_min_obs"]
    U, T, G = z.shape
    use = np.isfinite(z) & ~p.lockdown[None, :, None]
    lev = np.full_like(z, np.nan)
    for s in range(T):
        lo = max(0, s - L + 1)
        win = z[:, lo:s + 1]
        same = p.segment[:, lo:s + 1] == p.segment[:, s:s + 1]
        m = use[:, lo:s + 1] & same
        n = m.sum(1)
        tot = np.where(m, win, 0).sum(1)
        lev[:, s] = np.where(n >= mn, tot / np.maximum(n, 1), np.nan)
    return lev


def _fill(a):
    m = np.isfinite(a)
    return np.where(m, a, 0.0), m.astype(np.float32)


def build_rows(p: Panel, tf, h, fold, cfg, mode):
    z, met_z, firms_z = tf["z"], tf["met_z"], tf["firms_z"]
    lev = causal_level(p, z, cfg)
    K, KM = cfg["n_lags"], cfg["n_met_lags"]
    years = set(fold["train"]) | set(fold["eval"])
    first_t = h + max(K, KM, 1) - 1
    t_idx = np.array([t for t in range(first_t, len(p.dates)) if p.year[t] in years])
    if len(t_idx) == 0:
        return pd.DataFrame()
    s = t_idx - h                                                       # latest window a forecast may read
    frames = []
    for gi, gas in enumerate(p.gases):
        for ui, unit in enumerate(p.units):
            cols = {"unit": unit, "unit_type": p.unit_type[ui], "gas": gas,
                    "window_start": p.dates[t_idx], "t_index": t_idx, "year": p.year[t_idx]}
            if mode == "forecast":
                last = np.full(len(t_idx), np.nan)
                for k in range(K - 1, -1, -1):                          # oldest first; newer overwrite
                    v, m = _fill(z[ui, s - k, gi])
                    cols[f"own_lag{k}"], cols[f"own_mask{k}"] = v, m
                    src = z[ui, s - k, gi]
                    last = np.where(np.isfinite(src), src, last)
                cols["own_last_valid"], cols["own_last_valid_mask"] = _fill(last)
                cols["own_level"], cols["own_level_mask"] = _fill(lev[ui, s, gi])
                for gj, other in enumerate(p.gases):
                    if gj != gi:
                        cols[f"x_{other}_lag0"], cols[f"x_{other}_mask0"] = _fill(z[ui, s, gj])
                for k in range(KM):
                    for vi, name in enumerate(p.met_names):
                        cols[f"{name}_lag{k}"], cols[f"{name}_mask{k}"] = _fill(met_z[ui, s - k, vi])
                    if p.has_firms:
                        cols[f"firms_lag{k}"], cols[f"firms_mask{k}"] = _fill(firms_z[ui, s - k])
                cols["doy_sin"], cols["doy_cos"], cols["weekend_frac"] = p.cal[t_idx].T
            else:                                                       # perfect prognosis: target-window weather
                for vi, name in enumerate(p.met_names):
                    cols[f"pp_{name}"], cols[f"pp_{name}_mask"] = _fill(met_z[ui, t_idx, vi])
                if p.has_firms:
                    cols["pp_firms"], cols["pp_firms_mask"] = _fill(firms_z[ui, t_idx])
            y = z[ui, t_idx, gi]
            cols["y"] = y
            cols["y_phys"] = tf["value"][ui, t_idx, gi]
            cols["clim_phys"] = tf["clim"][ui, t_idx, gi]
            cols["std_phys"] = np.full(len(t_idx), tf["std"][ui, gi])
            cols["valid_frac"] = p.vfrac[ui, t_idx, gi]
            frames.append(pd.DataFrame(cols))
    df = pd.concat(frames, ignore_index=True)
    df["horizon"] = h
    df["fold"] = fold["name"]
    df["role"] = np.where(df["year"].isin(fold["train"]), "train", "eval")
    df["lockdown_target"] = p.lockdown[df["t_index"].values]
    df["train_ok"] = (df["role"] == "train") & ~df["lockdown_target"] & np.isfinite(df["y"])
    df["exploratory"] = True if fold["name"] != cfg["final_fold"]["name"] else False
    df["spent"] = p.spent[df["t_index"].values]
    return df


def feature_columns(df, mode):
    skip = {"unit", "unit_type", "gas", "window_start", "t_index", "year", "y", "y_phys", "clim_phys",
            "std_phys", "valid_frac", "horizon", "fold", "role", "lockdown_target", "train_ok", "exploratory",
            "spent"}
    return [c for c in df.columns if c not in skip]


def extract(t: pd.DataFrame, cfg, folds, horizons, modes):
    """Returns {(fold, h, mode): DataFrame} and fitted stats per fold. Pure function: used by the tests."""
    for fold in folds:
        if max(fold["eval"]) >= 2026:
            raise RuntimeError("the 2026 holdout fold is only built after the freeze (phase 5)")
    p = Panel(t)
    out, stats = {}, {}
    for fold in folds:
        tf, st = fit_transform(p, fold["train"], cfg)
        stats[fold["name"]] = st
        for h in horizons:
            for mode in modes:
                out[(fold["name"], h, mode)] = build_rows(p, tf, h, fold, cfg, mode)
    return out, stats, p


def write(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except ImportError:
        df.to_csv(path.with_suffix(".csv.gz"), index=False)
        return path.with_suffix(".csv.gz")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_v4", help="folder written by preprocessing.py")
    ap.add_argument("--out", default="features_v4")
    ap.add_argument("--horizons", type=int, nargs="+")
    ap.add_argument("--modes", nargs="+", default=["forecast", "perfect_prog"], choices=["forecast", "perfect_prog"])
    ap.add_argument("--config", help="JSON overriding DEFAULT_CONFIG keys")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))
    horizons = args.horizons or cfg["horizons"]
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t, man = read_table(data_dir)
    modes = list(args.modes)
    if "perfect_prog" in modes and not any(c.startswith("era5_") for c in t.columns) and "firms_count" not in t:
        log.warning("no ERA5/FIRMS columns in the window table: skipping perfect_prog mode")
        modes.remove("perfect_prog")
    tables, stats, p = extract(t, cfg, cfg["folds"], horizons, modes)

    files = {}
    for (fold, h, mode), df in tables.items():
        name = f"features_{fold}_h{h}" if mode == "forecast" else f"features_pp_{fold}_h{h}"
        path = write(df, out / name)
        files[path.name] = {"sha256": sha256_file(path), "rows": len(df), "fold": fold, "horizon": h, "mode": mode,
                            "train_ok_rows": int(df["train_ok"].sum()),
                            "eval_rows_with_target": int(((df["role"] == "eval") & np.isfinite(df["y"])).sum()),
                            "feature_columns": feature_columns(df, mode)}
        info = files[path.name]
        log.info("%s: %d rows (%d train_ok, %d eval with target)", path.name, len(df), info["train_ok_rows"],
                 info["eval_rows_with_target"])
        if info["eval_rows_with_target"] == 0:
            log.warning("%s: the eval year has no observed targets (data ends earlier?)", path.name)
    manifest = {
        "stage": "feature_extractor", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": sha256_file(Path(__file__)), "config": cfg,
        "input_table": {"path": str(data_dir / man["table"]["path"]), "sha256": man["table"]["sha256"],
                        "preprocessing_script_sha256": man["script_sha256"]},
        "units": p.units, "gases": p.gases, "met_features": p.met_names, "firms": p.has_firms,
        "folds": cfg["folds"], "fit_stats": stats, "files": files,
        "rules": ["statistics fitted on training years only, lockdown and low-coverage windows excluded",
                  "forecast features read windows <= t-h only (calendar and lockdown flag of t are known in advance)",
                  "perfect-prognosis features are in features_pp_* files only: upper bound, never a forecast claim",
                  "all folds are development folds on spent data: report as EXPLORATORY",
                  "the 2026 holdout fold is not built here"],
    }
    (out / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("wrote %d feature files and feature_manifest.json to %s", len(files), out)


if __name__ == "__main__":
    main()
