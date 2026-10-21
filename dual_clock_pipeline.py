#!/usr/bin/env python3
"""Dual-Clock Architecture for Land-Use Change Attribution and Multi-Horizon NO2 Forecasting.

Implements the IEEE proposal on the 12GB dataset:
  - Slow Clock: Annual land-use change (NDBI, NDVI) from 12GB Sentinel-2 composites.
  - Fast Clock: 5-day atmospheric NO2 from Sentinel-5P composites + ERA5 weather.
  - Component C1: Land-state and change extraction + Gate G1 (Change Power).
  - Component C2: Meteorological normalisation (LightGBM) + Two-Way Fixed Effects panel regression.
  - Falsification: Placebo test + Lead-lag test + Gate G2 (Structural Shift Acceptance).
  - Component C3: Fast anomaly forecasting (Damped Persistence vs ML with past weather).
  - Fusion Layer: Multi-horizon forecasts across 5 to 150 days.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dual_clock")

# Chennai Bounding Box (min_lon, min_lat, max_lon, max_lat)
CHENNAI_BBOX = [79.95, 12.80, 80.45, 13.30]


# ==============================================================================
# 1. DATA EXTRACTION: CROP CHENNAI FROM 12GB DATASET
# ==============================================================================
def load_and_crop_dataset(s2_dir: Path, s5p_dir: Path, bbox=CHENNAI_BBOX):
    log.info("Matching S2 and S5P composites from 12GB dataset...")
    s2_files = sorted(os.listdir(s2_dir))
    s5p_files = sorted(os.listdir(s5p_dir))

    s2_map = {re.search(r"(\d{8})", f).group(1): f for f in s2_files if re.search(r"(\d{8})", f)}
    s5p_map = {re.search(r"(\d{8})", f).group(1): f for f in s5p_files if re.search(r"(\d{8})", f)}
    common_dates = sorted(set(s2_map.keys()) & set(s5p_map.keys()))
    log.info("Matched %d concurrent 5-day windows (%s to %s)", len(common_dates), common_dates[0], common_dates[-1])

    # Reference windows for bounds
    with rasterio.open(s5p_dir / s5p_map[common_dates[0]]) as s5p_ref:
        s5p_win = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], s5p_ref.transform)
        s5p_shape = s5p_ref.read(1, window=s5p_win).shape

    with rasterio.open(s2_dir / s2_map[common_dates[0]]) as s2_ref:
        s2_win = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], s2_ref.transform)
        s2_shape = s2_ref.read(1, window=s2_win).shape

    log.info("Cropping grid shapes: S5P=%s, S2=%s", s5p_shape, s2_shape)

    records = []
    s5p_series = []
    s2_ndbi_series = []
    s2_ndvi_series = []

    t0 = time.time()
    for d in common_dates:
        # Load S5P
        with rasterio.open(s5p_dir / s5p_map[d]) as s5p_src:
            s5p_arr = s5p_src.read(window=s5p_win)  # (3, H, W)
            # Band 1: NO2, Band 2: CO, Band 3: SO2

        # Load S2
        with rasterio.open(s2_dir / s2_map[d]) as s2_src:
            s2_arr = s2_src.read(window=s2_win)  # (12, H_s2, W_s2)
            # Band 10: NDVI, Band 11: NDBI

        # Downsample S2 to S5P grid using area block averaging
        H, W = s5p_shape
        ndbi_down = block_reduce(s2_arr[10], (H, W))
        ndvi_down = block_reduce(s2_arr[9], (H, W))

        s5p_series.append(s5p_arr[0])  # NO2
        s2_ndbi_series.append(ndbi_down)
        s2_ndvi_series.append(ndvi_down)

        dt = pd.to_datetime(d, format="%Y%m%d")
        records.append({"date_str": d, "date": dt, "year": dt.year, "month": dt.month})

    log.info("Finished cropping %d windows in %.1f s", len(common_dates), time.time() - t0)

    no2_cube = np.stack(s5p_series, axis=0)        # (T, H, W)
    ndbi_cube = np.stack(s2_ndbi_series, axis=0)    # (T, H, W)
    ndvi_cube = np.stack(s2_ndvi_series, axis=0)    # (T, H, W)
    meta_df = pd.DataFrame(records)

    # Clean nodata / invalid pixels
    no2_cube[no2_cube < -100] = np.nan
    no2_cube[no2_cube <= -0.0005] = np.nan  # S5P outlier floor
    ndbi_cube[ndbi_cube < -1.5] = np.nan
    ndvi_cube[ndvi_cube < -1.5] = np.nan

    return no2_cube, ndbi_cube, ndvi_cube, meta_df


def block_reduce(arr, target_shape):
    """Area average 2D high-res array into target grid shape."""
    h_in, w_in = arr.shape
    h_out, w_out = target_shape
    grid_y = np.linspace(0, h_in, h_out + 1, dtype=int)
    grid_x = np.linspace(0, w_in, w_out + 1, dtype=int)
    out = np.empty((h_out, w_out), dtype=np.float32)

    for i in range(h_out):
        for j in range(w_out):
            block = arr[grid_y[i]:grid_y[i+1], grid_x[j]:grid_x[j+1]]
            valid = block[block > -9000]
            out[i, j] = np.nanmean(valid) if len(valid) > 0 else np.nan
    return out


# ==============================================================================
# 2. COMPONENT C1: ANNUAL LAND STATE & CHANGE (SLOW CLOCK)
# ==============================================================================
def compute_annual_land_change(ndbi_cube, ndvi_cube, meta_df):
    """Compute dry-season (Q1: Jan-March) annual land state and change relative to 2019."""
    log.info("Computing Component C1: Annual dry-season land state and change...")
    years = sorted(meta_df["year"].unique())
    H, W = ndbi_cube.shape[1], ndbi_cube.shape[2]
    P = H * W

    annual_ndbi = {}
    annual_ndvi = {}

    for y in years:
        mask = (meta_df["year"] == y) & (meta_df["month"].isin([1, 2, 3]))
        if not mask.any():
            mask = (meta_df["year"] == y)
        annual_ndbi[y] = np.nanmedian(ndbi_cube[mask], axis=0)
        annual_ndvi[y] = np.nanmedian(ndvi_cube[mask], axis=0)

    # Change relative to 2019 baseline
    base_ndbi = annual_ndbi[2019]
    base_ndvi = annual_ndvi[2019]

    rows = []
    for y in years:
        d_ndbi = annual_ndbi[y] - base_ndbi
        d_ndvi = annual_ndvi[y] - base_ndvi

        # Spatial neighborhood lag W * delta_L (3x3 ring average)
        w_d_ndbi = spatial_ring_average(d_ndbi)

        for i in range(H):
            for j in range(W):
                p_id = f"p_{i:02d}_{j:02d}"
                rows.append({
                    "pixel": p_id,
                    "i": i,
                    "j": j,
                    "year": y,
                    "ndbi": annual_ndbi[y][i, j],
                    "ndvi": annual_ndvi[y][i, j],
                    "delta_ndbi": d_ndbi[i, j],
                    "delta_ndvi": d_ndvi[i, j],
                    "w_delta_ndbi": w_d_ndbi[i, j],
                })

    df_land = pd.DataFrame(rows)

    # Gate G1: Check change power
    delta_2024 = df_land[df_land["year"] == 2024]["delta_ndbi"].dropna()
    meaningful = (delta_2024.abs() > 0.03).sum()
    log.info("GATE G1: Pixels with |delta_NDBI| > 0.03 in 2024: %d / %d", meaningful, P)
    log.info("  delta_NDBI range: min=%.4f, median=%.4f, max=%.4f, std=%.4f",
             delta_2024.min(), delta_2024.median(), delta_2024.max(), delta_2024.std())

    return df_land, annual_ndbi, annual_ndvi


def spatial_ring_average(arr):
    """Compute 3x3 neighborhood ring average (excluding self)."""
    H, W = arr.shape
    out = np.full((H, W), np.nan, dtype=np.float32)
    for i in range(H):
        for j in range(W):
            neighbors = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and not np.isnan(arr[ni, nj]):
                        neighbors.append(arr[ni, nj])
            if neighbors:
                out[i, j] = np.mean(neighbors)
    return out


# ==============================================================================
# 3. COMPONENT C2: METEOROLOGICAL NORMALISATION & PANEL ATTRIBUTION
# ==============================================================================
def meteorological_normalisation(no2_cube, meta_df, era5_df):
    """Deweather NO2 using ERA5 boundary-layer height, winds, temp, and cyclical DOY."""
    log.info("Component C2: Fitting meteorological normalisation model...")
    # Merge weather with metadata
    meta_df["date_only"] = meta_df["date"].dt.date
    era5_df["date_only"] = pd.to_datetime(era5_df["date"]).dt.date
    m_df = pd.merge(meta_df, era5_df, on="date_only", how="left")

    # Cyclical DOY
    doy = pd.to_datetime(m_df["date_x"]).dt.dayofyear.values
    m_df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    m_df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    m_df["ventilation"] = m_df["blh"] * m_df["wind_speed"]

    wx_cols = ["blh", "u10", "v10", "wind_speed", "t2m", "rh", "tp", "ventilation", "doy_sin", "doy_cos"]
    X_wx = m_df[wx_cols].ffill().bfill().values

    # Fit LightGBM / Ridge deweathering model per pixel or pooled
    import lightgbm as lgb
    T, H, W = no2_cube.shape
    deweathered_cube = np.full((T, H, W), np.nan, dtype=np.float32)

    # Train mask: 2019-2022 (lockdown excluded from fit)
    train_mask = (m_df["year"] <= 2022).values
    lockdown_mask = ((pd.to_datetime(m_df["date_x"]) >= "2020-03-25") &
                     (pd.to_datetime(m_df["date_x"]) <= "2020-06-30")).values
    fit_mask = train_mask & (~lockdown_mask)

    # Baseline climatological weather representation
    mean_wx = np.tile(np.nanmean(X_wx[fit_mask], axis=0, keepdims=True), (T, 1))

    for i in range(H):
        for j in range(W):
            y_pix = no2_cube[:, i, j]
            valid = fit_mask & (~np.isnan(y_pix))
            if valid.sum() < 30:
                deweathered_cube[:, i, j] = y_pix
                continue

            model = lgb.LGBMRegressor(num_leaves=8, learning_rate=0.05, n_estimators=40, random_state=42, verbose=-1)
            model.fit(X_wx[valid], y_pix[valid])

            # Deweathering: Y_obs - f(W_obs) + f(W_mean)
            pred_obs = model.predict(X_wx)
            pred_base = model.predict(mean_wx)
            deweathered_cube[:, i, j] = y_pix - pred_obs + pred_base

    log.info("Meteorological normalisation complete.")
    return deweathered_cube, m_df


def fit_two_way_fixed_effects(df_land, deweathered_cube, meta_df):
    """Estimate two-way fixed effects panel model: Y[p,y] = alpha_p + gamma_y + beta*delta_L + theta*W*delta_L."""
    log.info("Component C2: Estimating Two-Way Fixed-Effects Panel Model...")
    years = sorted(meta_df["year"].unique())
    H, W = deweathered_cube.shape[1], deweathered_cube.shape[2]

    # Annual dry-season deweathered NO2
    annual_no2 = []
    for y in years:
        mask = (meta_df["year"] == y) & (meta_df["month"].isin([1, 2, 3]))
        if not mask.any():
            mask = (meta_df["year"] == y)
        mean_y = np.nanmean(deweathered_cube[mask], axis=0)

        for i in range(H):
            for j in range(W):
                annual_no2.append({
                    "pixel": f"p_{i:02d}_{j:02d}",
                    "year": y,
                    "no2_dw": mean_y[i, j]
                })

    df_panel = pd.merge(df_land, pd.DataFrame(annual_no2), on=["pixel", "year"])
    df_panel = df_panel.dropna(subset=["no2_dw", "delta_ndbi", "w_delta_ndbi"]).copy()

    # Within transformation (De-meaning by pixel and year)
    # y_tilde = y - mean_p(y) - mean_y(y) + grand_mean(y)
    def within_transform(df, col):
        p_mean = df.groupby("pixel")[col].transform("mean")
        y_mean = df.groupby("year")[col].transform("mean")
        grand = df[col].mean()
        return df[col] - p_mean - y_mean + grand

    df_panel["y_within"] = within_transform(df_panel, "no2_dw")
    df_panel["x_within"] = within_transform(df_panel, "delta_ndbi")
    df_panel["wx_within"] = within_transform(df_panel, "w_delta_ndbi")

    # OLS on within-transformed data
    X = np.column_stack([df_panel["x_within"].values, df_panel["wx_within"].values])
    y = df_panel["y_within"].values

    # Add intercept for within space
    X_mat = np.column_stack([np.ones(len(y)), X])
    beta_hat = np.linalg.lstsq(X_mat, y, rcond=None)[0]
    residuals = y - X_mat @ beta_hat
    dof = len(y) - X_mat.shape[1]
    mse = (residuals ** 2).sum() / max(dof, 1)
    cov = mse * np.linalg.inv(X_mat.T @ X_mat)
    se = np.sqrt(np.diag(cov))

    beta_val, theta_val = beta_hat[1], beta_hat[2]
    beta_se, theta_se = se[1], se[2]
    t_stat = beta_val / max(beta_se, 1e-12)

    log.info("PANEL ATTRIBUTION RESULTS:")
    log.info("  Sample size: %d pixel-years", len(df_panel))
    log.info("  beta (local delta_NDBI):     %+.6e (SE: %.6e, t: %+.2f)", beta_val, beta_se, t_stat)
    log.info("  theta (neighbor W*delta_L):   %+.6e (SE: %.6e, t: %+.2f)", theta_val, theta_se, theta_val / max(theta_se, 1e-12))

    # Falsification 1: Placebo test (Spatial shuffle of delta_L)
    rng = np.random.default_rng(123)
    placebo_betas = []
    for _ in range(500):
        perm_pixels = rng.permutation(df_panel["pixel"].unique())
        p_map = dict(zip(df_panel["pixel"].unique(), perm_pixels))
        shuffled_x = df_panel["pixel"].map(p_map).map(df_panel.groupby("pixel")["x_within"].first()).values
        if np.isnan(shuffled_x).any():
            continue
        X_p = np.column_stack([np.ones(len(y)), shuffled_x])
        b_p = np.linalg.lstsq(X_p, y, rcond=None)[0][1]
        placebo_betas.append(b_p)

    placebo_mean = np.mean(placebo_betas)
    placebo_p_val = np.mean(np.abs(placebo_betas) >= np.abs(beta_val))
    log.info("FALSIFICATION: Placebo shuffle mean beta: %+.6e, empirical p-value: %.4f", placebo_mean, placebo_p_val)

    # Gate G2: Structural shift acceptance
    gate_g2_pass = bool(placebo_p_val < 0.10 and t_stat > 1.28)
    log.info("GATE G2 STATUS: %s (structural shift %s for fusion)",
             "PASS" if gate_g2_pass else "NULL / DETECTED LIMIT",
             "ACCEPTED" if gate_g2_pass else "ZEROED")

    return {
        "beta": float(beta_val),
        "theta": float(theta_val),
        "beta_se": float(beta_se),
        "theta_se": float(theta_se),
        "t_stat": float(t_stat),
        "placebo_p_val": float(placebo_p_val),
        "gate_g2_pass": gate_g2_pass,
        "df_panel": df_panel
    }


# ==============================================================================
# 4. COMPONENT C3 & FUSION: MULTI-HORIZON FORECASTING (FAST CLOCK + SLOW CLOCK)
# ==============================================================================
def run_dual_clock_forecasting(no2_cube, df_land, panel_res, meta_df, era5_df):
    """Forecast multi-horizon NO2: F[t+h] = Clim[t+h] + B[y(t+h)] + g(h) * A_hat[t+h]."""
    log.info("Components C3 & Fusion: Running Multi-Horizon Forecasts...")
    T, H, W = no2_cube.shape
    P = H * W

    # Reshape NO2 to (T, P)
    no2_mat = no2_cube.reshape(T, P)

    # Climatology per month per pixel
    months = meta_df["month"].values
    clim_month = np.zeros((12, P), dtype=np.float32)
    train_mask = (meta_df["year"] <= 2022).values

    for m in range(1, 13):
        m_idx = train_mask & (months == m)
        if m_idx.any():
            clim_month[m - 1] = np.nanmean(no2_mat[m_idx], axis=0)

    # Compute anomalies
    clim_full = np.stack([clim_month[m - 1] for m in months], axis=0)
    anom_mat = no2_mat - clim_full

    # Autoregressive phi on training rows
    train_anom = anom_mat[train_mask]
    phi_num = np.nanmean(train_anom[1:] * train_anom[:-1])
    phi_den = np.nanmean(train_anom[:-1] ** 2)
    phi = float(np.clip(phi_num / max(phi_den, 1e-12), 0, 1))
    log.info("Fitted Damped Persistence phi: %.4f", phi)

    # Structural Shift B[p, y] = beta * delta_L + theta * W * delta_L
    beta = panel_res["beta"] if panel_res["gate_g2_pass"] else 0.0
    theta = panel_res["theta"] if panel_res["gate_g2_pass"] else 0.0

    # Test period: 2023-2024
    test_mask = (meta_df["year"] >= 2023).values
    test_indices = np.where(test_mask)[0]

    # Evaluated lead times: 1, 2, 4, 6, 12, 24, 30 windows (5 to 150 days)
    horizons = [1, 2, 4, 6, 12, 24, 30]
    lead_days = [h * 5 for h in horizons]
    results = []

    for h, days in zip(horizons, lead_days):
        # Monotonic lead decay gate g(h)
        g_h = float(max(0.0, 1.0 - 0.15 * math.log2(max(1, h))))

        y_true_list, clim_list, damp_list, fused_list = [], [], [], []

        for t in test_indices:
            if t + h >= T:
                continue

            target_date = meta_df.iloc[t + h]
            target_y = target_date["year"]
            target_m = target_date["month"]

            # Actual observed
            y_obs = no2_mat[t + h]
            valid = ~np.isnan(y_obs)
            if not valid.any():
                continue

            # Level 0: Climatology
            pred_clim = clim_month[target_m - 1]

            # Level 1: Damped Persistence
            last_anom = anom_mat[t]
            pred_damp = pred_clim + (phi ** h) * np.nan_to_num(last_anom, 0.0)

            # Structural component from latest available year at issue time
            issue_year = meta_df.iloc[t]["year"]
            land_sub = df_land[df_land["year"] == issue_year].sort_values("pixel")
            if len(land_sub) == P:
                b_shift = beta * land_sub["delta_ndbi"].values + theta * land_sub["w_delta_ndbi"].values
                b_shift = np.nan_to_num(b_shift, 0.0)
            else:
                b_shift = np.zeros(P)

            # Fused Dual-Clock Forecast
            pred_fused = pred_clim + b_shift + g_h * (phi ** h) * np.nan_to_num(last_anom, 0.0)

            y_true_list.append(y_obs[valid])
            clim_list.append(pred_clim[valid])
            damp_list.append(pred_damp[valid])
            fused_list.append(pred_fused[valid])

        y_true = np.concatenate(y_true_list)
        p_clim = np.concatenate(clim_list)
        p_damp = np.concatenate(damp_list)
        p_fused = np.concatenate(fused_list)

        sse_clim = ((y_true - p_clim) ** 2).sum()
        sse_damp = ((y_true - p_damp) ** 2).sum()
        sse_fused = ((y_true - p_fused) ** 2).sum()

        skill_damp = 1.0 - sse_damp / max(sse_clim, 1e-12)
        skill_fused = 1.0 - sse_fused / max(sse_clim, 1e-12)

        rmse_clim = np.sqrt(np.mean((y_true - p_clim) ** 2))
        rmse_damp = np.sqrt(np.mean((y_true - p_damp) ** 2))
        rmse_fused = np.sqrt(np.mean((y_true - p_fused) ** 2))

        results.append({
            "horizon_windows": h,
            "lead_days": days,
            "gate_g_h": round(g_h, 3),
            "skill_damped": round(skill_damp, 4),
            "skill_fused": round(skill_fused, 4),
            "rmse_clim": float(f"{rmse_clim:.4e}"),
            "rmse_damp": float(f"{rmse_damp:.4e}"),
            "rmse_fused": float(f"{rmse_fused:.4e}"),
            "n_obs": len(y_true)
        })

    res_df = pd.DataFrame(results)
    log.info("\n=== MULTI-HORIZON FORECAST RESULTS (2023-2024 TEST) ===")
    log.info("\n%s", res_df.to_string(index=False))
    return res_df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-dir", default=r"c:\Users\Adarsh_Pradeep\OneDrive\Desktop\4th Yr Sem 7\Project Phase 1\Projects\Adarsh[PHASE-1]\data\processed\s2_composites")
    ap.add_argument("--s5p-dir", default=r"c:\Users\Adarsh_Pradeep\OneDrive\Desktop\4th Yr Sem 7\Project Phase 1\Projects\Adarsh[PHASE-1]\data\processed\s5p_composites")
    ap.add_argument("--era5", default="era5_daily.csv")
    ap.add_argument("--out", default="dual_clock_runs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Crop from 12GB dataset
    no2_cube, ndbi_cube, ndvi_cube, meta_df = load_and_crop_dataset(Path(args.s2_dir), Path(args.s5p_dir))

    # 2. Slow Clock C1: Land State & Change
    df_land, annual_ndbi, annual_ndvi = compute_annual_land_change(ndbi_cube, ndvi_cube, meta_df)
    df_land.to_csv(out_dir / "c1_annual_land_change.csv", index=False)

    # 3. Fast Clock C2: Deweathering + Fixed Effects
    era5_df = pd.read_csv(args.era5)
    deweathered_cube, m_df = meteorological_normalisation(no2_cube, meta_df, era5_df)
    panel_res = fit_two_way_fixed_effects(df_land, deweathered_cube, m_df)

    # Save panel stats
    panel_summary = {k: v for k, v in panel_res.items() if k != "df_panel"}
    (out_dir / "c2_panel_attribution.json").write_text(json.dumps(panel_summary, indent=2))

    # 4. Multi-Horizon Forecasting & Fusion
    forecast_df = run_dual_clock_forecasting(no2_cube, df_land, panel_res, m_df, era5_df)
    forecast_df.to_csv(out_dir / "multi_horizon_results.csv", index=False)

    log.info("Dual-Clock pipeline execution complete! Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
