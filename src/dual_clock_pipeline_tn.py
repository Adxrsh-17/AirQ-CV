#!/usr/bin/env python3
"""Dual-Clock Architecture: State-Wide Tamil Nadu Execution (11,685 Spatial Cells).

Executes the IEEE Dual-Clock Land-Emission Architecture across the full state of Tamil Nadu:
  - Grid: 123 x 95 cells (approx 5.5 km resolution, 11,685 cells)
  - Time Span: 415 five-day windows (2019-01-01 to 2024-12-31)
  - Meteorology: 5-Cluster Tamil Nadu ERA5 Reanalysis (Chennai, Coimbatore, Trichy, Madurai, Tirunelveli)
  - Component C1: Slow Clock annual dry-season land conversion (NDBI, NDVI) + Vectorized Spatial Lag W*dL
  - Component C2: Multi-Cluster Deweathering (LightGBM) + Two-Way Fixed Effects Panel OLS (~70,000 pixel-years)
  - Falsification: 500-Iteration Spatial Placebo Permutations + Gate G2
  - Component C3 & Fusion: Multi-Horizon Forecasting (5 to 150 days) on 2023-2024 Held-Out Test Set
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dual_clock_tn")

# 5 Regional Climate Clusters in Tamil Nadu
CLUSTERS = {
    "chennai": {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    "coimbatore": {"name": "Coimbatore", "lat": 11.0168, "lon": 76.9558},
    "trichy": {"name": "Trichy", "lat": 10.7905, "lon": 78.7047},
    "madurai": {"name": "Madurai", "lat": 9.9252, "lon": 78.1198},
    "tirunelveli": {"name": "Tirunelveli", "lat": 8.7139, "lon": 77.7567},
}


# ==============================================================================
# 1. LOAD FULL TAMIL NADU DATASET (123 x 95)
# ==============================================================================
def load_full_tn_dataset(s2_dir: Path, s5p_dir: Path):
    log.info("Matching S2 and S5P composites across full Tamil Nadu grid (123 x 95)...")
    s2_files = sorted(os.listdir(s2_dir))
    s5p_files = sorted(os.listdir(s5p_dir))

    s2_map = {re.search(r"(\d{8})", f).group(1): f for f in s2_files if re.search(r"(\d{8})", f)}
    s5p_map = {re.search(r"(\d{8})", f).group(1): f for f in s5p_files if re.search(r"(\d{8})", f)}
    common_dates = sorted(set(s2_map.keys()) & set(s5p_map.keys()))
    log.info("Matched %d concurrent 5-day windows (%s to %s)", len(common_dates), common_dates[0], common_dates[-1])

    # Reference grid parameters from S5P
    with rasterio.open(s5p_dir / s5p_map[common_dates[0]]) as s5p_ref:
        H, W = s5p_ref.shape
        transform = s5p_ref.transform
        bounds = s5p_ref.bounds

    with rasterio.open(s2_dir / s2_map[common_dates[0]]) as s2_ref:
        H_s2, W_s2 = s2_ref.shape

    log.info("Full State Grid Dimensions: S5P=(%d, %d), S2=(%d, %d)", H, W, H_s2, W_s2)
    log.info("Bounding Box: [%.4f, %.4f, %.4f, %.4f]", bounds.left, bounds.bottom, bounds.right, bounds.top)

    # Precompute cell centroids (lat, lon) and closest weather cluster
    cell_lats = np.array([bounds.top - (i + 0.5) * transform.e for i in range(H)])
    cell_lons = np.array([bounds.left + (j + 0.5) * transform.a for j in range(W)])
    lon_grid, lat_grid = np.meshgrid(cell_lons, cell_lats)

    cluster_keys = list(CLUSTERS.keys())
    c_lats = np.array([CLUSTERS[k]["lat"] for k in cluster_keys])
    c_lons = np.array([CLUSTERS[k]["lon"] for k in cluster_keys])

    # Map each cell to closest cluster index
    dist_sq = (lat_grid[:, :, None] - c_lats[None, None, :]) ** 2 + (lon_grid[:, :, None] - c_lons[None, None, :]) ** 2
    cell_cluster_idx = np.argmin(dist_sq, axis=-1)  # (H, W)

    # Block reduction grid indices
    grid_y = np.linspace(0, H_s2, H + 1, dtype=int)
    grid_x = np.linspace(0, W_s2, W + 1, dtype=int)

    s5p_series, s2_ndbi_series, s2_ndvi_series, records = [], [], [], []
    t0 = time.time()

    for d in common_dates:
        # Load S5P NO2 (Band 1)
        with rasterio.open(s5p_dir / s5p_map[d]) as s5p_src:
            s5p_arr = s5p_src.read(1)

        # Load S2 NDBI (Band 11) and NDVI (Band 10)
        with rasterio.open(s2_dir / s2_map[d]) as s2_src:
            s2_arr = s2_src.read()
            ndbi_raw = s2_arr[10]
            ndvi_raw = s2_arr[9]

        # Fast vectorized block reduce
        ndbi_down = np.empty((H, W), dtype=np.float32)
        ndvi_down = np.empty((H, W), dtype=np.float32)
        for i in range(H):
            y0, y1 = grid_y[i], grid_y[i + 1]
            for j in range(W):
                x0, x1 = grid_x[j], grid_x[j + 1]
                b_ndbi = ndbi_raw[y0:y1, x0:x1]
                b_ndvi = ndvi_raw[y0:y1, x0:x1]
                v_mask = (b_ndbi > -2.0) & (b_ndbi < 2.0)
                ndbi_down[i, j] = np.nanmean(b_ndbi[v_mask]) if v_mask.any() else np.nan
                ndvi_down[i, j] = np.nanmean(b_ndvi[v_mask]) if v_mask.any() else np.nan

        s5p_series.append(s5p_arr)
        s2_ndbi_series.append(ndbi_down)
        s2_ndvi_series.append(ndvi_down)

        dt = pd.to_datetime(d, format="%Y%m%d")
        records.append({"date_str": d, "date": dt, "year": dt.year, "month": dt.month})

    log.info("Finished ingesting %d full-state windows in %.1f s", len(common_dates), time.time() - t0)

    no2_cube = np.stack(s5p_series, axis=0)
    ndbi_cube = np.stack(s2_ndbi_series, axis=0)
    ndvi_cube = np.stack(s2_ndvi_series, axis=0)
    meta_df = pd.DataFrame(records)

    # Clean missing / invalid pixels
    no2_cube[no2_cube < -100] = np.nan
    no2_cube[no2_cube <= -0.0005] = np.nan
    ndbi_cube[ndbi_cube < -1.5] = np.nan
    ndvi_cube[ndvi_cube < -1.5] = np.nan

    grid_info = {
        "H": H, "W": W, "lat_grid": lat_grid, "lon_grid": lon_grid,
        "cell_cluster_idx": cell_cluster_idx, "cluster_keys": cluster_keys
    }
    return no2_cube, ndbi_cube, ndvi_cube, meta_df, grid_info


# ==============================================================================
# 2. COMPONENT C1: SLOW CLOCK (ANNUAL DYNAMICS ACROSS TAMIL NADU)
# ==============================================================================
def compute_annual_land_change_tn(ndbi_cube, ndvi_cube, meta_df, grid_info):
    log.info("Computing Component C1: Annual dry-season land state and change across Tamil Nadu...")
    years = sorted(meta_df["year"].unique())
    H, W = grid_info["H"], grid_info["W"]

    annual_ndbi = {}
    annual_ndvi = {}

    for y in years:
        mask = (meta_df["year"] == y) & (meta_df["month"].isin([1, 2, 3]))
        if not mask.any():
            mask = (meta_df["year"] == y)
        annual_ndbi[y] = np.nanmedian(ndbi_cube[mask], axis=0)
        annual_ndvi[y] = np.nanmedian(ndvi_cube[mask], axis=0)

    # Baseline 2019
    base_ndbi = annual_ndbi[2019]
    base_ndvi = annual_ndvi[2019]

    # Land mask: where multi-year NDBI is valid land (not deep water)
    mean_ndbi = np.nanmean(np.stack(list(annual_ndbi.values())), axis=0)
    mean_ndvi = np.nanmean(np.stack(list(annual_ndvi.values())), axis=0)
    land_mask = np.isfinite(mean_ndbi) & np.isfinite(mean_ndvi) & ~((mean_ndvi < 0.0) & (mean_ndbi < -0.1))
    log.info("Active Land Pixels in Tamil Nadu: %d / %d (%.1f%%)", land_mask.sum(), H * W, land_mask.mean() * 100)

    rows = []
    for y in years:
        d_ndbi = annual_ndbi[y] - base_ndbi
        d_ndvi = annual_ndvi[y] - base_ndvi

        # Fast vectorized spatial lag W * delta_L (3x3 ring average)
        w_d_ndbi = fast_spatial_ring_average(d_ndbi, land_mask)

        for i in range(H):
            for j in range(W):
                if not land_mask[i, j]:
                    continue
                p_id = f"p_{i:03d}_{j:03d}"
                c_idx = grid_info["cell_cluster_idx"][i, j]
                rows.append({
                    "pixel": p_id,
                    "i": i,
                    "j": j,
                    "lat": grid_info["lat_grid"][i, j],
                    "lon": grid_info["lon_grid"][i, j],
                    "cluster": grid_info["cluster_keys"][c_idx],
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
    log.info("GATE G1 (STATE-WIDE): Pixels with |delta_NDBI| > 0.03 in 2024: %d / %d (%.1f%%)",
             meaningful, len(delta_2024), (meaningful / max(len(delta_2024), 1)) * 100)
    log.info("  delta_NDBI range: min=%.4f, median=%.4f, max=%.4f, std=%.4f",
             delta_2024.min(), delta_2024.median(), delta_2024.max(), delta_2024.std())

    return df_land, annual_ndbi, annual_ndvi, land_mask


def fast_spatial_ring_average(arr, mask):
    """Vectorized 3x3 spatial ring average excluding self."""
    H, W = arr.shape
    arr_clean = np.where(mask & np.isfinite(arr), arr, 0.0)
    val_clean = np.where(mask & np.isfinite(arr), 1.0, 0.0)

    # Pad array
    arr_pad = np.pad(arr_clean, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    val_pad = np.pad(val_clean, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    sum_neighbors = np.zeros((H, W), dtype=np.float32)
    count_neighbors = np.zeros((H, W), dtype=np.float32)

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            sum_neighbors += arr_pad[1 + di:1 + di + H, 1 + dj:1 + dj + W]
            count_neighbors += val_pad[1 + di:1 + di + H, 1 + dj:1 + dj + W]

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count_neighbors > 0, sum_neighbors / count_neighbors, np.nan)
    return out


# ==============================================================================
# 3. COMPONENT C2: MULTI-CLUSTER DEWEATHERING & PANEL ATTRIBUTION
# ==============================================================================
def meteorological_normalisation_tn(no2_cube, meta_df, era5_clusters_df, grid_info, land_mask):
    log.info("Component C2: Multi-Cluster Meteorological Normalisation across Tamil Nadu...")
    import lightgbm as lgb

    meta_df["date_only"] = meta_df["date"].dt.date
    era5_clusters_df["date_only"] = pd.to_datetime(era5_clusters_df["date"]).dt.date

    # Add cyclical harmonics
    doy = pd.to_datetime(meta_df["date"]).dt.dayofyear.values
    meta_df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    meta_df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    T, H, W = no2_cube.shape
    deweathered_cube = np.full((T, H, W), np.nan, dtype=np.float32)

    train_mask = (meta_df["year"] <= 2022).values
    lockdown_mask = ((pd.to_datetime(meta_df["date"]) >= "2020-03-25") &
                     (pd.to_datetime(meta_df["date"]) <= "2020-06-30")).values
    fit_mask = train_mask & (~lockdown_mask)

    cluster_keys = grid_info["cluster_keys"]

    # Fit LightGBM deweathering model per cluster
    for c_idx, cid in enumerate(cluster_keys):
        c_weather = era5_clusters_df[era5_clusters_df["cluster"] == cid].copy()
        m_df = pd.merge(meta_df, c_weather, on="date_only", how="left")
        m_df["ventilation"] = m_df["blh"] * m_df["wind_speed"]

        wx_cols = ["blh", "u10", "v10", "wind_speed", "t2m", "rh", "tp", "ventilation", "doy_sin", "doy_cos"]
        X_wx = m_df[wx_cols].ffill().bfill().values
        mean_wx = np.tile(np.nanmean(X_wx[fit_mask], axis=0, keepdims=True), (T, 1))

        # Pixels belonging to this cluster
        pix_mask = (grid_info["cell_cluster_idx"] == c_idx) & land_mask
        n_cluster_pix = pix_mask.sum()
        if n_cluster_pix == 0:
            continue

        log.info("  Deweathering cluster '%s': %d active pixels...", cid, n_cluster_pix)
        y_cluster_mean = np.nanmean(no2_cube[:, pix_mask], axis=1)

        valid_fit = fit_mask & np.isfinite(y_cluster_mean)
        if valid_fit.sum() < 30:
            deweathered_cube[:, pix_mask] = no2_cube[:, pix_mask]
            continue

        # Train cluster-level deweathering model
        model = lgb.LGBMRegressor(num_leaves=12, learning_rate=0.04, n_estimators=50, random_state=42, verbose=-1)
        model.fit(X_wx[valid_fit], y_cluster_mean[valid_fit])

        pred_obs = model.predict(X_wx)
        pred_base = model.predict(mean_wx)
        delta_wx = (pred_obs - pred_base)[:, None]

        # Apply deweathering to all cluster pixels
        deweathered_cube[:, pix_mask] = no2_cube[:, pix_mask] - delta_wx

    log.info("Multi-Cluster Meteorological Normalisation complete.")
    return deweathered_cube, meta_df


def fit_two_way_fixed_effects_tn(df_land, deweathered_cube, meta_df, land_mask):
    log.info("Component C2: Estimating Two-Way Fixed-Effects Panel Model on Whole Tamil Nadu...")
    # Restrict training to <= 2022 to avoid data leakage into 2023-2024 test period
    years = sorted([y for y in meta_df["year"].unique() if y <= 2022])
    H, W = deweathered_cube.shape[1], deweathered_cube.shape[2]

    annual_no2 = []
    for y in years:
        mask = (meta_df["year"] == y) & (meta_df["month"].isin([1, 2, 3]))
        if not mask.any():
            mask = (meta_df["year"] == y)
        mean_y = np.nanmean(deweathered_cube[mask], axis=0)

        for i in range(H):
            for j in range(W):
                if not land_mask[i, j]:
                    continue
                annual_no2.append({
                    "pixel": f"p_{i:03d}_{j:03d}",
                    "year": y,
                    "no2_dw": mean_y[i, j]
                })

    df_panel = pd.merge(df_land, pd.DataFrame(annual_no2), on=["pixel", "year"])
    df_panel = df_panel.dropna(subset=["no2_dw", "delta_ndbi", "w_delta_ndbi"]).copy()

    log.info("Total Tamil Nadu Panel Sample Size: %d pixel-years", len(df_panel))

    # Fast Within transformation
    def within_transform(df, col):
        p_mean = df.groupby("pixel")[col].transform("mean")
        y_mean = df.groupby("year")[col].transform("mean")
        grand = df[col].mean()
        return df[col] - p_mean - y_mean + grand

    df_panel["y_within"] = within_transform(df_panel, "no2_dw")
    df_panel["x_within"] = within_transform(df_panel, "delta_ndbi")
    df_panel["wx_within"] = within_transform(df_panel, "w_delta_ndbi")

    X = np.column_stack([df_panel["x_within"].values, df_panel["wx_within"].values])
    y = df_panel["y_within"].values

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

    log.info("TAMIL NADU STATE PANEL ATTRIBUTION RESULTS:")
    log.info("  Sample size: %d pixel-years", len(df_panel))
    log.info("  beta (local delta_NDBI):     %+.6e (SE: %.6e, t: %+.2f)", beta_val, beta_se, t_stat)
    log.info("  theta (neighbor W*delta_L):   %+.6e (SE: %.6e, t: %+.2f)", theta_val, theta_se, theta_val / max(theta_se, 1e-12))

    # 500-iteration spatial placebo shuffle
    log.info("Running 500-iteration spatial placebo shuffle across Tamil Nadu...")
    rng = np.random.default_rng(123)
    placebo_betas = []
    unique_pixels = df_panel["pixel"].unique()
    x_by_pixel = df_panel.groupby("pixel")["x_within"].first()

    for _ in range(500):
        perm_pixels = rng.permutation(unique_pixels)
        p_map = dict(zip(unique_pixels, perm_pixels))
        shuffled_x = df_panel["pixel"].map(p_map).map(x_by_pixel).values
        if np.isnan(shuffled_x).any():
            continue
        X_p = np.column_stack([np.ones(len(y)), shuffled_x])
        b_p = np.linalg.lstsq(X_p, y, rcond=None)[0][1]
        placebo_betas.append(b_p)

    placebo_mean = np.mean(placebo_betas)
    placebo_p_val = np.mean(np.abs(placebo_betas) >= np.abs(beta_val))
    log.info("FALSIFICATION: Placebo shuffle mean beta: %+.6e, empirical p-value: %.4f", placebo_mean, placebo_p_val)

    gate_g2_pass = bool(placebo_p_val < 0.10 and abs(t_stat) > 1.96)
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
        "n_samples": len(df_panel),
        "df_panel": df_panel
    }


# ==============================================================================
# 4. COMPONENT C3 & FUSION: STATE-WIDE MULTI-HORIZON FORECASTING
# ==============================================================================
def run_dual_clock_forecasting_tn(no2_cube, df_land, panel_res, meta_df, land_mask):
    log.info("Components C3 & Fusion: Running Multi-Horizon Forecasts across Tamil Nadu...")
    T, H, W = no2_cube.shape
    P = H * W
    no2_mat = no2_cube.reshape(T, P)
    flat_mask = land_mask.reshape(P)

    # Climatology per month per pixel
    months = meta_df["month"].values
    clim_month = np.zeros((12, P), dtype=np.float32)
    train_mask = (meta_df["year"] <= 2022).values

    for m in range(1, 13):
        m_idx = train_mask & (months == m)
        if m_idx.any():
            clim_month[m - 1] = np.nanmean(no2_mat[m_idx], axis=0)

    clim_full = np.stack([clim_month[m - 1] for m in months], axis=0)
    anom_mat = no2_mat - clim_full

    # Autoregressive phi
    train_anom = anom_mat[train_mask]
    phi_num = np.nanmean(train_anom[1:, flat_mask] * train_anom[:-1, flat_mask])
    phi_den = np.nanmean(train_anom[:-1, flat_mask] ** 2)
    phi = float(np.clip(phi_num / max(phi_den, 1e-12), 0, 1))
    log.info("Fitted State-Wide Damped Persistence phi: %.4f", phi)

    beta = panel_res["beta"] if panel_res["gate_g2_pass"] else 0.0
    theta = panel_res["theta"] if panel_res["gate_g2_pass"] else 0.0

    test_mask = (meta_df["year"] >= 2023).values
    test_indices = np.where(test_mask)[0]

    horizons = [1, 2, 4, 6, 12, 24, 30]
    lead_days = [h * 5 for h in horizons]
    results = []

    for h, days in zip(horizons, lead_days):
        g_h = float(max(0.0, 1.0 - 0.15 * math.log2(max(1, h))))
        y_true_list, clim_list, damp_list, fused_list = [], [], [], []

        for t in test_indices:
            if t + h >= T:
                continue
            target_m = meta_df.iloc[t + h]["month"]
            y_obs = no2_mat[t + h]
            valid = flat_mask & np.isfinite(y_obs)
            if not valid.any():
                continue

            pred_clim = clim_month[target_m - 1]
            last_anom = anom_mat[t]
            pred_damp = pred_clim + (phi ** h) * np.nan_to_num(last_anom, 0.0)

            # Slow Clock: Annual structural land change from latest available year at issue time
            issue_year = meta_df.iloc[t]["year"]
            land_sub = df_land[df_land["year"] == issue_year]
            if len(land_sub) > 0 and (beta != 0.0 or theta != 0.0):
                b_shift = np.zeros(P, dtype=np.float32)
                p_flat = land_sub["i"].values * W + land_sub["j"].values
                b_vals = beta * land_sub["delta_ndbi"].values + theta * land_sub["w_delta_ndbi"].values
                b_shift[p_flat] = np.nan_to_num(b_vals, 0.0)
            else:
                b_shift = np.zeros(P, dtype=np.float32)

            # Fused Dual-Clock Forecast: Climatology + Gated Slow Clock + Gated Fast Persistence
            pred_fused = pred_clim + (1.0 - g_h) * b_shift + g_h * (phi ** h) * np.nan_to_num(last_anom, 0.0)

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
    log.info("\n=== TAMIL NADU STATE-WIDE MULTI-HORIZON FORECAST RESULTS (2023-2024 TEST) ===")
    log.info("\n%s", res_df.to_string(index=False))
    return res_df


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-dir", default=r"data\processed\s2_composites")
    ap.add_argument("--s5p-dir", default=r"data\processed\s5p_composites")
    ap.add_argument("--weather", default=r"weather\era5_tn_clusters.csv")
    ap.add_argument("--out", default=r"runs\dual_clock_tn_runs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Full TN
    no2_cube, ndbi_cube, ndvi_cube, meta_df, grid_info = load_full_tn_dataset(Path(args.s2_dir), Path(args.s5p_dir))

    # 2. Slow Clock C1: Annual Land Dynamics
    df_land, annual_ndbi, annual_ndvi, land_mask = compute_annual_land_change_tn(ndbi_cube, ndvi_cube, meta_df, grid_info)
    df_land.to_csv(out_dir / "c1_tn_annual_land_change.csv", index=False)

    # 3. Fast Clock C2: Deweathering + Fixed Effects Panel Attribution
    era5_clusters_df = pd.read_csv(args.weather)
    deweathered_cube, m_df = meteorological_normalisation_tn(no2_cube, meta_df, era5_clusters_df, grid_info, land_mask)
    panel_res = fit_two_way_fixed_effects_tn(df_land, deweathered_cube, m_df, land_mask)

    panel_summary = {k: v for k, v in panel_res.items() if k != "df_panel"}
    (out_dir / "c2_tn_panel_attribution.json").write_text(json.dumps(panel_summary, indent=2))

    # 4. Multi-Horizon Forecasting & Fusion
    forecast_df = run_dual_clock_forecasting_tn(no2_cube, df_land, panel_res, m_df, land_mask)
    forecast_df.to_csv(out_dir / "tn_multi_horizon_results.csv", index=False)

    log.info("Tamil Nadu State-Wide Dual-Clock pipeline execution complete! Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
