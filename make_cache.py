"""Build the aligned training cache from your notebook's GeoTIFFs (run once, CPU only).
Outputs (in --out): s5p_val.npy, s5p_mask.npy, s5p_month.npy, s5p_month_ok.npy, s2_month.npy (fp16, 8x grid),
s2_month_ok.npy, weather_5d.npy, meta.json.  Regular 5-day window axis: missing windows are kept (mask=0), never dropped."""
import argparse, os, json, datetime as dt, numpy as np, pandas as pd, rasterio
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from align_tn import s5p_grid, align_s2_to_s5p, R

def windows(start, end, step=5):
    cur, out = dt.datetime.strptime(start, "%Y-%m-%d"), []
    end = dt.datetime.strptime(end, "%Y-%m-%d")
    while cur + dt.timedelta(days=step) <= end:
        out.append(cur); cur += dt.timedelta(days=step)
    return out

def main(a):
    os.makedirs(a.out, exist_ok=True)
    W = windows(a.start, a.end); T = len(W)
    ref = next(os.path.join(a.s5p_dir, f) for f in sorted(os.listdir(a.s5p_dir)) if f.startswith("s5p_2"))
    tf, H5, W5, crs = s5p_grid(ref)
    win = None

    if getattr(a, "roi", None) == "chennai":
        bbox = [79.95, 12.80, 80.45, 13.30]
        with rasterio.open(ref) as s:
            w = rasterio.windows.from_bounds(*bbox, transform=s.transform).round_offsets().round_shape()
            c_off = max(0, min(int(w.col_off), s.width - 1))
            r_off = max(0, min(int(w.row_off), s.height - 1))
            w_w = min(int(w.width), s.width - c_off)
            w_h = min(int(w.height), s.height - r_off)
            win = rasterio.windows.Window(c_off, r_off, w_w, w_h)
            tf = rasterio.windows.transform(win, s.transform)
            H5, W5 = w_h, w_w
            print(f"[ROI Active: Chennai] Cropped S5P grid to ({H5}, {W5}) from bbox {bbox}.")

    val = np.zeros((T, 3, H5, W5), np.float32); msk = np.zeros((T, 3, H5, W5), np.uint8)
    for i, w in enumerate(tqdm(W, desc="S5P")):
        p = os.path.join(a.s5p_dir, f"s5p_{w:%Y%m%d}.tif")
        if not os.path.exists(p): continue
        with rasterio.open(p) as s:
            x = s.read(window=win).astype(np.float32)[:, :H5, :W5] if win else s.read().astype(np.float32)[:, :H5, :W5]
        ok = np.isfinite(x) & (x > 0 if a.zero_is_invalid else True)
        val[i, :, :x.shape[1], :x.shape[2]] = np.where(ok, x, 0); msk[i, :, :x.shape[1], :x.shape[2]] = ok
    months = pd.period_range(W[0], W[-1], freq="M"); M = len(months)
    mon_of = np.array([months.get_loc(pd.Period(w, "M")) for w in W])
    s5m = np.zeros((M, 3, H5, W5), np.float32); s5ok = np.zeros((M, 3, H5, W5), np.uint8)
    for m in range(M):
        ii = np.where(mon_of == m)[0]; c = msk[ii].sum(0)
        s5m[m] = (val[ii] * msk[ii]).sum(0) / np.maximum(c, 1); s5ok[m] = c > 0

    # Smart Resolution Selection & Cache Footprint Guard
    is_state_wide = (H5 > 25 or W5 > 25)
    if a.s2_ratio is not None and a.s2_ratio > 0:
        R_use = a.s2_ratio
    else:
        # Full-state Tamil Nadu raw S2 export is ~590m native -> R=12 preserves 100% of spatial data within ~2.9 GB.
        # Focused ROI (e.g. Chennai 13x13) has 100m native S2 -> R=32 yields ~300 MB.
        R_use = 12 if is_state_wide else 32

    H8, W8 = H5 * R_use, W5 * R_use
    est_bytes = M * 12 * H8 * W8 * 2  # float16
    est_gb = est_bytes / (1024 ** 3)
    print(f"[Cache Pre-Flight] S5P Grid: ({H5}, {W5}) | S2 Ratio: {R_use}x | High-Res S2 Grid: ({H8}, {W8})")
    print(f"[Cache Pre-Flight] Target s2_month.npy: {est_gb:.2f} GB ({est_bytes / (1024**2):.1f} MB)")

    if est_gb > 14.0 and not getattr(a, "force_large_cache", False):
        raise RuntimeError(
            f"Calculated cache size ({est_gb:.2f} GB) exceeds Kaggle's safe disk limit (~14 GB in /kaggle/working). "
            f"Grid ({H5}x{W5}) with R={R_use} creates {H8}x{W8} ({H8*W8:,} pixels) per month. "
            f"Solutions:\n"
            f"  1. For state-wide grids, use --s2_ratio 12 (~2.9 GB, matching native 500m S2 export);\n"
            f"  2. For high-res urban study (100m native), pass --roi chennai with --s2_ratio 32 (only ~300 MB);\n"
            f"  3. To bypass this guard if disk allows, add --force_large_cache 1."
        )

    s2m = np.lib.format.open_memmap(os.path.join(a.out, "s2_month.npy"), "w+", np.float16, (M, 12, H8, W8))
    s2ok = np.lib.format.open_memmap(os.path.join(a.out, "s2_month_ok.npy"), "w+", np.uint8, (M, H8, W8))

    def align_worker(item):
        i, w_date = item
        p = os.path.join(a.s2_dir, f"s2_{w_date:%Y%m%d}.tif")
        if not os.path.exists(p): return None
        x, v = align_s2_to_s5p(p, tf, H5, W5, crs, r=R_use); ok = v[0] > 0.5
        x[:, ~ok] = np.nan
        return x.astype(np.float16), ok

    with ThreadPoolExecutor(max_workers=4) as executor:
        for m in tqdm(range(M), desc="S2 monthly composites (4 workers)"):
            items = [(i, W[i]) for i in np.where(mon_of == m)[0]]
            results = list(executor.map(align_worker, items))
            stack = [r[0] for r in results if r is not None]
            if stack:
                med = np.nanmedian(np.stack(stack).astype(np.float32), axis=0); ok = np.isfinite(med[0])
                s2m[m] = np.nan_to_num(med, nan=0.0).astype(np.float16); s2ok[m] = ok

    for k, v in dict(s5p_val=val, s5p_mask=msk, s5p_month=s5m, s5p_month_ok=s5ok).items(): np.save(os.path.join(a.out, k + ".npy"), v)
    s2m.flush(); s2ok.flush()

    # Weather 5-day window extraction
    # Prefer era5_tn_clusters.csv for multi-cluster state-wide regional representation over Chennai-only era5_daily.csv
    wx_clusters_path = os.path.join(os.path.dirname(a.s5p_dir), "..", "weather", "era5_tn_clusters.csv")
    wx_default_path = os.path.join(os.path.dirname(a.s5p_dir), "..", "weather", "era5_daily.csv")
    wx_path = a.weather if (hasattr(a, "weather") and a.weather) else (wx_clusters_path if os.path.exists(wx_clusters_path) else wx_default_path)

    if os.path.exists(wx_path):
        df_wx = pd.read_csv(wx_path)
        df_wx["date"] = pd.to_datetime(df_wx["date"])
        wx_cols = ["blh", "u10", "v10", "wind_speed", "tp", "t2m", "rh"]

        # If multi-cluster dataset, aggregate across all regional clusters per day
        if "cluster" in df_wx.columns:
            print(f"Aggregating {df_wx['cluster'].nunique()} regional clusters from {wx_path}...")
            df_wx = df_wx.groupby("date")[wx_cols].mean().reset_index()

        # Handle missing 2024 blh values via causal DOY climatology imputation (strictly from training data)
        if df_wx["blh"].isna().any():
            nan_cnt = df_wx["blh"].isna().sum()
            print(f"Causally imputing {nan_cnt} missing ERA5 'blh' values using DOY training climatology...")
            tr_dates = df_wx[df_wx["date"] <= "2023-12-31"]
            doy_clim = tr_dates.groupby(tr_dates["date"].dt.dayofyear)["blh"].mean()

            for idx in df_wx[df_wx["blh"].isna()].index:
                d = df_wx.loc[idx, "date"].dayofyear
                if d in doy_clim and np.isfinite(doy_clim[d]):
                    val = doy_clim[d]
                else:
                    nearby = doy_clim[(doy_clim.index >= d - 15) & (doy_clim.index <= d + 15)]
                    val = nearby.mean() if len(nearby) > 0 else tr_dates["blh"].mean()
                df_wx.loc[idx, "blh"] = val

        wx_5d = np.zeros((T, len(wx_cols) + 1), np.float32)
        for i, w in enumerate(W):
            w_end = w + dt.timedelta(days=5)
            sub = df_wx[(df_wx["date"] >= w) & (df_wx["date"] < w_end)]
            if len(sub) > 0:
                vals = sub[wx_cols].mean().values
                vent = vals[0] * vals[3]
                wx_5d[i] = np.append(vals, vent)

        # Final check: ensure 0 NaNs remain
        if np.isnan(wx_5d).any():
            col_means = np.nanmean(wx_5d, axis=0)
            nan_rows = np.where(np.isnan(wx_5d).any(axis=1))[0]
            print(f"Warning: Imputing {len(nan_rows)} windows with remaining NaNs.")
            for r in nan_rows:
                for c_i in range(wx_5d.shape[1]):
                    if np.isnan(wx_5d[r, c_i]):
                        wx_5d[r, c_i] = col_means[c_i]

        np.save(os.path.join(a.out, "weather_5d.npy"), wx_5d)
        print("weather cache ready:", wx_5d.shape, "NaNs remaining:", int(np.isnan(wx_5d).sum()))

    json.dump(dict(T=T, M=M, H5=H5, W5=W5, H8=H8, W8=W8, s2_ratio=R_use, win_start=[f"{w:%Y-%m-%d}" for w in W],
                   month_start=[f"{p.start_time:%Y-%m-%d}" for p in months], month_end=[f"{p.end_time:%Y-%m-%d}" for p in months]),
              open(os.path.join(a.out, "meta.json"), "w"))
    print("cache ready:", T, "windows,", M, "months, S5P grid", (H5, W5), "S2 grid", (H8, W8), f"(ratio {R_use}x)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_dir"); ap.add_argument("--s5p_dir"); ap.add_argument("--out")
    ap.add_argument("--weather", default=None)
    ap.add_argument("--s2_ratio", type=int, default=0, help="S2 subgrid ratio per S5P cell (0=auto: 32 for Chennai ROI ~300MB, 12 for full TN ~2.9GB)")
    ap.add_argument("--roi", default="none", choices=["none", "chennai"], help="Crop ROI region (e.g. 'chennai' for high-res urban study)")
    ap.add_argument("--force_large_cache", type=int, default=0, help="Bypass the 14GB safety check if disk space allows")
    ap.add_argument("--start", default="2019-01-01"); ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--zero_is_invalid", type=int, default=1)
    main(ap.parse_args())
