# H=1 (5-Day Ahead) Parallel ML Benchmark Pipeline vs. ConvGRU Deep Learning Reference
# Pollutant-specific engineered features & models:
# 1. NO2: LightGBM (NDBI, industrial distance, Sobel edge gradients 5x5, trailing 30d anomaly, month sin/cos)
# 2. CO: Ridge Regression AND LightGBM (Multi-scale 3x3, 9x9, 27x27 spatial means/stds, ONI/DMI, trailing 90d anomaly, current anomaly)
# 3. SO2: Two-Stage Hurdle Model (Classifier for exceedance + Magnitude Regressor on elevated cases)

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import distance_transform_edt, uniform_filter, sobel
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, roc_auc_score,
    precision_score, recall_score, f1_score, brier_score_loss
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

class DifferentiableSSIMLoss(nn.Module):
    def __init__(self, window_size=7, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        gaussian = torch.tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ], dtype=torch.float32)
        gaussian = (gaussian / gaussian.sum()).unsqueeze(1)
        kernel_2d = gaussian.mm(gaussian.t()).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel_2d)

    def forward(self, pred, true, mask=None):
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
            true = true.unsqueeze(0).unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0).unsqueeze(0)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        k = self.kernel
        mu1 = F.conv2d(pred, k, padding=self.window_size // 2)
        mu2 = F.conv2d(true, k, padding=self.window_size // 2)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, k, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(true * true, k, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(pred * true, k, padding=self.window_size // 2) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
        )
        if mask is not None:
            w_mask = F.conv2d(mask.float(), k, padding=self.window_size // 2)
            valid = w_mask > 0.3
            if valid.sum() == 0:
                return 1.0
            return float(ssim_map[valid].mean().item())
        return float(ssim_map.mean().item())

def compute_fss(trues_ch, preds_ch, masks_ch=None, window_size=9, threshold=None):
    if masks_ch is not None:
        t_valid = trues_ch[masks_ch > 0.5]
    else:
        t_valid = trues_ch.flatten()
    if len(t_valid) == 0:
        return np.nan
    if threshold is None:
        threshold = np.percentile(t_valid, 75)
    bin_true = (trues_ch >= threshold).astype(np.float32)
    bin_pred = (preds_ch >= threshold).astype(np.float32)
    if masks_ch is not None:
        bin_true = bin_true * (masks_ch > 0.5)
        bin_pred = bin_pred * (masks_ch > 0.5)
    frac_true = uniform_filter(bin_true, size=window_size, mode='constant', cval=0.0)
    frac_pred = uniform_filter(bin_pred, size=window_size, mode='constant', cval=0.0)
    if masks_ch is not None:
        frac_mask = uniform_filter((masks_ch > 0.5).astype(np.float32), size=window_size, mode='constant', cval=0.0)
        valid = frac_mask > 0.2
    else:
        valid = np.ones_like(frac_true, dtype=bool)
    if np.sum(valid) == 0:
        return np.nan
    mse = np.mean((frac_true[valid] - frac_pred[valid]) ** 2)
    ref = np.mean(frac_true[valid] ** 2) + np.mean(frac_pred[valid] ** 2)
    if ref == 0:
        return 1.0
    return float(np.clip(1.0 - (mse / ref), 0.0, 1.0))

def compute_csi_q90(trues_ch, preds_ch, masks_ch=None):
    if masks_ch is not None:
        m = masks_ch > 0.5
    else:
        m = np.ones_like(trues_ch, dtype=bool)
    if np.sum(m) < 10:
        return np.nan, np.nan, np.nan
    t_v = trues_ch[m]
    p_v = preds_ch[m]
    thresh = np.percentile(t_v, 90)
    hits = np.sum((p_v >= thresh) & (t_v >= thresh))
    misses = np.sum((p_v < thresh) & (t_v >= thresh))
    false_alarms = np.sum((p_v >= thresh) & (t_v < thresh))
    csi = hits / (hits + misses + false_alarms + 1e-8)
    pod = hits / (hits + misses + 1e-8)
    far = false_alarms / (hits + false_alarms + 1e-8)
    return float(csi), float(pod), float(far)

CLIMATE_INDICES = {
    (2019, 1): (0.7, 0.1), (2019, 2): (0.7, 0.2), (2019, 3): (0.7, 0.3), (2019, 4): (0.7, 0.4),
    (2019, 5): (0.7, 0.6), (2019, 6): (0.5, 0.8), (2019, 7): (0.3, 1.0), (2019, 8): (0.1, 1.2),
    (2019, 9): (0.2, 1.5), (2019, 10): (0.3, 1.8), (2019, 11): (0.5, 1.4), (2019, 12): (0.5, 0.6),
    (2020, 1): (0.5, 0.2), (2020, 2): (0.5, 0.1), (2020, 3): (0.4, 0.0), (2020, 4): (0.2, -0.1),
    (2020, 5): (-0.1, -0.2), (2020, 6): (-0.4, -0.3), (2020, 7): (-0.4, -0.3), (2020, 8): (-0.6, -0.2),
    (2020, 9): (-0.9, -0.1), (2020, 10): (-1.2, 0.0), (2020, 11): (-1.3, 0.1), (2020, 12): (-1.2, 0.1),
    (2021, 1): (-1.0, 0.0), (2021, 2): (-0.9, 0.0), (2021, 3): (-0.8, -0.1), (2021, 4): (-0.7, -0.2),
    (2021, 5): (-0.5, -0.3), (2021, 6): (-0.4, -0.4), (2021, 7): (-0.4, -0.5), (2021, 8): (-0.5, -0.4),
    (2021, 9): (-0.7, -0.3), (2021, 10): (-0.8, -0.2), (2021, 11): (-1.0, -0.1), (2021, 12): (-1.0, 0.0),
    (2022, 1): (-1.0, 0.0), (2022, 2): (-0.9, 0.0), (2022, 3): (-1.0, -0.1), (2022, 4): (-1.1, -0.2),
    (2022, 5): (-1.0, -0.3), (2022, 6): (-0.9, -0.5), (2022, 7): (-0.8, -0.6), (2022, 8): (-0.9, -0.7),
    (2022, 9): (-1.0, -0.8), (2022, 10): (-1.0, -0.6), (2022, 11): (-0.9, -0.3), (2022, 12): (-0.8, -0.1),
    (2023, 1): (-0.7, 0.0), (2023, 2): (-0.4, 0.1), (2023, 3): (-0.1, 0.2), (2023, 4): (0.2, 0.3),
    (2023, 5): (0.5, 0.5), (2023, 6): (0.8, 0.7), (2023, 7): (1.1, 0.9), (2023, 8): (1.3, 1.2),
    (2023, 9): (1.6, 1.5), (2023, 10): (1.8, 1.7), (2023, 11): (2.0, 1.3), (2023, 12): (2.0, 0.8),
    (2024, 1): (1.8, 0.4), (2024, 2): (1.5, 0.2), (2024, 3): (1.2, 0.1), (2024, 4): (0.8, 0.0),
    (2024, 5): (0.4, -0.1), (2024, 6): (0.2, -0.1), (2024, 7): (0.1, -0.2), (2024, 8): (-0.1, -0.2),
    (2024, 9): (-0.3, -0.1), (2024, 10): (-0.4, 0.0), (2024, 11): (-0.4, 0.0), (2024, 12): (-0.4, 0.0)
}

def main():
    print("="*75)
    print("H=1 (5-DAY AHEAD) DIRECT ML BENCHMARK PIPELINE vs ConvGRU (DL)")
    print("="*75)
    
    cache_path = "data/processed/cached_dataset_256.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    df = cache["df"].copy()
    s2 = cache["s2"].numpy()
    s5p = cache["s5p"].numpy()
    s5p_mask = cache["s5p_mask"].numpy()
    
    T, C_s2, H, W = s2.shape
    df["date"] = pd.to_datetime(df["start_date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear
    df["oni"] = [CLIMATE_INDICES.get((r.year, r.month), (0.0, 0.0))[0] for _, r in df.iterrows()]
    df["dmi"] = [CLIMATE_INDICES.get((r.year, r.month), (0.0, 0.0))[1] for _, r in df.iterrows()]
    
    pollutants = ["NO2", "CO", "SO2"]
    ssim_calc = DifferentiableSSIMLoss(window_size=7)
    
    hist_idx = df[df["year"] <= 2022].index.values
    holdout_idx = df[df["year"] >= 2023].index.values
    
    # Monthly Spatial Climatology
    monthly_clim = np.zeros((12, 3, H, W), dtype=np.float32)
    for m in range(1, 13):
        m_hist_idx = df[(df["year"] <= 2022) & (df["month"] == m)].index.values
        for c in range(3):
            v = s5p[m_hist_idx, c, :, :]
            mask = s5p_mask[m_hist_idx, c, :, :]
            sum_v = np.sum(v * mask, axis=0)
            sum_m = np.sum(mask, axis=0)
            valid_p = sum_m > 0
            clim_map = np.zeros((H, W), dtype=np.float32)
            clim_map[valid_p] = sum_v[valid_p] / sum_m[valid_p]
            if np.sum(valid_p) > 0 and np.sum(~valid_p) > 0:
                clim_map[~valid_p] = np.mean(clim_map[valid_p])
            monthly_clim[m - 1, c] = clim_map

    # Anomaly maps
    anomalies = np.zeros((T, 3, H, W), dtype=np.float32)
    for t in range(T):
        m = df.loc[t, "month"]
        for c in range(3):
            anomalies[t, c] = s5p[t, c] - monthly_clim[m - 1, c]

    # Precompute Static Features (NDBI, NDVI, Distance to Industrial)
    mean_s2 = np.mean(s2, axis=0)
    b4 = mean_s2[3]
    b8 = mean_s2[7]
    b11 = mean_s2[10]
    b12 = mean_s2[11]
    static_ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    static_ndbi = (b11 - b8) / (b11 + b8 + 1e-6)
    mean_hist_no2 = np.mean(s5p[hist_idx, 0], axis=0)
    ind_mask = (static_ndbi > 0.05) & (mean_hist_no2 > np.percentile(mean_hist_no2, 85))
    dist_to_ind = distance_transform_edt(~ind_mask).astype(np.float32)
    dist_to_ind = dist_to_ind / (np.max(dist_to_ind) + 1e-6)

    # Precompute Spatial Sobel Edge Gradient Magnitude on Sentinel-2 optical bands
    s2_gray = 0.2989 * mean_s2[3] + 0.5870 * mean_s2[2] + 0.1140 * mean_s2[1]
    sx = sobel(s2_gray, axis=0)
    sy = sobel(s2_gray, axis=1)
    sobel_grad = np.hypot(sx, sy).astype(np.float32)
    sobel_grad = sobel_grad / (np.max(sobel_grad) + 1e-6)

    # H=1 Sequence Formation (t0 to t0+1 = 5 days)
    H_steps = 1
    min_history = 18
    train_samples = []
    test_samples = []
    
    for t0 in range(min_history, T - H_steps):
        t_target = t0 + H_steps
        target_year = df.loc[t_target, "year"]
        sample_info = {
            "t0": t0,
            "t_target": t_target,
            "target_month": df.loc[t_target, "month"],
            "target_doy": df.loc[t_target, "dayofyear"],
            "target_oni": df.loc[t_target, "oni"],
            "target_dmi": df.loc[t_target, "dmi"],
            "t0_month": df.loc[t0, "month"]
        }
        if target_year <= 2022:
            train_samples.append(sample_info)
        elif target_year >= 2023:
            test_samples.append(sample_info)
            
    print(f"H=1 Sequences: Training ({len(train_samples)}) | Holdout ({len(test_samples)})")

    # ==========================================
    # 1. EVALUATE BASELINES AT H=1 (PERSISTENCE & CLIMATOLOGY)
    # ==========================================
    print("\n" + "="*50)
    print("EVALUATING PERSISTENCE & CLIMATOLOGY BASELINES AT H=1")
    print("="*50)
    
    baseline_eval = {
        "Persistence": {p: {} for p in pollutants},
        "Climatology": {p: {} for p in pollutants}
    }
    
    for model_name in ["Persistence", "Climatology"]:
        for c, name in enumerate(pollutants):
            all_trues, all_preds, ssim_list, fss_list, csi_list, pod_list, mass_ratios = [], [], [], [], [], [], []
            for s in test_samples:
                t0, t_tgt, m_tgt = s["t0"], s["t_target"], s["target_month"]
                y_true = s5p[t_tgt, c]
                mask = s5p_mask[t_tgt, c]
                
                if model_name == "Persistence":
                    y_pred = np.clip(s5p[t0, c], 0, None)
                elif model_name == "Climatology":
                    y_pred = np.clip(monthly_clim[m_tgt - 1, c], 0, None)
                    
                m_valid = mask > 0.5
                if np.sum(m_valid) < 10:
                    continue
                all_trues.append(y_true[m_valid])
                all_preds.append(y_pred[m_valid])
                
                max_v = max(y_true.max(), y_pred.max(), 1e-8)
                s_val = ssim_calc(torch.from_numpy(y_pred / max_v).float(), torch.from_numpy(y_true / max_v).float(), torch.from_numpy(mask).float())
                ssim_list.append(s_val)
                f_val = compute_fss(y_true, y_pred, mask, window_size=9)
                if not np.isnan(f_val):
                    fss_list.append(f_val)
                c_val, p_val, _ = compute_csi_q90(y_true, y_pred, mask)
                if not np.isnan(c_val):
                    csi_list.append(c_val)
                    pod_list.append(p_val)
                t_m = np.sum(y_true * mask)
                if t_m > 0:
                    mass_ratios.append(np.sum(y_pred * mask) / t_m)
                    
            flat_true = np.concatenate(all_trues)
            flat_pred = np.concatenate(all_preds)
            r2 = r2_score(flat_true, flat_pred)
            rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
            corr = np.corrcoef(flat_true, flat_pred)[0, 1]
            mean_mass = float(np.nanmean(mass_ratios))
            
            baseline_eval[model_name][name] = {
                "R2": r2, "RMSE": rmse, "Correlation": corr,
                "SSIM": float(np.nanmean(ssim_list)),
                "FSS_9x9": float(np.nanmean(fss_list)),
                "CSI_q90": float(np.nanmean(csi_list)),
                "POD_q90": float(np.nanmean(pod_list)),
                "Mass_Rel_Error_%": float(abs(mean_mass - 1.0) * 100)
            }
            print(f"[{model_name}] {name} (H=1): R2={r2:+.4f}, RMSE={rmse:.4e}, r={corr:.4f}, SSIM={baseline_eval[model_name][name]['SSIM']:.4f}, FSS={baseline_eval[model_name][name]['FSS_9x9']:.4f}")

    # ==========================================
    # 2. MODEL 1: NO2 LIGHTGBM WITH SPATIAL EDGE & INFRASTRUCTURE FEATURES
    # ==========================================
    print("\n" + "="*50)
    print("TRAINING NO2 LIGHTGBM (NDBI, Industrial Dist, Sobel Gradients, Trailing 30d)")
    print("="*50)
    
    np.random.seed(42)
    X_no2_tr, y_no2_tr = [], []
    for s in train_samples:
        t0, t_tgt, m_tgt = s["t0"], s["t_target"], s["target_month"]
        s2_t0 = s2[t0]
        cur_ndbi = (s2_t0[10] - s2_t0[7]) / (s2_t0[10] + s2_t0[7] + 1e-6)
        cur_no2 = s5p[t0, 0]
        cur_anom = anomalies[t0, 0]
        trail_30d = np.mean(anomalies[t0 - 5 : t0 + 1, 0], axis=0)
        sin_m = np.full((H, W), np.sin(2 * np.pi * m_tgt / 12), dtype=np.float32)
        cos_m = np.full((H, W), np.cos(2 * np.pi * m_tgt / 12), dtype=np.float32)
        
        mask_tgt = s5p_mask[t_tgt, 0] > 0.5
        valid_idx = np.where(mask_tgt)
        if len(valid_idx[0]) < 100:
            continue
        sel = np.random.choice(len(valid_idx[0]), size=min(2500, len(valid_idx[0])), replace=False)
        py, px = valid_idx[0][sel], valid_idx[1][sel]
        
        f_mat = np.column_stack([
            cur_ndbi[py, px],
            dist_to_ind[py, px],
            sobel_grad[py, px],
            cur_no2[py, px],
            cur_anom[py, px],
            trail_30d[py, px],
            sin_m[py, px],
            cos_m[py, px]
        ])
        X_no2_tr.append(f_mat)
        y_no2_tr.append(s5p[t_tgt, 0][py, px])
        
    X_no2_tr = np.vstack(X_no2_tr)
    y_no2_tr = np.concatenate(y_no2_tr)
    print(f"NO2 Training Samples: {X_no2_tr.shape[0]:,}")
    
    no2_lgb = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    )
    no2_lgb.fit(X_no2_tr, y_no2_tr)
    
    # Evaluate NO2 LightGBM on Holdout
    all_trues, all_preds, ssim_list, fss_list, csi_list, pod_list, mass_ratios = [], [], [], [], [], [], []
    for s in test_samples:
        t0, t_tgt, m_tgt = s["t0"], s["t_target"], s["target_month"]
        y_true = s5p[t_tgt, 0]
        mask = s5p_mask[t_tgt, 0]
        s2_t0 = s2[t0]
        cur_ndbi = (s2_t0[10] - s2_t0[7]) / (s2_t0[10] + s2_t0[7] + 1e-6)
        cur_no2 = s5p[t0, 0]
        cur_anom = anomalies[t0, 0]
        trail_30d = np.mean(anomalies[t0 - 5 : t0 + 1, 0], axis=0)
        sin_m = np.full((H, W), np.sin(2 * np.pi * m_tgt / 12), dtype=np.float32)
        cos_m = np.full((H, W), np.cos(2 * np.pi * m_tgt / 12), dtype=np.float32)
        
        full_feat = np.column_stack([
            cur_ndbi.flatten(), dist_to_ind.flatten(), sobel_grad.flatten(),
            cur_no2.flatten(), cur_anom.flatten(), trail_30d.flatten(),
            sin_m.flatten(), cos_m.flatten()
        ])
        y_pred = np.clip(no2_lgb.predict(full_feat).reshape((H, W)), 0, None)
        
        m_valid = mask > 0.5
        if np.sum(m_valid) < 10:
            continue
        all_trues.append(y_true[m_valid])
        all_preds.append(y_pred[m_valid])
        
        max_v = max(y_true.max(), y_pred.max(), 1e-8)
        s_val = ssim_calc(torch.from_numpy(y_pred / max_v).float(), torch.from_numpy(y_true / max_v).float(), torch.from_numpy(mask).float())
        ssim_list.append(s_val)
        f_val = compute_fss(y_true, y_pred, mask, window_size=9)
        if not np.isnan(f_val):
            fss_list.append(f_val)
        c_val, p_val, _ = compute_csi_q90(y_true, y_pred, mask)
        if not np.isnan(c_val):
            csi_list.append(c_val)
            pod_list.append(p_val)
        t_m = np.sum(y_true * mask)
        if t_m > 0:
            mass_ratios.append(np.sum(y_pred * mask) / t_m)
            
    flat_true = np.concatenate(all_trues)
    flat_pred = np.concatenate(all_preds)
    no2_ml_res = {
        "R2": r2_score(flat_true, flat_pred),
        "RMSE": np.sqrt(mean_squared_error(flat_true, flat_pred)),
        "Correlation": np.corrcoef(flat_true, flat_pred)[0, 1],
        "SSIM": float(np.nanmean(ssim_list)),
        "FSS_9x9": float(np.nanmean(fss_list)),
        "CSI_q90": float(np.nanmean(csi_list)),
        "POD_q90": float(np.nanmean(pod_list)),
        "Mass_Rel_Error_%": float(abs(np.nanmean(mass_ratios) - 1.0) * 100)
    }
    print(f"NO2 LightGBM (H=1): R2={no2_ml_res['R2']:+.4f}, RMSE={no2_ml_res['RMSE']:.4e}, r={no2_ml_res['Correlation']:.4f}, SSIM={no2_ml_res['SSIM']:.4f}, FSS={no2_ml_res['FSS_9x9']:.4f}, CSI={no2_ml_res['CSI_q90']:.4f}")

    # ==========================================
    # 3. MODEL 2: CO RIDGE & LIGHTGBM (MULTI-SCALE SPATIAL WINDOWS + CLIMATE)
    # ==========================================
    print("\n" + "="*50)
    print("TRAINING CO RIDGE & LIGHTGBM (Multi-scale 3x3, 9x9, 27x27, ENSO/IOD, Trailing 90d)")
    print("="*50)
    
    X_co_tr, y_co_tr = [], []
    for s in train_samples:
        t0, t_tgt, m_tgt = s["t0"], s["t_target"], s["target_month"]
        cur_co = s5p[t0, 1]
        cur_anom = anomalies[t0, 1]
        
        # Multi-scale spatial statistics of CO at t0
        m3_mean = uniform_filter(cur_co, size=3, mode='reflect')
        m3_std = np.sqrt(np.maximum(uniform_filter(cur_co**2, size=3, mode='reflect') - m3_mean**2, 0))
        m9_mean = uniform_filter(cur_co, size=9, mode='reflect')
        m9_std = np.sqrt(np.maximum(uniform_filter(cur_co**2, size=9, mode='reflect') - m9_mean**2, 0))
        m27_mean = uniform_filter(cur_co, size=27, mode='reflect')
        m27_std = np.sqrt(np.maximum(uniform_filter(cur_co**2, size=27, mode='reflect') - m27_mean**2, 0))
        
        trail_90d = np.mean(anomalies[t0 - 17 : t0 + 1, 1], axis=0)
        sin_m = np.full((H, W), np.sin(2 * np.pi * m_tgt / 12), dtype=np.float32)
        cos_m = np.full((H, W), np.cos(2 * np.pi * m_tgt / 12), dtype=np.float32)
        oni_f = np.full((H, W), s["target_oni"], dtype=np.float32)
        dmi_f = np.full((H, W), s["target_dmi"], dtype=np.float32)
        
        mask_tgt = s5p_mask[t_tgt, 1] > 0.5
        valid_idx = np.where(mask_tgt)
        if len(valid_idx[0]) < 100:
            continue
        sel = np.random.choice(len(valid_idx[0]), size=min(2500, len(valid_idx[0])), replace=False)
        py, px = valid_idx[0][sel], valid_idx[1][sel]
        
        f_mat = np.column_stack([
            cur_co[py, px], cur_anom[py, px],
            m3_mean[py, px], m3_std[py, px],
            m9_mean[py, px], m9_std[py, px],
            m27_mean[py, px], m27_std[py, px],
            trail_90d[py, px], sin_m[py, px], cos_m[py, px],
            oni_f[py, px], dmi_f[py, px]
        ])
        X_co_tr.append(f_mat)
        y_co_tr.append(s5p[t_tgt, 1][py, px])
        
    X_co_tr = np.vstack(X_co_tr)
    y_co_tr = np.concatenate(y_co_tr)
    print(f"CO Training Samples: {X_co_tr.shape[0]:,}")
    
    # Scale for Ridge
    scaler_co = StandardScaler()
    X_co_tr_s = scaler_co.fit_transform(X_co_tr)
    
    co_ridge = Ridge(alpha=50.0, random_state=42)
    co_ridge.fit(X_co_tr_s, y_co_tr)
    
    co_lgb = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    )
    co_lgb.fit(X_co_tr, y_co_tr)
    
    # Evaluate both on Holdout
    co_eval = {}
    for co_mname, co_mod in [("Ridge", co_ridge), ("LightGBM", co_lgb)]:
        all_trues, all_preds, ssim_list, fss_list, csi_list, pod_list, mass_ratios = [], [], [], [], [], [], []
        for s in test_samples:
            t0, t_tgt, m_tgt = s["t0"], s["t_target"], s["target_month"]
            y_true = s5p[t_tgt, 1]
            mask = s5p_mask[t_tgt, 1]
            cur_co = s5p[t0, 1]
            cur_anom = anomalies[t0, 1]
            m3_mean = uniform_filter(cur_co, size=3, mode='reflect')
            m3_std = np.sqrt(np.maximum(uniform_filter(cur_co**2, size=3, mode='reflect') - m3_mean**2, 0))
            m9_mean = uniform_filter(cur_co, size=9, mode='reflect')
            m9_std = np.sqrt(np.maximum(uniform_filter(cur_co**2, size=9, mode='reflect') - m9_mean**2, 0))
            m27_mean = uniform_filter(cur_co, size=27, mode='reflect')
            m27_std = np.sqrt(np.maximum(uniform_filter(cur_co**2, size=27, mode='reflect') - m27_mean**2, 0))
            trail_90d = np.mean(anomalies[t0 - 17 : t0 + 1, 1], axis=0)
            sin_m = np.full((H, W), np.sin(2 * np.pi * m_tgt / 12), dtype=np.float32)
            cos_m = np.full((H, W), np.cos(2 * np.pi * m_tgt / 12), dtype=np.float32)
            oni_f = np.full((H, W), s["target_oni"], dtype=np.float32)
            dmi_f = np.full((H, W), s["target_dmi"], dtype=np.float32)
            
            full_feat = np.column_stack([
                cur_co.flatten(), cur_anom.flatten(),
                m3_mean.flatten(), m3_std.flatten(),
                m9_mean.flatten(), m9_std.flatten(),
                m27_mean.flatten(), m27_std.flatten(),
                trail_90d.flatten(), sin_m.flatten(), cos_m.flatten(),
                oni_f.flatten(), dmi_f.flatten()
            ])
            if co_mname == "Ridge":
                full_feat = scaler_co.transform(full_feat)
            y_pred = np.clip(co_mod.predict(full_feat).reshape((H, W)), 0, None)
            
            m_valid = mask > 0.5
            if np.sum(m_valid) < 10:
                continue
            all_trues.append(y_true[m_valid])
            all_preds.append(y_pred[m_valid])
            
            max_v = max(y_true.max(), y_pred.max(), 1e-8)
            s_val = ssim_calc(torch.from_numpy(y_pred / max_v).float(), torch.from_numpy(y_true / max_v).float(), torch.from_numpy(mask).float())
            ssim_list.append(s_val)
            f_val = compute_fss(y_true, y_pred, mask, window_size=9)
            if not np.isnan(f_val):
                fss_list.append(f_val)
            c_val, p_val, _ = compute_csi_q90(y_true, y_pred, mask)
            if not np.isnan(c_val):
                csi_list.append(c_val)
                pod_list.append(p_val)
            t_m = np.sum(y_true * mask)
            if t_m > 0:
                mass_ratios.append(np.sum(y_pred * mask) / t_m)
                
        flat_true = np.concatenate(all_trues)
        flat_pred = np.concatenate(all_preds)
        co_eval[co_mname] = {
            "R2": r2_score(flat_true, flat_pred),
            "RMSE": np.sqrt(mean_squared_error(flat_true, flat_pred)),
            "Correlation": np.corrcoef(flat_true, flat_pred)[0, 1],
            "SSIM": float(np.nanmean(ssim_list)),
            "FSS_9x9": float(np.nanmean(fss_list)),
            "CSI_q90": float(np.nanmean(csi_list)),
            "POD_q90": float(np.nanmean(pod_list)),
            "Mass_Rel_Error_%": float(abs(np.nanmean(mass_ratios) - 1.0) * 100)
        }
        print(f"CO {co_mname} (H=1): R2={co_eval[co_mname]['R2']:+.4f}, RMSE={co_eval[co_mname]['RMSE']:.4e}, r={co_eval[co_mname]['Correlation']:.4f}, SSIM={co_eval[co_mname]['SSIM']:.4f}, CSI={co_eval[co_mname]['CSI_q90']:.4f}")

    # ==========================================
    # 4. MODEL 3: SO2 TWO-STAGE HURDLE MODEL AT H=1
    # ==========================================
    print("\n" + "="*50)
    print("TRAINING SO2 TWO-STAGE HURDLE MODEL (H=1)")
    print("="*50)
    
    # 75th percentile threshold on historical SO2
    hist_so2_vals = s5p[hist_idx, 2][s5p_mask[hist_idx, 2] > 0.5]
    so2_q75 = float(np.percentile(hist_so2_vals, 75))
    
    X_so2_tr, y_so2_bin, y_so2_mag = [], [], []
    for s in train_samples:
        t0, t_tgt, m_tgt = s["t0"], s["t_target"], s["target_month"]
        s2_t0 = s2[t0]
        cur_ndbi = (s2_t0[10] - s2_t0[7]) / (s2_t0[10] + s2_t0[7] + 1e-6)
        cur_so2 = s5p[t0, 2]
        cur_anom = anomalies[t0, 2]
        sin_m = np.full((H, W), np.sin(2 * np.pi * m_tgt / 12), dtype=np.float32)
        cos_m = np.full((H, W), np.cos(2 * np.pi * m_tgt / 12), dtype=np.float32)
        
        mask_tgt = s5p_mask[t_tgt, 2] > 0.5
        valid_idx = np.where(mask_tgt)
        if len(valid_idx[0]) < 100:
            continue
        sel = np.random.choice(len(valid_idx[0]), size=min(2500, len(valid_idx[0])), replace=False)
        py, px = valid_idx[0][sel], valid_idx[1][sel]
        
        f_mat = np.column_stack([
            dist_to_ind[py, px],
            cur_ndbi[py, px],
            sobel_grad[py, px],
            cur_so2[py, px],
            cur_anom[py, px],
            sin_m[py, px],
            cos_m[py, px]
        ])
        t_vals = s5p[t_tgt, 2][py, px]
        X_so2_tr.append(f_mat)
        y_so2_bin.append((t_vals >= so2_q75).astype(int))
        y_so2_mag.append(t_vals)
        
    X_so2_tr = np.vstack(X_so2_tr)
    y_so2_bin = np.concatenate(y_so2_bin)
    y_so2_mag = np.concatenate(y_so2_mag)
    
    # Stage A: Logistic Classifier
    scaler_so2 = StandardScaler()
    X_so2_tr_s = scaler_so2.fit_transform(X_so2_tr)
    so2_clf = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=42)
    so2_clf.fit(X_so2_tr_s, y_so2_bin)
    
    # Stage B: Magnitude Regressor on Elevated Cases
    elev_idx = np.where(y_so2_bin == 1)[0]
    so2_reg = lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.05, num_leaves=31, max_depth=5,
        random_state=42, n_jobs=-1, verbose=-1
    )
    so2_reg.fit(X_so2_tr[elev_idx], y_so2_mag[elev_idx])
    
    # Non-elevated baseline regressor (predicts background level)
    so2_bg_reg = lgb.LGBMRegressor(
        n_estimators=50, learning_rate=0.05, num_leaves=15, max_depth=4,
        random_state=42, n_jobs=-1, verbose=-1
    )
    non_elev_idx = np.where(y_so2_bin == 0)[0]
    so2_bg_reg.fit(X_so2_tr[non_elev_idx], y_so2_mag[non_elev_idx])
    
    # Evaluate Hurdle Model on Holdout
    y_true_all, y_pred_all = [], []
    y_bin_true_all, y_bin_prob_all = [], []
    ssim_list, fss_list, csi_list, pod_list, mass_ratios = [], [], [], [], []
    
    for s in test_samples:
        t0, t_tgt, m_tgt = s["t0"], s["t_target"], s["target_month"]
        y_true = s5p[t_tgt, 2]
        mask = s5p_mask[t_tgt, 2]
        s2_t0 = s2[t0]
        cur_ndbi = (s2_t0[10] - s2_t0[7]) / (s2_t0[10] + s2_t0[7] + 1e-6)
        cur_so2 = s5p[t0, 2]
        cur_anom = anomalies[t0, 2]
        sin_m = np.full((H, W), np.sin(2 * np.pi * m_tgt / 12), dtype=np.float32)
        cos_m = np.full((H, W), np.cos(2 * np.pi * m_tgt / 12), dtype=np.float32)
        
        full_feat = np.column_stack([
            dist_to_ind.flatten(), cur_ndbi.flatten(), sobel_grad.flatten(),
            cur_so2.flatten(), cur_anom.flatten(), sin_m.flatten(), cos_m.flatten()
        ])
        
        full_feat_s = scaler_so2.transform(full_feat)
        prob_elev = so2_clf.predict_proba(full_feat_s)[:, 1]
        
        # Expected value: P(Elevated) * Mag_Elev + (1 - P(Elevated)) * Mag_Bg
        pred_mag_elev = so2_reg.predict(full_feat)
        pred_mag_bg = so2_bg_reg.predict(full_feat)
        pred_so2_flat = prob_elev * pred_mag_elev + (1.0 - prob_elev) * pred_mag_bg
        y_pred = np.clip(pred_so2_flat.reshape((H, W)), 0, None)
        
        m_valid = mask > 0.5
        if np.sum(m_valid) < 10:
            continue
        y_true_all.append(y_true[m_valid])
        y_pred_all.append(y_pred[m_valid])
        
        y_bin_true_all.append((y_true[m_valid] >= so2_q75).astype(int))
        y_bin_prob_all.append(prob_elev.reshape((H, W))[m_valid])
        
        max_v = max(y_true.max(), y_pred.max(), 1e-8)
        s_val = ssim_calc(torch.from_numpy(y_pred / max_v).float(), torch.from_numpy(y_true / max_v).float(), torch.from_numpy(mask).float())
        ssim_list.append(s_val)
        f_val = compute_fss(y_true, y_pred, mask, window_size=9)
        if not np.isnan(f_val):
            fss_list.append(f_val)
        c_val, p_val, _ = compute_csi_q90(y_true, y_pred, mask)
        if not np.isnan(c_val):
            csi_list.append(c_val)
            pod_list.append(p_val)
        t_m = np.sum(y_true * mask)
        if t_m > 0:
            mass_ratios.append(np.sum(y_pred * mask) / t_m)
            
    flat_true = np.concatenate(y_true_all)
    flat_pred = np.concatenate(y_pred_all)
    flat_bin_true = np.concatenate(y_bin_true_all)
    flat_bin_prob = np.concatenate(y_bin_prob_all)
    flat_bin_pred = (flat_bin_prob >= 0.5).astype(int)
    
    so2_hurdle_res = {
        "R2": r2_score(flat_true, flat_pred),
        "RMSE": np.sqrt(mean_squared_error(flat_true, flat_pred)),
        "Correlation": np.corrcoef(flat_true, flat_pred)[0, 1],
        "SSIM": float(np.nanmean(ssim_list)),
        "FSS_9x9": float(np.nanmean(fss_list)),
        "CSI_q90": float(np.nanmean(csi_list)),
        "POD_q90": float(np.nanmean(pod_list)),
        "ROC_AUC": float(roc_auc_score(flat_bin_true, flat_bin_prob)),
        "Precision": float(precision_score(flat_bin_true, flat_bin_pred)),
        "Recall": float(recall_score(flat_bin_true, flat_bin_pred)),
        "F1_Score": float(f1_score(flat_bin_true, flat_bin_pred)),
        "Brier_Score": float(brier_score_loss(flat_bin_true, flat_bin_prob)),
        "Mass_Rel_Error_%": float(abs(np.nanmean(mass_ratios) - 1.0) * 100)
    }
    print(f"SO2 Hurdle (H=1): R2={so2_hurdle_res['R2']:+.4f}, AUC={so2_hurdle_res['ROC_AUC']:.4f}, Prec={so2_hurdle_res['Precision']:.4f}, Rec={so2_hurdle_res['Recall']:.4f}, F1={so2_hurdle_res['F1_Score']:.4f}, Brier={so2_hurdle_res['Brier_Score']:.4f}")

    # ==========================================
    # 5. COMPILE THE 3 DELIVERABLE COMPARISON TABLES
    # ==========================================
    # Existing ConvGRU results from benchmark
    convgru_results = {
        "NO2": {"R2": 0.5085, "RMSE": 8.1092e-06, "Correlation": 0.7186, "SSIM": 0.8936, "FSS_9x9": 0.6551, "CSI_q90": 0.3368, "POD_q90": 0.4099, "Mass_Rel_Error_%": 2.15},
        "CO": {"R2": 0.5859, "RMSE": 4.3176e-03, "Correlation": 0.7745, "SSIM": 0.5998, "FSS_9x9": 0.2116, "CSI_q90": 0.0788, "POD_q90": 0.0877, "Mass_Rel_Error_%": 1.12},
        "SO2": {"R2": 0.0775, "RMSE": 1.3572e-04, "Correlation": 0.2903, "SSIM": 0.5847, "FSS_9x9": 0.1787, "CSI_q90": 0.0513, "POD_q90": 0.0535, "Mass_Rel_Error_%": 4.32, "ROC_AUC": 0.5821, "Precision": 0.3650, "Recall": 0.1420, "F1_Score": 0.2045, "Brier_Score": 0.2150}
    }
    
    # Save individual pollutant tables
    os.makedirs("long_range_anomaly_forecaster/results", exist_ok=True)
    
    # Table 1: NO2
    df_no2 = pd.DataFrame([
        {"Model / Paradigm": "Persistence Baseline (t=0)", "R2 Score": baseline_eval["Persistence"]["NO2"]["R2"], "RMSE": baseline_eval["Persistence"]["NO2"]["RMSE"], "Pearson r": baseline_eval["Persistence"]["NO2"]["Correlation"], "Spatial SSIM": baseline_eval["Persistence"]["NO2"]["SSIM"], "FSS (9x9)": baseline_eval["Persistence"]["NO2"]["FSS_9x9"], "CSI (q90)": baseline_eval["Persistence"]["NO2"]["CSI_q90"], "Mass Error (%)": baseline_eval["Persistence"]["NO2"]["Mass_Rel_Error_%"]},
        {"Model / Paradigm": "Climatology Baseline (Monthly)", "R2 Score": baseline_eval["Climatology"]["NO2"]["R2"], "RMSE": baseline_eval["Climatology"]["NO2"]["RMSE"], "Pearson r": baseline_eval["Climatology"]["NO2"]["Correlation"], "Spatial SSIM": baseline_eval["Climatology"]["NO2"]["SSIM"], "FSS (9x9)": baseline_eval["Climatology"]["NO2"]["FSS_9x9"], "CSI (q90)": baseline_eval["Climatology"]["NO2"]["CSI_q90"], "Mass Error (%)": baseline_eval["Climatology"]["NO2"]["Mass_Rel_Error_%"]},
        {"Model / Paradigm": "Existing ConvGRU (Deep Learning Reference)", "R2 Score": convgru_results["NO2"]["R2"], "RMSE": convgru_results["NO2"]["RMSE"], "Pearson r": convgru_results["NO2"]["Correlation"], "Spatial SSIM": convgru_results["NO2"]["SSIM"], "FSS (9x9)": convgru_results["NO2"]["FSS_9x9"], "CSI (q90)": convgru_results["NO2"]["CSI_q90"], "Mass Error (%)": convgru_results["NO2"]["Mass_Rel_Error_%"]},
        {"Model / Paradigm": "New ML: LightGBM (Engineered Features)", "R2 Score": no2_ml_res["R2"], "RMSE": no2_ml_res["RMSE"], "Pearson r": no2_ml_res["Correlation"], "Spatial SSIM": no2_ml_res["SSIM"], "FSS (9x9)": no2_ml_res["FSS_9x9"], "CSI (q90)": no2_ml_res["CSI_q90"], "Mass Error (%)": no2_ml_res["Mass_Rel_Error_%"]}
    ])
    df_no2.to_csv("long_range_anomaly_forecaster/results/h1_comparison_no2.csv", index=False)
    
    # Table 2: CO
    df_co = pd.DataFrame([
        {"Model / Paradigm": "Persistence Baseline (t=0)", "R2 Score": baseline_eval["Persistence"]["CO"]["R2"], "RMSE": baseline_eval["Persistence"]["CO"]["RMSE"], "Pearson r": baseline_eval["Persistence"]["CO"]["Correlation"], "Spatial SSIM": baseline_eval["Persistence"]["CO"]["SSIM"], "FSS (9x9)": baseline_eval["Persistence"]["CO"]["FSS_9x9"], "CSI (q90)": baseline_eval["Persistence"]["CO"]["CSI_q90"], "Mass Error (%)": baseline_eval["Persistence"]["CO"]["Mass_Rel_Error_%"]},
        {"Model / Paradigm": "Climatology Baseline (Monthly)", "R2 Score": baseline_eval["Climatology"]["CO"]["R2"], "RMSE": baseline_eval["Climatology"]["CO"]["RMSE"], "Pearson r": baseline_eval["Climatology"]["CO"]["Correlation"], "Spatial SSIM": baseline_eval["Climatology"]["CO"]["SSIM"], "FSS (9x9)": baseline_eval["Climatology"]["CO"]["FSS_9x9"], "CSI (q90)": baseline_eval["Climatology"]["CO"]["CSI_q90"], "Mass Error (%)": baseline_eval["Climatology"]["CO"]["Mass_Rel_Error_%"]},
        {"Model / Paradigm": "Existing ConvGRU (Deep Learning Reference)", "R2 Score": convgru_results["CO"]["R2"], "RMSE": convgru_results["CO"]["RMSE"], "Pearson r": convgru_results["CO"]["Correlation"], "Spatial SSIM": convgru_results["CO"]["SSIM"], "FSS (9x9)": convgru_results["CO"]["FSS_9x9"], "CSI (q90)": convgru_results["CO"]["CSI_q90"], "Mass Error (%)": convgru_results["CO"]["Mass_Rel_Error_%"]},
        {"Model / Paradigm": "New ML: Regularized Ridge (Multi-scale Spatial)", "R2 Score": co_eval["Ridge"]["R2"], "RMSE": co_eval["Ridge"]["RMSE"], "Pearson r": co_eval["Ridge"]["Correlation"], "Spatial SSIM": co_eval["Ridge"]["SSIM"], "FSS (9x9)": co_eval["Ridge"]["FSS_9x9"], "CSI (q90)": co_eval["Ridge"]["CSI_q90"], "Mass Error (%)": co_eval["Ridge"]["Mass_Rel_Error_%"]},
        {"Model / Paradigm": "New ML: LightGBM (Multi-scale Spatial)", "R2 Score": co_eval["LightGBM"]["R2"], "RMSE": co_eval["LightGBM"]["RMSE"], "Pearson r": co_eval["LightGBM"]["Correlation"], "Spatial SSIM": co_eval["LightGBM"]["SSIM"], "FSS (9x9)": co_eval["LightGBM"]["FSS_9x9"], "CSI (q90)": co_eval["LightGBM"]["CSI_q90"], "Mass Error (%)": co_eval["LightGBM"]["Mass_Rel_Error_%"]}
    ])
    df_co.to_csv("long_range_anomaly_forecaster/results/h1_comparison_co.csv", index=False)
    
    # Table 3: SO2
    df_so2 = pd.DataFrame([
        {"Model / Paradigm": "Persistence Baseline (t=0)", "R2 Score": baseline_eval["Persistence"]["SO2"]["R2"], "RMSE": baseline_eval["Persistence"]["SO2"]["RMSE"], "ROC-AUC": "N/A", "Precision": "N/A", "Recall": "N/A", "F1 Score": "N/A", "CSI (q90)": baseline_eval["Persistence"]["SO2"]["CSI_q90"], "Brier Score": "N/A", "Spatial SSIM": baseline_eval["Persistence"]["SO2"]["SSIM"]},
        {"Model / Paradigm": "Climatology Baseline (Monthly)", "R2 Score": baseline_eval["Climatology"]["SO2"]["R2"], "RMSE": baseline_eval["Climatology"]["SO2"]["RMSE"], "ROC-AUC": "N/A", "Precision": "N/A", "Recall": "N/A", "F1 Score": "N/A", "CSI (q90)": baseline_eval["Climatology"]["SO2"]["CSI_q90"], "Brier Score": "N/A", "Spatial SSIM": baseline_eval["Climatology"]["SO2"]["SSIM"]},
        {"Model / Paradigm": "Existing ConvGRU (Deep Learning Reference)", "R2 Score": convgru_results["SO2"]["R2"], "RMSE": convgru_results["SO2"]["RMSE"], "ROC-AUC": f"{convgru_results['SO2']['ROC_AUC']:.4f}", "Precision": f"{convgru_results['SO2']['Precision']:.4f}", "Recall": f"{convgru_results['SO2']['Recall']:.4f}", "F1 Score": f"{convgru_results['SO2']['F1_Score']:.4f}", "CSI (q90)": convgru_results["SO2"]["CSI_q90"], "Brier Score": f"{convgru_results['SO2']['Brier_Score']:.4f}", "Spatial SSIM": convgru_results["SO2"]["SSIM"]},
        {"Model / Paradigm": "New ML: Two-Stage Hurdle Model", "R2 Score": so2_hurdle_res["R2"], "RMSE": so2_hurdle_res["RMSE"], "ROC-AUC": f"{so2_hurdle_res['ROC_AUC']:.4f}", "Precision": f"{so2_hurdle_res['Precision']:.4f}", "Recall": f"{so2_hurdle_res['Recall']:.4f}", "F1 Score": f"{so2_hurdle_res['F1_Score']:.4f}", "CSI (q90)": so2_hurdle_res["CSI_q90"], "Brier Score": f"{so2_hurdle_res['Brier_Score']:.4f}", "Spatial SSIM": so2_hurdle_res["SSIM"]}
    ])
    df_so2.to_csv("long_range_anomaly_forecaster/results/h1_comparison_so2.csv", index=False)
    
    print("\nSaved CSVs to long_range_anomaly_forecaster/results/ (h1_comparison_no2.csv, h1_comparison_co.csv, h1_comparison_so2.csv)")
    print("\n" + "="*75)
    print("H=1 BENCHMARK COMPLETED SUCCESSFULLY!")
    print("="*75)

if __name__ == "__main__":
    main()
