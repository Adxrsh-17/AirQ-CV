# Statistical Low-Degrees-of-Freedom 150-Day Ahead (H=30) Forecasting Framework
# Methods:
# 1. Ridge Regression (Linear Regularized)
# 2. Harmonic Regression + Climate Indices (ENSO ONI & IOD DMI) on Regional Clusters
# 3. Two-Part Hurdle / Categorical Risk Model for SO2 (Low / Normal / Elevated Plume Risk)

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import distance_transform_edt, uniform_filter
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error, roc_auc_score,
    precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix
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

# NOAA / BOM Monthly Climate Index Lookup Table (2019-2024)
# ONI: Oceanic Nino Index (deg C anomaly in Nino 3.4)
# DMI: Dipole Mode Index (deg C anomaly for Indian Ocean Dipole)
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
    print("LOW-DEGREES-OF-FREEDOM STATISTICAL EXPERIMENTS FOR 150-DAY FORECASTING")
    print("="*75)
    
    # 1. Load Data
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
    
    # Attach ONI and DMI
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

    # Static features
    mean_s2 = np.mean(s2, axis=0)
    b4 = mean_s2[3]
    b8 = mean_s2[7]
    b11 = mean_s2[10]
    b12 = mean_s2[11]
    static_ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    static_ndbi = (b11 - b8) / (b11 + b8 + 1e-6)
    static_ui = (b11 + b12) / (b4 + b8 + 1e-6)
    
    mean_hist_no2 = np.mean(s5p[hist_idx, 0], axis=0)
    ind_mask = (static_ndbi > 0.05) & (mean_hist_no2 > np.percentile(mean_hist_no2, 85))
    dist_to_ind = distance_transform_edt(~ind_mask).astype(np.float32)
    dist_to_ind = dist_to_ind / (np.max(dist_to_ind) + 1e-6)

    # Spatial clustering into K=6 eco-climatic regions
    print("\n[Step 1] Spatial Clustering of Tamil Nadu Grid into K=6 Eco-Climatic Zones...")
    yy, xx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
    cluster_features = np.stack([
        yy.flatten(), xx.flatten(),
        static_ndvi.flatten(), static_ndbi.flatten(),
        dist_to_ind.flatten(),
        mean_hist_no2.flatten() / (np.max(mean_hist_no2) + 1e-8)
    ], axis=1)
    
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_features).reshape((H, W))
    
    H_steps = 30
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
            "target_year": df.loc[t_target, "year"],
            "target_doy": df.loc[t_target, "dayofyear"],
            "target_oni": df.loc[t_target, "oni"],
            "target_dmi": df.loc[t_target, "dmi"],
            "t0_oni": df.loc[t0, "oni"],
            "t0_dmi": df.loc[t0, "dmi"]
        }
        if target_year <= 2022:
            train_samples.append(sample_info)
        elif target_year >= 2023:
            test_samples.append(sample_info)
            
    print(f"H=30 Sequences: Training ({len(train_samples)}) | Holdout ({len(test_samples)})")
    
    # ==========================================
    # EXPERIMENT 1: REGULARIZED RIDGE REGRESSION
    # ==========================================
    print("\n" + "="*50)
    print("EXPERIMENT 1: LINEAR RIDGE REGRESSION ON TABULAR FEATURES")
    print("="*50)
    
    feature_names = [
        "ndvi", "ndbi", "industrial_index", "dist_to_industrial",
        "current_anomaly", "trailing_30d_mean_anom", "trailing_90d_mean_anom",
        "sin_month", "cos_month"
    ]
    
    ridge_results = {}
    ridge_models = {}
    ridge_coefs = {}
    
    for c, name in enumerate(pollutants):
        X_train_list, y_train_list = [], []
        for s in train_samples:
            t0, t_tgt, m_tgt = s["t0"], s["t_target"], s["target_month"]
            s2_t0 = s2[t0]
            cur_ndvi = (s2_t0[7] - s2_t0[3]) / (s2_t0[7] + s2_t0[3] + 1e-6)
            cur_ndbi = (s2_t0[10] - s2_t0[7]) / (s2_t0[10] + s2_t0[7] + 1e-6)
            cur_ui = (s2_t0[10] + s2_t0[11]) / (s2_t0[3] + s2_t0[7] + 1e-6)
            cur_anom = anomalies[t0, c]
            trail_30d = np.mean(anomalies[t0 - 5 : t0 + 1, c], axis=0)
            trail_90d = np.mean(anomalies[t0 - 17 : t0 + 1, c], axis=0)
            sin_m = np.full((H, W), np.sin(2 * np.pi * m_tgt / 12), dtype=np.float32)
            cos_m = np.full((H, W), np.cos(2 * np.pi * m_tgt / 12), dtype=np.float32)
            
            mask_tgt = s5p_mask[t_tgt, c] > 0.5
            valid_idx = np.where(mask_tgt)
            if len(valid_idx[0]) < 100:
                continue
            sel_choice = np.random.choice(len(valid_idx[0]), size=min(2500, len(valid_idx[0])), replace=False)
            py, px = valid_idx[0][sel_choice], valid_idx[1][sel_choice]
            
            f_mat = np.column_stack([
                cur_ndvi[py, px], cur_ndbi[py, px], cur_ui[py, px], dist_to_ind[py, px],
                cur_anom[py, px], trail_30d[py, px], trail_90d[py, px], sin_m[py, px], cos_m[py, px]
            ])
            X_train_list.append(f_mat)
            y_train_list.append(anomalies[t_tgt, c][py, px])
            
        X_tr = np.vstack(X_train_list)
        y_tr = np.concatenate(y_train_list)
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        
        # Heavy regularization (alpha=100.0)
        ridge = Ridge(alpha=100.0, random_state=42)
        ridge.fit(X_tr_s, y_tr)
        ridge_models[name] = (ridge, scaler)
        ridge_coefs[name] = {f: float(ridge.coef_[i]) for i, f in enumerate(feature_names)}
        
        # Evaluate on Holdout
        all_trues, all_preds, ssim_list, fss_list, csi_list, pod_list, mass_ratios = [], [], [], [], [], [], []
        for s in test_samples:
            t0, t_tgt, m_tgt = s["t0"], s["t_target"], s["target_month"]
            y_true = s5p[t_tgt, c]
            mask = s5p_mask[t_tgt, c]
            clim_tgt = monthly_clim[m_tgt - 1, c]
            
            s2_t0 = s2[t0]
            cur_ndvi = (s2_t0[7] - s2_t0[3]) / (s2_t0[7] + s2_t0[3] + 1e-6)
            cur_ndbi = (s2_t0[10] - s2_t0[7]) / (s2_t0[10] + s2_t0[7] + 1e-6)
            cur_ui = (s2_t0[10] + s2_t0[11]) / (s2_t0[3] + s2_t0[7] + 1e-6)
            cur_anom = anomalies[t0, c]
            trail_30d = np.mean(anomalies[t0 - 5 : t0 + 1, c], axis=0)
            trail_90d = np.mean(anomalies[t0 - 17 : t0 + 1, c], axis=0)
            sin_m = np.full((H, W), np.sin(2 * np.pi * m_tgt / 12), dtype=np.float32)
            cos_m = np.full((H, W), np.cos(2 * np.pi * m_tgt / 12), dtype=np.float32)
            
            full_feat = np.column_stack([
                cur_ndvi.flatten(), cur_ndbi.flatten(), cur_ui.flatten(), dist_to_ind.flatten(),
                cur_anom.flatten(), trail_30d.flatten(), trail_90d.flatten(), sin_m.flatten(), cos_m.flatten()
            ])
            full_feat_s = scaler.transform(full_feat)
            pred_anom = ridge.predict(full_feat_s).reshape((H, W))
            y_pred = np.clip(clim_tgt + pred_anom, 0, None)
            
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
        
        ridge_results[name] = {
            "R2": r2, "RMSE": rmse, "Correlation": corr,
            "SSIM": float(np.nanmean(ssim_list)),
            "FSS_9x9": float(np.nanmean(fss_list)),
            "CSI_q90": float(np.nanmean(csi_list)),
            "POD_q90": float(np.nanmean(pod_list)),
            "Mass_Rel_Error_%": float(abs(mean_mass - 1.0) * 100)
        }
        print(f"[Ridge Regression] {name}: R2={r2:+.4f}, RMSE={rmse:.4e}, r={corr:.4f}, SSIM={ridge_results[name]['SSIM']:.4f}, MassErr={ridge_results[name]['Mass_Rel_Error_%']:.2f}%")

    # ==========================================
    # EXPERIMENT 2: HARMONIC REGRESSION + CLIMATE INDICES (ONI & DMI)
    # ==========================================
    print("\n" + "="*50)
    print("EXPERIMENT 2: HARMONIC REGRESSION + ENSO/IOD CLIMATE INDICES")
    print("="*50)
    
    # Model per cluster k:
    # Y(t) = beta_0 + a1*sin(2pi*doy/365.25) + b1*cos(2pi*doy/365.25) + a2*sin(4pi*doy/365.25) + b2*cos(4pi*doy/365.25) + gamma_oni*ONI(t) + gamma_dmi*DMI(t)
    # Only 7 parameters per cluster!
    harmonic_results = {}
    harmonic_models = {p: {} for p in pollutants}
    
    for c, name in enumerate(pollutants):
        for k in range(6):
            k_mask = cluster_labels == k
            y_k_train = []
            X_k_train = []
            
            for s in train_samples:
                t_tgt = s["t_target"]
                doy = s["target_doy"]
                oni = s["target_oni"]
                dmi = s["target_dmi"]
                
                v_true = s5p[t_tgt, c]
                m_true = s5p_mask[t_tgt, c] > 0.5
                joint_k = k_mask & m_true
                if np.sum(joint_k) < 5:
                    continue
                mean_k_val = np.mean(v_true[joint_k])
                
                # 7 features
                feat = [
                    1.0, # Intercept
                    np.sin(2 * np.pi * doy / 365.25),
                    np.cos(2 * np.pi * doy / 365.25),
                    np.sin(4 * np.pi * doy / 365.25),
                    np.cos(4 * np.pi * doy / 365.25),
                    oni,
                    dmi
                ]
                X_k_train.append(feat)
                y_k_train.append(mean_k_val)
                
            X_mat = np.array(X_k_train)
            y_vec = np.array(y_k_train)
            # OLS with light L2 regularization (Ridge)
            beta = np.linalg.solve(X_mat.T @ X_mat + 1.0 * np.eye(7), X_mat.T @ y_vec)
            harmonic_models[name][k] = beta

        # Evaluate on Holdout
        all_trues, all_preds, ssim_list, fss_list, csi_list, pod_list, mass_ratios = [], [], [], [], [], [], []
        for s in test_samples:
            t_tgt = s["t_target"]
            doy = s["target_doy"]
            oni = s["target_oni"]
            dmi = s["target_dmi"]
            m_tgt = s["target_month"]
            
            y_true = s5p[t_tgt, c]
            mask = s5p_mask[t_tgt, c]
            
            # Predict cluster-level harmonic value
            y_pred = np.zeros((H, W), dtype=np.float32)
            feat = np.array([
                1.0,
                np.sin(2 * np.pi * doy / 365.25),
                np.cos(2 * np.pi * doy / 365.25),
                np.sin(4 * np.pi * doy / 365.25),
                np.cos(4 * np.pi * doy / 365.25),
                oni,
                dmi
            ])
            
            # Construct spatial prediction by applying cluster prediction + spatial climatology base
            clim_tgt = monthly_clim[m_tgt - 1, c]
            for k in range(6):
                k_mask = cluster_labels == k
                cluster_pred_val = float(feat @ harmonic_models[name][k])
                # Combine spatial climatology pattern with climate-driven cluster adjustment
                clim_k_mean = np.mean(clim_tgt[k_mask]) if np.sum(k_mask) > 0 else cluster_pred_val
                delta_k = cluster_pred_val - clim_k_mean
                y_pred[k_mask] = clim_tgt[k_mask] + delta_k
                
            y_pred = np.clip(y_pred, 0, None)
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
        
        harmonic_results[name] = {
            "R2": r2, "RMSE": rmse, "Correlation": corr,
            "SSIM": float(np.nanmean(ssim_list)),
            "FSS_9x9": float(np.nanmean(fss_list)),
            "CSI_q90": float(np.nanmean(csi_list)),
            "POD_q90": float(np.nanmean(pod_list)),
            "Mass_Rel_Error_%": float(abs(mean_mass - 1.0) * 100)
        }
        print(f"[Harmonic + ENSO/IOD] {name}: R2={r2:+.4f}, RMSE={rmse:.4e}, r={corr:.4f}, SSIM={harmonic_results[name]['SSIM']:.4f}, MassErr={harmonic_results[name]['Mass_Rel_Error_%']:.2f}%")

    # ==========================================
    # EXPERIMENT 3: SO2 TWO-PART HURDLE / RISK MODEL
    # ==========================================
    print("\n" + "="*50)
    print("EXPERIMENT 3: SO2 HURDLE RISK CLASSIFICATION (EARLY WARNING)")
    print("="*50)
    
    # Define threshold for elevated SO2 hazard (75th percentile of historical valid observations)
    hist_so2_vals = s5p[hist_idx, 2][s5p_mask[hist_idx, 2] > 0.5]
    so2_thresh_q75 = float(np.percentile(hist_so2_vals, 75))
    so2_thresh_q90 = float(np.percentile(hist_so2_vals, 90))
    print(f"SO2 Hazard Thresholds: 75th Percentile = {so2_thresh_q75:.4e}, 90th Percentile = {so2_thresh_q90:.4e}")
    
    # Train Hurdle Part 1: Logistic Classifier P(SO2 >= q75)
    X_so2_clf, y_so2_clf = [], []
    for s in train_samples:
        t_tgt = s["t_target"]
        doy = s["target_doy"]
        oni = s["target_oni"]
        dmi = s["target_dmi"]
        v_true = s5p[t_tgt, 2]
        m_true = s5p_mask[t_tgt, 2] > 0.5
        
        valid_idx = np.where(m_true)
        if len(valid_idx[0]) < 100:
            continue
        sel_choice = np.random.choice(len(valid_idx[0]), size=min(2500, len(valid_idx[0])), replace=False)
        py, px = valid_idx[0][sel_choice], valid_idx[1][sel_choice]
        
        f_mat = np.column_stack([
            static_ndbi[py, px],
            dist_to_ind[py, px],
            np.sin(2 * np.pi * doy / 365.25) * np.ones(len(py)),
            np.cos(2 * np.pi * doy / 365.25) * np.ones(len(py)),
            oni * np.ones(len(py)),
            dmi * np.ones(len(py))
        ])
        labels = (v_true[py, px] >= so2_thresh_q75).astype(int)
        X_so2_clf.append(f_mat)
        y_so2_clf.append(labels)
        
    X_so2_clf = np.vstack(X_so2_clf)
    y_so2_clf = np.concatenate(y_so2_clf)
    
    scaler_so2 = StandardScaler()
    X_so2_clf_s = scaler_so2.fit_transform(X_so2_clf)
    
    hurdle_clf = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=42)
    hurdle_clf.fit(X_so2_clf_s, y_so2_clf)
    
    # Evaluate Hurdle Risk Classifier on Holdout
    y_test_true_binary = []
    y_test_pred_prob = []
    
    for s in test_samples:
        t_tgt = s["t_target"]
        doy = s["target_doy"]
        oni = s["target_oni"]
        dmi = s["target_dmi"]
        v_true = s5p[t_tgt, 2]
        m_true = s5p_mask[t_tgt, 2] > 0.5
        
        valid_idx = np.where(m_true)
        if len(valid_idx[0]) < 10:
            continue
        py, px = valid_idx[0], valid_idx[1]
        f_mat = np.column_stack([
            static_ndbi[py, px],
            dist_to_ind[py, px],
            np.sin(2 * np.pi * doy / 365.25) * np.ones(len(py)),
            np.cos(2 * np.pi * doy / 365.25) * np.ones(len(py)),
            oni * np.ones(len(py)),
            dmi * np.ones(len(py))
        ])
        f_mat_s = scaler_so2.transform(f_mat)
        probs = hurdle_clf.predict_proba(f_mat_s)[:, 1]
        binary_labels = (v_true[py, px] >= so2_thresh_q75).astype(int)
        
        y_test_true_binary.extend(binary_labels)
        y_test_pred_prob.extend(probs)
        
    y_test_true_binary = np.array(y_test_true_binary)
    y_test_pred_prob = np.array(y_test_pred_prob)
    y_test_pred_bin = (y_test_pred_prob >= 0.5).astype(int)
    
    roc_auc = roc_auc_score(y_test_true_binary, y_test_pred_prob)
    prec = precision_score(y_test_true_binary, y_test_pred_bin)
    rec = recall_score(y_test_true_binary, y_test_pred_bin)
    f1 = f1_score(y_test_true_binary, y_test_pred_bin)
    brier = brier_score_loss(y_test_true_binary, y_test_pred_prob)
    
    print("\n--- SO2 150-Day Categorical Risk Performance ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Brier Calibration Score: {brier:.4f}")

    # ==========================================
    # COMPILE COMPREHENSIVE RESULTS
    # ==========================================
    # Load Stage 1 & Stage 2 results for comparison
    prev_csv = "long_range_anomaly_forecaster/results/h30_baseline_vs_rf_comparison.csv"
    df_prev = pd.read_csv(prev_csv)
    
    all_rows = []
    for _, r in df_prev.iterrows():
        all_rows.append(r.to_dict())
        
    for name in pollutants:
        r_res = ridge_results[name]
        all_rows.append({
            "Pollutant": name,
            "Model / Pipeline": "Regularized Ridge Regression (Linear)",
            "R2 Score": r_res["R2"],
            "RMSE (mol/m^2)": r_res["RMSE"],
            "Pearson r": r_res["Correlation"],
            "SSIM": r_res["SSIM"],
            "FSS (9x9)": r_res["FSS_9x9"],
            "CSI_q90": r_res["CSI_q90"],
            "POD_q90": r_res["POD_q90"],
            "Mass Conservation Error (%)": r_res["Mass_Rel_Error_%"]
        })
        h_res = harmonic_results[name]
        all_rows.append({
            "Pollutant": name,
            "Model / Pipeline": "Harmonic + Climate Indices (ENSO/IOD)",
            "R2 Score": h_res["R2"],
            "RMSE (mol/m^2)": h_res["RMSE"],
            "Pearson r": h_res["Correlation"],
            "SSIM": h_res["SSIM"],
            "FSS (9x9)": h_res["FSS_9x9"],
            "CSI_q90": h_res["CSI_q90"],
            "POD_q90": h_res["POD_q90"],
            "Mass Conservation Error (%)": h_res["Mass_Rel_Error_%"]
        })
        
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv("long_range_anomaly_forecaster/results/h30_comprehensive_low_dof_comparison.csv", index=False)
    print("\nSaved: long_range_anomaly_forecaster/results/h30_comprehensive_low_dof_comparison.csv")
    
    # Save Hurdle Metrics JSON
    hurdle_dict = {
        "ROC_AUC": float(roc_auc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1_Score": float(f1),
        "Brier_Score": float(brier),
        "Logistic_Coefficients": {
            "ndbi": float(hurdle_clf.coef_[0, 0]),
            "dist_to_industrial": float(hurdle_clf.coef_[0, 1]),
            "sin_annual": float(hurdle_clf.coef_[0, 2]),
            "cos_annual": float(hurdle_clf.coef_[0, 3]),
            "oni_enso": float(hurdle_clf.coef_[0, 4]),
            "dmi_iod": float(hurdle_clf.coef_[0, 5])
        }
    }
    with open("long_range_anomaly_forecaster/results/so2_hurdle_risk_metrics.json", "w") as f:
        json.dump(hurdle_dict, f, indent=2)
    print("Saved: long_range_anomaly_forecaster/results/so2_hurdle_risk_metrics.json")

    # Generate Visualization of Comprehensive Models
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_to_plot = ["R2 Score", "SSIM", "CSI_q90"]
    for i, m_name in enumerate(metrics_to_plot):
        ax = axes[i]
        chart_df = df_all[df_all["Model / Pipeline"] != "Stage 1: Persistence Baseline (Anom=Anom(t0))"]
        sns.barplot(data=chart_df, x="Pollutant", y=m_name, hue="Model / Pipeline", ax=ax, palette="tab10")
        ax.set_title(f"H=30 (150-Day Ahead) - {m_name}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        if i == 0:
            ax.legend(loc="lower left", fontsize=7)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig("long_range_anomaly_forecaster/plots/comprehensive_low_dof_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Plot saved: long_range_anomaly_forecaster/plots/comprehensive_low_dof_comparison.png")
    
    print("\n" + "="*75)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*75)

if __name__ == "__main__":
    main()
