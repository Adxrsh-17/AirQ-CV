# Stage 1 & Stage 2 Long-Range (H=30 / 150-day ahead) Anomaly-Based Forecasting Framework
# Pollutants: NO2, CO, SO2
# Models: Climatology-only Baseline, Persistence Baseline, LightGBM / Random Forest per pollutant

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
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

class DifferentiableSSIMLoss(nn.Module):
    """Computes Structural Similarity Index (SSIM) between forecast and observation."""
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
        elif pred.dim() == 3:
            pred = pred.unsqueeze(1)
            true = true.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

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

def main():
    print("="*70)
    print("STAGE 1 & STAGE 2: 150-DAY AHEAD (H=30) ANOMALY FORECASTING BENCHMARK")
    print("="*70)
    
    # 1. Load Data
    cache_path = "data/processed/cached_dataset_256.pt"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Missing {cache_path}")
        
    print("\n[Step 1] Loading full 415-composite dataset...")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    df = cache["df"].copy()
    s2 = cache["s2"].numpy()       # [415, 12, 256, 256]
    s5p = cache["s5p"].numpy()     # [415, 3, 256, 256]
    s5p_mask = cache["s5p_mask"].numpy() # [415, 3, 256, 256]
    
    T, C_s2, H, W = s2.shape
    _, C_s5p, _, _ = s5p.shape
    
    df["date"] = pd.to_datetime(df["start_date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    
    pollutants = ["NO2", "CO", "SO2"]
    ssim_calc = DifferentiableSSIMLoss(window_size=7)
    
    # 2. Compute Monthly Spatial Climatology on Historical Pool (<= 2022)
    print("\n[Step 2] Computing Monthly Spatial Climatology on Historical Pool (2019-2022)...")
    hist_idx = df[df["year"] <= 2022].index.values
    holdout_idx = df[df["year"] >= 2023].index.values
    print(f"Historical dates: {len(hist_idx)} | Holdout dates: {len(holdout_idx)}")
    
    monthly_clim = np.zeros((12, 3, H, W), dtype=np.float32)
    for m in range(1, 13):
        m_hist_idx = df[(df["year"] <= 2022) & (df["month"] == m)].index.values
        if len(m_hist_idx) == 0:
            print(f"Warning: No historical dates for month {m}")
            continue
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

    # Compute Anomaly maps for all T
    print("\n[Step 3] Computing Anomaly Maps for all time steps...")
    anomalies = np.zeros((T, 3, H, W), dtype=np.float32)
    for t in range(T):
        m = df.loc[t, "month"]
        for c in range(3):
            anomalies[t, c] = s5p[t, c] - monthly_clim[m - 1, c]

    # Compute static features: industrial map and distance transform
    print("\n[Step 4] Computing Static Spatial Features (NDBI, NDVI, Industrial Distance)...")
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
    if np.sum(ind_mask) == 0:
        ind_mask = static_ndbi > 0.0
    dist_to_ind = distance_transform_edt(~ind_mask).astype(np.float32)
    dist_to_ind = dist_to_ind / (np.max(dist_to_ind) + 1e-6)

    # Form H=30 (150-day) sequences
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
            "t0_month": df.loc[t0, "month"]
        }
        if target_year <= 2022:
            train_samples.append(sample_info)
        elif target_year >= 2023:
            test_samples.append(sample_info)
            
    print(f"Total H=30 Pairs -> Training Pairs (<=2022): {len(train_samples)} | Holdout Pairs (>=2023): {len(test_samples)}")
    
    # ==========================================
    # STAGE 1: BASELINE EVALUATION
    # ==========================================
    print("\n" + "="*50)
    print("EVALUATING STAGE 1 BASELINES ON HOLDOUT TEST SET")
    print("="*50)
    
    baseline_results = {
        "Climatology": {p: {} for p in pollutants},
        "Persistence": {p: {} for p in pollutants}
    }
    
    for model_name in ["Climatology", "Persistence"]:
        for c, name in enumerate(pollutants):
            all_trues = []
            all_preds = []
            all_masks = []
            ssim_list = []
            fss_list = []
            csi_list = []
            pod_list = []
            mass_ratios = []
            
            for s in test_samples:
                t0 = s["t0"]
                t_tgt = s["t_target"]
                m_tgt = s["target_month"]
                
                y_true = s5p[t_tgt, c]
                mask = s5p_mask[t_tgt, c]
                clim_tgt = monthly_clim[m_tgt - 1, c]
                
                if model_name == "Climatology":
                    y_pred = np.clip(clim_tgt, 0, None)
                elif model_name == "Persistence":
                    anom_t0 = anomalies[t0, c]
                    y_pred = np.clip(clim_tgt + anom_t0, 0, None)
                    
                m_valid = mask > 0.5
                if np.sum(m_valid) < 10:
                    continue
                    
                all_trues.append(y_true[m_valid])
                all_preds.append(y_pred[m_valid])
                all_masks.append(mask)
                
                # Spatial SSIM via DifferentiableSSIM
                max_v = max(y_true.max(), y_pred.max(), 1e-8)
                t_tensor = torch.from_numpy(y_true / max_v).float()
                p_tensor = torch.from_numpy(y_pred / max_v).float()
                m_tensor = torch.from_numpy(mask).float()
                s_val = ssim_calc(p_tensor, t_tensor, m_tensor)
                ssim_list.append(s_val)
                
                f_val = compute_fss(y_true, y_pred, mask, window_size=9)
                if not np.isnan(f_val):
                    fss_list.append(f_val)
                    
                c_val, p_val, _ = compute_csi_q90(y_true, y_pred, mask)
                if not np.isnan(c_val):
                    csi_list.append(c_val)
                    pod_list.append(p_val)
                    
                total_true_mass = np.sum(y_true * mask)
                total_pred_mass = np.sum(y_pred * mask)
                if total_true_mass > 0:
                    mass_ratios.append(total_pred_mass / total_true_mass)
                    
            flat_true = np.concatenate(all_trues)
            flat_pred = np.concatenate(all_preds)
            
            r2 = r2_score(flat_true, flat_pred)
            rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
            corr = np.corrcoef(flat_true, flat_pred)[0, 1]
            mean_ssim = float(np.nanmean(ssim_list))
            mean_fss = float(np.nanmean(fss_list))
            mean_csi = float(np.nanmean(csi_list))
            mean_pod = float(np.nanmean(pod_list))
            mean_mass_ratio = float(np.nanmean(mass_ratios))
            mass_rel_error = float(abs(mean_mass_ratio - 1.0) * 100)
            
            baseline_results[model_name][name] = {
                "R2": r2,
                "RMSE": rmse,
                "Correlation": corr,
                "SSIM": mean_ssim,
                "FSS_9x9": mean_fss,
                "CSI_q90": mean_csi,
                "POD_q90": mean_pod,
                "Mass_Ratio": mean_mass_ratio,
                "Mass_Rel_Error_%": mass_rel_error
            }
            print(f"[{model_name}] {name} (H=30 / 150d): R2={r2:+.4f}, RMSE={rmse:.4e}, r={corr:.4f}, SSIM={mean_ssim:.4f}, FSS={mean_fss:.4f}, CSI_q90={mean_csi:.4f}, MassErr={mass_rel_error:.2f}%")

    # ==========================================
    # STAGE 2: TABULAR GRADIENT BOOSTED REGRESSORS
    # ==========================================
    print("\n" + "="*50)
    print("STAGE 2: TRAINING PER-POLLUTANT DECOUPLED TABULAR MODELS")
    print("="*50)
    
    feature_names = [
        "ndvi", "ndbi", "industrial_index", "dist_to_industrial",
        "current_anomaly", "trailing_30d_mean_anom", "trailing_90d_mean_anom",
        "sin_month", "cos_month"
    ]
    
    np.random.seed(42)
    pixels_per_scene = 2500
    
    rf_models = {}
    feature_importances = {}
    
    for c, name in enumerate(pollutants):
        print(f"\n--- Extracting Stage 2 Features for Pollutant: {name} ---")
        X_train_list = []
        y_train_list = []
        
        for s in train_samples:
            t0 = s["t0"]
            t_tgt = s["t_target"]
            m_tgt = s["target_month"]
            
            s2_t0 = s2[t0]
            cur_ndvi = (s2_t0[7] - s2_t0[3]) / (s2_t0[7] + s2_t0[3] + 1e-6)
            cur_ndbi = (s2_t0[10] - s2_t0[7]) / (s2_t0[10] + s2_t0[7] + 1e-6)
            cur_ui = (s2_t0[10] + s2_t0[11]) / (s2_t0[3] + s2_t0[7] + 1e-6)
            
            cur_anom = anomalies[t0, c]
            trail_30d = np.mean(anomalies[t0 - 5 : t0 + 1, c], axis=0)
            trail_90d = np.mean(anomalies[t0 - 17 : t0 + 1, c], axis=0)
            
            sin_m = np.full((H, W), np.sin(2 * np.pi * m_tgt / 12), dtype=np.float32)
            cos_m = np.full((H, W), np.cos(2 * np.pi * m_tgt / 12), dtype=np.float32)
            
            target_anom = anomalies[t_tgt, c]
            mask_tgt = s5p_mask[t_tgt, c] > 0.5
            
            valid_idx = np.where(mask_tgt)
            if len(valid_idx[0]) < 100:
                continue
                
            n_sel = min(pixels_per_scene, len(valid_idx[0]))
            sel_choice = np.random.choice(len(valid_idx[0]), size=n_sel, replace=False)
            py = valid_idx[0][sel_choice]
            px = valid_idx[1][sel_choice]
            
            f_mat = np.column_stack([
                cur_ndvi[py, px],
                cur_ndbi[py, px],
                cur_ui[py, px],
                dist_to_ind[py, px],
                cur_anom[py, px],
                trail_30d[py, px],
                trail_90d[py, px],
                sin_m[py, px],
                cos_m[py, px]
            ])
            t_vec = target_anom[py, px]
            
            X_train_list.append(f_mat)
            y_train_list.append(t_vec)
            
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        print(f"{name} Training Set: {X_train.shape[0]:,} pixels across {len(train_samples)} scenes")
        
        print(f"Fitting Gradient Boosted Anomaly Regressor for {name}...")
        model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        rf_models[name] = model
        
        imp = model.feature_importances_
        imp_norm = imp / np.sum(imp)
        feature_importances[name] = {f: float(imp_norm[i]) for i, f in enumerate(feature_names)}
        print(f"Feature Importances for {name}:")
        for f, v in sorted(feature_importances[name].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {f:25s}: {v*100:.2f}%")

    # Evaluate Stage 2 Regressor on Holdout Test Set
    print("\n" + "="*50)
    print("EVALUATING STAGE 2 (GRADIENT BOOSTED ANOMALY FORECASTER) ON HOLDOUT")
    print("="*50)
    
    stage2_results = {p: {} for p in pollutants}
    
    for c, name in enumerate(pollutants):
        model = rf_models[name]
        all_trues = []
        all_preds = []
        ssim_list = []
        fss_list = []
        csi_list = []
        pod_list = []
        mass_ratios = []
        
        for s in test_samples:
            t0 = s["t0"]
            t_tgt = s["t_target"]
            m_tgt = s["target_month"]
            
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
            
            full_feat_mat = np.column_stack([
                cur_ndvi.flatten(),
                cur_ndbi.flatten(),
                cur_ui.flatten(),
                dist_to_ind.flatten(),
                cur_anom.flatten(),
                trail_30d.flatten(),
                trail_90d.flatten(),
                sin_m.flatten(),
                cos_m.flatten()
            ])
            
            pred_anom_flat = model.predict(full_feat_mat)
            pred_anom_map = pred_anom_flat.reshape((H, W))
            
            # Reconstruct raw prediction: Climatology(target_month) + Predicted Anomaly
            y_pred = np.clip(clim_tgt + pred_anom_map, 0, None)
            
            m_valid = mask > 0.5
            if np.sum(m_valid) < 10:
                continue
                
            all_trues.append(y_true[m_valid])
            all_preds.append(y_pred[m_valid])
            
            max_v = max(y_true.max(), y_pred.max(), 1e-8)
            t_tensor = torch.from_numpy(y_true / max_v).float()
            p_tensor = torch.from_numpy(y_pred / max_v).float()
            m_tensor = torch.from_numpy(mask).float()
            s_val = ssim_calc(p_tensor, t_tensor, m_tensor)
            ssim_list.append(s_val)
            
            f_val = compute_fss(y_true, y_pred, mask, window_size=9)
            if not np.isnan(f_val):
                fss_list.append(f_val)
                
            c_val, p_val, _ = compute_csi_q90(y_true, y_pred, mask)
            if not np.isnan(c_val):
                csi_list.append(c_val)
                pod_list.append(p_val)
                
            total_true_mass = np.sum(y_true * mask)
            total_pred_mass = np.sum(y_pred * mask)
            if total_true_mass > 0:
                mass_ratios.append(total_pred_mass / total_true_mass)
                
        flat_true = np.concatenate(all_trues)
        flat_pred = np.concatenate(all_preds)
        
        r2 = r2_score(flat_true, flat_pred)
        rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
        corr = np.corrcoef(flat_true, flat_pred)[0, 1]
        mean_ssim = float(np.nanmean(ssim_list))
        mean_fss = float(np.nanmean(fss_list))
        mean_csi = float(np.nanmean(csi_list))
        mean_pod = float(np.nanmean(pod_list))
        mean_mass_ratio = float(np.nanmean(mass_ratios))
        mass_rel_error = float(abs(mean_mass_ratio - 1.0) * 100)
        
        stage2_results[name] = {
            "R2": r2,
            "RMSE": rmse,
            "Correlation": corr,
            "SSIM": mean_ssim,
            "FSS_9x9": mean_fss,
            "CSI_q90": mean_csi,
            "POD_q90": mean_pod,
            "Mass_Ratio": mean_mass_ratio,
            "Mass_Rel_Error_%": mass_rel_error
        }
        print(f"[Stage 2 GBM/RF] {name} (H=30 / 150d): R2={r2:+.4f}, RMSE={rmse:.4e}, r={corr:.4f}, SSIM={mean_ssim:.4f}, FSS={mean_fss:.4f}, CSI_q90={mean_csi:.4f}, MassErr={mass_rel_error:.2f}%")

    # ==========================================
    # COMPILE COMPARISON TABLE & VISUALIZATIONS
    # ==========================================
    print("\n" + "="*50)
    print("COMPILING COMPARISON DELIVERABLE")
    print("="*50)
    
    rows = []
    for name in pollutants:
        # Climatology
        c_res = baseline_results["Climatology"][name]
        rows.append({
            "Pollutant": name,
            "Model / Pipeline": "Stage 1: Climatology Baseline (Anomaly=0)",
            "R2 Score": c_res["R2"],
            "RMSE (mol/m^2)": c_res["RMSE"],
            "Pearson r": c_res["Correlation"],
            "SSIM": c_res["SSIM"],
            "FSS (9x9)": c_res["FSS_9x9"],
            "CSI_q90": c_res["CSI_q90"],
            "POD_q90": c_res["POD_q90"],
            "Mass Conservation Error (%)": c_res["Mass_Rel_Error_%"]
        })
        # Persistence
        p_res = baseline_results["Persistence"][name]
        rows.append({
            "Pollutant": name,
            "Model / Pipeline": "Stage 1: Persistence Baseline (Anom=Anom(t0))",
            "R2 Score": p_res["R2"],
            "RMSE (mol/m^2)": p_res["RMSE"],
            "Pearson r": p_res["Correlation"],
            "SSIM": p_res["SSIM"],
            "FSS (9x9)": p_res["FSS_9x9"],
            "CSI_q90": p_res["CSI_q90"],
            "POD_q90": p_res["POD_q90"],
            "Mass Conservation Error (%)": p_res["Mass_Rel_Error_%"]
        })
        # Stage 2 GBM / RF
        rf_res = stage2_results[name]
        rows.append({
            "Pollutant": name,
            "Model / Pipeline": "Stage 2: Gradient Boosted Anomaly Regressor",
            "R2 Score": rf_res["R2"],
            "RMSE (mol/m^2)": rf_res["RMSE"],
            "Pearson r": rf_res["Correlation"],
            "SSIM": rf_res["SSIM"],
            "FSS (9x9)": rf_res["FSS_9x9"],
            "CSI_q90": rf_res["CSI_q90"],
            "POD_q90": rf_res["POD_q90"],
            "Mass Conservation Error (%)": rf_res["Mass_Rel_Error_%"]
        })

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv("long_range_anomaly_forecaster/results/h30_baseline_vs_rf_comparison.csv", index=False)
    print("\nSaved CSV: long_range_anomaly_forecaster/results/h30_baseline_vs_rf_comparison.csv")
    
    with open("long_range_anomaly_forecaster/results/feature_importances_h30.json", "w") as f:
        json.dump(feature_importances, f, indent=2)
    print("Saved JSON: long_range_anomaly_forecaster/results/feature_importances_h30.json")

    # Generate Feature Importance Bar Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for c, name in enumerate(pollutants):
        ax = axes[c]
        imp_df = pd.DataFrame(list(feature_importances[name].items()), columns=["Feature", "Importance"])
        imp_df = imp_df.sort_values(by="Importance", ascending=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(imp_df)))
        ax.barh(imp_df["Feature"], imp_df["Importance"] * 100, color=colors, edgecolor="black")
        ax.set_title(f"{name} 150-Day Anomaly Feature Importance", fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative Importance (%)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("long_range_anomaly_forecaster/plots/feature_importance_h30.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Plot saved: long_range_anomaly_forecaster/plots/feature_importance_h30.png")

    # Generate Model Comparison Bar Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_plot = ["R2 Score", "SSIM", "CSI_q90"]
    for i, m_name in enumerate(metrics_to_plot):
        ax = axes[i]
        chart_df = df_comp[["Pollutant", "Model / Pipeline", m_name]]
        sns.barplot(data=chart_df, x="Pollutant", y=m_name, hue="Model / Pipeline", ax=ax, palette="Set2")
        ax.set_title(f"H=30 (150-Day Ahead) - {m_name}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        if i == 0:
            ax.legend(loc="lower left", fontsize=8)
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig("long_range_anomaly_forecaster/plots/h30_baseline_vs_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Plot saved: long_range_anomaly_forecaster/plots/h30_baseline_vs_model_comparison.png")
    
    print("\n" + "="*70)
    print("STAGE 1 & 2 COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()
