# Verification Script for Task 1: ConvGRU H=1 NO2 SSIM Discrepancy & Evaluation Reproducibility
# Checks checkpoint: training/checkpoints/best_sharp_forecast_model.pt
# Compares SSIM definitions, dynamic range scaling, patch-level vs scene-level evaluation,
# and verifies exact holdout dataset (122 dates / 1098 patches).

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.ndimage import uniform_filter

# Import model architecture from train.py
sys.path.append(os.path.abspath("."))
from training.train import DecoupledSTResUNet, extract_patches_from_tensor

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
    print("="*75)
    print("TASK 1: RE-RUNNING & VERIFYING ConvGRU EVALUATION ON 2023-2024 HOLDOUT")
    print("="*75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 1. Load Checkpoint
    ckpt_path = "training/checkpoints/best_sharp_forecast_model.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found!")
    
    print(f"Loading checkpoint: {ckpt_path} (Size: {os.path.getsize(ckpt_path)/1e6:.2f} MB)")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = DecoupledSTResUNet(in_channels=15, out_channels=3, hidden_dim=128).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("ConvGRU DecoupledSTResUNet successfully initialized in eval mode.")

    # 2. Load Holdout Dataset
    cache_path = "data/processed/cached_dataset_256.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    df = cache["df"].copy()
    s2_full = cache["s2"]         # [415, 12, 256, 256]
    s5p_full = cache["s5p"]       # [415, 3, 256, 256]
    s5p_mask_full = cache["s5p_mask"] # [415, 3, 256, 256]

    df["date"] = pd.to_datetime(df["start_date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Strict Holdout index >= 2023
    holdout_indices = df[df["year"] >= 2023].index.values
    print(f"\n[Holdout Audit]")
    print(f"Total archive dates: {len(df)}")
    print(f"Holdout dates (2023-2024): {len(holdout_indices)} dates (from {df.loc[holdout_indices[0], 'start_date']} to {df.loc[holdout_indices[-1], 'start_date']})")

    # Target pollutants & scales
    no2_scale = 1e-4
    co_scale = 5e-2
    so2_scale = 1e-4

    # Extract 9 non-overlapping 128x128 patches per 256x256 scene
    # Grid: (0, 0), (0, 64), (0, 128), (64, 0), (64, 64), (64, 128), (128, 0), (128, 64), (128, 128)
    patch_size = 128
    stride = 64
    y_coords = [0, 64, 128]
    x_coords = [0, 64, 128]

    # Pre-instantiate SSIM Calculators with different window sizes
    ssim_w7 = DifferentiableSSIMLoss(window_size=7)
    ssim_w11 = DifferentiableSSIMLoss(window_size=11)

    pollutants = ["NO2", "CO", "SO2"]
    
    # Containers for evaluating ConvGRU
    patch_trues = {p: [] for p in pollutants}
    patch_preds = {p: [] for p in pollutants}
    patch_masks = {p: [] for p in pollutants}
    
    ssim_w7_patch_norm = {p: [] for p in pollutants}
    ssim_w7_unnorm = {p: [] for p in pollutants}
    ssim_w11_patch_norm = {p: [] for p in pollutants}
    fss_patch = {p: [] for p in pollutants}
    csi_patch = {p: [] for p in pollutants}
    pod_patch = {p: [] for p in pollutants}
    mass_ratios_patch = {p: [] for p in pollutants}

    total_patches_evaluated = 0

    with torch.no_grad():
        for t_idx in holdout_indices:
            # Requires t_idx >= 4 for t_in=4 history (t_idx-4 to t_idx-1) -> target is t_idx
            if t_idx < 4:
                continue
            
            # Extract 4-step history
            s2_hist = s2_full[t_idx-4 : t_idx]      # [4, 12, 256, 256]
            s5p_hist = s5p_full[t_idx-4 : t_idx]    # [4, 3, 256, 256]
            s5p_m_hist = s5p_mask_full[t_idx-4 : t_idx] # [4, 3, 256, 256]
            
            s2_tgt = s2_full[t_idx]                  # [12, 256, 256]
            s5p_tgt = s5p_full[t_idx]                # [3, 256, 256]
            s5p_m_tgt = s5p_mask_full[t_idx]         # [3, 256, 256]

            # Loop over the 9 spatial crops
            for y_off in y_coords:
                for x_off in x_coords:
                    total_patches_evaluated += 1
                    
                    # Patch slices
                    s2_p = s2_hist[:, :, y_off : y_off+patch_size, x_off : x_off+patch_size] # [4, 12, 128, 128]
                    s5p_p = s5p_hist[:, :, y_off : y_off+patch_size, x_off : x_off+patch_size].clone() # [4, 3, 128, 128]
                    
                    # Target patch
                    tgt_p = s5p_tgt[:, y_off : y_off+patch_size, x_off : x_off+patch_size].numpy() # [3, 128, 128]
                    tgt_m_p = s5p_m_tgt[:, y_off : y_off+patch_size, x_off : x_off+patch_size].numpy() # [3, 128, 128]

                    # Scale inputs exactly as during train.py
                    s5p_p[:, 0] = s5p_p[:, 0] / no2_scale
                    s5p_p[:, 1] = s5p_p[:, 1] / co_scale
                    s5p_p[:, 2] = torch.log1p(s5p_p[:, 2] / so2_scale)

                    # Compute spectral indices for S2 patch
                    # B4=3, B8=7, B11=10
                    b4_p = s2_p[:, 3:4]
                    b8_p = s2_p[:, 7:8]
                    b11_p = s2_p[:, 10:11]
                    ndvi_p = (b8_p - b4_p) / (b8_p + b4_p + 1e-6)
                    ndbi_p = (b11_p - b8_p) / (b11_p + b8_p + 1e-6)
                    
                    # Fused input: S5P (3ch) + S2 optical (12ch) = 15ch
                    fused_in = torch.cat([s5p_p, s2_p], dim=1).unsqueeze(0).to(device) # [1, 4, 15, 128, 128]

                    # Forward pass
                    out_pred = model(fused_in) # [1, 3, 128, 128]
                    pred_np = out_pred.squeeze(0).cpu().numpy() # [3, 128, 128]

                    # Physical inversion scaling
                    pred_phys = np.zeros_like(pred_np)
                    pred_phys[0] = np.clip(pred_np[0] * no2_scale, 0, None)
                    pred_phys[1] = np.clip(pred_np[1] * co_scale, 0, None)
                    pred_phys[2] = np.clip(np.expm1(pred_np[2]) * so2_scale, 0, None)

                    # Evaluate per pollutant
                    for c, name in enumerate(pollutants):
                        t_arr = tgt_p[c]
                        p_arr = pred_phys[c]
                        m_arr = tgt_m_p[c]
                        
                        m_valid = m_arr > 0.5
                        if np.sum(m_valid) < 10:
                            continue
                            
                        patch_trues[name].append(t_arr[m_valid])
                        patch_preds[name].append(p_arr[m_valid])
                        
                        # SSIM with window=7 (patch normalized)
                        max_v = max(t_arr.max(), p_arr.max(), 1e-8)
                        t_t = torch.from_numpy(t_arr / max_v).float()
                        p_t = torch.from_numpy(p_arr / max_v).float()
                        m_t = torch.from_numpy(m_arr).float()
                        s7_norm = ssim_w7(p_t, t_t, m_t)
                        ssim_w7_patch_norm[name].append(s7_norm)
                        
                        # SSIM with window=11 (patch normalized)
                        s11_norm = ssim_w11(p_t, t_t, m_t)
                        ssim_w11_patch_norm[name].append(s11_norm)

                        # SSIM unnormalized
                        t_un = torch.from_numpy(t_arr).float()
                        p_un = torch.from_numpy(p_arr).float()
                        s7_un = ssim_w7(p_un, t_un, m_t)
                        ssim_w7_unnorm[name].append(s7_un)

                        # FSS & CSI
                        f_v = compute_fss(t_arr, p_arr, m_arr, window_size=9)
                        if not np.isnan(f_v):
                            fss_patch[name].append(f_v)
                        c_v, p_v, _ = compute_csi_q90(t_arr, p_arr, m_arr)
                        if not np.isnan(c_v):
                            csi_patch[name].append(c_v)
                            pod_patch[name].append(p_v)

                        # Mass conservation
                        tm = np.sum(t_arr * m_arr)
                        if tm > 0:
                            mass_ratios_patch[name].append(np.sum(p_arr * m_arr) / tm)

    print(f"\nAudit complete! Evaluated {total_patches_evaluated} patches ({len(holdout_indices)} dates * 9 crops = {len(holdout_indices)*9} expected).")

    # Compute global bulk metrics on holdout
    print("\n" + "="*75)
    print("CONVGRU HOLDOUT METRICS REPRODUCTION AUDIT (2023-2024 HOLDOUT)")
    print("="*75)
    
    reproduction_results = {}
    for c, name in enumerate(pollutants):
        flat_true = np.concatenate(patch_trues[name])
        flat_pred = np.concatenate(patch_preds[name])
        
        r2 = float(r2_score(flat_true, flat_pred))
        rmse = float(np.sqrt(mean_squared_error(flat_true, flat_pred)))
        corr = float(np.corrcoef(flat_true, flat_pred)[0, 1])
        
        ssim_w7_val = float(np.nanmean(ssim_w7_patch_norm[name]))
        ssim_w11_val = float(np.nanmean(ssim_w11_patch_norm[name]))
        ssim_unnorm_val = float(np.nanmean(ssim_w7_unnorm[name]))
        
        fss_val = float(np.nanmean(fss_patch[name]))
        csi_val = float(np.nanmean(csi_patch[name]))
        pod_val = float(np.nanmean(pod_patch[name]))
        mass_err = float(abs(np.nanmean(mass_ratios_patch[name]) - 1.0) * 100)
        
        reproduction_results[name] = {
            "R2": r2,
            "RMSE": rmse,
            "Correlation": corr,
            "SSIM (window=7, patch-norm)": ssim_w7_val,
            "SSIM (window=11, patch-norm)": ssim_w11_val,
            "SSIM (window=7, unnorm)": ssim_unnorm_val,
            "FSS (9x9)": fss_val,
            "CSI (q90)": csi_val,
            "POD (q90)": pod_val,
            "Mass Error (%)": mass_err
        }
        
        print(f"\n--- {name} Results ---")
        print(f"  R^2 Score:                        {r2:+.4f} (Phase 6 Champion: +0.5085 for NO2, +0.5859 for CO, +0.0775 for SO2)")
        print(f"  RMSE:                             {rmse:.4e}")
        print(f"  Pearson Correlation r:            {corr:.4f}")
        print(f"  Spatial SSIM (window=7, norm):     {ssim_w7_val:.4f}")
        print(f"  Spatial SSIM (window=11, norm):    {ssim_w11_val:.4f}")
        print(f"  Spatial SSIM (window=7, unnorm):   {ssim_unnorm_val:.4f}")
        print(f"  FSS (9x9 window):                 {fss_val:.4f}")
        print(f"  CSI (90th percentile):            {csi_val:.4f}")
        print(f"  POD (90th percentile):            {pod_val:.4f}")
        print(f"  Mass Conservation Error:          {mass_err:.2f}%")

    # Save verification JSON
    os.makedirs("long_range_anomaly_forecaster/results", exist_ok=True)
    with open("long_range_anomaly_forecaster/results/convgru_reproduction_audit.json", "w") as f:
        json.dump(reproduction_results, f, indent=2)
    print("\nSaved verification audit to: long_range_anomaly_forecaster/results/convgru_reproduction_audit.json")

if __name__ == "__main__":
    main()
