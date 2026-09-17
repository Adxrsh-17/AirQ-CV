# -*- coding: utf-8 -*-
"""
evaluation/metrics.py
Physical, spatial, and meteorological evaluation metrics:
- Standard: MAE, RMSE, R² Score, Relative Accuracy, Spatial SSIM
- Meteorological Plume Metrics: Pearson r, Fractions Skill Score (FSS 3x3, 9x9),
  Critical Success Index (CSI), Probability of Detection (POD), False Alarm Ratio (FAR),
  and Log-space MAE (for SO2).
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableSSIMLoss(nn.Module):
    """Computes Structural Similarity Index (SSIM) between forecast and observation."""
    def __init__(self, window_size=7):
        super().__init__()
        self.window_size = window_size
        gaussian = torch.tensor([
            math.exp(-(x - window_size // 2) ** 2 / (2 * 1.5 ** 2))
            for x in range(window_size)
        ])
        gaussian = (gaussian / gaussian.sum()).unsqueeze(1)
        kernel = gaussian.mm(gaussian.t()).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def forward(self, pred, target, mask=None):
        b, c, h, w = pred.shape
        k = self.kernel.repeat(c, 1, 1, 1).to(pred.device)

        mu1 = F.conv2d(pred, k, padding=self.window_size // 2, groups=c)
        mu2 = F.conv2d(target, k, padding=self.window_size // 2, groups=c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, k, padding=self.window_size // 2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(target * target, k, padding=self.window_size // 2, groups=c) - mu2_sq
        sigma12 = F.conv2d(pred * target, k, padding=self.window_size // 2, groups=c) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        loss_map = 1.0 - ssim_map

        if mask is not None:
            m = mask.to(pred.device)
            w_mask = F.conv2d(m, k, padding=self.window_size // 2, groups=c)
            valid_w = (w_mask > 0.3).float() * w_mask
            v_sum = valid_w.sum() + 1e-8
            return (loss_map * valid_w).sum() / v_sum
        return loss_map.mean()


def compute_fss(trues_ch, preds_ch, masks_ch=None, window_size=3, threshold=None):
    """
    Computes Fractions Skill Score (FSS) (Roberts and Lean, 2008) at given window size.
    Evaluates spatial plume overlap over neighborhood windows.
    Args:
        trues_ch: (N, H, W) numpy array of true observations
        preds_ch: (N, H, W) numpy array of predictions
        masks_ch: (N, H, W) numpy array of valid masks (1=valid, 0=masked)
        window_size: int, neighborhood kernel size (3, 9, etc.)
        threshold: float, plume threshold (default: 90th percentile of valid pixels)
    Returns:
        float: FSS in range [0.0, 1.0]
    """
    t = torch.from_numpy(trues_ch).float()
    p = torch.from_numpy(preds_ch).float()
    if masks_ch is not None:
        m = torch.from_numpy(masks_ch).float()
    else:
        m = (t > 0).float()

    valid_mask = (~torch.isnan(t)) & (~torch.isnan(p)) & (~torch.isinf(t)) & (~torch.isinf(p)) & (m > 0.5) & (t > 0)
    m = valid_mask.float()

    if threshold is None:
        valid_t = t[valid_mask]
        if len(valid_t) == 0:
            return 0.0
        threshold = float(torch.quantile(valid_t, 0.90).item())

    # Binary thresholded indicators
    b_t = ((t >= threshold) & (m > 0.5)).float().unsqueeze(1)
    b_p = ((p >= threshold) & (m > 0.5)).float().unsqueeze(1)
    m_4d = m.unsqueeze(1)

    kernel = torch.ones(1, 1, window_size, window_size)
    padding = window_size // 2

    sum_t = F.conv2d(b_t, kernel, padding=padding)
    sum_p = F.conv2d(b_p, kernel, padding=padding)
    sum_m = F.conv2d(m_4d, kernel, padding=padding)

    frac_t = sum_t / (sum_m + 1e-8)
    frac_p = sum_p / (sum_m + 1e-8)

    eval_pts = (m_4d > 0.5) & (sum_m > 0)
    if eval_pts.sum() == 0:
        return 0.0

    mse = torch.sum(((frac_p - frac_t) ** 2) * eval_pts)
    mse_ref = torch.sum(((frac_p ** 2) + (frac_t ** 2)) * eval_pts)

    if mse_ref < 1e-10:
        return 1.0

    fss = 1.0 - (mse / (mse_ref + 1e-10))
    return float(torch.clamp(fss, 0.0, 1.0).item())


def compute_all_metrics(trues, preds, masks=None, pollutant_names=None, so2_scale=1.2324e-4):
    """
    Computes standard physical and meteorological evaluation metrics,
    strictly excluding masked/nodata pixels.
    Args:
        trues: (N, 3, H, W) numpy array of true observations (in physical units)
        preds: (N, 3, H, W) numpy array of forecasted predictions (in physical units)
        masks: (N, 3, H, W) optional boolean/float mask (1 = valid, 0 = nodata)
        pollutant_names: list of pollutant names
        so2_scale: physical scale for SO2 log1p transformation
    Returns:
        pd.DataFrame with:
        [Target Pollutant, Mean True Obs, MAE, RMSE, R² Score, Relative Accuracy, Spatial SSIM,
         Pearson r, FSS (3x3), FSS (9x9), CSI (q90), POD (q90), FAR (q90), Log-space MAE]
    """
    if pollutant_names is None:
        pollutant_names = ["NO2 (mol/m²)", "CO (mol/m²)", "SO2 (mol/m²)"]

    ssim_module = DifferentiableSSIMLoss(window_size=7)
    records = []

    for c, name in enumerate(pollutant_names):
        p_ch = preds[:, c]
        t_ch = trues[:, c]
        m_ch = masks[:, c] if masks is not None else None

        p = p_ch.flatten()
        t = t_ch.flatten()

        # Build mask: exclude NaNs, Infs, and zero-padded nodata
        if m_ch is not None:
            m = m_ch.flatten()
            valid = (~np.isnan(t)) & (~np.isnan(p)) & (~np.isinf(t)) & (~np.isinf(p)) & (m > 0.5) & (t > 0)
        else:
            valid = (~np.isnan(t)) & (~np.isnan(p)) & (~np.isinf(t)) & (~np.isinf(p)) & (t > 0)

        t_val = t[valid]
        p_val = p[valid]

        if len(t_val) == 0:
            mean_true = 0.0
            mae = 0.0
            rmse = 0.0
            r2 = 0.0
            rel_acc = 0.0
            pearson_r = 0.0
            q90 = 0.0
            csi = 0.0
            pod = 0.0
            far = 0.0
            fss_3x3 = 0.0
            fss_9x9 = 0.0
            log_mae_str = "N/A"
        else:
            mean_true = float(np.mean(t_val))
            mae = float(np.mean(np.abs(p_val - t_val)))
            rmse = float(np.sqrt(np.mean((p_val - t_val) ** 2)))
            ss_tot = float(np.sum((t_val - mean_true) ** 2))
            ss_res = float(np.sum((t_val - p_val) ** 2))
            r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
            rel_acc = max(0.0, (1.0 - mae / (abs(mean_true) + 1e-10))) * 100.0

            # 1. Pearson Correlation Coefficient (valid pixels only)
            if len(t_val) >= 2 and np.std(t_val) > 1e-12 and np.std(p_val) > 1e-12:
                r_mat = np.corrcoef(p_val, t_val)
                pearson_r = float(r_mat[0, 1])
                if np.isnan(pearson_r):
                    pearson_r = 0.0
            else:
                pearson_r = 0.0

            # 2. 90th-percentile Plume Threshold & Contingency Metrics (CSI, POD, FAR)
            q90 = float(np.percentile(t_val, 90.0))
            obs_event = (t_val >= q90)
            pred_event = (p_val >= q90)

            hits = float(np.sum(obs_event & pred_event))
            misses = float(np.sum(obs_event & (~pred_event)))
            false_alarms = float(np.sum((~obs_event) & pred_event))

            pod = hits / (hits + misses + 1e-10) if (hits + misses) > 0 else 0.0
            far = false_alarms / (hits + false_alarms + 1e-10) if (hits + false_alarms) > 0 else 0.0
            csi = hits / (hits + misses + false_alarms + 1e-10) if (hits + misses + false_alarms) > 0 else 0.0

            # 3. Fractions Skill Score (FSS) at 3x3 and 9x9 window sizes
            fss_3x3 = compute_fss(t_ch, p_ch, masks_ch=m_ch, window_size=3, threshold=q90)
            fss_9x9 = compute_fss(t_ch, p_ch, masks_ch=m_ch, window_size=9, threshold=q90)

            # 4. Log-space MAE for SO2 only
            if "SO2" in name:
                t_log = np.log1p(np.maximum(0.0, t_val) / so2_scale)
                p_log = np.log1p(np.maximum(0.0, p_val) / so2_scale)
                log_mae = float(np.mean(np.abs(p_log - t_log)))
                log_mae_str = f"{log_mae:.4f}"
            else:
                log_mae_str = "N/A"

        # SSIM calculation (normalized channel, masked)
        p_t = torch.from_numpy(preds[:, c:c+1]).float()
        t_t = torch.from_numpy(trues[:, c:c+1]).float()

        if len(t_val) > 0:
            t_min, t_max = float(t_val.min()), float(t_val.max())
        else:
            t_min, t_max = float(t_t.min()), float(t_t.max())

        p_norm = (p_t - t_min) / (t_max - t_min + 1e-8)
        t_norm = (t_t - t_min) / (t_max - t_min + 1e-8)

        if masks is not None:
            m_t = torch.from_numpy(masks[:, c:c+1]).float()
            ssim_loss = ssim_module(p_norm, t_norm, mask=m_t).item()
        else:
            m_t = (t_t > 0).float()
            ssim_loss = ssim_module(p_norm, t_norm, mask=m_t).item()

        ssim_val = max(0.0, 1.0 - ssim_loss)

        records.append({
            # Original backward-compatible columns
            "Target Pollutant": name,
            "Mean True Obs": f"{mean_true:.4e}",
            "MAE": f"{mae:.4e}",
            "RMSE": f"{rmse:.4e}",
            "R² Score": f"{r2:.4f}",
            "Relative Accuracy": f"{rel_acc:.2f}%",
            "Spatial SSIM": f"{ssim_val:.4f}",
            # Newly added meteorological & plume metrics
            "Pearson r": f"{pearson_r:.4f}",
            "FSS (3x3)": f"{fss_3x3:.4f}",
            "FSS (9x9)": f"{fss_9x9:.4f}",
            "CSI (q90)": f"{csi:.4f}",
            "POD (q90)": f"{pod:.4f}",
            "FAR (q90)": f"{far:.4f}",
            "Log-space MAE": log_mae_str
        })

    return pd.DataFrame(records)
