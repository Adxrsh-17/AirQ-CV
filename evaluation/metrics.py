# -*- coding: utf-8 -*-
"""
evaluation/metrics.py
Evaluation metrics: MAE, RMSE, R2 Score, Relative Accuracy, and SSIM.
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


def compute_all_metrics(trues, preds, masks=None, pollutant_names=None):
    """
    Computes standard physical and statistical evaluation metrics,
    strictly excluding masked/nodata pixels from MAE, RMSE, R2, Rel-Acc, and SSIM.
    Args:
        trues: (N, 3, H, W) numpy array of true observations
        preds: (N, 3, H, W) numpy array of forecasted predictions
        masks: (N, 3, H, W) optional boolean/float mask (1 = valid, 0 = nodata)
    Returns:
        pd.DataFrame with Mean True Obs, MAE, RMSE, R2, Rel-Acc, and SSIM.
    """
    if pollutant_names is None:
        pollutant_names = ["NO2 (mol/m²)", "CO (mol/m²)", "SO2 (mol/m²)"]

    ssim_module = DifferentiableSSIMLoss(window_size=7)
    records = []

    for c, name in enumerate(pollutant_names):
        p = preds[:, c].flatten()
        t = trues[:, c].flatten()

        # Build mask: exclude NaNs, Infs, and zero-padded nodata
        if masks is not None:
            m = masks[:, c].flatten()
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
        else:
            mean_true = np.mean(t_val)
            mae = np.mean(np.abs(p_val - t_val))
            rmse = np.sqrt(np.mean((p_val - t_val) ** 2))
            ss_tot = np.sum((t_val - mean_true) ** 2)
            ss_res = np.sum((t_val - p_val) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
            rel_acc = max(0.0, (1.0 - mae / (abs(mean_true) + 1e-10))) * 100.0

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
            "Target Pollutant": name,
            "Mean True Obs": f"{mean_true:.4e}",
            "MAE": f"{mae:.4e}",
            "RMSE": f"{rmse:.4e}",
            "R² Score": f"{r2:.4f}",
            "Relative Accuracy": f"{rel_acc:.2f}%",
            "Spatial SSIM": f"{ssim_val:.4f}"
        })

    return pd.DataFrame(records)
