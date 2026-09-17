import os
import sys
import pandas as pd
import numpy as np
import torch

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

data_dir = r"..\Projec-Code\data\processed"
cache_pt = os.path.join(data_dir, "cached_dataset_256.pt")
cached = torch.load(cache_pt, map_location="cpu", weights_only=False)

df = cached["df"]
s5p = cached["s5p"] # (N, 3, 256, 256)
mask = cached["s5p_mask"] # (N, 3, 256, 256)

df["start_date"] = pd.to_datetime(df["start_date"])
df["year"] = df["start_date"].dt.year
df["month"] = df["start_date"].dt.month

hist_indices = df[df["year"] <= 2022].index.tolist()

monthly_climatology = {}
for m in range(1, 13):
    m_hist_idx = [i for i in hist_indices if df.iloc[i]["month"] == m]
    if len(m_hist_idx) == 0:
        m_hist_idx = hist_indices
    m_s5p = s5p[m_hist_idx]
    m_mask = mask[m_hist_idx]
    sum_vals = (m_s5p * m_mask).sum(dim=0)
    sum_weights = m_mask.sum(dim=0).clamp(min=1.0)
    monthly_climatology[m] = sum_vals / sum_weights

overall_hist_mean = (s5p[hist_indices] * mask[hist_indices]).sum(dim=0) / mask[hist_indices].sum(dim=0).clamp(min=1.0)

sys.path.insert(0, r"..\Projec-Code")
from evaluation.metrics import compute_all_metrics

offsets = [(ty, tx) for ty in [0, 64, 128] for tx in [0, 64, 128]]

def eval_climatology(horizon=1):
    preds_all = []
    trues_all = []
    masks_all = []
    
    t_in = 4
    for i in range(len(df) - (t_in - 1) - horizon):
        target_idx = i + t_in - 1 + horizon
        target_year = df.iloc[target_idx]["year"]
        target_month = df.iloc[target_idx]["month"]
        
        if target_year > 2022:
            clim_pred = monthly_climatology.get(target_month, overall_hist_mean)
            true_obs = s5p[target_idx]
            true_mask = mask[target_idx]
            
            for (ty, tx) in offsets:
                preds_all.append(clim_pred[:, ty:ty+128, tx:tx+128].numpy())
                trues_all.append(true_obs[:, ty:ty+128, tx:tx+128].numpy())
                masks_all.append(true_mask[:, ty:ty+128, tx:tx+128].numpy())
                
    preds_np = np.stack(preds_all)
    trues_np = np.stack(trues_all)
    masks_np = np.stack(masks_all)
    
    return compute_all_metrics(preds_np, trues_np, masks_np)

print("=" * 80)
print("SEASONAL-CLIMATOLOGY BASELINE EVALUATION ON 2023-2024 HOLDOUT")
print("=" * 80)

for hz in [1, 30, 60]:
    res_df = eval_climatology(horizon=hz)
    print(f"\n--- Seasonal-Climatology Baseline (Horizon={hz:2d} / {hz*5:3d} Days, N=1,098 Patches) ---")
    cols = ["Target Pollutant", "MAE", "RMSE", "R² Score", "Pearson r", "Spatial SSIM", "FSS (9x9)", "CSI (q90)", "POD (q90)", "FAR (q90)", "Log-space MAE"]
    avail_cols = [c for c in cols if c in res_df.columns]
    print(res_df[avail_cols].to_string(index=False))
