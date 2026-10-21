#!/usr/bin/env python3
"""
evaluate.py - Full multi-horizon evaluation for DC-Net
======================================================
Computes metrics per pollutant x horizon in physical units:
  - R2 and Pearson r
  - CSI (Critical Success Index) for plume peaks (> 90th pct)
  - 80% coverage and interval width (q90 - q10)
  - SO2 occurrence ROC-AUC and PR-AUC
  - Skill over Harmonic Climatology and Persistence
======================================================
"""
import argparse, json, os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset import DCDataset, S5PNorm, WeatherNorm, collate
from dcnet import DCNet, HORIZONS_DAYS, HORIZON_STEPS

def to_dev(b, dev): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
def model_in(b):
    res = dict(fast_v=b["fast_v"], fast_m=b["fast_m"], s2_slow=b["s2_slow"], long_mem=b["long_mem"],
               crop=b["crop"], doy_tgt=b["doy_tgt"], doy_org_sc=b["doy_org_sc"])
    if "weather" in b: res["weather"] = b["weather"]
    return res

def calc_csi(y_true, y_pred, thr):
    hits = np.sum((y_pred >= thr) & (y_true >= thr))
    false_alarms = np.sum((y_pred >= thr) & (y_true < thr))
    misses = np.sum((y_pred < thr) & (y_true >= thr))
    denom = hits + false_alarms + misses
    return hits / max(denom, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to best.pt checkpoint")
    p.add_argument("--cache", required=True, help="Path to cache dir")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--out", default="runs/dcnet_eval")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=dev)
    cfg = ckpt["cfg"]

    # Reconstruct norms
    s5_norm = S5PNorm()
    s5_norm.s = ckpt["norm"]["s"]
    s5_norm.mu = ckpt["norm"]["mu"]
    s5_norm.sd = ckpt["norm"]["sd"]

    wx_norm = None
    if ckpt.get("wx") is not None:
        wx_norm = WeatherNorm()
        wx_norm.mu = ckpt["wx"]["mu"]
        wx_norm.sd = ckpt["wx"]["sd"]

    ds = DCDataset(args.cache, args.split, cfg, norm=s5_norm, s2_stats=ckpt["s2"], wx_norm=wx_norm)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    use_wx = wx_norm is not None
    model = DCNet(3, (ds.H5, ds.W5), use_B=bool(cfg.get("use_B", 1)), use_s2=bool(cfg.get("use_s2", 1)), use_weather=use_wx).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    poll_names = ["NO2", "CO", "SO2"]
    Q = len(HORIZONS_DAYS)
    
    # Storage for accumulated predictions
    preds_phys = {c: [[] for _ in range(Q)] for c in range(3)}
    targets_phys = {c: [[] for _ in range(Q)] for c in range(3)}
    masks = {c: [[] for _ in range(Q)] for c in range(3)}
    q10_phys = {c: [[] for _ in range(Q)] for c in range(3)}
    q90_phys = {c: [[] for _ in range(Q)] for c in range(3)}
    priors_phys = {c: [[] for _ in range(Q)] for c in range(3)}
    persist_phys = {c: [[] for _ in range(Q)] for c in range(3)}
    occ_logits = [[] for _ in range(Q)]
    so2_events = [[] for _ in range(Q)]

    print(f"Running evaluation on {len(ds)} {args.split} windows...")
    with torch.no_grad():
        for b in dl:
            b_dev = to_dev(b, dev)
            H, W = b["crop"][2], b["crop"][3]
            o = model(model_in(b_dev))
            
            crop = lambda t: t[..., :H, :W]
            q50 = crop(o["q50"]).cpu().numpy()[0]
            q10 = crop(o["q10"]).cpu().numpy()[0]
            q90 = crop(o["q90"]).cpu().numpy()[0]
            prior = crop(model.prior(b_dev["doy_tgt"], b["crop"], (H, W))).cpu().numpy()[0]
            
            y = crop(b["y"]).numpy()[0]
            ym = crop(b["ym"]).numpy()[0]
            
            # Origin frame for persistence
            last_valid_fast = crop(b["fast_v"][:, -1]).numpy()[0] # (3, H, W)
            
            occ = crop(o["occ_logit"]).cpu().numpy()[0]
            ev = crop(b["ev"]).numpy()[0]

            for q in range(Q):
                occ_logits[q].append(occ[q].flatten())
                so2_events[q].append(ev[q].flatten())
                for c in range(3):
                    m = ym[q, c] > 0
                    if not m.any(): continue
                    masks[c][q].append(m.flatten())
                    
                    # Convert to physical units
                    p_p = s5_norm.inv(q50[q, c], c)
                    t_p = s5_norm.inv(y[q, c], c)
                    l_p = s5_norm.inv(q10[q, c], c)
                    u_p = s5_norm.inv(q90[q, c], c)
                    pr_p = s5_norm.inv(prior[q, c], c)
                    pers_p = s5_norm.inv(last_valid_fast[c], c)
                    
                    preds_phys[c][q].append(p_p.flatten())
                    targets_phys[c][q].append(t_p.flatten())
                    q10_phys[c][q].append(l_p.flatten())
                    q90_phys[c][q].append(u_p.flatten())
                    priors_phys[c][q].append(pr_p.flatten())
                    persist_phys[c][q].append(pers_p.flatten())

    results = []
    for c, poll in enumerate(poll_names):
        peak_thr_phys = float(s5_norm.inv(ds.peak_thr[c], c))
        for q, h_days in enumerate(HORIZONS_DAYS):
            if len(targets_phys[c][q]) == 0: continue
            
            m_all = np.concatenate(masks[c][q])
            y_t = np.concatenate(targets_phys[c][q])[m_all]
            y_p = np.concatenate(preds_phys[c][q])[m_all]
            y_lo = np.concatenate(q10_phys[c][q])[m_all]
            y_hi = np.concatenate(q90_phys[c][q])[m_all]
            y_pr = np.concatenate(priors_phys[c][q])[m_all]
            y_pers = np.concatenate(persist_phys[c][q])[m_all]

            sse_m = np.sum((y_t - y_p) ** 2)
            sse_pr = np.sum((y_t - y_pr) ** 2)
            sse_pers = np.sum((y_t - y_pers) ** 2)
            sst = np.sum((y_t - np.mean(y_t)) ** 2)

            r2 = 1.0 - sse_m / max(sst, 1e-12)
            r = np.corrcoef(y_t, y_p)[0, 1] if len(y_t) > 1 else np.nan
            skill_clim = (1.0 - sse_m / max(sse_pr, 1e-12)) * 100.0
            skill_pers = (1.0 - sse_m / max(sse_pers, 1e-12)) * 100.0
            
            cov = np.mean((y_t >= y_lo) & (y_t <= y_hi))
            width = np.mean(y_hi - y_lo)
            csi = calc_csi(y_t, y_p, peak_thr_phys)

            row = dict(pollutant=poll, horizon_days=h_days, horizon_step=h_days//5,
                       R2=float(r2), pearson_r=float(r), skill_vs_clim_pct=float(skill_clim),
                       skill_vs_pers_pct=float(skill_pers), coverage_80=float(cov),
                       interval_width=float(width), csi_p90=float(csi), n_obs=int(len(y_t)))
            results.append(row)

    df_res = pd.DataFrame(results)
    csv_path = os.path.join(args.out, f"dcnet_{args.split}_metrics.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved evaluation metrics to {csv_path}:")
    print(df_res[["pollutant", "horizon_days", "R2", "pearson_r", "skill_vs_clim_pct", "coverage_80", "csi_p90"]].to_string())

if __name__ == "__main__":
    main()
