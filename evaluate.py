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
    model = DCNet(ds.n_poll, (ds.H5, ds.W5), use_B=bool(cfg.get("use_B", 1)), use_s2=bool(cfg.get("use_s2", 1)), use_weather=use_wx).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_poll_names = ["NO2", "CO", "SO2"]
    poll_names = [all_poll_names[c] for c in ds.poll_idx]
    Q = len(HORIZONS_DAYS)

    # Compute Empirical Training-Set Pixel Climatology
    tr_idx = [i for i in range(ds.T) if ds.ws[i] <= pd.Timestamp(cfg["splits"]["train"][1])]
    clim_empirical = np.zeros((12, ds.n_poll, ds.H5, ds.W5), dtype=np.float32)
    ws_months = np.array([d.month for d in ds.ws])
    for m in range(1, 13):
        m_idx = [i for i in tr_idx if ws_months[i] == m]
        if len(m_idx) > 0:
            for p_i, c in enumerate(ds.poll_idx):
                v_sub = ds.val[m_idx, c]
                m_sub = ds.msk[m_idx, c] > 0
                cnt = m_sub.sum(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    c_mean = np.where(cnt > 0, (v_sub * m_sub).sum(axis=0) / np.maximum(cnt, 1), 0.0)
                clim_empirical[m - 1, p_i] = np.nan_to_num(c_mean, nan=0.0)

    # Fill any missing pixel-months with overall training pixel mean
    for p_i, c in enumerate(ds.poll_idx):
        v_sub = ds.val[tr_idx, c]
        m_sub = ds.msk[tr_idx, c] > 0
        cnt_all = m_sub.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ov_mean = np.where(cnt_all > 0, (v_sub * m_sub).sum(axis=0) / np.maximum(cnt_all, 1), 0.0)
        for m in range(12):
            missing = clim_empirical[m, p_i] == 0.0
            clim_empirical[m, p_i] = np.where(missing, ov_mean, clim_empirical[m, p_i])
    
    # Storage for accumulated predictions
    preds_phys = {p_i: [[] for _ in range(Q)] for p_i in range(ds.n_poll)}
    targets_phys = {p_i: [[] for _ in range(Q)] for p_i in range(ds.n_poll)}
    masks = {p_i: [[] for _ in range(Q)] for p_i in range(ds.n_poll)}
    q10_phys = {p_i: [[] for _ in range(Q)] for p_i in range(ds.n_poll)}
    q90_phys = {p_i: [[] for _ in range(Q)] for p_i in range(ds.n_poll)}
    clim_phys_list = {p_i: [[] for _ in range(Q)] for p_i in range(ds.n_poll)}
    priors_phys = {p_i: [[] for _ in range(Q)] for p_i in range(ds.n_poll)}
    persist_phys = {p_i: [[] for _ in range(Q)] for p_i in range(ds.n_poll)}
    occ_logits = [[] for _ in range(Q)]
    so2_events = [[] for _ in range(Q)]

    print(f"Running evaluation on {len(ds)} {args.split} windows (Pollutants: {poll_names})...")
    with torch.no_grad():
        for b_idx, b in enumerate(dl):
            b_dev = to_dev(b, dev)
            if torch.is_tensor(b["crop"]):
                if b["crop"].dim() == 2:
                    H, W = int(b["crop"][0, 2]), int(b["crop"][0, 3])
                    i0, j0 = int(b["crop"][0, 0]), int(b["crop"][0, 1])
                else:
                    H, W = int(b["crop"][2]), int(b["crop"][3])
                    i0, j0 = int(b["crop"][0]), int(b["crop"][1])
            else:
                H, W = int(b["crop"][2]), int(b["crop"][3])
                i0, j0 = int(b["crop"][0]), int(b["crop"][1])

            o = model(model_in(b_dev))
            crop = lambda t: t[..., :H, :W]
            q50 = crop(o["q50"]).cpu().numpy()[0]
            q10 = crop(o["q10"]).cpu().numpy()[0]
            q90 = crop(o["q90"]).cpu().numpy()[0]
            prior = crop(model.prior(b_dev["doy_tgt"], b["crop"], (H, W))).cpu().numpy()[0]
            
            y = crop(b["y"]).numpy()[0]
            ym = crop(b["ym"]).numpy()[0]
            
            # Origin frame for persistence
            # Last *valid* observation per pixel (not just the last window, which is 0/climatology where cloudy)
            fv_all, fm_all = b["fast_v"][0].numpy(), b["fast_m"][0].numpy()          # (L, P, H, W)
            t_idx = np.arange(fv_all.shape[0])[:, None, None, None]
            last_i = np.where(fm_all > 0, t_idx, -1).max(0)                           # (P, H, W)
            last_valid_fast = np.take_along_axis(fv_all, np.maximum(last_i, 0)[None], 0)[0][..., :H, :W]
            has_hist = (last_i >= 0)[..., :H, :W]
            
            occ = crop(o["occ_logit"]).cpu().numpy()[0]
            ev = crop(b["ev"]).numpy()[0]

            orig_idx = ds.origins[b_idx]
            steps = cfg["steps"]

            for q in range(Q):
                occ_logits[q].append(occ[q].flatten())
                so2_events[q].append(ev[q].flatten())
                st = steps[q]
                tgt_ts = ds.ws[orig_idx] + pd.Timedelta(days=5 * st)
                tgt_m = tgt_ts.month
                clim_sub = clim_empirical[tgt_m - 1, :, i0:i0 + H, j0:j0 + W]

                for p_i in range(ds.n_poll):
                    c = ds.poll_idx[p_i]
                    m = ym[q, p_i] > 0
                    if not m.any(): continue
                    masks[p_i][q].append(m.flatten())
                    
                    # Convert to physical units
                    p_p = s5_norm.inv(q50[q, p_i], c)
                    t_p = s5_norm.inv(y[q, p_i], c)
                    l_p = s5_norm.inv(q10[q, p_i], c)
                    u_p = s5_norm.inv(q90[q, p_i], c)
                    pr_p = s5_norm.inv(prior[q, p_i], c)
                    cl_p = clim_sub[p_i]
                    pers_p = np.where(has_hist[p_i], s5_norm.inv(last_valid_fast[p_i], c), cl_p)
                    
                    preds_phys[p_i][q].append(p_p.flatten())
                    targets_phys[p_i][q].append(t_p.flatten())
                    q10_phys[p_i][q].append(l_p.flatten())
                    q90_phys[p_i][q].append(u_p.flatten())
                    priors_phys[p_i][q].append(pr_p.flatten())
                    persist_phys[p_i][q].append(pers_p.flatten())
                    clim_phys_list[p_i][q].append(cl_p.flatten())

    results = []
    for p_i, poll in enumerate(poll_names):
        c = ds.poll_idx[p_i]
        peak_thr_phys = float(s5_norm.inv(ds.peak_thr[p_i], c))
        for q, h_days in enumerate(HORIZONS_DAYS):
            if len(targets_phys[p_i][q]) == 0: continue
            
            m_all = np.concatenate(masks[p_i][q])
            y_t = np.concatenate(targets_phys[p_i][q])[m_all]
            y_p = np.concatenate(preds_phys[p_i][q])[m_all]
            y_lo = np.concatenate(q10_phys[p_i][q])[m_all]
            y_hi = np.concatenate(q90_phys[p_i][q])[m_all]
            y_clim = np.concatenate(clim_phys_list[p_i][q])[m_all]
            y_pr = np.concatenate(priors_phys[p_i][q])[m_all]
            y_pers = np.concatenate(persist_phys[p_i][q])[m_all]

            sse_m = np.sum((y_t - y_p) ** 2)
            sse_clim = np.sum((y_t - y_clim) ** 2)
            sse_pr = np.sum((y_t - y_pr) ** 2)
            sse_pers = np.sum((y_t - y_pers) ** 2)
            sst = np.sum((y_t - np.mean(y_t)) ** 2)

            r2 = 1.0 - sse_m / max(sst, 1e-12)
            r = np.corrcoef(y_t, y_p)[0, 1] if len(y_t) > 1 else np.nan
            skill_clim = (1.0 - sse_m / max(sse_clim, 1e-12)) * 100.0
            skill_pers = (1.0 - sse_m / max(sse_pers, 1e-12)) * 100.0
            skill_prior = (1.0 - sse_m / max(sse_pr, 1e-12)) * 100.0
            
            cov = np.mean((y_t >= y_lo) & (y_t <= y_hi))
            width = np.mean(y_hi - y_lo)
            csi = calc_csi(y_t, y_p, peak_thr_phys)

            row = dict(pollutant=poll, horizon_days=h_days, horizon_step=h_days//5,
                       R2=float(r2), pearson_r=float(r), skill_vs_clim_pct=float(skill_clim),
                       skill_vs_pers_pct=float(skill_pers), skill_vs_prior_pct=float(skill_prior),
                       coverage_80=float(cov), interval_width=float(width), csi_p90=float(csi),
                       n_obs=int(len(y_t)))
            results.append(row)

    df_res = pd.DataFrame(results)
    csv_path = os.path.join(args.out, f"dcnet_{args.split}_metrics.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved evaluation metrics to {csv_path}:")
    if len(df_res) > 0:
        cols_to_print = [c for c in ["pollutant", "horizon_days", "R2", "pearson_r", "skill_vs_clim_pct", "coverage_80", "csi_p90"] if c in df_res.columns]
        print(df_res[cols_to_print].to_string())
    else:
        print("No evaluation records produced for this split.")

if __name__ == "__main__":
    main()
