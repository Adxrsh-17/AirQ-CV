#!/usr/bin/env python3
"""evaluate.py - Stage B evaluation (Rule.md 7.5), physical units, per pollutant x horizon x region.

Headline metrics : SS_clim (vs per-pixel monthly climatology of the training years), SS_pers (vs last valid
                   observation), anomaly Pearson r, with 95% block-bootstrap CIs (blocks of 6 origins ~ 30 d).
Also reported    : pooled R2, RMSE, Pearson r, 80% interval coverage/width, CSI@monthly p90, SSIM.
Blob metrics     : hotspots = valid pixels above the training p90 of the same calendar month; 8-connected
                   blobs >= 2 px; matched at IoU >= 0.3 -> precision/recall/F1, IoU, centroid error (km),
                   area ratio, peak bias. Same metrics for the climatology and persistence maps.
Regions          : all valid pixels, and the spatial hold-out blocks never used in the training loss.
Ensemble         : --ckpt a.pt,b.pt,... averages the quantiles of several seeds.
"""
import argparse, os, numpy as np, pandas as pd, torch
from scipy.ndimage import gaussian_filter, label
from torch.utils.data import DataLoader
from dataset import DCDataset, collate, POLL_NAMES
from dcnet import DCNet

STRUCT = np.ones((3, 3), int)


def to_dev(b, dev): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


# ----------------------------------------------------------------------------- map metrics
def masked_ssim(a, b, m, rng, sigma=1.5):
    m = m.astype(np.float64)
    if m.mean() < 0.3 or rng <= 0: return np.nan
    G = lambda x: gaussian_filter(x, sigma, mode="constant")
    a, b = np.where(m > 0, a, 0.0), np.where(m > 0, b, 0.0)
    wm = G(m); ok = (m > 0) & (wm > 0.5); wm = np.maximum(wm, 1e-6)
    mu_a, mu_b = G(a * m) / wm, G(b * m) / wm
    va, vb = G(a * a * m) / wm - mu_a ** 2, G(b * b * m) / wm - mu_b ** 2
    cv = G(a * b * m) / wm - mu_a * mu_b
    C1, C2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    s = ((2 * mu_a * mu_b + C1) * (2 * cv + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s[ok].mean()) if ok.any() else np.nan


def _blobs(hot, min_px=2):
    lab, n = label(hot, structure=STRUCT)
    if n == 0: return lab, []
    sizes = np.bincount(lab.ravel())
    return lab, [i for i in range(1, n + 1) if sizes[i] >= min_px]


def blob_stats(obs, pred, valid, thr, px_km, iou_thr=0.3):
    """obs/pred physical maps, valid mask, scalar threshold. Returns additive counts."""
    lo, ko = _blobs((obs > thr) & valid); lp, kp = _blobs((pred > thr) & valid)
    r = dict(n_obs=len(ko), n_pred=len(kp), tp=0, iou=0.0, cdist=0.0, larea=0.0, pbias=0.0)
    if not ko or not kp: return r
    om = {i: lo == i for i in ko}; pm = {j: lp == j for j in kp}
    pairs = []
    for i in ko:
        for j in kp:
            inter = np.logical_and(om[i], pm[j]).sum()
            if inter: pairs.append((inter / np.logical_or(om[i], pm[j]).sum(), i, j))
    used_o, used_p = set(), set()
    for iou, i, j in sorted(pairs, reverse=True):
        if iou < iou_thr: break
        if i in used_o or j in used_p: continue
        used_o.add(i); used_p.add(j)
        ci, cj = np.argwhere(om[i]).mean(0), np.argwhere(pm[j]).mean(0)
        pk_o = obs[om[i]].max()
        r["tp"] += 1; r["iou"] += iou; r["cdist"] += float(np.hypot(*(ci - cj))) * px_km
        r["larea"] += float(np.log2(pm[j].sum() / om[i].sum()))
        r["pbias"] += float((pred[pm[j]].max() - pk_o) / (abs(pk_o) + 1e-30))
    return r


def summarise_blobs(acc):
    tp, no, npd = acc["tp"], acc["n_obs"], acc["n_pred"]
    prec = tp / npd if npd else np.nan; rec = tp / no if no else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0) else 0.0
    return dict(blob_precision=prec, blob_recall=rec, blob_F1=f1, blob_IoU=acc["iou"] / tp if tp else np.nan,
                centroid_err_km=acc["cdist"] / tp if tp else np.nan, area_ratio=2 ** (acc["larea"] / tp) if tp else np.nan,
                peak_bias=acc["pbias"] / tp if tp else np.nan, n_obs_blobs=no, n_pred_blobs=npd)


def block_bootstrap(num, den, block=6, n=1000, seed=0):
    """CI of 1 - sum(num)/sum(den) with moving-block resampling over consecutive origins."""
    num, den = np.asarray(num), np.asarray(den); T = len(num)
    if T < 2: return np.nan, np.nan
    rng = np.random.default_rng(seed); nb = int(np.ceil(T / block)); out = np.empty(n)
    for k in range(n):
        st = rng.integers(0, max(1, T - block + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, T)) for s in st])[:T]
        out[k] = 1 - num[idx].sum() / max(den[idx].sum(), 1e-30)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt, or comma-separated list for a seed ensemble")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="runs/dcnet_eval")
    ap.add_argument("--px_km", type=float, default=5.5, help="S5P grid spacing in km (centroid error)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    paths = [p for p in a.ckpt.split(",") if p]
    cks = [torch.load(p, map_location=dev, weights_only=False) for p in paths]
    ck = cks[0]; cfg = ck["cfg"]
    models = []
    for c in cks:
        m = DCNet(**c["model_kwargs"]).to(dev); m.load_state_dict(c["model"]); m.eval(); models.append(m)
    ds = DCDataset(a.cache, a.split, cfg, stats=ck["stats"])
    if not ds.origins: raise SystemExit(f"no origins in split '{a.split}' {cfg['splits'][a.split]} - is that period in the cache?")
    dl = DataLoader(ds, 1, shuffle=False, num_workers=2, collate_fn=collate)
    H5, W5, P, L = ds.H5, ds.W5, ds.n_poll, ds.L
    horizons, steps = list(cfg["horizons"]), list(cfg["steps"]); Q = len(steps)
    pn = [POLL_NAMES[c] for c in ds.poll_idx]
    tag = f"_{a.tag}" if a.tag else ""

    # reference climatology: per-pixel calendar-month mean of the training windows (physical units)
    tr = ds.tr_idx; mon = np.array([w.month for w in ds.ws])
    clim = np.zeros((12, P, H5, W5), np.float32)
    for p_i, c in enumerate(ds.poll_idx):
        v, mk = ds.val[tr, c], ds.msk[tr, c]
        ov = (v * mk).sum(0) / np.maximum(mk.sum(0), 1)
        for mth in range(1, 13):
            ii = [k for k, t in enumerate(tr) if mon[t] == mth]
            cnt = mk[ii].sum(0) if ii else np.zeros((H5, W5))
            clim[mth - 1, p_i] = np.where(cnt > 0, (v[ii] * mk[ii]).sum(0) / np.maximum(cnt, 1), ov) if ii else ov
    thr_phys = np.stack([[ds.norm.inv(ds.thr[m, c], c) for c in ds.poll_idx] for m in range(12)])   # (12, P)
    hold = ds.hold > 0
    rng_obs = {}

    keys = [(q, p) for q in range(Q) for p in range(P)]
    pooled = {k: dict(y=[], f=[], c=[], s=[], lo=[], hi=[], th=[], h=[]) for k in keys}
    per_origin = {k: dict(m=[], c=[], s=[], am=[]) for k in keys}
    ssim = {k: dict(m=[], c=[], s=[]) for k in keys}
    blobs = {(q, p, src): dict(n_obs=0, n_pred=0, tp=0, iou=0.0, cdist=0.0, larea=0.0, pbias=0.0)
             for q in range(Q) for p in range(P) for src in ("model", "climatology", "persistence")}
    occ_all = {q: ([], []) for q in range(Q)}
    sel_o = set(ds.origins[i] for i in np.linspace(0, len(ds.origins) - 1, min(3, len(ds.origins))).astype(int))
    panels = []

    print(f"evaluating {len(paths)} model(s) on {len(ds.origins)} {a.split} origins | pollutants {pn} | horizons {horizons} d", flush=True)
    with torch.no_grad():
        for b in dl:
            o_idx = int(b["origin"][0]); bd = to_dev(b, dev)
            outs = [m(bd) for m in models]
            get = lambda k: torch.stack([x[k] for x in outs]).mean(0)[0, ..., :H5, :W5].float().cpu().numpy()
            q50, q10, q90 = get("q50"), get("q10"), get("q90")
            occ = get("occ_logit")
            y = b["y"][0, ..., :H5, :W5].numpy(); ym = b["ym"][0, ..., :H5, :W5].numpy() > 0
            # raw persistence: last valid observation within the input window (physical), else climatology
            ids = np.clip(np.arange(o_idx - L + 1, o_idx + 1), 0, None)
            for q, st in enumerate(steps):
                tmon = (ds.ws[o_idx] + pd.Timedelta(days=5 * st)).month
                if 2 in ds.poll_idx:
                    sp = ds.poll_idx.index(2); mk = ym[q, sp]
                    occ_all[q][0].append(occ[q][mk]); occ_all[q][1].append(b["ev"][0, q, :H5, :W5].numpy()[mk])
                for p_i, c in enumerate(ds.poll_idx):
                    mk = ym[q, p_i]
                    if not mk.any(): continue
                    vv, mm = ds.val[ids, c], ds.msk[ids, c] > 0
                    li = np.where(mm, np.arange(L)[:, None, None], -1).max(0)
                    last = np.take_along_axis(vv, np.maximum(li, 0)[None], 0)[0]
                    cl = clim[tmon - 1, p_i]
                    pers = np.where(li >= 0, last, cl)
                    yt = ds.norm.inv(y[q, p_i], c); f = ds.norm.inv(q50[q, p_i], c)
                    lo, hi = ds.norm.inv(q10[q, p_i], c), ds.norm.inv(q90[q, p_i], c)
                    th = thr_phys[tmon - 1, p_i]
                    k = (q, p_i); d = pooled[k]
                    for nm, arr in (("y", yt), ("f", f), ("c", cl), ("s", pers), ("lo", lo), ("hi", hi)): d[nm].append(arr[mk])
                    d["th"].append(np.full(mk.sum(), th, np.float32)); d["h"].append(hold[mk])
                    po = per_origin[k]
                    po["m"].append(((yt - f)[mk] ** 2).sum()); po["c"].append(((yt - cl)[mk] ** 2).sum())
                    po["s"].append(((yt - pers)[mk] ** 2).sum())
                    if p_i not in rng_obs: rng_obs[p_i] = float(np.percentile(yt[mk], 99) - np.percentile(yt[mk], 1))
                    for nm, mp in (("m", f), ("c", cl), ("s", pers)):
                        ssim[k][nm].append(masked_ssim(yt, mp, mk, rng_obs[p_i]))
                    for src, mp in (("model", f), ("climatology", cl), ("persistence", pers)):
                        r = blob_stats(yt, mp, mk, th, a.px_km)
                        for kk, vv_ in r.items(): blobs[(q, p_i, src)][kk] += vv_
                    if o_idx in sel_o and horizons[q] in (5, 30, 120):
                        panels.append(dict(o=o_idx, q=q, p=p_i, y=np.where(mk, yt, np.nan), f=f, w=hi - lo, th=th,
                                           date=f"{ds.ws[o_idx] + pd.Timedelta(days=5 * st):%Y-%m-%d}"))

    rows, brows = [], []
    for (q, p_i), d in pooled.items():
        if not d["y"]: continue
        cat = {k: np.concatenate(v) for k, v in d.items()}
        po = {k: np.array(v) for k, v in per_origin[(q, p_i)].items()}
        for region, sel in (("all", np.ones_like(cat["h"], bool)), ("heldout_blocks", cat["h"])):
            if sel.sum() < 50: continue
            y, f, c_, s_ = cat["y"][sel], cat["f"][sel], cat["c"][sel], cat["s"][sel]
            sse = ((y - f) ** 2).sum()
            row = dict(pollutant=pn[p_i], horizon_days=horizons[q], region=region, n=int(sel.sum()),
                       SS_clim=1 - sse / max(((y - c_) ** 2).sum(), 1e-30),
                       SS_pers=1 - sse / max(((y - s_) ** 2).sum(), 1e-30),
                       anomaly_r=float(np.corrcoef(y - c_, f - c_)[0, 1]),
                       R2=1 - sse / max(((y - y.mean()) ** 2).sum(), 1e-30), RMSE=float(np.sqrt(sse / len(y))),
                       pearson_r=float(np.corrcoef(y, f)[0, 1]),
                       R2_climatology=1 - ((y - c_) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30),
                       coverage_80=float(np.mean((y >= cat["lo"][sel]) & (y <= cat["hi"][sel]))),
                       interval_width=float(np.mean(cat["hi"][sel] - cat["lo"][sel])))
            th = cat["th"][sel]; hit = ((f > th) & (y > th)).sum(); fa = ((f > th) & (y <= th)).sum(); mi = ((f <= th) & (y > th)).sum()
            row["CSI_p90"] = hit / max(hit + fa + mi, 1)
            if region == "all":
                row["SS_clim_lo"], row["SS_clim_hi"] = block_bootstrap(po["m"], po["c"], n=a.n_boot)
                row["SS_pers_lo"], row["SS_pers_hi"] = block_bootstrap(po["m"], po["s"], n=a.n_boot)
                for nm, lab in (("m", "SSIM"), ("c", "SSIM_climatology"), ("s", "SSIM_persistence")):
                    row[lab] = float(np.nanmean(ssim[(q, p_i)][nm])) if ssim[(q, p_i)][nm] else np.nan
                if pn[p_i] == "SO2" and occ_all[q][0]:
                    try:
                        from sklearn.metrics import roc_auc_score, average_precision_score
                        lg, ev = np.concatenate(occ_all[q][0]), np.concatenate(occ_all[q][1])
                        if 0 < ev.mean() < 1: row["SO2_occ_ROC_AUC"], row["SO2_occ_PR_AUC"] = roc_auc_score(ev, lg), average_precision_score(ev, lg)
                    except Exception as e:
                        print("AUC skipped:", e)
            rows.append(row)
        for src in ("model", "climatology", "persistence"):
            brows.append(dict(pollutant=pn[p_i], horizon_days=horizons[q], source=src, **summarise_blobs(blobs[(q, p_i, src)])))

    df, bf = pd.DataFrame(rows), pd.DataFrame(brows)
    f1, f2 = os.path.join(a.out, f"dcnet_{a.split}_metrics{tag}.csv"), os.path.join(a.out, f"dcnet_{a.split}_blobs{tag}.csv")
    df.to_csv(f1, index=False); bf.to_csv(f2, index=False)
    pd.set_option("display.width", 220)
    show = ["pollutant", "horizon_days", "SS_clim", "SS_clim_lo", "SS_clim_hi", "SS_pers", "anomaly_r", "R2", "R2_climatology",
            "SSIM", "CSI_p90", "coverage_80"]
    print("\n=== skill (region = all valid pixels) ===")
    print(df[df.region == "all"][[c for c in show if c in df]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if (df.region == "heldout_blocks").any():
        print("\n=== spatial hold-out blocks (never in the training loss) ===")
        print(df[df.region == "heldout_blocks"][["pollutant", "horizon_days", "SS_clim", "SS_pers", "anomaly_r", "R2"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== hotspot blobs ===")
    print(bf.pivot_table(index=["pollutant", "horizon_days"], columns="source", values="blob_F1").to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nsaved {f1}\nsaved {f2}")

    # ------------------------------------------------------------------ figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        al = df[df.region == "all"]
        fig, ax = plt.subplots(P, 4, figsize=(18, 3.6 * P), squeeze=False)
        for p_i, pname in enumerate(pn):
            g = al[al.pollutant == pname].sort_values("horizon_days"); gb = bf[bf.pollutant == pname]
            if g.empty: continue
            x = g.horizon_days.values
            ax[p_i, 0].plot(x, g.SS_clim, "o-", label="model"); ax[p_i, 0].fill_between(x, g.SS_clim_lo, g.SS_clim_hi, alpha=.25)
            ax[p_i, 0].axhline(0, c="k", lw=.8); ax[p_i, 0].set_title(f"{pname}: skill vs climatology (95% CI)")
            ax[p_i, 1].plot(x, g.SS_pers, "o-"); ax[p_i, 1].axhline(0, c="k", lw=.8); ax[p_i, 1].set_title("skill vs persistence")
            for col, lab in (("SSIM", "model"), ("SSIM_climatology", "climatology"), ("SSIM_persistence", "persistence")):
                ax[p_i, 2].plot(x, g[col], "o-", label=lab)
            ax[p_i, 2].set_title("SSIM"); ax[p_i, 2].legend(fontsize=8)
            for src in ("model", "climatology", "persistence"):
                s_ = gb[gb.source == src].sort_values("horizon_days"); ax[p_i, 3].plot(s_.horizon_days, s_.blob_F1, "o-", label=src)
            ax[p_i, 3].set_title("hotspot blob F1"); ax[p_i, 3].legend(fontsize=8)
            for k_ in range(4): ax[p_i, k_].set_xscale("log"); ax[p_i, k_].set_xticks(horizons); ax[p_i, k_].set_xticklabels(horizons); ax[p_i, k_].set_xlabel("lead (days)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"summary_{a.split}{tag}.png"), dpi=110); plt.close(fig)

        for p_i, pname in enumerate(pn):
            pp = [d for d in panels if d["p"] == p_i]
            if not pp: continue
            fig, ax = plt.subplots(len(pp), 3, figsize=(13, 3.6 * len(pp)), squeeze=False)
            for r_, d in enumerate(pp):
                vmin, vmax = np.nanpercentile(d["y"], 2), np.nanpercentile(d["y"], 98)
                ax[r_, 0].imshow(d["y"], vmin=vmin, vmax=vmax, cmap="magma"); ax[r_, 0].set_title(f"observed {d['date']}")
                ax[r_, 1].imshow(d["f"], vmin=vmin, vmax=vmax, cmap="magma")
                ax[r_, 1].contour((np.nan_to_num(d["y"], nan=-np.inf) > d["th"]).astype(float), levels=[.5], colors="cyan", linewidths=.8)
                ax[r_, 1].contour((d["f"] > d["th"]).astype(float), levels=[.5], colors="lime", linewidths=.8)
                ax[r_, 1].set_title(f"q50 +{horizons[d['q']]} d  (cyan obs / green fcst hotspots)")
                im = ax[r_, 2].imshow(d["w"], cmap="viridis"); ax[r_, 2].set_title("q90 - q10"); plt.colorbar(im, ax=ax[r_, 2], fraction=.04)
                for k_ in range(3): ax[r_, k_].axis("off")
            fig.tight_layout(); fig.savefig(os.path.join(a.out, f"panels_{pname}_{a.split}{tag}.png"), dpi=100); plt.close(fig)
        print(f"figures saved in {a.out}")
    except Exception as e:
        print("figure step skipped:", e)


if __name__ == "__main__":
    main()
