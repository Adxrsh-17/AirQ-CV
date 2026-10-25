#!/usr/bin/env python3
"""
Stage A (Rule.md): land-use-inferred pollution baseline.

Question: can Sentinel-2 surface reflectance ALONE explain the spatial pattern of the
long-term S5P column (NO2 by default)?

  Input       : static S2 dry-season composite (median of Jan-Mar months), 12 bands + NDVI/NDBI/MNDWI/NDRE.
                No timestamps, no weather, no S5P history (Rule.md rules 1-2).
  Target      : log of the multi-year mean S5P column per S5P cell (supervision only, Rule 4).
                S5P is a ~3.5x5.5 km tropospheric COLUMN, not ground-level concentration (Rule 5).
  Validation  : spatial block K-fold with a buffer ring removed from training (no geographic leakage).
  Baselines   : global mean, geo_knn (smooth geography / "near Chennai = high" shortcut),
                ridge on land-use index means.  S2 models must beat geo_knn on held-out blocks.
  Models      : lgbm_s2 (per-cell S2 statistics + neighbourhood context), cnn_s2 (S2 patch),
                lgbm_s2_geo (diagnostic: does S2 add information on top of geography?).

Usage (Kaggle):
  python landuse_baseline.py --cache /kaggle/tmp/cache --out /kaggle/working/runs/landuse --poll NO2 --cnn 1
"""
import argparse, json, os, time, warnings
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12", "NDVI", "NDBI", "NDMI"]   # export channel order (GEE: 9 reflectances /1e4 + NDVI, NDBI, NDMI)
POLL = {"NO2": 0, "CO": 1, "SO2": 2}
IDX_NAMES = ["NDVI", "NDBI", "MNDWI", "NDRE"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- data
def s2_composite(cache, meta, months):
    """Per-subpixel median over valid S2 months whose calendar month is in `months`."""
    s2m = np.load(os.path.join(cache, "s2_month.npy"), mmap_mode="r")
    s2ok = np.load(os.path.join(cache, "s2_month_ok.npy"), mmap_mode="r")
    idx = [k for k, d in enumerate(meta["month_start"]) if int(d[5:7]) in months]
    ok = np.stack([np.asarray(s2ok[k]) > 0 for k in idx])                  # (n, H8, W8)
    C, H8, W8 = s2m.shape[1:]
    comp = np.full((C, H8, W8), np.nan, np.float32)
    for c in range(C):
        x = np.stack([np.asarray(s2m[k, c], np.float32) for k in idx])
        x[~ok] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            comp[c] = np.nanmedian(x, 0)
    return comp, ok.sum(0), len(idx)


def spectral_indices(comp):
    b = lambda n: comp[BANDS.index(n)]
    nd = lambda a, c: np.clip((a - c) / (a + c + 1e-6), -1, 1)
    with np.errstate(all="ignore"):
        return np.stack([nd(b("B8"), b("B4")), nd(b("B11"), b("B8")), nd(b("B3"), b("B11")), nd(b("B8"), b("B5"))]).astype(np.float32)


def s5p_mean(cache, meta, p, years):
    v = np.load(os.path.join(cache, "s5p_val.npy"), mmap_mode="r")
    m = np.load(os.path.join(cache, "s5p_mask.npy"), mmap_mode="r")
    yrs = np.array([int(d[:4]) for d in meta["win_start"]])
    sel = np.where(np.isin(yrs, list(years)))[0]
    vv, mm = np.asarray(v[sel, p], np.float64), np.asarray(m[sel, p]) > 0
    cnt = mm.sum(0)
    with np.errstate(all="ignore"):
        mean = np.where(cnt > 0, (vv * mm).sum(0) / np.maximum(cnt, 1), np.nan)
    return mean, cnt


# ----------------------------------------------------------------------------- features
def to_cells(a, R, H5, W5):
    """(K, H8, W8) -> (K, H5, W5, R*R) sub-pixels per S5P cell."""
    K = a.shape[0]
    return a[:, :H5 * R, :W5 * R].reshape(K, H5, R, W5, R).transpose(0, 1, 3, 2, 4).reshape(K, H5, W5, R * R)


def box_mean(a, w, r):
    """Weighted mean over a (2r+1)^2 cell neighbourhood. a: (K,H,W) may contain NaN, w: (H,W)."""
    K, H, W = a.shape
    a0 = np.nan_to_num(a) * w
    pa, pw = np.pad(a0, ((0, 0), (r, r), (r, r))), np.pad(w, r)
    s, sw = np.zeros_like(a0), np.zeros_like(w)
    for di in range(2 * r + 1):
        for dj in range(2 * r + 1):
            s += pa[:, di:di + H, dj:dj + W]; sw += pw[di:di + H, dj:dj + W]
    return s / np.maximum(sw, 1e-6)


def build_features(comp, ind, R, H5, W5, ctx_radii=(1, 3)):
    stack = np.concatenate([comp, ind])                                      # (16, H8, W8), NaN = no data
    names_ch = BANDS + IDX_NAMES
    x = to_cells(stack, R, H5, W5)
    valid_sub = np.isfinite(x[0])
    land_frac = valid_sub.mean(-1)                                           # (H5, W5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, std = np.nanmean(x, -1), np.nanstd(x, -1)
        p10, p90 = np.nanpercentile(x, [10, 90], axis=-1)
        ndvi, ndbi, mndwi = x[16 - 4], x[16 - 3], x[16 - 2]
        f = lambda cond: np.where(valid_sub, cond, np.nan)
        fr = np.stack([np.nanmean(f((ndbi > 0) & (ndvi < 0.2)), -1),         # built-up / bare impervious
                       np.nanmean(f(ndvi > 0.4), -1),                          # dense vegetation
                       np.nanmean(f(mndwi > 0), -1)])                          # water
    fr_names = ["frac_built", "frac_veg", "frac_water"]
    feats = [mean, std, p10, p90, fr]
    names = [f"{s}_{n}" for s in ("mean", "std", "p10", "p90") for n in names_ch] + fr_names
    for r in ctx_radii:                                                      # transport context: neighbouring land use
        ctx = box_mean(np.concatenate([mean, fr]), land_frac, r)
        feats.append(ctx); names += [f"ctx{2*r+1}_{n}" for n in names_ch + fr_names]
    F = np.concatenate(feats).astype(np.float32)                             # (nf, H5, W5)
    return F, names, land_frac


# ----------------------------------------------------------------------------- CV
def spatial_folds(H, W, block, k, seed):
    bid = (np.arange(H)[:, None] // block) * 100000 + (np.arange(W)[None, :] // block)
    ub = np.unique(bid)
    perm = np.random.default_rng(seed).permutation(len(ub))
    lut = {b: perm[i] % k for i, b in enumerate(ub)}
    return np.vectorize(lut.get)(bid)


def dilate(mask, r):
    H, W = mask.shape; p = np.pad(mask, r); out = np.zeros_like(mask)
    for di in range(2 * r + 1):
        for dj in range(2 * r + 1):
            out |= p[di:di + H, dj:dj + W]
    return out


def ring_exclude(cells, idx, H, W, r):
    """bool per cell: within r cells of any cell in idx (used to buffer inner folds)."""
    m = np.zeros((H, W), bool); m[cells[idx, 0], cells[idx, 1]] = True
    d = dilate(m, r); return d[cells[:, 0], cells[:, 1]]


def rank(a):
    r = np.empty(len(a)); r[np.argsort(a)] = np.arange(len(a)); return r


def metrics(y, p):
    ok = np.isfinite(p); y, p = y[ok], p[ok]
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    top, ptop = y >= np.quantile(y, 0.9), p >= np.quantile(p, 0.9)
    return dict(R2=r2, RMSE=float(np.sqrt(((y - p) ** 2).mean())), pearson=float(np.corrcoef(y, p)[0, 1]),
                spearman=float(np.corrcoef(rank(y), rank(p))[0, 1]),
                hotspot_precision_top10=float((top & ptop).sum() / max(ptop.sum(), 1)), n=int(ok.sum()))


# ----------------------------------------------------------------------------- models
def fit_predict(name, Xtr, ytr, Xte, seed):
    if name == "mean":
        return np.full(len(Xte), ytr.mean())
    if name == "geo_knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=10, weights="distance").fit(Xtr, ytr).predict(Xte)
    if name == "ridge_indices":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(Xtr, ytr).predict(Xte)
    if name.startswith("lgbm"):
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                              random_state=seed, verbose=-1)
        return m.fit(Xtr, ytr).predict(Xte)
    raise ValueError(name)


def cnn_oof(comp, ind, land_ok_sub, R, ctx, cells, y, folds_cell, train_sel_fn, epochs, seed):
    """Small CNN on the raw S2 patch around each cell. Trained per fold, fixed epochs (no test peeking)."""
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    valid = np.isfinite(comp[0])
    bands = comp.copy()
    for c in range(bands.shape[0]):                       # robust per-band scaling (inputs only, unsupervised)
        v = bands[c][valid]; med, iqr = np.median(v), np.subtract(*np.percentile(v, [75, 25])) + 1e-6
        bands[c] = (bands[c] - med) / iqr
    X = np.concatenate([np.nan_to_num(np.clip(bands, -5, 5)), np.nan_to_num(ind), valid[None].astype(np.float32)])
    pad = ctx * R; P = (2 * ctx + 1) * R
    X = torch.from_numpy(np.pad(X, ((0, 0), (pad, pad), (pad, pad)))).float().to(dev)
    cin = X.shape[0]

    def patches(cc):
        return torch.stack([X[:, i * R:i * R + P, j * R:j * R + P] for i, j in cc])

    def blk(a, b): return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.GELU())

    oof = np.full(len(y), np.nan)
    for f in np.unique(folds_cell):
        tr, te = train_sel_fn(f), np.where(folds_cell == f)[0]
        mu, sd = y[tr].mean(), y[tr].std() + 1e-8
        net = nn.Sequential(blk(cin, 32), blk(32, 32), nn.MaxPool2d(2), blk(32, 64), blk(64, 64), nn.MaxPool2d(2),
                            blk(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = epochs * int(np.ceil(len(tr) / 128)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        yt = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32, device=dev)
        net.train()
        for ep in range(epochs):
            perm = np.random.permutation(len(tr))
            for s in range(0, len(tr), 128):
                b = perm[s:s + 128]; xb = patches(cells[tr[b]])
                if np.random.rand() < 0.5: xb = xb.flip(-1)            # flips only: keep N/S-E/W layout mostly intact
                if np.random.rand() < 0.5: xb = xb.flip(-2)
                loss = nn.functional.huber_loss(net(xb).squeeze(1), yt[b], delta=1.0)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        net.eval(); pr = []
        with torch.no_grad():
            for s in range(0, len(te), 256):
                pr.append(net(patches(cells[te[s:s + 256]])).squeeze(1).cpu().numpy())
        oof[te] = np.concatenate(pr) * sd + mu
        log(f"  cnn fold {f}: train {len(tr)} test {len(te)} R2={metrics(y[te], oof[te])['R2']:.3f}")
    return oof


# ----------------------------------------------------------------------------- main
def main(a):
    os.makedirs(a.out, exist_ok=True)
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    H5, W5, R = meta["H5"], meta["W5"], int(meta.get("s2_ratio", meta["H8"] // meta["H5"]))
    p = POLL[a.poll]; years = list(range(a.y0, a.y1 + 1)); months = [int(m) for m in a.months.split(",")]

    log(f"S2 composite: calendar months {months} ...")
    comp, nobs, nmon = s2_composite(a.cache, meta, months)
    ind = spectral_indices(comp)
    log(f"  used {nmon} monthly composites; sub-pixels with >=1 valid month: {np.isfinite(comp[0]).mean():.3f}")

    tgt, cnt = s5p_mean(a.cache, meta, p, years)
    F, names, land_frac = build_features(comp, ind, R, H5, W5)

    valid = (cnt >= a.min_obs) & (land_frac >= a.min_land) & np.isfinite(tgt) & (tgt > 0) & np.isfinite(F).all(0)
    cells = np.argwhere(valid)
    y = np.log(tgt[valid]) if a.log_target else tgt[valid]
    X = F[:, valid].T
    log(f"target {a.poll} mean {years[0]}-{years[-1]}: {len(y)} valid land cells of {H5*W5} "
        f"(min_obs={a.min_obs}, min_land={a.min_land}); features {X.shape[1]}")

    if a.prior_only:
        # Stage B input (Rule.md 7.3): S2-only land-use prior map, SPATIAL OUT-OF-FOLD (each cell predicted by a model
        # that never saw its block or the buffer ring), target period y0..y1 (set y1 to the last TRAINING year).
        fold_map = spatial_folds(H5, W5, a.prior_block, a.k, a.seed); fc = fold_map[valid]
        oof = np.full(len(y), np.nan)
        for f in range(a.k):
            te = np.where(fc == f)[0]
            if len(te) == 0: continue
            tr = np.where(~dilate((fold_map == f) & valid, a.buffer)[valid])[0]
            oof[te] = fit_predict("lgbm_s2", X[tr], y[tr], X[te], a.seed)
        m = metrics(y, oof)
        prior = np.full((H5, W5), np.nan, np.float32)
        prior[valid] = np.exp(oof) if a.log_target else oof              # physical units, NaN where no prediction
        os.makedirs(a.prior_out or a.out, exist_ok=True)
        fn = os.path.join(a.prior_out or a.out, f"prior_{a.poll}.npy"); np.save(fn, prior)
        log(f"prior_{a.poll}: OOF lgbm_s2, block {a.prior_block} cells, years {years[0]}-{years[-1]}: "
            f"R2={m['R2']:.3f} spearman={m['spearman']:.3f} coverage={valid.mean():.2f} -> {fn}")
        return

    # reliability ceiling: how reproducible is the target map itself? (split-half by years)
    h = len(years) // 2
    t1, _ = s5p_mean(a.cache, meta, p, years[:h]); t2, _ = s5p_mean(a.cache, meta, p, years[h:])
    ok = valid & np.isfinite(t1) & np.isfinite(t2) & (t1 > 0) & (t2 > 0)
    f_ = np.log if a.log_target else (lambda z: z)
    r_half = float(np.corrcoef(f_(t1[ok]), f_(t2[ok]))[0, 1]); rel = 2 * r_half / (1 + r_half)
    log(f"target reliability: split-half r={r_half:.3f} (years {years[:h]} vs {years[h:]}) -> Spearman-Brown {rel:.3f} "
        f"(approx. ceiling on explainable variance)")

    geo = cells.astype(np.float32)
    idx_cols = [i for i, n in enumerate(names) if any(k in n for k in IDX_NAMES + ["frac_"]) and not n.startswith(("std_", "p10_", "p90_"))]
    loc_cols = [i for i, n in enumerate(names) if not n.startswith("ctx")]            # own cell only (~5 km)
    c3_cols = [i for i, n in enumerate(names) if not n.startswith("ctx7")]            # + 3x3 cells (~16 km)
    designs = {"mean": X[:, :1], "geo_knn": geo, "ridge_indices": X[:, idx_cols], "lgbm_s2": X,
               "lgbm_s2_local": X[:, loc_cols], "lgbm_s2_ctx3": X[:, c3_cols],
               "lgbm_s2_geo": np.concatenate([X, geo], 1)}

    rows, oof_all = [], {}
    for block in [int(b) for b in a.blocks.split(",")]:
        fold_map = spatial_folds(H5, W5, block, a.k, a.seed)
        fc = fold_map[valid]

        def train_sel(f, fold_map=fold_map):
            test = (fold_map == f) & valid
            excl = dilate(test, a.buffer)
            return np.where(~excl[valid])[0]

        log(f"--- spatial block {block} cells (~{block*5} km), {a.k} folds, buffer {a.buffer} cells ---")
        for name, D in designs.items():
            oof = np.full(len(y), np.nan)
            for f in range(a.k):
                tr, te = train_sel(f), np.where(fc == f)[0]
                if len(te) == 0: continue
                oof[te] = fit_predict(name, D[tr], y[tr], D[te], a.seed)
            m = metrics(y, oof); m.update(model=name, block=block); rows.append(m); oof_all[f"{name}_b{block}"] = oof
            log(f"  {name:14s} R2={m['R2']:.3f} spearman={m['spearman']:.3f} hotspot@10%={m['hotspot_precision_top10']:.2f}")
        if a.cnn:
            oof = cnn_oof(comp, ind, None, R, a.ctx, cells, y, fc, train_sel, a.epochs, a.seed)
            m = metrics(y, oof); m.update(model="cnn_s2", block=block); rows.append(m); oof_all[f"cnn_s2_b{block}"] = oof
            log(f"  {'cnn_s2':14s} R2={m['R2']:.3f} spearman={m['spearman']:.3f} hotspot@10%={m['hotspot_precision_top10']:.2f}")

        # stacked test (leak-free): geography trend + S2 correction of the trend's error.
        # Residual targets for training cells come from an INNER spatial CV with the same block size,
        # so the S2 model learns to fix hole-filling errors of the size it will face on the test block.
        oof_trend, oof_fin = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for f in range(a.k):
            te, tr = np.where(fc == f)[0], train_sel(f)
            if len(te) == 0: continue
            trend_tr = np.full(len(y), np.nan)
            for g in range(a.k):
                if g == f: continue
                it = tr[fc[tr] == g]
                if len(it) == 0: continue
                itr = tr[~ring_exclude(cells, it, H5, W5, a.buffer)[tr]]
                trend_tr[it] = fit_predict("geo_knn", geo[itr], y[itr], geo[it], a.seed)
            tr2 = tr[np.isfinite(trend_tr[tr])]
            trend_te = fit_predict("geo_knn", geo[tr], y[tr], geo[te], a.seed)
            res = fit_predict("lgbm", X[tr2], y[tr2] - trend_tr[tr2], X[te], a.seed)
            oof_trend[te], oof_fin[te] = trend_te, trend_te + res
        m = metrics(y, oof_fin); m.update(model="stack_geo+s2", block=block); rows.append(m)
        mr = metrics(y - oof_trend, oof_fin - oof_trend); mr.update(model="stack_resid_only", block=block); rows.append(mr)
        oof_all[f"stack_b{block}"] = oof_fin
        log(f"  {'stack_geo+s2':14s} R2={m['R2']:.3f} residual-only R2={mr['R2']:.3f}")

    import pandas as pd
    df = pd.DataFrame(rows)[["block", "model", "R2", "RMSE", "pearson", "spearman", "hotspot_precision_top10", "n"]]
    df.to_csv(os.path.join(a.out, f"landuse_{a.poll}_results.csv"), index=False)
    np.savez_compressed(os.path.join(a.out, f"landuse_{a.poll}_oof.npz"), cells=cells, y=y, **oof_all)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== VERDICT (per block size) ===")
    for block, g in df.groupby("block"):
        r = g.set_index("model")["R2"]
        best_s2 = max((m for m in r.index if m in ("lgbm_s2", "cnn_s2", "ridge_indices")), key=lambda m: r[m])
        print(f"block {block}: best S2-only = {best_s2} R2 {r[best_s2]:.3f} | geography baseline geo_knn {r['geo_knn']:.3f} "
              f"| S2 beats geography: {'YES' if r[best_s2] > r['geo_knn'] else 'NO'} "
              f"| S2 adds over geography (lgbm_s2_geo - geo_knn): {r['lgbm_s2_geo'] - r['geo_knn']:+.3f} "
              f"| reliability ceiling ~{rel:.3f}")
        if "stack_geo+s2" in r.index:
            print(f"   stacking: geo_knn {r['geo_knn']:.3f} -> geo + S2 correction {r['stack_geo+s2']:.3f} "
                  f"(gain {r['stack_geo+s2'] - r['geo_knn']:+.3f}) | R2 on the residual alone {r['stack_resid_only']:.3f}")
            print(f"   scale ablation (S2 only): own cell {r['lgbm_s2_local']:.3f} | +16 km {r['lgbm_s2_ctx3']:.3f} | +38 km {r['lgbm_s2']:.3f}")

    # feature importance of the S2-only LightGBM fitted on all cells (descriptive only)
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.7, random_state=a.seed, verbose=-1, importance_type="gain").fit(X, y)
    imp = sorted(zip(m.feature_importances_, names), reverse=True)[:15]
    print("\nTop S2 features by gain (descriptive, all cells):"); [print(f"  {n:24s} {v:.1f}") for v, n in imp]

    grp = {"own cell": 0.0, "+3x3 (~16 km)": 0.0, "+7x7 (~38 km)": 0.0}
    for v, n in zip(m.feature_importances_, names):
        grp["+7x7 (~38 km)" if n.startswith("ctx7") else "+3x3 (~16 km)" if n.startswith("ctx3") else "own cell"] += v
    tot = sum(grp.values()); print("Gain share by spatial scale:", {k: f"{100 * v / tot:.0f}%" for k, v in grp.items()})

    # error by built-up fraction quartile (S2-derived; an INDEPENDENT land-cover map is the proper check, see notes)
    fb = X[:, names.index("frac_built")]; edges = np.quantile(fb, [.25, .5, .75]); qb = np.digitize(fb, edges)
    out = []
    for block in [int(b) for b in a.blocks.split(",")]:
        for b in range(4):
            sel = qb == b
            if sel.sum() == 0: continue
            row = dict(block=block, built_quartile=b + 1, frac_built=float(fb[sel].mean()), target_mean=float(y[sel].mean()), n=int(sel.sum()))
            for k_ in ("geo_knn", "lgbm_s2", "stack"):
                p_ = oof_all[f"{k_}_b{block}"][sel]
                row[f"rmse_{k_}"] = float(np.sqrt(np.nanmean((y[sel] - p_) ** 2))); row[f"bias_{k_}"] = float(np.nanmean(p_ - y[sel]))
            out.append(row)
    bf = pd.DataFrame(out); bf.to_csv(os.path.join(a.out, f"landuse_{a.poll}_by_builtfrac.csv"), index=False)
    print("\nError by built-up fraction quartile (1 = least built):"); print(bf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        blk = int(a.blocks.split(",")[0]); key = f"cnn_s2_b{blk}" if a.cnn else f"lgbm_s2_b{blk}"
        maps = {}
        for k, v in (("target", y), ("S2 prediction (" + key + ")", oof_all[key]), ("residual", y - oof_all[key])):
            M = np.full((H5, W5), np.nan); M[cells[:, 0], cells[:, 1]] = v; maps[k] = M
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        vmin, vmax = np.nanpercentile(maps["target"], [2, 98])
        for i, (k, M) in enumerate(maps.items()):
            im = ax[i].imshow(M, cmap="RdBu_r" if k == "residual" else "viridis",
                              **({} if k == "residual" else dict(vmin=vmin, vmax=vmax)))
            ax[i].set_title(k); fig.colorbar(im, ax=ax[i], shrink=0.7)
        fig.suptitle(f"{a.poll} {years[0]}-{years[-1]} mean column (log={a.log_target}), spatial block {blk}")
        fig.savefig(os.path.join(a.out, f"landuse_{a.poll}_maps.png"), dpi=110, bbox_inches="tight")
    except Exception as e:
        log("map plot skipped:", e)
    log("done ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--out", default="runs/landuse")
    ap.add_argument("--poll", default="NO2", choices=list(POLL)); ap.add_argument("--y0", type=int, default=2019)
    ap.add_argument("--y1", type=int, default=2024); ap.add_argument("--months", default="1,2,3")
    ap.add_argument("--min_obs", type=int, default=40); ap.add_argument("--min_land", type=float, default=0.5)
    ap.add_argument("--log_target", type=int, default=1); ap.add_argument("--blocks", default="8,4,16")
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--buffer", type=int, default=2)
    ap.add_argument("--cnn", type=int, default=1); ap.add_argument("--ctx", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prior_only", type=int, default=0, help="1 = only export the OOF S2 prior map for Stage B")
    ap.add_argument("--prior_block", type=int, default=8); ap.add_argument("--prior_out", default=None)
    main(ap.parse_args())
