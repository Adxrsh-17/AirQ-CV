#!/usr/bin/env python3
"""attribution_test.py - which inputs does the trained Stage B model rely on, and does S2 alone explain the hotspots?

TEST 1  Input attribution on the trained checkpoint (no retraining). The validation split is re-run with one input
        group destroyed at a time; the drop in R2 / SS_clim versus the intact model is how much the network relies on it.
          weather_shuffle  ERA5 history taken from another, random validation origin (breaks weather <-> target link)
          s2_shuffle       S2 land-use image permuted in 8x8-cell blocks (~44 km)   (breaks land use <-> location link)
          prior_shuffle    Stage A prior map permuted with the same blocks
          landuse_shuffle  S2 and prior permuted together
          history_zero     all S5P history removed (fast clock, monthly memory, persistence)
          clim_flat        per-pixel climatology replaced by its spatial mean per horizon/pollutant
                           (keeps the seasonal cycle, removes the per-pixel S5P map memory)
        Caveat: this measures reliance of THIS network, not the information in an input. Inputs that carry the same
        information (S2/prior and climatology both encode the hotspot map) can mask each other; Config B retraining
        (notebook ABLATE_B=True) is the complementary test.

TEST 2  S2-only hotspot test on static maps, against the observed mean map of the validation year:
          prior        Stage A S2-only map (spatial out-of-fold, target years = training years)
          climatology  training-years S5P mean (pure S5P memory)
          model        mean 5-day forecast q50 over the validation origins
        Spatial Pearson r, Spearman rho, top-10% precision, top-10% hotspot blob precision/recall/F1/IoU/centroid error.
        Full grid and the Kongu box (Coimbatore, Palakkad, Tiruppur, Erode, Salem, Namakkal).
"""
import argparse, os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from dataset import DCDataset, collate, POLL_NAMES
from dcnet import DCNet
from evaluate import to_dev, blob_stats, summarise_blobs, region_slices

VARIANTS = ["intact", "weather_shuffle", "s2_shuffle", "prior_shuffle", "landuse_shuffle", "history_zero", "clim_flat"]


def permute_blocks(x, blk, perm):
    """x: tensor (..., H, W), H and W multiples of blk; blocks of blk x blk are reordered by perm."""
    *lead, H, W = x.shape
    nbi, nbj = H // blk, W // blk
    v = x.reshape(*lead, nbi, blk, nbj, blk).transpose(-3, -2).reshape(*lead, nbi * nbj, blk, blk)
    v = v[..., perm, :, :]
    return v.reshape(*lead, nbi, nbj, blk, blk).transpose(-3, -2).reshape(*lead, H, W)


def apply_variant(b, name, ds, o_idx, rng, blk=8):
    """returns a modified copy of the (device) batch dict."""
    b = dict(b)
    Hp, Wp = b["y"].shape[-2:]
    r = b["s2"].shape[-1] // Wp
    perm = torch.as_tensor(rng.permutation((Hp // blk) * (Wp // blk)), device=b["y"].device)
    if name == "weather_shuffle" and "wx" in b:
        others = [o for o in ds.origins if abs(o - o_idx) > 6]
        o2 = int(rng.choice(others)); L = ds.L
        ids = np.arange(o2 - L + 1, o2 + 1); okid = (ids >= 0).astype(np.float32); ids = np.clip(ids, 0, None)
        b["wx"] = torch.as_tensor((ds.wx_feat[ids] * okid[:, None]).reshape(1, -1), device=b["wx"].device)
    if name in ("s2_shuffle", "landuse_shuffle"):
        b["s2"] = permute_blocks(b["s2"], blk * r, perm)
    if name in ("prior_shuffle", "landuse_shuffle"):
        b["prior"] = permute_blocks(b["prior"], blk, perm)
    if name == "history_zero":
        for k in ("fast_a", "fast_m", "long_a", "long_m", "last_a", "has"): b[k] = torch.zeros_like(b[k])
        b["age"] = torch.ones_like(b["age"])
    if name == "clim_flat":
        H5, W5 = ds.H5, ds.W5
        c = b["clim_tgt"]
        mu = c[..., :H5, :W5].mean((-2, -1), keepdim=True)
        b["clim_tgt"] = torch.zeros_like(c) + mu
    return b


def calendar_clim(ds):
    tr = ds.tr_idx; mon = np.array([w.month for w in ds.ws]); P, H5, W5 = ds.n_poll, ds.H5, ds.W5
    clim = np.zeros((12, P, H5, W5), np.float32)
    for p_i, c in enumerate(ds.poll_idx):
        v, mk = ds.val[tr, c], ds.msk[tr, c]
        ov = (v * mk).sum(0) / np.maximum(mk.sum(0), 1)
        for mth in range(1, 13):
            ii = [k for k, t in enumerate(tr) if mon[t] == mth]
            cnt = mk[ii].sum(0) if ii else np.zeros((H5, W5))
            clim[mth - 1, p_i] = np.where(cnt > 0, (v[ii] * mk[ii]).sum(0) / np.maximum(cnt, 1), ov) if ii else ov
    return clim


def rank_map(x, valid):
    out = np.zeros_like(x, dtype=np.float64)
    v = x[valid]
    if v.size < 2: return out
    out[valid] = np.argsort(np.argsort(v)) / (v.size - 1)
    return out


def static_map_scores(obs, cand, valid, px_km):
    o, c = obs[valid], cand[valid]
    r = float(np.corrcoef(o, c)[0, 1])
    ro, rc = rank_map(obs, valid), rank_map(cand, valid)
    rho = float(np.corrcoef(ro[valid], rc[valid])[0, 1])
    top_o, top_c = (ro >= 0.9) & valid, (rc >= 0.9) & valid
    prec10 = float((top_o & top_c).sum() / max(top_c.sum(), 1))
    bl = summarise_blobs(blob_stats(ro, rc, valid, 0.9 - 1e-9, px_km))
    return dict(pearson_r=r, spearman=rho, top10_precision=prec10, blob_F1=bl["blob_F1"], blob_precision=bl["blob_precision"],
                blob_recall=bl["blob_recall"], blob_IoU=bl["blob_IoU"], centroid_err_km=bl["centroid_err_km"],
                n_obs_blobs=bl["n_obs_blobs"], n_pred_blobs=bl["n_pred_blobs"], n_pixels=int(valid.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", default="runs/attribution")
    ap.add_argument("--px_km", type=float, default=5.5)
    ap.add_argument("--block", type=int, default=8, help="shuffle block size in S5P cells (8 = ~44 km)")
    ap.add_argument("--min_obs", type=int, default=10, help="min valid windows for a pixel of the observed mean map")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bbox", default="kongu"); ap.add_argument("--bbox_rc", default=None); ap.add_argument("--ref_tif", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); cfg = ck["cfg"]
    model = DCNet(**ck["model_kwargs"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
    ds = DCDataset(a.cache, a.split, cfg, stats=ck["stats"])
    dl = DataLoader(ds, 1, shuffle=False, num_workers=2, collate_fn=collate)
    H5, W5, P = ds.H5, ds.W5, ds.n_poll
    horizons, steps = list(cfg["horizons"]), list(cfg["steps"]); Q = len(steps)
    pn = [POLL_NAMES[c] for c in ds.poll_idx]
    variants = [v for v in VARIANTS if not (v == "weather_shuffle" and not ds.has_weather)]
    if not model.use_s2: variants = [v for v in variants if v not in ("s2_shuffle", "landuse_shuffle")]
    if not model.use_prior: variants = [v for v in variants if v not in ("prior_shuffle", "landuse_shuffle")]
    RS = region_slices(a, H5, W5)
    box = np.zeros((H5, W5), bool)
    if RS is not None: box[RS] = True
    clim = calendar_clim(ds)

    # accumulators: [variant][region] -> arrays (Q, P)
    regs = ["all"] + (["box"] if RS is not None else [])
    z = lambda: {r: {k: np.zeros((Q, P)) for k in ("sse", "ssc", "sy", "syy", "n")} for r in regs}
    acc = {v: z() for v in variants}
    fmap_sum, fmap_n = np.zeros((P, H5, W5)), 0
    print(f"attribution test: {len(ds.origins)} {a.split} origins x {len(variants)} variants {variants}", flush=True)
    with torch.no_grad():
        for k, b in enumerate(dl):
            o_idx = int(b["origin"][0]); bd = to_dev(b, dev)
            y = b["y"][0, ..., :H5, :W5].numpy(); ym = b["ym"][0, ..., :H5, :W5].numpy() > 0
            for v in variants:
                rng = np.random.default_rng(a.seed * 100003 + o_idx)
                bv = bd if v == "intact" else apply_variant(bd, v, ds, o_idx, rng, a.block)
                q50 = model(bv)["q50"][0, ..., :H5, :W5].float().cpu().numpy()
                for q, st in enumerate(steps):
                    tmon = (ds.ws[o_idx] + pd.Timedelta(days=5 * st)).month
                    for p_i, c in enumerate(ds.poll_idx):
                        f = ds.norm.inv(q50[q, p_i], c)
                        if v == "intact" and q == 0: fmap_sum[p_i] += f
                        mk = ym[q, p_i]
                        if not mk.any(): continue
                        yt = ds.norm.inv(y[q, p_i], c); cl = clim[tmon - 1, p_i]
                        for r in regs:
                            m = mk if r == "all" else (mk & box)
                            A = acc[v][r]
                            A["sse"][q, p_i] += ((yt - f)[m] ** 2).sum(); A["ssc"][q, p_i] += ((yt - cl)[m] ** 2).sum()
                            A["sy"][q, p_i] += yt[m].sum(); A["syy"][q, p_i] += (yt[m] ** 2).sum(); A["n"][q, p_i] += m.sum()
            fmap_n += 1
            if (k + 1) % 10 == 0: print(f"  {k + 1}/{len(ds.origins)} origins", flush=True)

    rows = []
    for v in variants:
        for r in regs:
            A = acc[v][r]
            with np.errstate(all="ignore"):
                sst = A["syy"] - A["sy"] ** 2 / np.maximum(A["n"], 1)
                r2 = 1 - A["sse"] / np.maximum(sst, 1e-30); ss = 1 - A["sse"] / np.maximum(A["ssc"], 1e-30)
            for q in range(Q):
                for p_i in range(P):
                    if A["n"][q, p_i] < 50: continue
                    rows.append(dict(variant=v, region=r, pollutant=pn[p_i], horizon_days=horizons[q], R2=r2[q, p_i], SS_clim=ss[q, p_i]))
    df = pd.DataFrame(rows)
    base = df[df.variant == "intact"].set_index(["region", "pollutant", "horizon_days"])
    df["dR2_vs_intact"] = df.apply(lambda x: x.R2 - base.loc[(x.region, x.pollutant, x.horizon_days), "R2"], axis=1)
    df["dSS_vs_intact"] = df.apply(lambda x: x.SS_clim - base.loc[(x.region, x.pollutant, x.horizon_days), "SS_clim"], axis=1)
    f1 = os.path.join(a.out, f"attribution_{a.split}.csv"); df.to_csv(f1, index=False)

    pd.set_option("display.width", 220)
    for r in regs:
        sub = df[(df.region == r) & (df.variant != "intact")]
        piv = sub.pivot_table(index="variant", columns="pollutant", values="dR2_vs_intact", aggfunc="mean").reindex([v for v in variants if v != "intact"])
        print(f"\n=== TEST 1 ({r}): drop in R2 when an input is destroyed (mean over horizons; more negative = model relies on it more) ===")
        print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))
        print(f"intact R2 (mean over horizons): " + ", ".join(f"{p} {df[(df.region == r) & (df.variant == 'intact') & (df.pollutant == p)].R2.mean():.3f}" for p in pn))
    print("\nper horizon (all pixels):")
    print(df[(df.region == "all")].pivot_table(index=["pollutant", "variant"], columns="horizon_days", values="dR2_vs_intact")
          .to_string(float_format=lambda x: f"{x:+.4f}"))

    # ---------------------------------------------------------------- TEST 2: static maps
    sel = [i for i in range(ds.T) if ds.lo <= ds.ws[i] <= ds.hi]
    srows, maps = [], {}
    fmean = fmap_sum / max(fmap_n, 1)
    for p_i, c in enumerate(ds.poll_idx):
        v_, m_ = ds.val[sel, c], ds.msk[sel, c]
        cnt = m_.sum(0); obs = np.where(cnt >= a.min_obs, (v_ * m_).sum(0) / np.maximum(cnt, 1), np.nan)
        vt, mt = ds.val[ds.tr_idx, c], ds.msk[ds.tr_idx, c]
        clim_map = np.where(mt.sum(0) > 0, (vt * mt).sum(0) / np.maximum(mt.sum(0), 1), np.nan)
        pr_ok = ds.prior[-1] > 0
        prior = np.where(pr_ok, ds.norm.inv(ds.prior[p_i], c), np.nan) if pr_ok.any() else np.full((H5, W5), np.nan)
        valid = np.isfinite(obs) & np.isfinite(clim_map) & np.isfinite(prior)
        maps[pn[p_i]] = dict(obs=obs, prior=prior, climatology=clim_map, model=fmean[p_i], valid=valid)
        for r in regs:
            vm = valid if r == "all" else (valid & box)
            if vm.sum() < 30: continue
            for name, cand in (("prior (S2 only)", prior), ("climatology (S5P memory)", clim_map), ("model 5 d mean forecast", fmean[p_i])):
                srows.append(dict(region=r, pollutant=pn[p_i], map=name, **static_map_scores(obs, cand, vm, a.px_km)))
    sf = pd.DataFrame(srows)
    f2 = os.path.join(a.out, f"prior_hotspot_test_{a.split}.csv"); sf.to_csv(f2, index=False)
    print(f"\n=== TEST 2: static maps vs observed {a.split} mean map (top-10% hotspots) ===")
    print(sf[["region", "pollutant", "map", "pearson_r", "spearman", "top10_precision", "blob_F1", "blob_IoU", "centroid_err_km", "n_obs_blobs", "n_pred_blobs", "n_pixels"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------------------------------------------------------------- verdict lines (plain reading of the numbers)
    print("\n=== READING ===")
    al = df[(df.region == "all") & (df.variant != "intact")].groupby(["pollutant", "variant"]).dR2_vs_intact.mean()
    for p in pn:
        d = al.loc[p].sort_values()
        print(f"{p}: inputs ranked by reliance (R2 drop): " + ", ".join(f"{k} {v:+.3f}" for k, v in d.items()))
    for p in pn:
        s = sf[(sf.region == "all") & (sf.pollutant == p)].set_index("map")
        if len(s) == 3:
            print(f"{p}: hotspot map vs observed - prior(S2-only) rho {s.loc['prior (S2 only)', 'spearman']:.3f} F1 {s.loc['prior (S2 only)', 'blob_F1']:.3f} | "
                  f"climatology rho {s.loc['climatology (S5P memory)', 'spearman']:.3f} F1 {s.loc['climatology (S5P memory)', 'blob_F1']:.3f}")

    # ---------------------------------------------------------------- figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        sub = df[(df.region == "all") & (df.variant != "intact")]
        fig, ax = plt.subplots(1, P, figsize=(6 * P, 4), squeeze=False)
        for p_i, p in enumerate(pn):
            for v in [v for v in variants if v != "intact"]:
                g = sub[(sub.pollutant == p) & (sub.variant == v)].sort_values("horizon_days")
                ax[0, p_i].plot(g.horizon_days, g.dR2_vs_intact, "o-", label=v)
            ax[0, p_i].axhline(0, c="k", lw=.8); ax[0, p_i].set_xscale("log"); ax[0, p_i].set_xticks(horizons); ax[0, p_i].set_xticklabels(horizons)
            ax[0, p_i].set_title(f"{p}: R2 change when input destroyed"); ax[0, p_i].set_xlabel("lead (days)")
        ax[0, 0].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(a.out, f"attribution_{a.split}.png"), dpi=110); plt.close(fig)

        fig, ax = plt.subplots(P, 4, figsize=(16, 4.4 * P), squeeze=False)
        for p_i, p in enumerate(pn):
            mp = maps[p]; vm = mp["valid"]; ro = rank_map(np.nan_to_num(mp["obs"]), vm)
            for j, key in enumerate(("obs", "prior", "climatology", "model")):
                x = np.where(vm, mp[key], np.nan)
                ax[p_i, j].imshow(x, cmap="magma", vmin=np.nanpercentile(x, 2), vmax=np.nanpercentile(x, 98))
                ax[p_i, j].contour(((ro >= 0.9) & vm).astype(float), levels=[.5], colors="cyan", linewidths=.7)
                if key != "obs":
                    rc = rank_map(np.nan_to_num(mp[key]), vm); ax[p_i, j].contour(((rc >= 0.9) & vm).astype(float), levels=[.5], colors="lime", linewidths=.7)
                if RS is not None:
                    ax[p_i, j].add_patch(plt.Rectangle((RS[1].start - .5, RS[0].start - .5), RS[1].stop - RS[1].start, RS[0].stop - RS[0].start, fill=False, ec="white", lw=1, ls="--"))
                title = {"obs": f"{p} observed {a.split} mean", "prior": "S2-only prior", "climatology": "training climatology", "model": "model 5 d mean forecast"}[key]
                ax[p_i, j].set_title(title); ax[p_i, j].axis("off")
        fig.suptitle("cyan = observed top-10% hotspots, green = map's own top-10%, dashed box = Kongu region", y=1.0)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"prior_hotspot_maps_{a.split}.png"), dpi=100); plt.close(fig)
        print(f"figures saved in {a.out}")
    except Exception as e:
        print("figure step skipped:", e)
    print(f"saved {f1}\nsaved {f2}")


if __name__ == "__main__":
    main()
