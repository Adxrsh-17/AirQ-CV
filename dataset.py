"""dataset.py - Stage B data pipeline (Rule.md section 7).

S2 land-use prior + S5P history (+ weather up to the origin) -> future S5P composites at 5..120 days.

Everything the model sees is known at the forecast origin:
  * S5P history as ANOMALIES w.r.t. a per-pixel harmonic climatology fitted on the training years only
  * last valid anomaly per pixel and its age (for the damped-persistence path)
  * 12 months of monthly S5P anomalies (long memory)
  * S2 land use: previous-year annual composite + latest 3-month seasonal composite (bands + 5 indices),
    year-on-year NDBI/NDVI change, valid masks  (38 channels at native S2 resolution)
  * Stage A land-use prior map (spatial out-of-fold, target period ends at the training cutoff), optional
  * ERA5 history over the input window (never the target date: rule B4)
Targets are used only if the target window lies in the same split (rule B1).
"""
import json, os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12", "NDVI", "NDBI", "NDMI"]   # export channel order (GEE: 9 reflectances /1e4 + NDVI, NDBI, NDMI)
POLL_NAMES = ["NO2", "CO", "SO2"]
IDX_NAMES = ["NDBI", "NDVI", "MNDWI", "NDRE", "BSI"]
N_IDX = len(IDX_NAMES)
# channel layout of the S2 input (keep in sync with dcnet.LandUseEncoder)
S2_ANN, S2_ANN_IDX = slice(0, 12), slice(12, 17)          # annual bands, annual indices (ch 12 = annual NDBI)
S2_SEA, S2_SEA_IDX = slice(17, 29), slice(29, 34)         # seasonal bands, seasonal indices
S2_ANN_OK, S2_SEA_OK, S2_DNDBI, S2_DNDVI = 34, 35, 36, 37
S2_CHANNELS = 38
K_HARM = 3


# ----------------------------------------------------------------------------- normalisers
class S5PNorm:
    def fit(s, val, mask, idx):
        s.s, s.mu, s.sd = [], [], []
        for c in range(val.shape[1]):
            x = val[idx, c][mask[idx, c] > 0]
            x = x[np.isfinite(x)]
            sc = float(np.median(np.abs(x))) + 1e-12 if len(x) else 1.0
            z = np.arcsinh(x / sc) if len(x) else np.zeros(1)
            s.s.append(sc); s.mu.append(float(z.mean())); s.sd.append(float(z.std()) + 1e-6)
        return s
    def fwd(s, x, c): return (np.arcsinh(x / s.s[c]) - s.mu[c]) / s.sd[c]
    def inv(s, z, c): return np.sinh(z * s.sd[c] + s.mu[c]) * s.s[c]
    def state(s): return dict(s=s.s, mu=s.mu, sd=s.sd)
    @classmethod
    def from_state(cls, st):
        n = cls(); n.s, n.mu, n.sd = list(st["s"]), list(st["mu"]), list(st["sd"]); return n


class WeatherNorm:
    def fit(s, wx, idx):
        sub = wx[idx]; s.mu = sub.mean(0).tolist(); s.sd = (sub.std(0) + 1e-6).tolist(); return s
    def fwd(s, x): return (x - np.array(s.mu, np.float32)) / np.array(s.sd, np.float32)
    def state(s): return dict(mu=s.mu, sd=s.sd)
    @classmethod
    def from_state(cls, st):
        n = cls(); n.mu, n.sd = list(st["mu"]), list(st["sd"]); return n


# ----------------------------------------------------------------------------- helpers
def s2_indices(b):
    """b: (12, H, W) reflectance (any scale). Returns (5, H, W): NDBI, NDVI, MNDWI, NDRE, BSI."""
    b = np.nan_to_num(np.asarray(b, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    B2, B3, B4, B5, B8, B11 = (b[BANDS.index(k)] for k in ("B2", "B3", "B4", "B5", "B8", "B11"))
    nd = lambda x, y: np.clip((x - y) / (x + y + 1e-6), -1.0, 1.0)
    with np.errstate(all="ignore"):
        out = np.stack([nd(B11, B8),               # NDBI  built-up / impervious
                        nd(B8, B4),                # NDVI  vegetation
                        nd(B3, B11),               # MNDWI water
                        nd(B8, B5),                # NDRE  red-edge chlorophyll
                        nd(B11 + B4, B8 + B2)])    # BSI   bare soil (separates bare/arid land from built-up)
    return np.nan_to_num(out).astype(np.float32)


def harm_basis(doy, K=K_HARM):
    doy = np.asarray(doy, np.float64)[..., None]
    ang = 2 * np.pi * doy * np.arange(1, K + 1) / 365.25
    return np.concatenate([np.ones_like(doy), np.sin(ang), np.cos(ang)], -1).astype(np.float32)


def _box3(a, w):
    """count-weighted 3x3 mean of a (..., H, W) with weights w (H, W)."""
    H, W = w.shape
    pa = np.pad(a * w, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)]); pw = np.pad(w, 1)
    s = np.zeros_like(a, dtype=np.float64); sw = np.zeros((H, W))
    for di in range(3):
        for dj in range(3):
            s += pa[..., di:di + H, dj:dj + W]; sw += pw[di:di + H, dj:dj + W]
    return s / np.maximum(sw, 1e-9)


def fit_harmonic_clim(z, m, doy, ridge=2.0):
    """Per-pixel masked ridge fit of z on [1, sin/cos(k*doy)], k=1..K, using TRAINING windows only.
    z, m: (N, P, H, W). Returns beta (P, 2K+1, H, W). Seasonal coefficients are 3x3-smoothed
    (seasonal shape is regionally coherent); the intercept (spatial pattern, hotspots) is not smoothed."""
    X = harm_basis(doy).astype(np.float64)                       # (N, D)
    N, P, H, W = z.shape; D = X.shape[1]
    XX = (X[:, :, None] * X[:, None, :]).reshape(N, D * D)       # (N, D*D)
    R = np.eye(D) * ridge; R[0, 0] = 1e-6
    beta = np.zeros((P, D, H, W), np.float32)
    for p in range(P):
        mp = (m[:, p].reshape(N, -1) > 0).astype(np.float64)    # (N, S)
        zp = np.where(mp > 0, z[:, p].reshape(N, -1), 0.0)
        A = (XX.T @ mp).T.reshape(-1, D, D) + R[None]            # (S, D, D)
        rhs = X.T @ zp                                           # (D, S)
        b = np.linalg.solve(A, rhs.T[..., None])[..., 0]         # (S, D)
        cnt = mp.sum(0)
        good = cnt >= 3 * D
        if good.any(): b[~good] = np.where(cnt[~good, None] > 0, b[~good], b[good].mean(0))
        b = b.T.reshape(D, H, W)
        w = np.minimum(cnt.reshape(H, W), 50.0)
        b[1:] = _box3(b[1:], w)
        beta[p] = b
    return beta


def holdout_mask(H, W, block, frac, seed):
    """Spatial hold-out blocks (excluded from the training loss, reported separately)."""
    if frac <= 0: return np.zeros((H, W), np.float32)
    bi, bj = np.arange(H)[:, None] // block, np.arange(W)[None, :] // block
    nb_i, nb_j = bi.max() + 1, bj.max() + 1
    rng = np.random.default_rng(seed)
    sel = rng.random((nb_i, nb_j)) < frac
    return sel[bi, bj].astype(np.float32)


# ----------------------------------------------------------------------------- dataset
class DCDataset(Dataset):
    def __init__(s, cache, split, cfg, stats=None):
        s.c, s.cfg, s.split = cache, cfg, split
        st = stats or {}
        s.meta = json.load(open(os.path.join(cache, "meta.json")))
        ld = lambda n, mm=None: np.load(os.path.join(cache, n + ".npy"), mmap_mode=mm)
        s.val = np.nan_to_num(ld("s5p_val").astype(np.float32))
        s.msk = (ld("s5p_mask") > 0).astype(np.float32)
        s5m, s5ok = ld("s5p_month"), ld("s5p_month_ok")
        s.s2m, s.s2ok = ld("s2_month", "r"), ld("s2_month_ok", "r")
        s.ws = pd.to_datetime(s.meta["win_start"]); s.we = s.ws + pd.Timedelta(days=5)
        s.ms, s.me = pd.to_datetime(s.meta["month_start"]), pd.to_datetime(s.meta["month_end"])
        s.T, s.M, s.H5, s.W5 = s.meta["T"], s.meta["M"], s.meta["H5"], s.meta["W5"]
        s.s2_ratio = int(s.meta.get("s2_ratio", s.meta["H8"] // s.meta["H5"]))
        s.lo, s.hi = [pd.Timestamp(x) for x in cfg["splits"][split]]
        pi = cfg.get("poll_idx", [0]); s.poll_idx = [pi] if isinstance(pi, int) else list(pi)
        s.n_poll = len(s.poll_idx)
        s.L, s.steps = cfg["L"], list(cfg["steps"])
        s.origins = [i for i in range(s.T) if s.lo <= s.ws[i] <= s.hi and i >= cfg["min_origin_idx"]]
        tr_end = pd.Timestamp(cfg["splits"]["train"][1])
        tr_idx = [i for i in range(s.T) if s.we[i] <= tr_end + pd.Timedelta(days=1)]
        s.tr_idx = tr_idx

        # --- S5P normalisation (train only) and anomalies w.r.t. harmonic climatology (train only)
        s.norm = S5PNorm.from_state(st["norm"]) if "norm" in st else S5PNorm().fit(s.val, s.msk, tr_idx)
        s.zs5 = np.stack([s.norm.fwd(s.val[:, c], c) for c in range(s.val.shape[1])], 1).astype(np.float32) * s.msk
        s.doy_win = np.array([(w + pd.Timedelta(hours=60)).dayofyear for w in s.ws], np.float32)
        s.beta = np.asarray(st["clim_beta"], np.float32) if "clim_beta" in st else \
            fit_harmonic_clim(s.zs5[tr_idx], s.msk[tr_idx], s.doy_win[tr_idx], cfg.get("clim_ridge", 2.0))
        s.clim_win = np.einsum("td,pdhw->tphw", harm_basis(s.doy_win), s.beta).astype(np.float32)
        s.za = (s.zs5 - s.clim_win) * s.msk
        doy_m = np.array([(m + pd.Timedelta(days=14)).dayofyear for m in s.ms], np.float32)
        clim_m = np.einsum("md,pdhw->mphw", harm_basis(doy_m), s.beta).astype(np.float32)
        s5ok = (np.asarray(s5ok) > 0) & np.isfinite(np.asarray(s5m))
        zm = np.stack([s.norm.fwd(np.nan_to_num(np.asarray(s5m[:, c], np.float32)), c) for c in range(s5m.shape[1])], 1)
        s.lma = np.where(s5ok, zm - clim_m, 0.0).astype(np.float32)          # monthly anomalies
        s.lmk = s5ok.astype(np.float32)

        # --- hotspot thresholds: training-period p90 per pollutant and calendar month (z units)
        if "thr" in st: s.thr = np.asarray(st["thr"], np.float32)
        else:
            s.thr = np.zeros((12, s.zs5.shape[1]), np.float32)
            mon = np.array([w.month for w in s.ws])
            for c in range(s.zs5.shape[1]):
                allv = s.zs5[tr_idx, c][s.msk[tr_idx, c] > 0]
                for mth in range(1, 13):
                    ii = [i for i in tr_idx if mon[i] == mth]
                    v = s.zs5[ii, c][s.msk[ii, c] > 0] if ii else np.zeros(0)
                    s.thr[mth - 1, c] = np.quantile(v if len(v) > 100 else allv, 0.9) if len(allv) else 1.0
        if "so2_thr" in st: s.so2_thr = float(st["so2_thr"])
        else:
            v = s.zs5[tr_idx, 2][s.msk[tr_idx, 2] > 0] if s.zs5.shape[1] > 2 else np.zeros(0)
            s.so2_thr = float(np.quantile(v, cfg.get("so2_event_q", 0.85))) if len(v) else 0.0

        # --- S2: normalisation stats, annual composites (built once, cached on disk)
        s.s2_stats = st.get("s2") or s._fit_s2(tr_end)
        s._mu = np.array(s.s2_stats["mu"], np.float32)[:, None, None]
        s._sd = np.array(s.s2_stats["sd"], np.float32)[:, None, None]
        s._load_annual()

        # --- weather (state-mean ERA5 per 5-day window), history only
        wx_file = os.path.join(cache, "weather_5d.npy")
        s.has_weather = os.path.exists(wx_file) and cfg.get("use_weather", True)
        if s.has_weather:
            wx_raw = np.nan_to_num(np.load(wx_file).astype(np.float32))
            s.wx_norm = WeatherNorm.from_state(st["wx"]) if st.get("wx") else WeatherNorm().fit(wx_raw, tr_idx)
            s.wx_feat = s.wx_norm.fwd(wx_raw).astype(np.float32)
            s.n_wx = s.wx_feat.shape[1]
        else:
            s.wx_norm, s.wx_feat, s.n_wx = None, None, 0

        # --- Stage A land-use prior (static map per pollutant, z units) + availability channel
        if "prior" in st: s.prior = np.asarray(st["prior"], np.float32)
        else: s.prior = s._load_prior(cfg.get("prior_dir"))

        # --- spatial hold-out blocks (Config C vs B comparison on unseen places)
        s.hold = np.asarray(st["hold"], np.float32) if "hold" in st else \
            holdout_mask(s.H5, s.W5, cfg.get("holdout_block", 8), cfg.get("holdout_frac", 0.0), cfg.get("holdout_seed", 0))

        s.stats = dict(norm=s.norm.state(), s2=s.s2_stats, wx=s.wx_norm.state() if s.wx_norm else None,
                       clim_beta=s.beta, thr=s.thr, so2_thr=s.so2_thr, prior=s.prior, hold=s.hold)

    # ------------------------------------------------------------------ S2
    def _fit_s2(s, tr_end):
        mm = [m for m in range(s.M) if s.me[m] <= tr_end]
        mm = mm[::max(1, len(mm) // 6)]
        mu, sd = np.zeros(12), np.ones(12)
        acc = [[] for _ in range(12)]
        for m in mm:
            x = np.asarray(s.s2m[m, :, ::4, ::4], np.float32); ok = (np.asarray(s.s2ok[m, ::4, ::4]) > 0) & np.isfinite(x).all(0)
            for c in range(12): acc[c].append(x[c][ok])
        for c in range(12):
            v = np.concatenate(acc[c]) if acc[c] else np.zeros(1)
            mu[c], sd[c] = float(v.mean()), float(v.std()) + 1e-6
        return dict(mu=mu.tolist(), sd=sd.tolist())

    def _aux_dir(s):
        d = s.cfg.get("aux_dir") or s.c
        if not os.access(d, os.W_OK): d = os.path.join("/tmp", "airq_aux")
        os.makedirs(d, exist_ok=True)
        return d

    def _load_annual(s):
        """Mean of valid monthly S2 composites per calendar year -> s2_annual.npy (Y, 12, H8, W8) float16."""
        years = sorted({d.year for d in s.ms})
        s.years = years
        for d in (s.cfg.get("aux_dir"), s.c, os.path.join("/tmp", "airq_aux")):
            if d and os.path.exists(os.path.join(d, "s2_annual.npy")) and os.path.exists(os.path.join(d, "s2_annual_ok.npy")):
                a = np.load(os.path.join(d, "s2_annual.npy"), mmap_mode="r")
                if a.shape[0] == len(years):
                    s.s2a, s.s2a_ok = a, np.load(os.path.join(d, "s2_annual_ok.npy"), mmap_mode="r"); return
        d = s._aux_dir()
        C, H8, W8 = s.s2m.shape[1:]
        print(f"[dataset] building annual S2 composites for {years} -> {d} (one-off)", flush=True)
        a = np.lib.format.open_memmap(os.path.join(d, "s2_annual.tmp.npy"), "w+", np.float16, (len(years), C, H8, W8))
        ok = np.lib.format.open_memmap(os.path.join(d, "s2_annual_ok.tmp.npy"), "w+", np.uint8, (len(years), H8, W8))
        for yi, y in enumerate(years):
            sm = np.zeros((C, H8, W8), np.float32); cnt = np.zeros((H8, W8), np.float32)
            for m in [k for k in range(s.M) if s.ms[k].year == y]:
                x = np.asarray(s.s2m[m], np.float32)
                v = (np.asarray(s.s2ok[m]) > 0) & np.isfinite(x).all(0)
                sm += np.where(v[None], x, 0.0); cnt += v
            a[yi] = (sm / np.maximum(cnt, 1.0)[None]).astype(np.float16); ok[yi] = (cnt > 0)
        a.flush(); ok.flush(); del a, ok
        os.replace(os.path.join(d, "s2_annual.tmp.npy"), os.path.join(d, "s2_annual.npy"))
        os.replace(os.path.join(d, "s2_annual_ok.tmp.npy"), os.path.join(d, "s2_annual_ok.npy"))
        s.s2a = np.load(os.path.join(d, "s2_annual.npy"), mmap_mode="r")
        s.s2a_ok = np.load(os.path.join(d, "s2_annual_ok.npy"), mmap_mode="r")

    def _s2n(s, x): return np.nan_to_num((x - s._mu) / s._sd)

    def _nanmean_months(s, months, sl):
        a = np.stack([np.asarray(s.s2m[m][(slice(None),) + sl], np.float32) for m in months])
        ok = np.stack([np.asarray(s.s2ok[m][sl]) > 0 for m in months]) & np.isfinite(a).all(1)
        cnt = ok.sum(0)
        med = np.where(ok[:, None], a, 0.0).sum(0) / np.maximum(cnt[None], 1.0)
        return med.astype(np.float32), cnt > 0

    def slow(s, t_end, sl, hw):
        """38-channel S2 land-use input for an origin ending at t_end (only imagery available by then)."""
        out = np.zeros((S2_CHANNELS,) + hw, np.float32)
        lag = pd.Timedelta(days=s.cfg["s2_lag_days"])
        # annual composite of the previous calendar year (first year: that year itself - land state only, train split only)
        yi = int(np.clip(s.years.index((t_end - lag).year) - 1 if (t_end - lag).year in s.years else 0, 0, len(s.years) - 1))
        a = np.asarray(s.s2a[yi][(slice(None),) + sl], np.float32); aok = np.asarray(s.s2a_ok[yi][sl]) > 0
        ia = s2_indices(a)
        out[S2_ANN] = s._s2n(a) * aok; out[S2_ANN_IDX] = ia * aok; out[S2_ANN_OK] = aok
        if yi > 0:
            p = np.asarray(s.s2a[yi - 1][(slice(None),) + sl], np.float32); both = aok & (np.asarray(s.s2a_ok[yi - 1][sl]) > 0)
            ip = s2_indices(p)
            out[S2_DNDBI] = (ia[0] - ip[0]) * both; out[S2_DNDVI] = (ia[1] - ip[1]) * both
        # latest 3-month seasonal composite
        use = [m for m in range(s.M) if s.me[m] + lag <= t_end][-3:]
        if use:
            now, ok = s._nanmean_months(use, sl)
            out[S2_SEA] = s._s2n(now) * ok; out[S2_SEA_IDX] = s2_indices(now) * ok; out[S2_SEA_OK] = ok
        return out

    # ------------------------------------------------------------------ prior
    def _load_prior(s, prior_dir):
        pr = np.zeros((s.n_poll + 1, s.H5, s.W5), np.float32)
        if not prior_dir or not s.cfg.get("use_prior", True): return pr
        avail = np.zeros((s.H5, s.W5), bool); found = 0
        for p_i, c in enumerate(s.poll_idx):
            f = os.path.join(prior_dir, f"prior_{POLL_NAMES[c]}.npy")
            if not os.path.exists(f): print(f"[dataset] prior missing: {f} (channel left at 0)"); continue
            x = np.load(f).astype(np.float32)
            if x.shape != (s.H5, s.W5): print(f"[dataset] prior {f} shape {x.shape} != grid; skipped"); continue
            ok = np.isfinite(x)
            pr[p_i] = np.where(ok, s.norm.fwd(np.where(ok, x, 0.0), c), 0.0); avail |= ok; found += 1
        pr[-1] = avail
        print(f"[dataset] land-use prior: {found}/{s.n_poll} pollutants loaded from {prior_dir}, coverage {avail.mean():.2f}")
        return pr

    # ------------------------------------------------------------------ items
    def __len__(s): return s.cfg["samples_per_epoch"] if s.split == "train" else len(s.origins)

    def clim_at(s, doy, i0, h, j0, w):
        return np.einsum("d,pdhw->phw", harm_basis(doy), s.beta[s.poll_idx, :, i0:i0 + h, j0:j0 + w]).astype(np.float32)

    def __getitem__(s, k):
        cfg, rng = s.cfg, np.random
        o = s.origins[rng.randint(len(s.origins))] if s.split == "train" else s.origins[k]
        P = min(cfg["patch"], s.H5, s.W5)
        if s.split == "train":
            i0 = rng.randint(0, s.H5 - P + 1); j0 = rng.randint(0, s.W5 - P + 1); h = w = P
        else:
            i0 = j0 = 0; h, w = s.H5, s.W5
        Hp, Wp = (-(-h // 8)) * 8, (-(-w // 8)) * 8
        pad = lambda a, r=1: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, Hp * r - a.shape[-2]), (0, Wp * r - a.shape[-1])])
        pidx, L = s.poll_idx, s.L
        cs = (slice(i0, i0 + h), slice(j0, j0 + w))

        # fast clock: anomalies of the last L windows
        ids = np.arange(o - L + 1, o + 1); okid = (ids >= 0).astype(np.float32); ids = np.clip(ids, 0, None)
        fa = s.za[ids][:, pidx][(Ellipsis,) + cs]
        fm = s.msk[ids][:, pidx][(Ellipsis,) + cs] * okid[:, None, None, None]
        if s.split == "train" and cfg.get("mask_aug", 0.0) > 0:
            fm = fm * (rng.rand(*fm.shape) >= cfg["mask_aug"])
        fa = (fa * fm).astype(np.float32)
        t_idx = np.arange(L)[:, None, None, None]
        last_i = np.where(fm > 0, t_idx, -1).max(0)                                # (P, h, w)
        has = (last_i >= 0).astype(np.float32)
        last_a = np.take_along_axis(fa, np.maximum(last_i, 0)[None], 0)[0] * has
        age = np.where(has > 0, (L - 1 - last_i) / L, 1.0).astype(np.float32)

        # long memory: last 12 monthly anomalies completed by the origin
        t_end = s.we[o]
        use = [m for m in range(s.M) if s.me[m] <= t_end][-12:][::-1]
        la = np.zeros((12, s.n_poll, h, w), np.float32); lk = np.zeros_like(la)
        if use:
            la[:len(use)] = s.lma[use][:, pidx][(Ellipsis,) + cs]; lk[:len(use)] = s.lmk[use][:, pidx][(Ellipsis,) + cs]
        if s.split == "train" and rng.rand() < cfg.get("long_mem_dropout", 0.0):
            la[:] = 0.0; lk[:] = 0.0

        # land use (native S2 resolution)
        r = s.s2_ratio
        s2 = s.slow(t_end, (slice(i0 * r, (i0 + h) * r), slice(j0 * r, (j0 + w) * r)), (h * r, w * r))

        # targets
        Q = len(s.steps)
        y = np.zeros((Q, s.n_poll, h, w), np.float32); ym = np.zeros_like(y); clim_t = np.zeros_like(y)
        thr_t = np.zeros((Q, s.n_poll), np.float32); doy_t = np.zeros(Q, np.float32)
        for q, stp in enumerate(s.steps):
            t0 = s.ws[o] + pd.Timedelta(days=5 * stp)
            doy_t[q] = (t0 + pd.Timedelta(hours=60)).dayofyear
            clim_t[q] = s.clim_at(doy_t[q], i0, h, j0, w)
            thr_t[q] = s.thr[t0.month - 1, pidx]
            t = o + stp
            if t < s.T and s.lo <= s.ws[t] <= s.hi:
                y[q] = s.zs5[t][pidx][(Ellipsis,) + cs]; ym[q] = s.msk[t][pidx][(Ellipsis,) + cs]
        hold = s.hold[cs][None]
        if s.split == "train": ym = ym * (1.0 - hold[None])          # held-out blocks never enter the loss
        if 2 in pidx:
            sp = pidx.index(2); ev = ((y[:, sp] > s.so2_thr) & (ym[:, sp] > 0)).astype(np.float32)
        else:
            ev = np.zeros((Q, h, w), np.float32)

        d_org = t_end.dayofyear
        b = dict(fast_a=pad(fa), fast_m=pad(fm), long_a=pad(la), long_m=pad(lk), last_a=pad(last_a), has=pad(has),
                 age=pad(age), s2=pad(s2, r), prior=pad(s.prior[(slice(None),) + cs]), clim_tgt=pad(clim_t),
                 y=pad(y), ym=pad(ym), ev=pad(ev), hold=pad(hold), thr_tgt=thr_t, doy_tgt=doy_t,
                 doy_org_sc=np.array([np.sin(2 * np.pi * d_org / 365.25), np.cos(2 * np.pi * d_org / 365.25)], np.float32),
                 crop=np.array([i0, j0, h, w], np.int64), origin=np.array(o, np.int64))
        if s.has_weather:
            b["wx"] = (s.wx_feat[ids] * okid[:, None]).reshape(-1).astype(np.float32)
        for kk in ("fast_a", "fast_m", "long_a", "last_a", "s2", "prior", "clim_tgt", "y", "ym"):
            if not np.isfinite(b[kk]).all():
                raise ValueError(f"non-finite values in '{kk}' (origin={o}, crop={i0},{j0}) - check the cache")
        return {kk: torch.from_numpy(np.ascontiguousarray(v)) for kk, v in b.items()}


def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}
