import json, os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class S5PNorm:
    def fit(s, val, mask, idx):
        s.s, s.mu, s.sd = [], [], []
        for c in range(val.shape[1]):
            x = val[idx, c][mask[idx, c] > 0]
            sc = float(np.median(np.abs(x))) + 1e-12; z = np.arcsinh(x / sc)
            s.s.append(sc); s.mu.append(float(z.mean())); s.sd.append(float(z.std()) + 1e-6)
        return s
    def fwd(s, x, c): return (np.arcsinh(x / s.s[c]) - s.mu[c]) / s.sd[c]
    def inv(s, z, c): return np.sinh(z * s.sd[c] + s.mu[c]) * s.s[c]
    def state(s): return dict(s=s.s, mu=s.mu, sd=s.sd)

class WeatherNorm:
    def fit(s, wx, idx):
        sub = wx[idx]
        s.mu = sub.mean(0).tolist()
        s.sd = (sub.std(0) + 1e-6).tolist()
        return s
    def fwd(s, x):
        return (x - np.array(s.mu, np.float32)) / np.array(s.sd, np.float32)
    def state(s): return dict(mu=s.mu, sd=s.sd)

class DCDataset(Dataset):
    def __init__(s, cache, split, cfg, norm=None, s2_stats=None, wx_norm=None):
        s.c, s.cfg, s.split = cache, cfg, split
        s.meta = json.load(open(os.path.join(cache, "meta.json")))
        ld = lambda n, mm=None: np.load(os.path.join(cache, n + ".npy"), mmap_mode=mm)
        s.val, s.msk, s.s5m, s.s5ok = ld("s5p_val"), ld("s5p_mask"), ld("s5p_month"), ld("s5p_month_ok")
        s.s2m, s.s2ok = ld("s2_month", "r"), ld("s2_month_ok", "r")
        s.ws = pd.to_datetime(s.meta["win_start"]); s.we = s.ws + pd.Timedelta(days=5)
        s.ms, s.me = pd.to_datetime(s.meta["month_start"]), pd.to_datetime(s.meta["month_end"])
        s.T, s.M, s.H5, s.W5 = s.meta["T"], s.meta["M"], s.meta["H5"], s.meta["W5"]
        lo, hi = [pd.Timestamp(x) for x in cfg["splits"][split]]
        s.lo, s.hi = lo, hi
        poll_idx = cfg.get("poll_idx", [0])
        if isinstance(poll_idx, int): poll_idx = [poll_idx]
        s.poll_idx = poll_idx
        s.n_poll = len(poll_idx)
        s.origins = [i for i in range(s.T) if lo <= s.ws[i] <= hi and i >= cfg["min_origin_idx"]]
        tr_idx = [i for i in range(s.T) if s.ws[i] <= pd.Timestamp(cfg["splits"]["train"][1])]
        s.norm = norm or S5PNorm().fit(s.val, s.msk, tr_idx)
        s.zs5 = np.stack([s.norm.fwd(s.val[:, c], c) for c in range(s.val.shape[1])], 1).astype(np.float32) * s.msk
        s.s2_stats = s2_stats or s._fit_s2(tr_idx)
        z = [s.zs5[tr_idx][:, c][s.msk[tr_idx][:, c] > 0] for c in s.poll_idx]
        s.peak_thr = np.array([np.quantile(x, 0.9) if len(x) > 0 else 1.0 for x in z], np.float32)
        if 2 in s.poll_idx:
            so2_pos = s.poll_idx.index(2)
            s.so2_thr = float(np.quantile(z[so2_pos], cfg.get("so2_event_q", 0.85))) if len(z[so2_pos]) > 0 else 0.0
        else:
            s.so2_thr = 0.0

        # Weather conditioning
        wx_file = os.path.join(cache, "weather_5d.npy")
        s.has_weather = os.path.exists(wx_file)
        if s.has_weather:
            s.wx_raw = np.load(wx_file)
            s.wx_norm = wx_norm or WeatherNorm().fit(s.wx_raw, tr_idx)
            s.wx_feat = s.wx_norm.fwd(s.wx_raw)
        else:
            s.wx_feat = None
            s.wx_norm = None

    def _fit_s2(s, tr_idx):
        mm = [m for m in range(s.M) if s.me[m] <= pd.Timestamp(s.cfg["splits"]["train"][1])]
        x = np.asarray(s.s2m[mm[::max(1, len(mm)//6)]], np.float32); ok = np.asarray(s.s2ok[mm[::max(1, len(mm)//6)]]) > 0
        mu = [float(x[:, c][ok].mean()) for c in range(12)]; sd = [float(x[:, c][ok].std()) + 1e-6 for c in range(12)]
        return dict(mu=mu, sd=sd)

    def __len__(s): return s.cfg["samples_per_epoch"] if s.split == "train" else len(s.origins)

    def _s2_norm(s, x):
        return (x - np.array(s.s2_stats["mu"], np.float32)[:, None, None]) / np.array(s.s2_stats["sd"], np.float32)[:, None, None]

    def _nanmed(s, months, sl):
        a = np.stack([np.asarray(s.s2m[m][(slice(None),) + sl], np.float32) for m in months])
        ok = np.stack([np.asarray(s.s2ok[m][sl]) > 0 for m in months])
        a[~np.broadcast_to(ok[:, None], a.shape)] = np.nan
        with np.errstate(all="ignore"): med = np.nanmedian(a, 0)
        return med, np.isfinite(med[0])

    def slow(s, t_end, sl, hw8):
        lag = pd.Timedelta(days=s.cfg["s2_lag_days"])
        use = [m for m in range(s.M) if s.me[m] + lag <= t_end][-3:]
        out = np.zeros((26,) + hw8, np.float32)
        if not use: return out
        now, ok = s._nanmed(use, sl); out[:12] = np.nan_to_num(s._s2_norm(now), nan=0.0) * ok; out[24] = ok
        prev = [m - 12 for m in use]
        if min(prev) >= 0:
            pv, pok = s._nanmed(prev, sl); both = ok & pok
            out[12:24] = np.nan_to_num(s._s2_norm(now) - s._s2_norm(pv), nan=0.0) * both; out[25] = 1.0
        return out

    def __getitem__(s, k):
        cfg = s.cfg; rng = np.random
        o = s.origins[rng.randint(len(s.origins))] if s.split == "train" else s.origins[k]
        P = min(cfg["patch"], s.H5, s.W5)
        if s.split == "train":
            i0 = rng.randint(0, max(1, s.H5 - P + 1)) if s.H5 > P else 0
            j0 = rng.randint(0, max(1, s.W5 - P + 1)) if s.W5 > P else 0
            h = w = P
        else: i0 = j0 = 0; h, w = s.H5, s.W5
        Hp, Wp = (-(-h // 8)) * 8, (-(-w // 8)) * 8
        pad = lambda a, k=1: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, Hp * k - a.shape[-2]), (0, Wp * k - a.shape[-1])])
        L = cfg["L"]; ids = np.arange(o - L + 1, o + 1); okid = ids >= 0; ids = np.clip(ids, 0, None)
        fv = s.zs5[ids][:, s.poll_idx, i0:i0 + h, j0:j0 + w]
        fm = s.msk[ids][:, s.poll_idx, i0:i0 + h, j0:j0 + w].astype(np.float32) * okid[:, None, None, None]
        if s.split == "train" and cfg.get("mask_aug", 0.0) > 0:
            drop = (rng.rand(*fm.shape) < cfg["mask_aug"]); fm = fm * (~drop)
        fv = fv * fm
        t_end = s.we[o]
        use = [m for m in range(s.M) if s.me[m] <= t_end][-12:]
        lv = np.zeros((12, s.n_poll, h, w), np.float32); lk = np.zeros((12, s.n_poll, h, w), np.float32)
        for q, m in enumerate(use[::-1]):
            for p_i, c in enumerate(s.poll_idx):
                mk = s.s5ok[m][c, i0:i0 + h, j0:j0 + w].astype(np.float32)
                lv[q, p_i] = s.norm.fwd(s.s5m[m][c, i0:i0 + h, j0:j0 + w], c) * mk
                lk[q, p_i] = mk
        long_mem = np.concatenate([lv.reshape(12 * s.n_poll, h, w), lk.reshape(12 * s.n_poll, h, w)])

        # Spatial / Historical Dropout on long_mem to prevent S2 shortcut
        if s.split == "train" and cfg.get("long_mem_dropout", 0.0) > 0:
            if rng.rand() < cfg["long_mem_dropout"]:
                long_mem = np.zeros_like(long_mem)

        sl = (slice(i0 * 8, (i0 + h) * 8), slice(j0 * 8, (j0 + w) * 8))
        s2 = s.slow(t_end, sl, (h * 8, w * 8))
        steps = cfg["steps"]; y = np.zeros((len(steps), s.n_poll, h, w), np.float32); ym = np.zeros_like(y)
        for q, st in enumerate(steps):
            t = o + st
            if t < s.T and s.lo <= s.ws[t] <= s.hi:
                for p_i, c in enumerate(s.poll_idx):
                    y[q, p_i] = s.zs5[t][c, i0:i0 + h, j0:j0 + w]
                    ym[q, p_i] = s.msk[t][c, i0:i0 + h, j0:j0 + w]
        if 2 in s.poll_idx:
            so2_pos = s.poll_idx.index(2)
            ev = ((y[:, so2_pos] > s.so2_thr) & (ym[:, so2_pos] > 0)).astype(np.float32)
        else:
            ev = np.zeros((len(steps), h, w), np.float32)
        doy = lambda ts: ts.dayofyear + ts.hour / 24.0
        d_org = doy(s.we[o]); d_tgt = np.array([(s.we[o] + pd.Timedelta(days=5 * st)).dayofyear for st in steps], np.float32)
        b = dict(fast_v=pad(fv), fast_m=pad(fm), s2_slow=pad(s2, 8), long_mem=pad(long_mem), y=pad(y), ym=pad(ym), ev=pad(ev),
                 doy_tgt=d_tgt, doy_org_sc=np.array([np.sin(2 * np.pi * d_org / 365.25), np.cos(2 * np.pi * d_org / 365.25)], np.float32),
                 crop=np.array([i0, j0, h, w], dtype=np.int64))
        if s.has_weather and s.wx_feat is not None:
            b["weather"] = s.wx_feat[o].astype(np.float32)
        return {k: torch.from_numpy(np.asarray(v)) for k, v in b.items()}

def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in b[0]}
