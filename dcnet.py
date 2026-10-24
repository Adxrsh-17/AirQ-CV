"""DC-Net v2: land-use-prior temporal forecaster (Rule.md section 7, Stage B).

Forecast in z-space (arcsinh-normalised S5P) at horizons 5..120 days:

    q50(x, t+h) = clim(x, doy(t+h))                          fixed harmonic climatology (training years)
                + alpha_h * decay(age) * last_anomaly(x)     damped anomaly persistence (learned per horizon)
                + Delta_h(x)                                 network residual (zero-initialised)

so training starts exactly at "climatology + damped persistence" and the network only has to learn what
the baselines miss. Components and why they are here:

  * LandUseEncoder   native-resolution S2 (annual + seasonal bands, NDBI/NDVI/MNDWI/NDRE/BSI, change),
                     within-cell index statistics and built/veg/water fractions (Stage A's best features),
                     Stage A out-of-fold prior map, and a context pyramid at ~16/38/80 km (Stage A found the
                     land-use signal lives at 16-38 km neighbourhood scale, not in the own cell).
  * Dynamics encoder mask-aware partial-conv stem per window -> SimVP-style time-as-channels encoder
                     (parallel, no recurrence) + 12-month anomaly memory.
  * TAU translator   Temporal Attention Unit blocks (Tan et al., CVPR 2023): large-kernel depthwise
                     "statical" attention x channel "dynamical" attention, FiLM-conditioned on calendar and
                     weather history (origin and before only).
  * Per-horizon decoder  shared U-Net decoder run once per lead time with MetNet-style lead-time FiLM, and
                     land-use-adaptive normalisation (SPADE-style): the land-use map sets how each pixel's
                     anomaly is scaled/shifted, i.e. "same weather, different response over a city vs a forest".
  * Heads            non-crossing quantiles q10/q50/q90; SO2 occurrence logit.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

HORIZONS_DAYS = (5, 10, 15, 30, 60, 120)              # Rule.md 7.2 (no next-day: 5-day composites)
HORIZON_STEPS = tuple(d // 5 for d in HORIZONS_DAYS)   # 1, 2, 3, 6, 12, 24
S2_IDX_CH = slice(12, 17)    # annual NDBI, NDVI, MNDWI, NDRE, BSI (dataset.S2_ANN_IDX)
S2_ANN_OK_CH = 34


def gn(c, g=8): return nn.GroupNorm(min(g, c), c)


class ResBlock(nn.Module):
    def __init__(s, cin, cout, p=0.0):
        super().__init__()
        s.n1, s.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        s.n2, s.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        s.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        s.drop = nn.Dropout2d(p)
    def forward(s, x):
        h = s.c1(F.silu(s.n1(x))); h = s.c2(s.drop(F.silu(s.n2(h))))
        return h + s.skip(x)


class PartialStem(nn.Module):
    """Mask-aware first conv: invalid (cloudy / no retrieval) pixels never leak in as zeros."""
    def __init__(s, c_in, cout):
        super().__init__()
        s.conv = nn.Conv2d(c_in, cout, 3, padding=1, bias=False)
        s.bias = nn.Parameter(torch.zeros(1, cout, 1, 1))
        s.register_buffer("ones", torch.ones(1, 1, 3, 3))
        s.mix = nn.Conv2d(cout + c_in, cout, 1)
    def forward(s, v, m):
        y = s.conv(v * m)
        msum = F.conv2d(m.mean(1, keepdim=True), s.ones, padding=1)
        y = y * (9.0 / msum.clamp(min=1.0)) * (msum > 0) + s.bias
        return F.silu(s.mix(torch.cat([y, m], 1)))


class FiLM(nn.Module):
    def __init__(s, cond, ch):
        super().__init__(); s.fc = nn.Linear(cond, 2 * ch); nn.init.zeros_(s.fc.weight); nn.init.zeros_(s.fc.bias)
    def forward(s, x, c):
        g, b = s.fc(c).chunk(2, 1)
        return x * (1 + g[..., None, None]) + b[..., None, None]


class TAU(nn.Module):
    """Temporal Attention Unit (SimVP v2 / TAU): statical (large-kernel depthwise) x dynamical (channel) attention."""
    def __init__(s, c, p=0.1):
        super().__init__()
        s.n1, s.n2 = gn(c), gn(c)
        s.sa = nn.Sequential(nn.Conv2d(c, c, 5, padding=2, groups=c),
                             nn.Conv2d(c, c, 7, padding=9, dilation=3, groups=c),
                             nn.Conv2d(c, c, 1))
        s.da = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(8, c // 4), 1), nn.SiLU(), nn.Conv2d(max(8, c // 4), c, 1))
        s.proj = nn.Conv2d(c, c, 1)
        s.mlp = nn.Sequential(nn.Conv2d(c, 2 * c, 1), nn.GELU(), nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c),
                              nn.Dropout(p), nn.Conv2d(2 * c, c, 1))
    def forward(s, x):
        h = s.n1(x)
        x = x + s.proj(h * s.sa(h) * torch.sigmoid(s.da(h)))
        return x + s.mlp(s.n2(x))


class ContextPyramid(nn.Module):
    """Neighbourhood land-use context at several scales (stride-1 average pools, ~16/38/80 km on a 5.5 km grid)."""
    def __init__(s, c, scales=(3, 7, 15)):
        super().__init__(); s.scales = scales
        s.proj = nn.Sequential(nn.Conv2d(c * (1 + len(scales)), c, 1), nn.SiLU())
    def forward(s, x):
        f = [x] + [F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False) for k in s.scales]
        return s.proj(torch.cat(f, 1))


class LUMod(nn.Module):
    """Land-use-adaptive normalisation (SPADE-style): x -> GN(x) * (1 + gamma(u)) + beta(u)."""
    def __init__(s, c, cu):
        super().__init__()
        s.cu = cu
        s.norm = nn.GroupNorm(min(8, c), c, affine=cu == 0)
        if cu:
            s.sh = nn.Sequential(nn.Conv2d(cu, 64, 3, padding=1), nn.SiLU())
            s.g, s.b = nn.Conv2d(64, c, 3, padding=1), nn.Conv2d(64, c, 3, padding=1)
            for m in (s.g, s.b): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
    def forward(s, x, u):
        x = s.norm(x)
        if not s.cu or u is None: return x
        a = s.sh(F.interpolate(u, size=x.shape[-2:], mode="bilinear", align_corners=False) if u.shape[-2:] != x.shape[-2:] else u)
        return x * (1 + s.g(a)) + s.b(a)


class LandUseEncoder(nn.Module):
    def __init__(s, c_s2, c_prior, ratio=8, cu=96, p=0.05):
        super().__init__()
        s.c_s2, s.c_prior, s.ratio = c_s2, c_prior, ratio
        c_coarse = c_prior
        if c_s2:
            chans = [24, 32, 48, 64]
            s.stem = nn.Sequential(nn.Conv2d(c_s2, chans[0], 3, padding=1), ResBlock(chans[0], chans[0], p))
            n_down = max(0, int(round(math.log2(ratio))))
            downs, c = [], chans[0]
            for i in range(n_down):
                co = chans[min(i + 1, 3)]
                downs += [nn.Conv2d(c, co, 3, 2, 1), ResBlock(co, co, p)]; c = co
            s.down = nn.Sequential(*downs)
            c_coarse += c + 2 * 5 + 3      # learned + index mean/std + built/veg/water fractions
        s.mix = nn.Sequential(nn.Conv2d(c_coarse, cu, 3, padding=1), ResBlock(cu, cu, p))
        s.ctx = ContextPyramid(cu)

    def cell_stats(s, x, hw):
        """Stage A features computed on the fly: within-cell mean/std of indices + land-cover fractions."""
        r = s.ratio
        ok = x[:, S2_ANN_OK_CH:S2_ANN_OK_CH + 1]
        idx = x[:, S2_IDX_CH]
        pool = lambda t: F.adaptive_avg_pool2d(t, hw) if r < 1 or (t.shape[-2] // hw[0]) != r else F.avg_pool2d(t, r)
        w = pool(ok).clamp(min=1e-3)
        mu = pool(idx * ok) / w
        sd = (pool(idx ** 2 * ok) / w - mu ** 2).clamp(min=0).sqrt()
        ndbi, ndvi, mndwi = idx[:, 0:1], idx[:, 1:2], idx[:, 2:3]
        fr = torch.cat([((ndbi > 0) & (ndvi < 0.2)).float(), (ndvi > 0.4).float(), (mndwi > 0).float()], 1)
        frac = pool(fr * ok) / w
        return torch.cat([mu, sd, frac], 1)

    def forward(s, s2, prior, hw):
        feats = []
        if s.c_s2:
            f = s.down(s.stem(s2))
            if f.shape[-2:] != hw: f = F.adaptive_avg_pool2d(f, hw)
            feats += [f, s.cell_stats(s2, hw)]
        if s.c_prior: feats.append(prior)
        return s.ctx(s.mix(torch.cat(feats, 1)))


class DCNet(nn.Module):
    def __init__(s, n_poll=1, horizons=HORIZONS_DAYS, L=12, n_wx=0, s2_ch=38, s2_ratio=8,
                 use_s2=True, use_prior=True, use_weather=True, dim=160, n_tau=4, p=0.1, cu=96):
        super().__init__()
        P, Q = n_poll, len(horizons)
        s.P, s.Q, s.L, s.horizons = P, Q, L, tuple(horizons)
        s.use_s2, s.use_prior, s.use_weather = bool(use_s2), bool(use_prior), bool(use_weather and n_wx > 0)
        s.n_wx = n_wx
        # dynamics
        s.stem = PartialStem(P, 16)
        c_dyn = L * 16 + 12 * P * 2 + 3 * P
        s.enc0 = nn.Sequential(nn.Conv2d(c_dyn, 64, 3, padding=1), ResBlock(64, 64, p))
        s.down1, s.enc1 = nn.Conv2d(64, 96, 3, 2, 1), ResBlock(96, 96, p)
        s.down2, s.enc2 = nn.Conv2d(96, 128, 3, 2, 1), ResBlock(128, 128, p)
        # land use
        s.has_lu = s.use_s2 or s.use_prior
        cu = cu if s.has_lu else 0
        s.cu = cu
        if s.has_lu:
            s.lu = LandUseEncoder(s2_ch if s.use_s2 else 0, (P + 1) if s.use_prior else 0, s2_ratio, cu)
        # conditioning: calendar + weather history (origin and before only)
        n_cond = 2 + (L * n_wx if s.use_weather else 0)
        s.cond = nn.Sequential(nn.Linear(n_cond, 64), nn.SiLU(), nn.Linear(64, 64))
        s.hcond = nn.Sequential(nn.Linear(64 + 3 + Q, 64), nn.SiLU(), nn.Linear(64, 64))
        # translator
        s.tin = nn.Conv2d(128 + cu, dim, 1)
        s.film_t = FiLM(64, dim)
        s.tau = nn.ModuleList([TAU(dim, p) for _ in range(n_tau)])
        # per-horizon decoder
        s.film_z = FiLM(64, dim)
        s.up2, s.mod2, s.dec2, s.film2 = nn.Conv2d(dim, 96, 1), LUMod(192, cu), ResBlock(192, 96, p), FiLM(64, 96)
        s.up1, s.mod1, s.dec1, s.film1 = nn.Conv2d(96, 64, 1), LUMod(128, cu), ResBlock(128, 64, p), FiLM(64, 64)
        s.head = nn.Conv2d(64, 3 * P, 1)
        with torch.no_grad():
            s.head.weight.zero_(); s.head.bias.zero_()
        s.occ = nn.Conv2d(64, 1, 1)
        # damped anomaly persistence: alpha_h init exp(-h/15 d), learned per horizon x pollutant
        h = torch.tensor(horizons, dtype=torch.float32)
        a0 = torch.exp(-h / 15.0).clamp(1e-3, 0.999)
        s.alpha_raw = nn.Parameter(torch.log(a0 / (1 - a0))[:, None].repeat(1, P).contiguous())
        s.age_k = nn.Parameter(torch.zeros(P))
        s.register_buffer("h_feat", torch.log(h / 5.0)[:, None] / math.log(24.0))
        s.register_buffer("h_onehot", torch.eye(Q))

    def forward(s, b):
        fa, fm = b["fast_a"], b["fast_m"]                                   # (B, L, P, H, W)
        B, L, P, H, W = fa.shape
        x = s.stem(fa.flatten(0, 1), fm.flatten(0, 1)).view(B, L * 16, H, W)
        dyn = torch.cat([x, b["long_a"].flatten(1, 2), b["long_m"].flatten(1, 2), b["last_a"], b["has"], b["age"]], 1)
        e0 = s.enc0(dyn); e1 = s.enc1(s.down1(e0)); e2 = s.enc2(s.down2(e1))

        u = s.lu(b.get("s2") if s.use_s2 else None, b.get("prior") if s.use_prior else None, (H, W)) if s.has_lu else None

        c_items = [b["doy_org_sc"]]
        if s.use_weather: c_items.append(b["wx"])
        c = s.cond(torch.cat(c_items, 1))                                   # (B, 64)

        z = s.tin(torch.cat([e2] + ([F.adaptive_avg_pool2d(u, e2.shape[-2:])] if u is not None else []), 1))
        z = s.film_t(z, c)
        for blk in s.tau: z = blk(z)

        Q = s.Q
        ang = 2 * math.pi * b["doy_tgt"] / 365.25                           # (B, Q)
        hf = torch.cat([s.h_feat[None].expand(B, Q, 1), torch.sin(ang)[..., None], torch.cos(ang)[..., None],
                        s.h_onehot[None].expand(B, Q, Q)], -1)
        hc = s.hcond(torch.cat([c[:, None].expand(B, Q, c.size(-1)), hf], -1)).flatten(0, 1)   # (B*Q, 64)
        rep = lambda t: t[:, None].expand(B, Q, *t.shape[1:]).flatten(0, 1)
        uq = rep(u) if u is not None else None

        zq = s.film_z(rep(z), hc)
        d2 = s.up2(F.interpolate(zq, size=e1.shape[-2:], mode="bilinear", align_corners=False))
        d2 = s.film2(s.dec2(s.mod2(torch.cat([d2, rep(e1)], 1), uq)), hc)
        d1 = s.up1(F.interpolate(d2, size=e0.shape[-2:], mode="bilinear", align_corners=False))
        d1 = s.film1(s.dec1(s.mod1(torch.cat([d1, rep(e0)], 1), uq)), hc)
        trunk = F.silu(d1)

        out = s.head(trunk).view(B, Q, P, 3, H, W)
        delta, wlo, whi = out[:, :, :, 0], out[:, :, :, 1], out[:, :, :, 2]
        alpha = torch.sigmoid(s.alpha_raw)[None, :, :, None, None]                       # (1, Q, P, 1, 1)
        decay = torch.exp(-F.softplus(s.age_k)[None, :, None, None] * b["age"])          # (B, P, H, W)
        pers = alpha * (b["last_a"] * b["has"] * decay)[:, None]
        base = b["clim_tgt"] + pers
        q50 = base + delta
        return {"q50": q50, "q10": q50 - F.softplus(wlo) - 1e-3, "q90": q50 + F.softplus(whi) + 1e-3,
                "base": base, "occ_logit": s.occ(trunk).view(B, Q, H, W)}


# ----------------------------------------------------------------------------- loss
def _per_h(num, den):
    """mean over horizons that have targets of (num/den) per horizon; num, den: (Q,)"""
    ok = den > 0
    return (num[ok] / den[ok]).mean() if ok.any() else num.sum() * 0.0


def dcnet_loss(out, b, so2_pos=None, lam_peak=0.5, lam_grad=0.05, lam_blob=0.2, lam_occ=0.2, tau_blob=0.1):
    """Quantile loss (each horizon weighted equally) + hotspot-weighted pinball + gradient (edge) loss
    + soft-Dice on hotspot exceedance (directly targets blob F1) + SO2 occurrence BCE."""
    y, ym = b["y"], b["ym"]
    thr = b["thr_tgt"][..., None, None]                                       # (B, Q, P, 1, 1)
    w = ym * (1 + lam_peak * torch.sigmoid((y - thr) / 0.25))
    red = (0, 2, 3, 4)
    den = w.sum(red)

    def pin(pred, tau):
        e = y - pred
        return _per_h((torch.maximum(tau * e, (tau - 1) * e) * w).sum(red), den)

    q50 = out["q50"]
    tot = pin(q50, 0.5) + 0.5 * pin(out["q10"], 0.1) + 0.5 * pin(out["q90"], 0.9)

    mx, my = ym[..., :, 1:] * ym[..., :, :-1], ym[..., 1:, :] * ym[..., :-1, :]
    gx = ((q50[..., :, 1:] - q50[..., :, :-1]) - (y[..., :, 1:] - y[..., :, :-1])).abs() * mx
    gy = ((q50[..., 1:, :] - q50[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])).abs() * my
    tot = tot + lam_grad * (_per_h(gx.sum(red), mx.sum(red)) + _per_h(gy.sum(red), my.sum(red)))

    if lam_blob > 0:
        sp = torch.sigmoid((q50 - thr) / tau_blob) * ym
        so = (y > thr).float() * ym
        inter, dsum = (sp * so).sum(red), sp.sum(red) + so.sum(red)
        ok = ym.sum(red) > 0
        dice = 1 - (2 * inter + 1) / (dsum + 1)
        tot = tot + lam_blob * (dice[ok].mean() if ok.any() else dice.sum() * 0)

    if so2_pos is not None and lam_occ > 0:
        mk = ym[:, :, so2_pos]
        bce = F.binary_cross_entropy_with_logits(out["occ_logit"], b["ev"], reduction="none") * mk
        tot = tot + lam_occ * bce.sum() / mk.sum().clamp(min=1)
    return tot


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
