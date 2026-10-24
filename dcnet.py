"""DC-Net: Dual-Clock S2+S5P multi-horizon forecaster (pure deep learning)."""
import math, torch, torch.nn as nn, torch.nn.functional as F

HORIZONS_DAYS = (5, 10, 20, 30, 60, 120, 150)      # 5-day steps: 1,2,4,6,12,24,30
HORIZON_STEPS = tuple(d // 5 for d in HORIZONS_DAYS)

def gn(c, g=8): return nn.GroupNorm(min(g, c), c)

class ResBlock(nn.Module):
    def __init__(s, cin, cout, p_drop=0.0):
        super().__init__()
        s.n1, s.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1, padding_mode="reflect")
        s.n2, s.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1, padding_mode="reflect")
        s.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        s.drop = nn.Dropout2d(p_drop)
    def forward(s, x):
        h = s.c1(F.silu(s.n1(x))); h = s.c2(s.drop(F.silu(s.n2(h))))
        return h + s.skip(x)

class ConvGRUCell(nn.Module):
    def __init__(s, cin, ch, k=3):
        super().__init__()
        s.ch = ch
        s.gates = nn.Conv2d(cin + ch, 2 * ch, k, padding=k // 2, padding_mode="reflect")
        s.cand = nn.Conv2d(cin + ch, ch, k, padding=k // 2, padding_mode="reflect")
    def forward(s, x, h):
        if h is None: h = x.new_zeros(x.size(0), s.ch, *x.shape[-2:])
        z, r = torch.sigmoid(s.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(s.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n

class PartialStem(nn.Module):
    """Mask-aware first conv (partial-conv style): invalid S5P pixels never leak as zeros."""
    def __init__(s, c_val, cout):
        super().__init__()
        s.conv = nn.Conv2d(c_val, cout, 3, padding=1, padding_mode="reflect", bias=False)
        s.bias = nn.Parameter(torch.zeros(1, cout, 1, 1))
        s.register_buffer("ones", torch.ones(1, 1, 3, 3))
        s.mix = nn.Conv2d(cout + c_val, cout, 1)
    def forward(s, v, m):
        vm = v * m
        y = s.conv(vm)
        msum = F.conv2d(m.mean(1, keepdim=True), s.ones, padding=1)
        y = y * (9.0 / msum.clamp(min=1.0)) * (msum > 0) + s.bias
        return F.silu(s.mix(torch.cat([y, m], 1)))

class FiLM(nn.Module):
    def __init__(s, cond, ch):
        super().__init__(); s.fc = nn.Linear(cond, 2 * ch); nn.init.zeros_(s.fc.weight); nn.init.zeros_(s.fc.bias)
    def forward(s, x, c):
        g, b = s.fc(c).chunk(2, 1)
        return x * (1 + g[..., None, None]) + b[..., None, None]

class RelPosAttnBlock(nn.Module):
    def __init__(s, dim, heads=6, p=0.1):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        s.attn = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(p), nn.Linear(2 * dim, dim))
        s.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, padding_mode="reflect")
    def forward(s, x):
        B, C, H, W = x.shape
        x = x + s.dw(x)
        t = x.flatten(2).transpose(1, 2)
        h = s.n1(t); t = t + s.attn(h, h, h, need_weights=False)[0]
        t = t + s.mlp(s.n2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)

class FastEncoder(nn.Module):
    def __init__(s, c_in=3, ch=(32, 64, 128, 192), gru=(64, 128, 128), p=0.05):
        super().__init__()
        s.stem = PartialStem(c_in, ch[0])
        s.e1 = ResBlock(ch[0], ch[0], p)
        s.d1, s.e2 = nn.Conv2d(ch[0], ch[1], 3, 2, 1), ResBlock(ch[1], ch[1], p)
        s.d2, s.e3 = nn.Conv2d(ch[1], ch[2], 3, 2, 1), ResBlock(ch[2], ch[2], p)
        s.d3, s.e4 = nn.Conv2d(ch[2], ch[3], 3, 2, 1), ResBlock(ch[3], ch[3], p)
        s.g2, s.g3, s.g4 = ConvGRUCell(ch[1], gru[0]), ConvGRUCell(ch[2], gru[1]), ConvGRUCell(ch[3], gru[2])
        s.last1 = nn.Conv2d(ch[0], ch[0], 1)
    def forward(s, v, m):
        B, L = v.shape[:2]
        h2 = h3 = h4 = None
        for t in range(L):
            x1 = s.e1(s.stem(v[:, t], m[:, t]))
            x2 = s.e2(s.d1(x1)); x3 = s.e3(s.d2(x2)); x4 = s.e4(s.d3(x3))
            h2, h3, h4 = s.g2(x2, h2), s.g3(x3, h3), s.g4(x4, h4)
        return s.last1(x1), h2, h3, h4

class S2SlowEncoder(nn.Module):
    def __init__(s, c_in=32, ch=(24, 32, 48, 64), p=0.05):
        super().__init__()
        s.stem = nn.Conv2d(c_in, ch[0], 3, padding=1, padding_mode="reflect")
        s.b1 = ResBlock(ch[0], ch[0], p)
        s.d1, s.b2 = nn.Conv2d(ch[0], ch[1], 3, 2, 1), ResBlock(ch[1], ch[1], p)
        s.d2, s.b4 = nn.Conv2d(ch[1], ch[2], 3, 2, 1), ResBlock(ch[2], ch[2], p)
        s.d3, s.b8 = nn.Conv2d(ch[2], ch[3], 3, 2, 1), ResBlock(ch[3], ch[3], p)
    def forward(s, x):
        # Adaptive handling for both 32 biophysical stacked channels and 26 legacy channels
        if x.size(1) != s.stem.in_channels:
            if x.size(1) < s.stem.in_channels:
                x = F.pad(x, (0, 0, 0, 0, 0, s.stem.in_channels - x.size(1)))
            else:
                x = x[:, :s.stem.in_channels]
        f1 = s.b1(F.silu(s.stem(x)))
        f2 = s.b2(s.d1(f1)); f4 = s.b4(s.d2(f2)); f8 = s.b8(s.d3(f4))
        return f1, f2, f4, f8

class FineDisaggregationHead(nn.Module):
    """
    Rank 1 Physics-Guided Fine Disaggregation Head:
    Fuses high-res S2 biophysical features (NDBI/NDVI/SWIR) with coarse regional latent state
    to predict fine-scale emission modulation at native spatial resolution.
    """
    def __init__(s, c_trunk=64, c_f1=24, n_poll=1, p=0.05):
        super().__init__()
        s.fuse = nn.Sequential(
            nn.Conv2d(c_trunk + c_f1, 48, 3, padding=1, padding_mode="reflect"),
            nn.SiLU(),
            ResBlock(48, 48, p),
            nn.Conv2d(48, n_poll, 1)
        )
        nn.init.zeros_(s.fuse[-1].weight)
        nn.init.zeros_(s.fuse[-1].bias)
    def forward(s, trunk, f1):
        trunk_up = F.interpolate(trunk, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        return s.fuse(torch.cat([trunk_up, f1], 1))

class LongMemoryEncoder(nn.Module):
    def __init__(s, c_in=72, ch=64, p=0.05):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(c_in, ch, 3, padding=1, padding_mode="reflect"), nn.SiLU(),
                              ResBlock(ch, ch, p), ResBlock(ch, ch, p))
    def forward(s, x): return s.net(x)

class AttnGate(nn.Module):
    def __init__(s, cx, cg, ci):
        super().__init__()
        s.wx, s.wg, s.psi = nn.Conv2d(cx, ci, 1), nn.Conv2d(cg, ci, 1), nn.Conv2d(ci, 1, 1)
    def forward(s, x, g):
        return x * torch.sigmoid(s.psi(F.silu(s.wx(x) + s.wg(g))))

class SeasonalPrior(nn.Module):
    def __init__(s, n_ch, H, W, K=3):
        super().__init__()
        s.K = K
        s.W = nn.Parameter(torch.zeros(n_ch, 1 + 2 * K, H, W))
        s.smooth = nn.Conv2d(n_ch * (1 + 2 * K), n_ch * (1 + 2 * K), 5, padding=2, groups=n_ch * (1 + 2 * K), bias=False)
        with torch.no_grad(): s.smooth.weight.fill_(1 / 25.)
    def basis(s, doy):
        k = torch.arange(1, s.K + 1, device=doy.device).float()
        ang = 2 * math.pi * doy[..., None] / 365.25 * k
        return torch.cat([torch.ones_like(doy)[..., None], ang.sin(), ang.cos()], -1)
    def forward(s, doy, crop, out_hw):
        Wm = s.W.flatten(0, 1)[None]
        Wm = s.smooth(Wm)[0].view_as(s.W)
        basis = s.basis(doy)
        B = doy.size(0)
        if torch.is_tensor(crop) and crop.dim() == 2 and crop.size(0) == B:
            slices = []
            for b in range(B):
                i0, j0, h, w = int(crop[b, 0]), int(crop[b, 1]), int(crop[b, 2]), int(crop[b, 3])
                sub = Wm[:, :, i0:i0 + h, j0:j0 + w]
                ph, pw = out_hw[0] - sub.shape[-2], out_hw[1] - sub.shape[-1]
                if ph > 0 or pw > 0: sub = F.pad(sub, (0, max(pw, 0), 0, max(ph, 0)), mode="replicate")
                slices.append(sub)
            Wm_b = torch.stack(slices, 0)
            return torch.einsum("bqk,bckhw->bqchw", basis, Wm_b)
        else:
            if torch.is_tensor(crop) and crop.dim() == 2:
                i0, j0, h, w = int(crop[0, 0]), int(crop[0, 1]), int(crop[0, 2]), int(crop[0, 3])
            elif torch.is_tensor(crop) and crop.dim() == 1:
                i0, j0, h, w = int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3])
            else:
                i0, j0, h, w = [int(v) for v in crop]
            sub = Wm[:, :, i0:i0 + h, j0:j0 + w]
            ph, pw = out_hw[0] - sub.shape[-2], out_hw[1] - sub.shape[-1]
            if ph > 0 or pw > 0: sub = F.pad(sub, (0, max(pw, 0), 0, max(ph, 0)), mode="replicate")
            return torch.einsum("bqk,ckhw->bqchw", basis, sub)

class MonotoneGate(nn.Module):
    def __init__(s, n_ch):
        super().__init__(); s.raw = nn.Parameter(torch.full((n_ch,), -0.5))
    def forward(s, h_days):
        return torch.exp(-F.softplus(s.raw)[None] * (h_days[:, None] / 150.0))

class DCNet(nn.Module):
    def __init__(s, n_poll=1, full_hw=(123, 95), horizons=HORIZONS_DAYS, fine_ratio=8,
                 dim=192, attn_layers=2, heads=6, p=0.1, use_B=True, use_s2=True, use_weather=True, so2_idx=2,
                 long_mem_dropout=0.0):
        super().__init__()
        s.n_poll, s.horizons, s.use_B, s.use_s2, s.use_weather, s.so2_idx = n_poll, horizons, use_B, use_s2, use_weather, so2_idx
        s.long_mem_dropout = long_mem_dropout
        s.Q = len(horizons)
        s.register_buffer("h_days", torch.tensor(horizons, dtype=torch.float32))
        s.fast = FastEncoder(c_in=n_poll)
        s.slow = S2SlowEncoder(c_in=32)
        s.fine_head = FineDisaggregationHead(c_trunk=64, c_f1=24, n_poll=n_poll, p=p)
        s.long = LongMemoryEncoder(c_in=12 * n_poll * 2)
        n_cond = 2 + 2 * s.Q + (8 if use_weather else 0)
        cond = 32
        s.cal = nn.Sequential(nn.Linear(n_cond, cond), nn.SiLU(), nn.Linear(cond, cond))
        s.fuse = nn.Conv2d(128 + 64 + 64, dim, 1)
        s.film_b = FiLM(cond, dim)
        s.attn = nn.ModuleList([RelPosAttnBlock(dim, heads, p) for _ in range(attn_layers)])
        s.up3, s.ag3, s.dec3 = nn.Conv2d(dim, 128, 1), AttnGate(128, 64, 64), ResBlock(128 + 128, 128, 0.05)
        s.up2, s.ag2, s.dec2 = nn.Conv2d(128, 64, 1), AttnGate(64, 64, 32), ResBlock(64 + 64, 64, 0.05)
        s.up1, s.ag1, s.dec1 = nn.Conv2d(64, 32, 1), AttnGate(32, 64, 32), ResBlock(32 + 32, 32, 0.05)
        s.film_d = FiLM(cond, 32)
        s.mix_native = nn.Conv2d(32 + 64 + 64, 64, 3, padding=1, padding_mode="reflect")
        s.head = nn.Conv2d(64, s.Q * n_poll * 3, 1)
        s.occ = nn.Conv2d(64, s.Q, 1)
        s.prior = SeasonalPrior(n_poll, *full_hw)
        s.gate = MonotoneGate(n_poll)
        s.Bhead = nn.Sequential(nn.Conv2d(64 + 64, 32, 3, padding=1, padding_mode="reflect"), nn.SiLU(),
                                nn.Conv2d(32, n_poll, 1))
        nn.init.zeros_(s.Bhead[-1].weight); nn.init.zeros_(s.Bhead[-1].bias)
        nn.init.zeros_(s.head.bias)

    def forward(s, batch):
        v, m = batch["fast_v"], batch["fast_m"]
        s2, lm = batch["s2_slow"], batch["long_mem"]
        crop, doy_tgt = batch["crop"], batch["doy_tgt"]
        B = v.size(0)

        # Spatial / Historical Dropout on long_mem to prevent S2 shortcut
        if s.training and s.long_mem_dropout > 0:
            drop_mask = (torch.rand(B, 1, 1, 1, device=lm.device) >= s.long_mem_dropout).float()
            lm = lm * drop_mask

        cal_items = [batch["doy_org_sc"], torch.sin(2*math.pi*doy_tgt/365.25), torch.cos(2*math.pi*doy_tgt/365.25)]
        if s.use_weather and "weather" in batch and batch["weather"] is not None:
            cal_items.append(batch["weather"])
        cal = s.cal(torch.cat(cal_items, 1))

        f1, h2, h3, h4 = s.fast(v, m)
        if s.use_s2: s1, s2f, s4f, s8f = s.slow(s2)
        else:
            H, W = v.shape[-2:]
            s8f = v.new_zeros(B, 64, H, W); s4f = v.new_zeros(B, 48, 2*H, 2*W); s2f = v.new_zeros(B, 32, 4*H, 4*W)
            s1 = v.new_zeros(B, 24, H, W)
        lmf = s.long(lm)
        s8_native = F.interpolate(s8f, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        s8_h2 = F.interpolate(s8f, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        s8_h3 = F.interpolate(s8f, size=h3.shape[-2:], mode="bilinear", align_corners=False)
        z = s.fuse(torch.cat([h4, F.adaptive_avg_pool2d(s8f, h4.shape[-2:]), F.adaptive_avg_pool2d(lmf, h4.shape[-2:])], 1))
        z = s.film_b(z, cal)
        for blk in s.attn: z = blk(z)
        d3 = s.up3(F.interpolate(z, size=h3.shape[-2:], mode="bilinear", align_corners=False))
        d3 = s.dec3(torch.cat([d3, s.ag3(h3, s8_h3)], 1))
        d2 = s.up2(F.interpolate(d3, size=h2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = s.dec2(torch.cat([d2, s.ag2(h2, s8_h2)], 1))
        d1 = s.up1(F.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = s.dec1(torch.cat([d1, s.ag1(f1, s8_native)], 1))
        d1 = s.film_d(d1, cal)
        trunk = F.silu(s.mix_native(torch.cat([d1, s8_native, lmf], 1)))
        out = s.head(trunk).view(B, s.Q, s.n_poll, 3, *trunk.shape[-2:])
        a, wlo, whi = out[:, :, :, 0], F.softplus(out[:, :, :, 1]) + 1e-3, F.softplus(out[:, :, :, 2]) + 1e-3
        prior = s.prior(doy_tgt, crop, trunk.shape[-2:])
        g = s.gate(s.h_days)[None, :, :, None, None]
        Bterm = s.Bhead(torch.cat([trunk, s8_native], 1))[:, None] if (s.use_B and s.use_s2) else 0.0
        bscale = (1 - g)
        med = prior + bscale * Bterm + g * a

        # Rank 1: High-Resolution Disaggregation & Fine-Scale Prediction
        fine_mod = s.fine_head(trunk, s1) if s.use_s2 else torch.zeros(B, s.n_poll, *s1.shape[-2:], device=med.device)
        med_up = F.interpolate(med.flatten(0, 1), size=s1.shape[-2:], mode="bilinear", align_corners=False).view(B, s.Q, s.n_poll, *s1.shape[-2:])
        fine_q50 = med_up + fine_mod[:, None]

        return {"q50": med, "q10": med - wlo, "q90": med + whi, "fine_q50": fine_q50, "occ_logit": s.occ(trunk), "B": Bterm}

def masked_pinball(pred, target, mask, tau):
    e = target - pred
    l = torch.maximum(tau * e, (tau - 1) * e) * mask
    return l.sum() / mask.sum().clamp(min=1.0)

def dcnet_loss(out, y, ymask, so2_event=None, so2_idx=2, lam_grad=0.1, lam_peak=0.5, lam_B=1e-3, lam_occ=0.2,
               lam_mass=0.1, peak_thr=None, hweights=None):
    tot = 0.0
    q50, q10, q90 = out["q50"], out["q10"], out["q90"]
    n_poll = y.size(2)
    w = 1.0
    if peak_thr is not None:
        peak_t = torch.as_tensor(peak_thr, dtype=y.dtype, device=y.device)
        if peak_t.numel() == n_poll:
            w = 1.0 + lam_peak * torch.sigmoid((y - peak_t[None, None, :, None, None]) / 0.25)
    m = ymask * w
    tot = tot + masked_pinball(q50, y, m, 0.5) + 0.5 * masked_pinball(q10, y, m, 0.1) + 0.5 * masked_pinball(q90, y, m, 0.9)
    gx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    gy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    mx, my = ymask[..., :, 1:] * ymask[..., :, :-1], ymask[..., 1:, :] * ymask[..., :-1, :]
    tot = tot + lam_grad * ((gx(q50) - gx(y)).abs() * mx).sum() / mx.sum().clamp(min=1) \
                + lam_grad * ((gy(q50) - gy(y)).abs() * my).sum() / my.sum().clamp(min=1)

    # Rank 1: Area-to-Point Mass Conservation Integral Loss (Late Pooling)
    if "fine_q50" in out and out["fine_q50"] is not None:
        fine_pred = out["fine_q50"]
        coarse_pool = F.adaptive_avg_pool2d(fine_pred.flatten(0, 1), y.shape[-2:]).view_as(y)
        mass_loss = ((coarse_pool - y).abs() * m).sum() / m.sum().clamp(min=1.0)
        tot = tot + lam_mass * mass_loss

    if so2_event is not None and so2_idx is not None and so2_idx < n_poll:
        mk = ymask[:, :, so2_idx]
        bce = F.binary_cross_entropy_with_logits(out["occ_logit"], so2_event, reduction="none") * mk
        tot = tot + lam_occ * bce.sum() / mk.sum().clamp(min=1)
    if torch.is_tensor(out["B"]): tot = tot + lam_B * out["B"].abs().mean()
    return tot

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
