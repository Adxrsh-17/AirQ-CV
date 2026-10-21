# DC-Net: Dual-Clock S2 + S5P Multi-Horizon Forecaster (pure deep learning)

**Project:** Satellite-Driven Multi-Pollutant Forecasting, Tamil Nadu (Team B6, Amrita School of AI).
**Rule:** deep learning only. No ML/TS models (no Ridge/XGBoost/ARIMA/Prophet). Climatology and persistence may be computed **only as scoring yardsticks** in evaluation, never as the model.

## 0. Instructions to the training agent (read first)

1. Install: `torch>=2.2 numpy pandas rasterio tqdm scikit-image scikit-learn(metrics only)`.
2. Put the notebook GeoTIFFs in `data2/s2_composites/s2_YYYYMMDD.tif` and `data2/s5p_composites/s5p_YYYYMMDD.tif` (S2: 12 bands, S5P: NO2, CO, SO2).
3. Save the five code files from section 12 into one folder.
4. `python make_cache.py --s2_dir data2/s2_composites --s5p_dir data2/s5p_composites --out cache --start 2019-01-01 --end 2024-12-31`
5. `python train.py --cache cache --out runs/main --workers 4` (single GPU, >= 8 GB).
6. Run the ablations in section 9. Report results per horizon and pollutant, never one pooled number.
7. Do **not** change splits, thresholds or lag rules after seeing validation or test scores.

**Verification status (be honest in reports):**

| Item | Status |
|---|---|
| `dcnet.py` forward/backward, shapes, params, quantile order, all-invalid-batch safety, full-domain inference 123x95 padded to 128x96 | tested on CPU |
| Mass-conserving fine head (block mean error 5e-7, outputs >= 0) | tested |
| `align_tn.py` (S2 -> exact 8x8 per S5P cell) | tested on synthetic rasters at your grid sizes |
| `make_cache.py` -> `dataset.py` -> `train.py` | ran end to end on a small synthetic cache (2 epochs, loss decreasing). **Never run on real data or GPU.** |
| GPU memory, real-data accuracy, SSIM/CSI/AUC code, Phase C Wald training, SSL pretraining | **not implemented or not measured** (specified below) |

## 1. Data facts and mandatory fixes

Your extraction notebook: AOI `[76.20, 8.05, 80.40, 13.55]`, 5-day median composites 2019-01-01..2024-12-31 (438 windows), S5P at 5000 m (about 123x95 px, 0.0449 deg), S2 at 600 m (12 channels: B2 B3 B4 B5 B6 B7 B8 B11 B12 + NDVI NDBI NDMI), S5P channels NO2, CO, SO2 (GEE OFFL L3, QA-filtered at ingestion: qa>0.5, NO2 0.75).

| # | Problem in notebook | Consequence | Fix in this pipeline |
|---|---|---|---|
| 1 | `unmask(0).max(0)` on S5P | missing pixels = zeros, negatives clamped; zeros indistinguishable from data | `make_cache.py` treats 0 as invalid (`--zero_is_invalid 1`). SO2 true clamped zeros are lost, biasing SO2 upward. **Preferred:** re-export (snippet below) |
| 2 | S2 tiers fill with +-10/+-20 day imagery | file for window t can contain up to 20 days of **future** imagery = leakage | S2 used only from months with `month_end + 20 d <= forecast origin` (`s2_lag_days=20`) |
| 3 | S5P/S2 ratio = 5000/600 = 8.32 (not an integer) | block averaging misaligned | resample S2 by area-average to exact `S5P_res/8` sub-grid (984x760 for 123x95) |
| 4 | Notebook Dataset: random S2 crop but full-domain S5P target, stale 11x11 shape | S2 and target not co-located | crops always taken from the same geographic window (`dataset.py`) |
| 5 | Sequences built from filtered "complete" rows | irregular time steps | regular 438-window axis; missing windows kept with mask=0 |
| 6 | Manifest had 301 rows / 22 incomplete when run | missing data | finish the run; incomplete windows stay as masked gaps |

Preferred S5P re-export (small, ~45 KB/file): keep QA filter, **no unmask, no clamp**, add a count band:
```python
col = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2').filterBounds(aoi).filterDate(d0, d1)
img = col.select(['tropospheric_NO2_column_number_density'],['NO2']).median()
cnt = col.select(['tropospheric_NO2_column_number_density'],['n']).count()   # repeat for CO, SO2
# export bands: NO2, CO, SO2 (masked where no data) + n_NO2, n_CO, n_SO2; then mask = n > 0, keep negatives
```
If re-exported, change the mask line in `make_cache.py` to use the count bands. **2025:** data now exists; re-run the notebook for 2025-01-01..2025-12-31 and use it as the clean final test (2024 was already used in earlier experiments).

## 2. Dual-clock definition (assumption: confirm with the team)

* **Fast clock (5 days):** S5P NO2/CO/SO2 + validity masks, last L=12 windows (60 days).
* **Slow clock (monthly + yearly):** S2 causal composite: median of the last 3 complete usable months (`S_now`, 12 ch), its year-over-year difference (`dS`, 12 ch), plus `valid` and `has_yoy` flags = **26 channels at 8x native resolution**.
* **Long memory:** last 12 monthly S5P means + validity = 72 channels at native resolution.
* **Calendar:** sin/cos of origin day-of-year, target day-of-year per horizon.

## 3. Strict alignment spec

* Canonical grid = S5P grid (H5=123, W5=95, EPSG:4326, origin at AOI NW corner). S5P cell (i,j) <-> S2 rows `8i..8i+7`, cols `8j..8j+7`.
* S2 reprojected with `Resampling.average`, valid-pixel weighted (all 9 reflectance bands == 0 means invalid/ocean fill). S2 stored fp16, monthly, about 1.3 GB for 72 months.
* Normalisation fitted on **train-period valid pixels only**: S5P `z=(asinh(x/s)-mu)/sd` (invertible), S2 per-channel mean/std.
* Training crops: 64x64 native cells (512x512 S2) from the same window. Inference: full domain padded to 128x96, then cropped back.
* Raw size note: 438 x 12 x 984 x 760 fp16 = 7.9 GB if every window were stored; the cache stores monthly composites (1.3 GB) instead, so the 11 GB raw set is only read once.

## 4. Architecture (DC-Net, 5.70 M parameters)

```
FAST (5-day)  S5P values+masks (B,12,3,H,W) -> PartialStem(mask-aware conv) -> shared ResUNet encoder
              (32,64,128,192 ch; strides 1,2,4,8) -> ConvGRU @1/2 (64), @1/4 (128), @1/8 (128), unrolled over 12 steps
SLOW (monthly) S2 26ch @8x -> ResNet encoder (24,32,48,64) -> feats @8x,4x,2x and 1x(native = S5P grid)
LONG MEMORY   72ch @native -> conv + 2 ResBlocks (64)
BOTTLENECK    concat(GRU@1/8, pooled S2, pooled long-mem) -> 1x1 -> 192 ch -> FiLM(calendar, horizons) -> 2x global attention
              (6 heads, relative/conv positional bias only: no absolute position, so no pixel-ID memorisation)
DECODER       3 up-stages with attention-gated skips (gate signal = S2 slow features), FiLM(calendar) -> trunk (64 ch)
HEADS         7 horizons x 3 pollutants x (anomaly a, width_lo, width_hi); SO2 occurrence logit per horizon
SEASONAL PRIOR  learned per-pixel harmonic climatology (K=3 -> 7 basis maps x 3 pollutants), smoothed 5x5, inside the network
STRUCTURAL SHIFT B  small head on [trunk, S2 native feats], last layer zero-init (null result = default), L1-penalised
FUSION        q50 = Prior(doy_target) + (1-g(h))*B + g(h)*a ;  q10 = q50 - softplus(w_lo) ;  q90 = q50 + softplus(w_hi)
              g_c(h) = exp(-softplus(raw_c) * h/150): in (0,1], monotone decreasing with lead (learned per pollutant)
```
Horizons (days -> 5-day steps): 5,10,20,30,60,120,150 -> 1,2,4,6,12,24,30. "Short-term 1-5 d" = step 1.

**Parameter table (measured, full_hw=(123,95)):**

| Module | Params |
|---|---|
| FastEncoder (stem + ResUNet + 3 ConvGRU) | 3,582,752 |
| S2SlowEncoder | 199,176 |
| LongMemoryEncoder | 189,760 |
| Bottleneck attention (2 blocks, dim 192) | 597,888 |
| SeasonalPrior (3 x 7 x 123 x 95 + smoothing) | 245,910 |
| B head | 36,995 |
| Output head (63 ch) | 4,095 |
| Decoder, gates, FiLM, fuse, occurrence head (remainder) | ~838,925 |
| **Total** | **5,695,501** |
| Optional GuidedSoftmaxDownscaler (Phase C) | 202,820 |

Estimated activation memory about 400 MB per 64x64 sample in fp32 (about half with AMP). Batch 8 + AMP should fit an 8 GB GPU; if OOM, use batch 4 and `L=8`. This is an estimate, not measured.

**Why these choices (from research):** decoupled ConvGRU ResUNet was your best NO2 short-term model; mask-aware (partial) convolution is established for TROPOMI NO2 gap handling (Lops et al.); FiLM conditioning follows GreenEarthNet-style satellite forecasting; Earthformer shows attention helps Earth-system forecasting but needs large data, so attention is kept tiny (192 tokens) at the 1/8-res bottleneck; the fine head uses the softmax constraint layer of Harder et al. (JMLR 2023) for exact mass conservation. Prithvi-EO-2.0 (30 m, 6 bands, 300M/600M params) does not match 600 m / 12 channels / 8 GB, so S2 encoder is trained from scratch.

## 5. Loss (all masked; zero-safe)

`L = sum_h,c [ pinball_0.5 + 0.5*pinball_0.1 + 0.5*pinball_0.9 (peak-weighted) ] + 0.1*(|grad_x| + |grad_y| error on q50) + 0.2*BCE(SO2 occurrence, y>85th pct) + 1e-3*|B|_1`

* Peak weight `w = 1 + 0.5*sigmoid((y - q90_train)/0.25)` counters plume blurring (your earlier peak recovery was 73%).
* Pinball weights the 80% interval; target empirical coverage about 0.80.
* No hand-set per-horizon weights (targets are already normalised per pollutant).
* Never fill missing with 0 as a value: masked everywhere.

## 6. Hyperparameters (defaults in `train.py`)

| Param | Value | Param | Value |
|---|---|---|---|
| Optimiser | AdamW, wd 0.05 (0 for norms/biases/prior) | LR | 3e-4 peak, 3-epoch warmup, cosine |
| Batch | 8 (patch 64) | Samples/epoch | 1000 random (origin, crop) |
| Epochs | 120, early stop patience 15 on val loss | Grad clip | 1.0 |
| EMA | 0.999 (evaluate and save EMA weights) | Precision | fp16 autocast (CUDA) |
| Dropout2d | 0.05-0.1 | Mask augmentation | 10% extra random S5P gaps |
| L (history) | 12 windows | S2 lag | 20 days |
| Seeds | 3 minimum (0,1,2), report mean +- std | Flips/rot90 | **off** (breaks wind/coast geometry); test as ablation only |

## 7. Splits and leakage rules (already enforced in `dataset.py`)

* Train 2019-01..2022-12, Val 2023, Test 2024 (replace by 2025 when exported). Origins are in-split; **a horizon target is used only if the target date is inside the same split** (150-day targets crossing a boundary are masked, not leaked).
* Scalers, thresholds (90th pct peak, 85th pct SO2 event) fit on train only.
* S2 causal rule above. No concurrent meteorology. ERA5/ENSO are **not** used (optional later, past-only).
* Spatial check (recommended): hold out 2 of 12 spatial blocks (about 40x32 cells, with 8-cell buffer) by zeroing their loss mask; report held-out-block metrics separately.
* Touch the test split once, after freezing everything.

## 8. Evaluation (implement in a separate `evaluate.py`; `train.py` logs only pinball, R2, coverage)

Per pollutant x horizon, on valid pixels, physical units after `norm.inv`: R2, Pearson r, SSIM (`skimage`, on 123x95 maps), CSI for events above the train 90th percentile, 80% interval coverage and width; SO2: ROC-AUC, PR-AUC on the occurrence head. Report **skill vs climatology** `1 - L_model/L_clim` and vs persistence (yardsticks only), because seasonality inflates raw R2.

Previous results to compare against (your Review-1 deck): NO2 short R2 0.651/r 0.810/SSIM 0.896/CSI 0.361; CO short R2 0.693/r 0.835/SSIM 0.719; SO2 short AUC 0.616; NO2 long (monthly climatology) R2 0.532; CO long 0.687; SO2 long ROC-AUC 0.634. Expect 150-day skill over climatology near zero; a positive result must be shown, not assumed.

## 9. Experiment plan

| Phase | Run | Purpose |
|---|---|---|
| A | `train.py` default (joint 3 pollutants) | main model |
| A1 | `--use_s2 0` | does S2 help? |
| A2 | `--use_B 0` | structural shift on/off (null result is acceptable) |
| A3 | remove year-over-year channels in `dataset.slow` | static vs change features |
| A4 | train 3 single-pollutant models (set n_poll=1) | **single vs pollutant-specific DNN** (your research gap) |
| A5 | enable flips/rot90 | augmentation ablation |
| A6 | ConvLSTM cell instead of ConvGRU; attention_layers=0 | temporal/attention ablation |
| B | optional SSL pretrain of `S2SlowEncoder`: masked-patch reconstruction on all S2 crops, 30 epochs, then fine-tune | uses the large S2 volume |
| C | **fine head (specification only, not implemented in training):** train `GuidedSoftmaxDownscaler` with the Wald protocol: coarsen native S5P by 2x and 4x, guidance = S2 pooled to native resolution, target = native S5P, loss = L1 (+ SSIM); then apply at 8x with S2 at 0.62 km | optional |

**Honest limit of Phase C:** no fine-resolution ground truth exists, so the sub-5 km redistribution cannot be verified beyond Wald scale-consistency, block-mean consistency (exact by construction) and sanity checks on known sources (Manali, Ennore, Tuticorin, Neyveli). Present it as an estimate.

## 10. Known risks

* About 290 training windows: overfitting is the main risk. Watch the train/val gap; use 3 seeds, EMA, early stopping.
* SO2 is noise dominated; expect weak scores. Do not tune on test.
* If NO2/CO collapse to climatology at long horizons, that is a valid finding.
* Output maps may be blurry; the gradient and peak terms help; compare cross-section peak recovery as in your slide 14.
* `python train.py ... --max_epochs 2` first, to confirm data loading on real files.

## 11. References

Harder et al., Hard-Constrained Deep Learning for Climate Downscaling, JMLR 24(365), 2023. Lops et al., Depthwise Partial Convolutional NN for TROPOMI NO2, Neural Comput. Appl. 35, 2023. Gao et al., Earthformer, NeurIPS 2022. Szwarcman et al., Prithvi-EO-2.0, arXiv:2412.02732. Google Earth Engine S5P OFFL L3 catalog (QA thresholds). Team's dual-clock proposal and Review decks.

## 12. Code

### `dcnet.py`
```python
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
        s.mix = nn.Conv2d(cout + c_val, cout, 1)       # re-injects the raw mask as features
    def forward(s, v, m):                               # v,m: (N,C,H,W); m in {0,1}
        vm = v * m
        y = s.conv(vm)
        msum = F.conv2d(m.mean(1, keepdim=True), s.ones, padding=1)         # valid-neighbour count
        y = y * (9.0 / msum.clamp(min=1.0)) * (msum > 0) + s.bias
        return F.silu(s.mix(torch.cat([y, m], 1)))

class FiLM(nn.Module):
    def __init__(s, cond, ch):
        super().__init__(); s.fc = nn.Linear(cond, 2 * ch); nn.init.zeros_(s.fc.weight); nn.init.zeros_(s.fc.bias)
    def forward(s, x, c):
        g, b = s.fc(c).chunk(2, 1)
        return x * (1 + g[..., None, None]) + b[..., None, None]

class RelPosAttnBlock(nn.Module):
    """Global self-attention on the 1/8-res grid; relative (not absolute) position -> no pixel-ID memorisation."""
    def __init__(s, dim, heads=6, p=0.1):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        s.attn = nn.MultiheadAttention(dim, heads, dropout=p, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(p), nn.Linear(2 * dim, dim))
        s.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, padding_mode="reflect")   # conv positional bias
    def forward(s, x):
        B, C, H, W = x.shape
        x = x + s.dw(x)
        t = x.flatten(2).transpose(1, 2)
        h = s.n1(t); t = t + s.attn(h, h, h, need_weights=False)[0]
        t = t + s.mlp(s.n2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)

class FastEncoder(nn.Module):
    """Decoupled spatial ResUNet encoder (shared over time) + ConvGRU at 1/2, 1/4, 1/8."""
    def __init__(s, c_in=3, ch=(32, 64, 128, 192), gru=(64, 128, 128), p=0.05):
        super().__init__()
        s.stem = PartialStem(c_in, ch[0])
        s.e1 = ResBlock(ch[0], ch[0], p)
        s.d1, s.e2 = nn.Conv2d(ch[0], ch[1], 3, 2, 1), ResBlock(ch[1], ch[1], p)
        s.d2, s.e3 = nn.Conv2d(ch[1], ch[2], 3, 2, 1), ResBlock(ch[2], ch[2], p)
        s.d3, s.e4 = nn.Conv2d(ch[2], ch[3], 3, 2, 1), ResBlock(ch[3], ch[3], p)
        s.g2, s.g3, s.g4 = ConvGRUCell(ch[1], gru[0]), ConvGRUCell(ch[2], gru[1]), ConvGRUCell(ch[3], gru[2])
        s.last1 = nn.Conv2d(ch[0], ch[0], 1)
    def forward(s, v, m):            # v,m: (B,L,C,H,W)
        B, L = v.shape[:2]
        h2 = h3 = h4 = None
        for t in range(L):
            x1 = s.e1(s.stem(v[:, t], m[:, t]))
            x2 = s.e2(s.d1(x1)); x3 = s.e3(s.d2(x2)); x4 = s.e4(s.d3(x3))
            h2, h3, h4 = s.g2(x2, h2), s.g3(x3, h3), s.g4(x4, h4)
        return s.last1(x1), h2, h3, h4        # last-frame native feats + GRU states at 1/2,1/4,1/8

class S2SlowEncoder(nn.Module):
    """Slow clock. Input at 8x native: [S_now(12), dS_yoy(12), valid(1), has_yoy(1)] = 26 ch."""
    def __init__(s, c_in=26, ch=(24, 32, 48, 64), p=0.05):
        super().__init__()
        s.stem = nn.Conv2d(c_in, ch[0], 3, padding=1, padding_mode="reflect")
        s.b1 = ResBlock(ch[0], ch[0], p)
        s.d1, s.b2 = nn.Conv2d(ch[0], ch[1], 3, 2, 1), ResBlock(ch[1], ch[1], p)
        s.d2, s.b4 = nn.Conv2d(ch[1], ch[2], 3, 2, 1), ResBlock(ch[2], ch[2], p)
        s.d3, s.b8 = nn.Conv2d(ch[2], ch[3], 3, 2, 1), ResBlock(ch[3], ch[3], p)
    def forward(s, x):
        f1 = s.b1(F.silu(s.stem(x)))
        f2 = s.b2(s.d1(f1)); f4 = s.b4(s.d2(f2)); f8 = s.b8(s.d3(f4))
        return f1, f2, f4, f8          # f8 is exactly native S5P resolution

class LongMemoryEncoder(nn.Module):
    """12 monthly S5P means (+validity) -> native-res context."""
    def __init__(s, c_in=72, ch=64, p=0.05):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(c_in, ch, 3, padding=1, padding_mode="reflect"), nn.SiLU(),
                              ResBlock(ch, ch, p), ResBlock(ch, ch, p))
    def forward(s, x): return s.net(x)

class AttnGate(nn.Module):
    """Attention-gated skip: gating signal g (S2 slow + decoder) modulates the skip x."""
    def __init__(s, cx, cg, ci):
        super().__init__()
        s.wx, s.wg, s.psi = nn.Conv2d(cx, ci, 1), nn.Conv2d(cg, ci, 1), nn.Conv2d(ci, 1, 1)
    def forward(s, x, g):
        return x * torch.sigmoid(s.psi(F.silu(s.wx(x) + s.wg(g))))

class SeasonalPrior(nn.Module):
    """Learned per-pixel harmonic climatology INSIDE the network: P(c,doy)=sum_k phi_k(doy) W[c,k,:,:]."""
    def __init__(s, n_ch, H, W, K=3):
        super().__init__()
        s.K = K
        s.W = nn.Parameter(torch.zeros(n_ch, 1 + 2 * K, H, W))
        s.smooth = nn.Conv2d(n_ch * (1 + 2 * K), n_ch * (1 + 2 * K), 5, padding=2, groups=n_ch * (1 + 2 * K), bias=False)
        with torch.no_grad():
            s.smooth.weight.fill_(1 / 25.)
    def basis(s, doy):                       # doy: (B,H') float
        k = torch.arange(1, s.K + 1, device=doy.device).float()
        ang = 2 * math.pi * doy[..., None] / 365.25 * k
        return torch.cat([torch.ones_like(doy)[..., None], ang.sin(), ang.cos()], -1)   # (B,Hh,1+2K)
    def forward(s, doy, crop, out_hw):       # crop=(i0,j0,h,w) in full-domain pixels; out_hw = (H,W) of trunk (may be padded)
        i0, j0, h, w = crop
        Wm = s.W.flatten(0, 1)[None]
        Wm = s.smooth(Wm)[0].view_as(s.W)[:, :, i0:i0 + h, j0:j0 + w]
        ph, pw = out_hw[0] - Wm.shape[-2], out_hw[1] - Wm.shape[-1]
        if ph > 0 or pw > 0: Wm = F.pad(Wm, (0, max(pw, 0), 0, max(ph, 0)), mode="replicate")
        return torch.einsum("bqk,ckhw->bqchw", s.basis(doy), Wm)

class MonotoneGate(nn.Module):
    """g_c(h)=exp(-lambda_c * h/150), lambda_c>0  => 0<g<=1 and monotone decreasing in lead."""
    def __init__(s, n_ch):
        super().__init__(); s.raw = nn.Parameter(torch.full((n_ch,), -0.5))
    def forward(s, h_days):                  # (Q,) -> (Q,C)
        return torch.exp(-F.softplus(s.raw)[None] * (h_days[:, None] / 150.0))

class DCNet(nn.Module):
    def __init__(s, n_poll=3, full_hw=(123, 95), horizons=HORIZONS_DAYS, fine_ratio=8,
                 dim=192, attn_layers=2, heads=6, p=0.1, use_B=True, use_s2=True, so2_idx=2):
        super().__init__()
        s.n_poll, s.horizons, s.use_B, s.use_s2, s.so2_idx = n_poll, horizons, use_B, use_s2, so2_idx
        s.Q = len(horizons)
        s.register_buffer("h_days", torch.tensor(horizons, dtype=torch.float32))
        s.fast = FastEncoder(c_in=n_poll)
        s.slow = S2SlowEncoder()
        s.long = LongMemoryEncoder(c_in=12 * n_poll * 2)
        cond = 32
        s.cal = nn.Sequential(nn.Linear(2 + 2 * s.Q, cond), nn.SiLU(), nn.Linear(cond, cond))
        s.fuse = nn.Conv2d(128 + 64 + 64, dim, 1)
        s.film_b = FiLM(cond, dim)
        s.attn = nn.ModuleList([RelPosAttnBlock(dim, heads, p) for _ in range(attn_layers)])
        # decoder (native-res U-Net decoder with attention-gated, S2-tapped skips)
        s.up3, s.ag3, s.dec3 = nn.Conv2d(dim, 128, 1), AttnGate(128, 64, 64), ResBlock(128 + 128, 128, 0.05)
        s.up2, s.ag2, s.dec2 = nn.Conv2d(128, 64, 1), AttnGate(64, 64, 32), ResBlock(64 + 64, 64, 0.05)
        s.up1, s.ag1, s.dec1 = nn.Conv2d(64, 32, 1), AttnGate(32, 64, 32), ResBlock(32 + 32, 32, 0.05)
        s.film_d = FiLM(cond, 32)
        s.mix_native = nn.Conv2d(32 + 64 + 64, 64, 3, padding=1, padding_mode="reflect")
        s.head = nn.Conv2d(64, s.Q * n_poll * 3, 1)         # (anomaly a, width_lo, width_hi) per horizon x pollutant
        s.occ = nn.Conv2d(64, s.Q, 1)                       # SO2 hurdle: occurrence logit per horizon
        s.prior = SeasonalPrior(n_poll, *full_hw)
        s.gate = MonotoneGate(n_poll)
        s.Bhead = nn.Sequential(nn.Conv2d(64 + 64, 32, 3, padding=1, padding_mode="reflect"), nn.SiLU(),
                                nn.Conv2d(32, n_poll, 1))
        nn.init.zeros_(s.Bhead[-1].weight); nn.init.zeros_(s.Bhead[-1].bias)    # B starts at 0 (null-hypothesis safe)
        nn.init.zeros_(s.head.bias)

    def forward(s, batch):
        v, m = batch["fast_v"], batch["fast_m"]             # (B,L,C,H,W)
        s2, lm = batch["s2_slow"], batch["long_mem"]        # (B,26,8H,8W), (B,72,H,W)
        crop, doy_tgt = batch["crop"], batch["doy_tgt"]     # tuple, (B,Q)
        B = v.size(0)
        cal = s.cal(torch.cat([batch["doy_org_sc"], torch.sin(2*math.pi*doy_tgt/365.25), torch.cos(2*math.pi*doy_tgt/365.25)], 1))
        f1, h2, h3, h4 = s.fast(v, m)
        if s.use_s2: s1, s2f, s4f, s8f = s.slow(s2)
        else:
            H, W = v.shape[-2:]
            s8f = v.new_zeros(B, 64, H, W); s4f = v.new_zeros(B, 48, 2*H, 2*W); s2f = v.new_zeros(B, 32, 4*H, 4*W)
        lmf = s.long(lm)
        # bring S2 slow feats to the 1/2,1/4 scales of the trunk
        s8_h2 = F.avg_pool2d(s8f, 2); s8_h3 = F.avg_pool2d(s8f, 4)
        z = s.fuse(torch.cat([h4, F.adaptive_avg_pool2d(s8f, h4.shape[-2:]), F.adaptive_avg_pool2d(lmf, h4.shape[-2:])], 1))
        z = s.film_b(z, cal)
        for blk in s.attn: z = blk(z)
        d3 = s.up3(F.interpolate(z, size=h3.shape[-2:], mode="bilinear", align_corners=False))
        d3 = s.dec3(torch.cat([d3, s.ag3(h3, s8_h3)], 1))
        d2 = s.up2(F.interpolate(d3, size=h2.shape[-2:], mode="bilinear", align_corners=False))
        d2 = s.dec2(torch.cat([d2, s.ag2(h2, s8_h2)], 1))
        d1 = s.up1(F.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False))
        d1 = s.dec1(torch.cat([d1, s.ag1(f1, s8f)], 1))
        d1 = s.film_d(d1, cal)
        trunk = F.silu(s.mix_native(torch.cat([d1, s8f, lmf], 1)))
        out = s.head(trunk).view(B, s.Q, s.n_poll, 3, *trunk.shape[-2:])
        a, wlo, whi = out[:, :, :, 0], F.softplus(out[:, :, :, 1]) + 1e-3, F.softplus(out[:, :, :, 2]) + 1e-3
        prior = s.prior(doy_tgt, crop, trunk.shape[-2:])                                        # (B,Q,C,H,W)
        g = s.gate(s.h_days)[None, :, :, None, None]                          # (1,Q,C,1,1)
        Bterm = s.Bhead(torch.cat([trunk, s8f], 1))[:, None] if (s.use_B and s.use_s2) else 0.0
        bscale = (1 - g)                                                      # structural shift matters more at long lead
        med = prior + bscale * Bterm + g * a
        return {"q50": med, "q10": med - wlo, "q90": med + whi,
                "occ_logit": s.occ(trunk), "B": Bterm}

def masked_pinball(pred, target, mask, tau):
    e = target - pred
    l = torch.maximum(tau * e, (tau - 1) * e) * mask
    return l.sum() / mask.sum().clamp(min=1.0)

def dcnet_loss(out, y, ymask, so2_event=None, so2_idx=2, lam_grad=0.1, lam_peak=0.5, lam_B=1e-3, lam_occ=0.2,
               peak_thr=None, hweights=None):
    """y,ymask: (B,Q,C,H,W). Loss is zero-safe for all-invalid batches."""
    tot = 0.0
    q50, q10, q90 = out["q50"], out["q10"], out["q90"]
    w = 1.0
    if peak_thr is not None:
        w = 1.0 + lam_peak * torch.sigmoid((y - peak_thr[None, None, :, None, None]) / 0.25)
    m = ymask * w
    tot = tot + masked_pinball(q50, y, m, 0.5) + 0.5 * masked_pinball(q10, y, m, 0.1) + 0.5 * masked_pinball(q90, y, m, 0.9)
    gx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    gy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    mx, my = ymask[..., :, 1:] * ymask[..., :, :-1], ymask[..., 1:, :] * ymask[..., :-1, :]
    tot = tot + lam_grad * ((gx(q50) - gx(y)).abs() * mx).sum() / mx.sum().clamp(min=1) \
                + lam_grad * ((gy(q50) - gy(y)).abs() * my).sum() / my.sum().clamp(min=1)
    if so2_event is not None:
        mk = ymask[:, :, so2_idx]
        bce = F.binary_cross_entropy_with_logits(out["occ_logit"], so2_event, reduction="none") * mk
        tot = tot + lam_occ * bce.sum() / mk.sum().clamp(min=1)
    if torch.is_tensor(out["B"]): tot = tot + lam_B * out["B"].abs().mean()
    return tot

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class GuidedSoftmaxDownscaler(nn.Module):
    """Phase C. Mass-conserving fine-resolution head (Harder et al. softmax constraint layer).
    fine = coarse * r^2 * softmax_over_rxr_block(logits)  ->  block-mean(fine) == coarse EXACTLY, fine >= 0."""
    def __init__(s, c_poll=3, r=8, c_trunk=64, ch=(24, 32, 48, 64)):
        super().__init__()
        s.r = r
        s.t = nn.Conv2d(c_trunk + ch[3], 64, 3, padding=1, padding_mode="reflect")
        s.u4 = ResBlock(64 + ch[2], 48); s.u2 = ResBlock(48 + ch[1], 32); s.u1 = ResBlock(32 + ch[0], 24)
        s.out = nn.Conv2d(24, c_poll, 1)
        s.temp = nn.Parameter(torch.zeros(1))          # softmax temperature (log-scale), init 1.0
    def forward(s, coarse, trunk, s1, s2, s4, s8):
        x = F.silu(s.t(torch.cat([trunk, s8], 1)))
        x = s.u4(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), s4], 1))
        x = s.u2(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), s2], 1))
        x = s.u1(torch.cat([F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), s1], 1))
        lg = s.out(x) * torch.exp(s.temp)                                   # (B,C,rH,rW)
        B, C, RH, RW = lg.shape; r = s.r
        w = F.pixel_unshuffle(lg, r).view(B, C, r * r, RH // r, RW // r).softmax(2) * (r * r)
        fine = F.pixel_shuffle(w.reshape(B, C * r * r, RH // r, RW // r), r)  # relative weights, mean 1 per block
        return fine * F.interpolate(coarse, scale_factor=r, mode="nearest")
```

### `align_tn.py`
```python
"""Strict S2<->S5P alignment: S2 is area-averaged onto an exact 8x8-per-S5P-cell grid (valid-pixel weighted)."""
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine

R = 8   # S2 sub-pixels per S5P cell, per axis

def s5p_grid(s5p_path):
    with rasterio.open(s5p_path) as src:
        return src.transform, src.height, src.width, src.crs

def align_s2_to_s5p(s2_path, s5p_transform, H5, W5, crs, r=R):
    """returns S2 (12, r*H5, r*W5) float32 and valid (1, r*H5, r*W5) in [0,1] = fraction of valid sub-pixels."""
    dst_tf = s5p_transform * Affine.scale(1.0 / r, 1.0 / r)           # exact sub-grid of the S5P grid
    with rasterio.open(s2_path) as src:
        x = src.read().astype(np.float32)                              # (12,h,w)
        valid = (np.nan_to_num(x[:9]) != 0).any(0).astype(np.float32)  # all 9 reflectance bands == 0  -> NoData/ocean fill
        x = np.nan_to_num(x, nan=0.0) * valid[None]
        out = np.zeros((x.shape[0], H5 * r, W5 * r), np.float32); vout = np.zeros((1, H5 * r, W5 * r), np.float32)
        for b in range(x.shape[0]):
            reproject(x[b], out[b], src_transform=src.transform, src_crs=src.crs, dst_transform=dst_tf,
                      dst_crs=crs, resampling=Resampling.average)
        reproject(valid, vout[0], src_transform=src.transform, src_crs=src.crs, dst_transform=dst_tf,
                  dst_crs=crs, resampling=Resampling.average)
    out = out / np.clip(vout, 1e-6, None)                              # valid-weighted mean
    out[:, vout[0] < 0.05] = 0.0
    return out, vout
```

### `make_cache.py`
```python
"""Build the aligned training cache from your notebook's GeoTIFFs (run once, CPU only).
Outputs (in --out): s5p_val.npy, s5p_mask.npy, s5p_month.npy, s5p_month_ok.npy, s2_month.npy (fp16, 8x grid),
s2_month_ok.npy, meta.json.  Regular 5-day window axis: missing windows are kept (mask=0), never dropped."""
import argparse, os, json, datetime as dt, numpy as np, pandas as pd, rasterio
from tqdm import tqdm
from align_tn import s5p_grid, align_s2_to_s5p, R

def windows(start, end, step=5):
    cur, out = dt.datetime.strptime(start, "%Y-%m-%d"), []
    end = dt.datetime.strptime(end, "%Y-%m-%d")
    while cur + dt.timedelta(days=step) <= end:
        out.append(cur); cur += dt.timedelta(days=step)
    return out

def main(a):
    os.makedirs(a.out, exist_ok=True)
    W = windows(a.start, a.end); T = len(W)
    ref = next(os.path.join(a.s5p_dir, f) for f in sorted(os.listdir(a.s5p_dir)) if f.startswith("s5p_2"))
    tf, H5, W5, crs = s5p_grid(ref)
    val = np.zeros((T, 3, H5, W5), np.float32); msk = np.zeros((T, 3, H5, W5), np.uint8)
    for i, w in enumerate(tqdm(W, desc="S5P")):
        p = os.path.join(a.s5p_dir, f"s5p_{w:%Y%m%d}.tif")
        if not os.path.exists(p): continue
        with rasterio.open(p) as s: x = s.read().astype(np.float32)[:, :H5, :W5]
        ok = np.isfinite(x) & (x > 0 if a.zero_is_invalid else True)    # notebook wrote unmask(0): 0 == "no data"
        val[i, :, :x.shape[1], :x.shape[2]] = np.where(ok, x, 0); msk[i, :, :x.shape[1], :x.shape[2]] = ok
    months = pd.period_range(W[0], W[-1], freq="M"); M = len(months)
    mon_of = np.array([months.get_loc(pd.Period(w, "M")) for w in W])
    s5m = np.zeros((M, 3, H5, W5), np.float32); s5ok = np.zeros((M, 3, H5, W5), np.uint8)
    for m in range(M):
        ii = np.where(mon_of == m)[0]; c = msk[ii].sum(0)
        s5m[m] = (val[ii] * msk[ii]).sum(0) / np.maximum(c, 1); s5ok[m] = c > 0
    H8, W8 = H5 * R, W5 * R
    s2m = np.lib.format.open_memmap(os.path.join(a.out, "s2_month.npy"), "w+", np.float16, (M, 12, H8, W8))
    s2ok = np.lib.format.open_memmap(os.path.join(a.out, "s2_month_ok.npy"), "w+", np.uint8, (M, H8, W8))
    for m in tqdm(range(M), desc="S2 monthly (causal-safe) composites"):
        stack, oks = [], []
        for i in np.where(mon_of == m)[0]:
            p = os.path.join(a.s2_dir, f"s2_{W[i]:%Y%m%d}.tif")
            if not os.path.exists(p): continue
            x, v = align_s2_to_s5p(p, tf, H5, W5, crs); ok = v[0] > 0.5
            x[:, ~ok] = np.nan; stack.append(x.astype(np.float16)); oks.append(ok)
        if stack:
            med = np.nanmedian(np.stack(stack).astype(np.float32), axis=0); ok = np.isfinite(med[0])
            s2m[m] = np.nan_to_num(med, nan=0.0).astype(np.float16); s2ok[m] = ok
    for k, v in dict(s5p_val=val, s5p_mask=msk, s5p_month=s5m, s5p_month_ok=s5ok).items(): np.save(os.path.join(a.out, k + ".npy"), v)
    s2m.flush(); s2ok.flush()
    json.dump(dict(T=T, M=M, H5=H5, W5=W5, H8=H8, W8=W8, win_start=[f"{w:%Y-%m-%d}" for w in W],
                   month_start=[f"{p.start_time:%Y-%m-%d}" for p in months], month_end=[f"{p.end_time:%Y-%m-%d}" for p in months]),
              open(os.path.join(a.out, "meta.json"), "w"))
    print("cache ready:", T, "windows,", M, "months, S5P grid", (H5, W5), "S2 grid", (H8, W8))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2_dir"); ap.add_argument("--s5p_dir"); ap.add_argument("--out")
    ap.add_argument("--start", default="2019-01-01"); ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--zero_is_invalid", type=int, default=1)
    main(ap.parse_args())
```

### `dataset.py`
```python
import json, os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class S5PNorm:
    """Invertible per-pollutant transform: z=(asinh(x/s)-mu)/sd, fit on TRAIN-period valid pixels only."""
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

class DCDataset(Dataset):
    def __init__(s, cache, split, cfg, norm=None, s2_stats=None):
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
        s.origins = [i for i in range(s.T) if lo <= s.ws[i] <= hi and i >= cfg["min_origin_idx"]]
        tr_idx = [i for i in range(s.T) if s.ws[i] <= pd.Timestamp(cfg["splits"]["train"][1])]
        s.norm = norm or S5PNorm().fit(s.val, s.msk, tr_idx)
        s.zs5 = np.stack([s.norm.fwd(s.val[:, c], c) for c in range(3)], 1).astype(np.float32) * s.msk
        s.s2_stats = s2_stats or s._fit_s2(tr_idx)
        z = [s.zs5[tr_idx][:, c][s.msk[tr_idx][:, c] > 0] for c in range(3)]
        s.peak_thr = np.array([np.quantile(x, 0.9) for x in z], np.float32)
        s.so2_thr = float(np.quantile(z[2], cfg["so2_event_q"]))
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
        lag = pd.Timedelta(days=s.cfg["s2_lag_days"])       # S2 files contain up to +20 d of future imagery
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
        P = cfg["patch"]
        if s.split == "train": i0, j0 = rng.randint(0, s.H5 - P + 1), rng.randint(0, s.W5 - P + 1); h = w = P
        else: i0 = j0 = 0; h, w = s.H5, s.W5
        Hp, Wp = (-(-h // 8)) * 8, (-(-w // 8)) * 8
        pad = lambda a, k=1: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, Hp * k - a.shape[-2]), (0, Wp * k - a.shape[-1])])
        L = cfg["L"]; ids = np.arange(o - L + 1, o + 1); okid = ids >= 0; ids = np.clip(ids, 0, None)
        fv = s.zs5[ids][..., i0:i0 + h, j0:j0 + w]; fm = s.msk[ids][..., i0:i0 + h, j0:j0 + w].astype(np.float32) * okid[:, None, None, None]
        if s.split == "train" and cfg["mask_aug"] > 0:                       # simulate extra cloud gaps
            drop = (rng.rand(*fm.shape) < cfg["mask_aug"]); fm = fm * (~drop)
        fv = fv * fm
        t_end = s.we[o]
        use = [m for m in range(s.M) if s.me[m] <= t_end][-12:]
        lv = np.zeros((12, 3, h, w), np.float32); lk = np.zeros((12, 3, h, w), np.float32)
        for q, m in enumerate(use[::-1]):
            mk = s.s5ok[m][:, i0:i0 + h, j0:j0 + w].astype(np.float32)
            lv[q] = np.stack([s.norm.fwd(s.s5m[m][c, i0:i0 + h, j0:j0 + w], c) for c in range(3)]) * mk; lk[q] = mk
        long_mem = np.concatenate([lv.reshape(36, h, w), lk.reshape(36, h, w)])
        sl = (slice(i0 * 8, (i0 + h) * 8), slice(j0 * 8, (j0 + w) * 8))
        s2 = s.slow(t_end, sl, (h * 8, w * 8))
        steps = cfg["steps"]; y = np.zeros((len(steps), 3, h, w), np.float32); ym = np.zeros_like(y)
        for q, st in enumerate(steps):
            t = o + st
            if t < s.T and s.lo <= s.ws[t] <= s.hi:
                y[q] = s.zs5[t][:, i0:i0 + h, j0:j0 + w]; ym[q] = s.msk[t][:, i0:i0 + h, j0:j0 + w]
        ev = ((y[:, 2] > s.so2_thr) & (ym[:, 2] > 0)).astype(np.float32)
        doy = lambda ts: ts.dayofyear + ts.hour / 24.0
        d_org = doy(s.we[o]); d_tgt = np.array([(s.we[o] + pd.Timedelta(days=5 * st)).dayofyear for st in steps], np.float32)
        b = dict(fast_v=pad(fv), fast_m=pad(fm), s2_slow=pad(s2, 8), long_mem=pad(long_mem), y=pad(y), ym=pad(ym), ev=pad(ev),
                 doy_tgt=d_tgt, doy_org_sc=np.array([np.sin(2 * np.pi * d_org / 365.25), np.cos(2 * np.pi * d_org / 365.25)], np.float32),
                 crop=np.array([i0, j0, h, w]))
        return {k: torch.from_numpy(np.asarray(v)) for k, v in b.items()}

def collate(b):
    out = {k: torch.stack([x[k] for x in b]) for k in b[0] if k != "crop"}
    out["crop"] = tuple(int(v) for v in b[0]["crop"])       # same crop for the whole batch (prior-layer indexing)
    return out
```

### `train.py`
```python
import argparse, copy, json, math, os, numpy as np, torch
from torch.utils.data import DataLoader
from dataset import DCDataset, collate
from dcnet import DCNet, dcnet_loss, count_params, HORIZON_STEPS

CFG = dict(L=12, steps=list(HORIZON_STEPS), patch=64, s2_lag_days=20, so2_event_q=0.85, mask_aug=0.1, min_origin_idx=12,
           samples_per_epoch=1000, batch=8, epochs=120, lr=3e-4, wd=0.05, warmup=3, ema=0.999, clip=1.0, patience=15,
           splits=dict(train=["2019-01-01", "2022-12-31"], val=["2023-01-01", "2023-12-31"], test=["2024-01-01", "2024-12-31"]))

def to_dev(b, dev): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
def model_in(b): return dict(fast_v=b["fast_v"], fast_m=b["fast_m"], s2_slow=b["s2_slow"], long_mem=b["long_mem"],
                             crop=b["crop"], doy_tgt=b["doy_tgt"], doy_org_sc=b["doy_org_sc"])

@torch.no_grad()
def evaluate(model, dl, dev, peak):
    model.eval(); tot, n = 0.0, 0; cov = np.zeros((7, 3)); cn = np.zeros((7, 3)); sse = np.zeros((7, 3)); sst = np.zeros((7, 3))
    for b in dl:
        b = to_dev(b, dev); H, W = b["crop"][2], b["crop"][3]
        o = model(model_in(b)); crop = lambda t: t[..., :H, :W]
        y, ym = crop(b["y"]), crop(b["ym"]); out = {k: (crop(v) if torch.is_tensor(v) and v.dim() >= 4 else v) for k, v in o.items()}
        tot += dcnet_loss(out, y, ym, crop(b["ev"]), peak_thr=peak).item(); n += 1
        inside = ((y >= out["q10"]) & (y <= out["q90"])).float() * ym
        cov += inside.sum((0, 3, 4)).cpu().numpy(); cn += ym.sum((0, 3, 4)).cpu().numpy()
        e = ((y - out["q50"]) ** 2 * ym).sum((0, 3, 4)).cpu().numpy(); sse += e
        mu = (y * ym).sum((0, 3, 4)) / ym.sum((0, 3, 4)).clamp(min=1); sst += (((y - mu[None, :, :, None, None]) ** 2) * ym).sum((0, 3, 4)).cpu().numpy()
    r2 = np.where(sst > 1e-9, 1 - sse / np.maximum(sst, 1e-9), np.nan)   # NaN where a horizon has no valid targets
    return tot / max(n, 1), r2, cov / np.maximum(cn, 1)

def main(a):
    cfg = copy.deepcopy(CFG)
    if a.cfg: cfg.update(json.load(open(a.cfg)))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = DCDataset(a.cache, "train", cfg); va = DCDataset(a.cache, "val", cfg, tr.norm, tr.s2_stats)
    tl = DataLoader(tr, cfg["batch"], num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    vl = DataLoader(va, 1, num_workers=0, collate_fn=collate)
    model = DCNet(3, (tr.H5, tr.W5), use_B=bool(a.use_B), use_s2=bool(a.use_s2)).to(dev); print("params", f"{count_params(model):,}")
    ema = copy.deepcopy(model).eval()
    nd = [p for n, p in model.named_parameters() if p.ndim <= 1 or n.startswith("prior.W")]; dd = [p for n, p in model.named_parameters() if not (p.ndim <= 1 or n.startswith("prior.W"))]
    opt = torch.optim.AdamW([dict(params=dd, weight_decay=cfg["wd"]), dict(params=nd, weight_decay=0.0)], lr=cfg["lr"])
    iters = cfg["samples_per_epoch"] // cfg["batch"]; total = iters * cfg["epochs"]; step = 0
    sched = lambda s: min(1, (s + 1) / (cfg["warmup"] * iters)) * 0.5 * (1 + math.cos(math.pi * min(1, s / total)))
    amp = dev == "cuda"; scaler = torch.amp.GradScaler(enabled=amp); peak = torch.tensor(tr.peak_thr, device=dev)
    os.makedirs(a.out, exist_ok=True); best, bad = 1e9, 0
    for ep in range(cfg["epochs"]):
        model.train(); run = 0.0
        for b in tl:
            for g in opt.param_groups: g["lr"] = cfg["lr"] * sched(step)
            b = to_dev(b, dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                o = model(model_in(b))
            o = {k: (v.float() if torch.is_tensor(v) else v) for k, v in o.items()}
            loss = dcnet_loss(o, b["y"], b["ym"], b["ev"], peak_thr=peak)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]); scaler.step(opt); scaler.update(); step += 1; run += loss.item()
            with torch.no_grad():
                d = min(cfg["ema"], (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        vloss, r2, cov = evaluate(ema, vl, dev, peak)
        print(f"ep {ep} train {run/iters:.4f} val {vloss:.4f} | R2[h=5d,150d] NO2 {r2[0,0]:.3f}/{r2[-1,0]:.3f} CO {r2[0,1]:.3f}/{r2[-1,1]:.3f} | 80%-cov {cov.mean():.3f}", flush=True)
        if vloss < best: best, bad = vloss, 0; torch.save(dict(model=ema.state_dict(), norm=tr.norm.state(), s2=tr.s2_stats, cfg=cfg), os.path.join(a.out, "best.pt"))
        else:
            bad += 1
            if bad >= cfg["patience"]: print("early stop"); break
        if a.max_epochs and ep + 1 >= a.max_epochs: break

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cache"); ap.add_argument("--out", default="runs/dcnet"); ap.add_argument("--cfg")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use_B", type=int, default=1); ap.add_argument("--use_s2", type=int, default=1); ap.add_argument("--max_epochs", type=int, default=0)
    main(ap.parse_args())
```

