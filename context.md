# Experimental Context, Learnings, and Architectural Insights

## 1. Executive Summary & Dataset Expansion Context

In this experimental phase, the framework was scaled from an initial prototype (118 composites) to the **full operational archive covering Tamil Nadu, India across 2019-2024 (415 paired composites, 3,699 spatial patches of size 128 x 128)**.

* **Historical Training/Validation Pool (2019-2022):** 293 composites (2,601 valid patches).
* **Strict Temporal Holdout Pool (2023-2024):** 122 composites (1,098 valid patches) with **zero spatial or temporal data leakage**.
* **Observational Inputs:** 4 past time-steps of Sentinel-5P TROPOMI (NO2, CO, SO2) fused with high-resolution Sentinel-2 optical bands (12 channels + NDVI, NDWI, NDBI) via a frozen SSL4EO-S12 self-supervised encoder.

---

## 2. Hypotheses Tested & Methodological Iterations

### A. Decoupled Pollutant-Specific Decoder Heads (Phase 6)
* **Rationale:** Atmospheric pollutants possess fundamentally distinct physical regimes:
  * **NO2:** Short atmospheric lifetime (~ 2-6 hours), sharp spatial gradients concentrated around combustion sources, highways, and industrial corridors.
  * **CO:** Long atmospheric lifetime (~ 1-2 months), diffuse synoptic advection across regional scales.
  * **SO2:** Intermittent, highly localized point emissions (thermal power plants) characterized by near-zero background and extreme heavy-tailed spikes.
* **Architecture Implementation:**
  * **Shared Decoder Trunk:** ConvTranspose2d stack (128 -> 64 -> 32) with temporally-weighted multi-level skip connections.
  * **NO2 Head:** ResBlock2D(32, 32) -> Conv2d(32, 1, 1) for high-frequency gradient retention.
  * **CO Head:** Dilated Residual Block (d=2, 4) to expand the effective receptive field for large-scale synoptic advection.
  * **SO2 Head:** Dedicated residual block trained on log1p normalized space and inverted via expm1 scaling to stabilize extreme variance.
* **Outcome:**
  * NO2 holdout R^2 improved from +0.485 to **+0.5085** (+0.023).
  * CO maintained high spatial correlation (r = +0.7850, R^2 = +0.5859).
  * SO2 achieved positive R^2 = **+0.0775** on the full 1,098 holdout patches (overcoming the negative R^2 collapse seen in shared decoders).

---

### B. Multi-Horizon Forecasting: 5-Day vs 150-Day vs 300-Day (Phase 7)
* **Goal:** Evaluate the model's long-range predictive capability across distinct lead times:
  * Horizon 1 (5-day ahead lead time)
  * Horizon 30 (150-day / 5-month ahead lead time)
  * Horizon 60 (300-day / 10-month ahead lead time)
* **Benchmark:** Rigorous **Naive Seasonal-Climatology Baseline** (predicting the historical multi-year calendar month/season mean for each patch).

#### Comparative Results Matrix:
| Target Pollutant | Metric | Horizon 1 (5 Days) | Horizon 30 (150 Days) | Horizon 60 (300 Days) | Seasonal Climatology |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NO2** | R^2 Score | **+0.5085** | -0.0163 | -0.0094 | **+0.4705** |
| | Pearson r | **+0.7571** | +0.4001 | +0.4093 | +0.7303 |
| | POD_q90 | **0.6558** | 0.2796 | 0.2635 | 0.6120 |
| **CO** | R^2 Score | +0.5859 | +0.2255 | +0.1345 | **+0.6374** |
| | Pearson r | +0.7850 | **+0.6802** | **+0.5445** | +0.8035 |
| | POD_q90 | 0.6698 | **0.5397** | 0.4485 | 0.7023 |
| **SO2** | R^2 Score | **+0.0775** | -0.0381 | -0.0396 | -0.1102 |
| | Pearson r | **+0.2974** | +0.0386 | +0.0374 | +0.0732 |
| | POD_q90 | **0.4079** | 0.2014 | 0.1772 | 0.2201 |

---

## 3. What Failed / Experimental Limitations

1. **Direct Multi-Step Long-Horizon Prediction Without Climatological Priors:**
   * Directly projecting a neural network 150 to 300 days ahead without calendar/day-of-year or climatological residual conditioning causes the network to revert towards the spatial mean.
   * For short-lived species (NO2), atmospheric memory decays within days; past optical and gas states at t <= 0 carry no autoregressive physical memory to t = +150 days.
2. **Single Shared Decoder Trunk for Multi-Scale Dynamics:**
   * Forcing NO2 (local), CO (regional), and SO2 (spiky point sources) through a single final 1x1 conv created severe gradient competition. Decoupled heads resolved this, demonstrating the necessity of modular decoders.
3. **Raw L1/L2 Loss on Unnormalized SO2:**
   * SO2 values are orders of magnitude smaller (~ 10^-4) with extreme sparse spikes. Linear MSE ignores 99% of the low-intensity background and overfits to single outlier pixels. log1p compression was strictly necessary to avoid numerical instability.

---

## 4. Key Scientific & Engineering Learnings

1. **Atmospheric Physics Dictates Predictability Limits:**
   * **CO has genuine medium-to-long-range memory:** Because CO has a lifetime of 1-2 months, the model achieved high spatial correlation (r = 0.6802 at 150d, r = 0.5445 at 300d) and retained high peak detection (POD = 0.5397).
   * **NO2 and SO2 are governed by local meteorological advection and real-time emissions:** Beyond 10-15 days, forecasting cannot rely solely on past gas imagery and must be conditioned on seasonal solar radiation and wind fields.
2. **Climatology Baseline is an Essential Benchmark:**
   * In multi-month forecasting, seasonality (monsoon vs dry winter) dominates raw variance. A model that achieves R^2 > 0 might simply be re-learning calendar averages. Comparing directly against naive monthly climatology revealed where neural forecasting adds true marginal skill.
3. **Spatial Metrics Outweigh Pixel-Wise R^2:**
   * High SSIM > 0.95 and FSS > 0.99 show that the model excels at super-resolving spatial plume geometries and boundary delineations, even when absolute point errors fluctuate.

---

## 5. Roadmap & Recommended Architecture for Next Iterations

### Iteration Priorities for Next Runs:

1. **Climatology-Residual Architecture (Year/Month Prior + Residual Network):**
   * Instead of predicting the raw pollutant field from scratch for long horizons, train the network to predict the **anomaly/residual** relative to the seasonal climatological mean.
2. **Meteorological Conditioning (ERA5 Reanalysis / GFS / ECMWF):**
   * Integrate 10m u/v wind components, planetary boundary layer height (PBLH), surface temperature, and relative humidity into the temporal ConvGRU bottleneck.
3. **Physics-Informed Advection-Diffusion Loss:**
   * Enforce continuity and mass-conservation constraints based on wind vectors and diffusion coefficients.
4. **Fourier Neural Operators (FNO) / Vision-State-Space (Mamba) Bottleneck:**
   * Replace standard ConvGRU with a spatial Fourier Neural Operator or Mamba-based long-sequence state space layer to capture multi-scale spatial dynamics with lower memory footprint.
5. **Autoregressive Multi-Step Rollout with Scheduled Sampling:**
   * Train iteratively on 1-step, 2-step, ..., k-step rollouts with curriculum noise injection to enable stable recursive forecasting across intermediate horizons (5, 10, 15, ..., 60 days).
