TODO — Land-Use-Informed Pollution Baseline

0. Do not drop any meaningful data in the name of hardware constraints, local laptop performance, or aggressive preprocessing/feature extraction. This is not a time series model.

- Critical Preprocessing Audit :
  The raw GeoTIFFs exported from Google Earth Engine are already high-resolution (558 x 558 pixels, in order of ~10^2m, 12 spectral bands).
  Aggressive downsampling occurred inside `align_tn.py` via `R = 8` (8 S2 sub-pixels per S5P cell). Because Sentinel-5P has a coarse resolution (~5.5 km, 13 x 13 pixels over Chennai), the script downscaled the 558 x 558 Sentinel-2 optical images down to 13 x 8 = 104 x 104 pixels (~700m per cell) 
  
  At lower resolution, individual road networks, vehicular traffic corridors, and narrow industrial buildup become severely blurred into broad background averages, destroying the fine-grained land-use predictive signal. This downsampling violates Rule 0 and is strictly prohibited. Instead explore methods that utilise full res data (Prefer high res data so high quality learning) while ensuring efficiency and objectivity. (THIS NEEDS TO BE ADDRESSED BEFORE CONTINUING OTHERE SUGESSIONS WERE IMPLEMENTED)

1. Objective & Architectural Context

Build a pure spatial baseline model that learns pollution signatures inferred from land use using Sentinel-2 surface reflectance, rather than memorizing temporal or climatic patterns.

- Why this exists (Legacy Failure Mode): Previous iterations overfitted on temporal autocorrelation and meteorology, acting effectively as pure time-series models while ignoring surface/land-use significance. Furthermore, over-downsampling S2 optical features to 700m blinded the spatial encoder to fine-grained land use.
- How this fits the Core Model: We already have a planned ML/time-series counterpart to handle temporal dynamics in the final core architecture. This stage isolates the spatial encoder. It must prove it can extract static/quasi-static land-use pollution signatures on its own before we fuse it with the temporal counterpart.


---

2. Non-Negotiable Rules

1. Land-use signal is the sole objective of this stage. The model must map spatial/surface characteristics (industrial zones, road networks, urban density, vegetation) to baseline pollution levels.
2. Zero temporal/climate shortcuts in the baseline. Do not feed lag features, timestamps, or meteorological variables into the Sentinel-2 baseline branch.
3. Do not equate high time-series $R^2$ with success. A lower, honest spatial $R^2$ on held-out locations is acceptable and expected; a high $R^2$ driven by temporal leakage will be rejected.
4. Use temporally aggregated targets to stabilize the spatial signal. Frame the baseline target over monthly, quarterly, or long-term means (following the national-no2-cnn and Khan et al. aggregation approaches) so high-frequency weather noise does not drown out the land-use signature.
5. Document Sentinel-5P constraints. Where Sentinel-5P column density is used as supervision, a target, or a secondary fusion branch, explicitly document its spatial resolution gap and column-to-ground mismatch.
6. Follow rule.md strictly. Any deviation from rule.md must be logged and justified before running experiments.
7. Strictly Preserve Fine-Grained Optical Resolution: Sentinel-2 optical features must NOT be aggressively downscaled to coarse resolutions (such as the legacy R=8 ~700m downsampling in align_tn.py). The spatial encoder requires native or fine-scale resolution (e.g., 60m–100m, retaining the 558 x 558 grid) so that emission hot spots (power plants, refineries, major highway corridors) retain distinct structural contrast rather than being blurred into ambient background noise.


---

3. Action Items (TODO)

- [ ] Review Context & Rules: Read rule.md, the legacy failure notes, and the reference papers/repos below before writing or modifying model code.
- [ ] Lock Problem Definition: Define the target pollutant ($\text{NO}_2$), spatial patch size (e.g., $120 \times 120\text{ px}$ at $10\text{ m}$ resolution / $1.2 \times 1.2\text{ km}$), temporal aggregation window (daily vs. monthly/quarterly mean), and Sentinel-2 spectral bands.
- [ ] Enforce Leak-Free Spatial Splits: Implement spatial block cross-validation or distance-buffered splits so training and validation/test Sentinel-2 patches never overlap geographically.
- [ ] Build Pure Sentinel-2 Baseline: Train a clean, reproducible CNN/spatial baseline using only Sentinel-2 surface reflectance to predict the aggregated pollutant target.
- [ ] Run Shortcut & Fusion Ablations:
  - Config A: Sentinel-2 only (Pure Land-Use Baseline — Primary Focus)
  - Config B: Temporal / Climate / S5P only (Shortcut / Atmospheric Baseline)
  - Config C: Combined inputs (Evaluate genuine marginal gain from Sentinel-2)
- [ ] Verify Land-Use Learning: Provide direct evidence that the model learned surface patterns (e.g., performance across unseen spatial blocks, error breakdown by land-cover class, or spatial attribution/Grad-CAM checks).
- [ ] Document Failure Boundaries (Old vs. New):
  - Document where the older time-series approach failed (spatial blindness / unseen locations).
  - Document where this new land-use change still does not help (e.g., transient meteorological trapping, wind transport, short-term traffic spikes) so we know exactly what the core ML counterpart must cover next.


---

4. Acceptance Criteria

The baseline is accepted only if:

1. Experiments demonstrate that Sentinel-2 surface reflectance provides predictive signal on spatially held-out locations with zero temporal/meteorological inputs.
2. When combined with atmospheric/temporal inputs in ablations, the Sentinel-2 branch demonstrably improves spatial generalization rather than being ignored by the network.
3. Failure cases where land-use alone is insufficient are clearly logged before integrating the ML/time-series counterpart.


---

5. References

- Project Rulebook (rule.md): Local source of truth for engineering and modeling constraints.
- Reference Paper (Land-Use CNN): Frontiers in Environmental Science (2023) — https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2023.1285471/full
- Reference Codebase (national-no2-cnn): https://github.com/eltonc01/national-no2-cnn
- Multisource Fusion & Temporal Aggregation Reference: Solaiman Khan et al. (2026), High-resolution ground $\text{NO}_2$ estimation at hyperlocal level using deep learning with Sentinel-2 and Sentinel-5P data (refer to project summary for daily vs. monthly/quarterly $R^2$ scaling and patch setup; verify full bibliographic details before formal citation).


---

6. Scope Boundary

This document defines the overall objective, stage boundaries, and validation guardrails. Detailed preprocessing pipelines, hyperparameter tables, and core-model fusion architecture belong in separate implementation docs.
-Sentinel-2-only inputs do not automatically prove that the model learned land-use pollution signatures. It could learn other spatial correlations. Your ablations, held-out spatial validation, and land-cover analysis help test that distinction.
-Monthly or quarterly target aggregation stabilizes the spatial signal but does not, by itself, remove meteorological or other influences.


---

## 7. Stage B — Land-Use-Prior Temporal Forecaster (Core Model)

*Added 2026-10-25. Stage A (sections 1-6) is unchanged. Stage B does not start until the Stage A acceptance evidence (section 4) is logged.*

### 7.1 Objective

Forecast **future Sentinel-5P composites** at 5, 10, 15, 30, 60 and 120 days of lead time, using the Stage A land-use spatial prior (from Sentinel-2) together with S5P history, ERA5 meteorology and calendar terms. The primary pollutant is NO2; CO and SO2 are secondary. The output is a **map** per origin and horizon, with uncertainty bands and hotspot blobs, not a single pooled number.

### 7.2 Scope decisions (these deviate from DCNet_Training_Spec.md and are logged here per rule 6)

1. **Horizons: 5, 10, 15, 30, 60, 120 days** = steps 1, 2, 3, 6, 12, 24 of the 5-day composite axis. This replaces (5, 10, 20, 30, 60, 120, 150).
2. **No next-day forecasting.** Land use and surface state do not change at daily scale, so a land-use prior has nothing to add there; the data cadence is the 5-day composite and the shortest lead is one composite (5 days).
3. **Horizon groups:** short = 5-15 d (synoptic memory, where real skill is expected); medium = 30 d; long = 60-120 d (seasonal outlook).
4. **Long horizons are scored as skill over climatology, never as raw R2.** The seasonal cycle alone explains much of the variance at long lead (the corrected per-pixel monthly climatology scores about R2 0.45 for NO2 on the 2024 validation year). Converging to climatology at 60-120 d is an acceptable, documented outcome.

### 7.3 Model contract

- **Forecast form:** y_hat(x, t+h) = clim(x, month(t+h)) + a_hat_h(x, t). The network predicts the **anomaly** relative to the training-period per-pixel monthly climatology; the Stage A prior is an input channel.
- **Prior leakage rule:** the prior used for training and validation pixels must be a **spatial out-of-fold** prediction (blocks >= 20 km with a buffer ring), and its target period must end at or before the training cutoff (2023-12-31 for Stage 1). An in-sample prior memorises each pixel's own mean and makes the S2 branch look useful when it is not. A prior fit on all cells may be used only for final deployment maps.
- **Resolution:** keep S2 at the finest scale justified by the Stage A resolution ablation; do not downsample below it for convenience (rule 7 still applies).

### 7.4 Non-negotiable rules for Stage B

- **B1. Temporal splits only.** Stage 1: train 2019-2023, validate 2024, test 2025. A horizon target is used only if its date falls inside the same split. The test split is evaluated once, after configuration is frozen.
- **B2. Baselines are mandatory, per horizon and pollutant:**
  (i) per-pixel monthly climatology from the training period;
  (ii) persistence from the **last valid observation per pixel** (not simply the last window);
  (iii) damped persistence of the anomaly;
  (iv) Config B: the same temporal model **without** the S2 prior.
  A result with no baselines is not a result.
- **B3. Valid pixels only.** Report metrics on valid target pixels and state the valid-pixel percentage per split and horizon.
- **B4. Weather policy.** ERA5 for the target date is not available at forecast time. Use meteorology only up to the origin date, or label the run **oracle-weather (upper bound)**. ERA5 `blh` for 2024-01-01 to 2024-06-30 is missing and currently imputed with climatology; it must be replaced with real values before any weather ablation is reported.
- **B5. Data prerequisites** (each must be checked off before the final run):
  - cache extended to 2025 so a true held-out test exists;
  - real `blh` for 2024 H1 and 2025;
  - SO2 negative retrievals kept (apply the `> 0` validity rule to NO2 and CO only);
  - S2 files with missing georeferencing counted and handled (S2 valid-pixel audit passes);
  - `check_cache.py` run and its output stored with the results.
- **B6. Report per horizon x pollutant.** Never one pooled number. Report one seed plus a block-bootstrap confidence interval (blocks of 6 consecutive windows, about 30 days) for every headline metric.

### 7.5 Metrics: what "good" means

Raw R2 is **not** the success criterion (rule 3): a static spatial pattern plus the seasonal cycle already gives a high pooled R2. Success is skill over the baselines.

**Per horizon and pollutant (headline):**

| Metric | Definition | Reading |
|---|---|---|
| SS_clim | 1 - MSE(model) / MSE(climatology); equals the anomaly R2 about zero | > 0 beats climatology |
| SS_pers | 1 - MSE(model) / MSE(persistence) | > 0 beats persistence |
| Anomaly Pearson r | correlation of (y - clim) with (y_hat - clim) | pattern skill |
| Pooled R2, RMSE | physical units | reported, **not** the headline |
| 80% interval coverage and width | share of targets inside [q10, q90] | target 0.80 +/- 0.05 |

**Map and blob metrics (the "blob output"):**

- **SSIM** between observed and forecast maps on valid pixels.
- **Hotspot blobs:** threshold at the training-period 90th percentile for the same calendar month; connected components (8-neighbour, minimum 2 pixels); match forecast blobs to observed blobs by IoU >= 0.3. Report blob detection precision, recall and F1, mean matched IoU, **centroid error in km** (about 5 km per pixel), area ratio and peak-amplitude bias.
- **Blob baselines:** the same blob metrics for the climatology map and the persistence map. The model must beat both.

**Output deliverables per run:**

1. For each (origin, horizon): a panel with observed, forecast q50 with blob contours overlaid, and the q90-q10 uncertainty map.
2. A blob table (CSV) per horizon with the metrics above.
3. A summary figure: SS_clim, SS_pers, SSIM and blob F1 against horizon, with climatology and persistence as reference lines.

### 7.6 Aims and acceptance criteria

These are aims defined as skill over baselines, not raw-R2 thresholds. They are accepted only if all leakage checks (B1-B5) pass.

1. **Short range (5, 10, 15 d), NO2:** SS_clim and SS_pers > 0 on the test split, with the bootstrap confidence interval excluding 0 for at least the 5-day horizon. (Earlier damped-persistence results in docs/ showed about +4.8% at 5 d, which is the scale of gain to expect.)
2. **Medium range (30 d):** SS_clim >= -0.01. Gains are welcome, not assumed.
3. **Long range (60, 120 d):** SS_clim >= -0.01, plus blob F1 and SSIM at least as good as the climatology map. If the model collapses to climatology, state this explicitly.
4. **Land-use value (Config C vs Config B):** the S2-prior model beats the no-prior model on spatially held-out blocks at one or more horizons, or the absence of a gain is documented as a failure boundary.
5. **Blob output:** blob F1 and centroid error beat the climatology and persistence blob baselines at the short horizons.

### 7.7 Entry gate and implementation TODO

- [ ] Stage A evidence logged (section 4 criteria 1-3), including the stacking and scale-ablation results.
- [ ] `dcnet.py` and `DCNet_Training_Spec.md`: change `HORIZONS_DAYS` to (5, 10, 15, 30, 60, 120).
- [ ] `evaluate.py`: add SS_clim, SS_pers, anomaly r, block-bootstrap CIs, SSIM and the blob metrics; use the last-valid-observation persistence (done in commit 834c562).
- [ ] `train.py`: replace the per-window spatial R2 with the pooled or anomaly metric above for checkpoint reporting.
- [ ] Stage A prior export: spatial out-of-fold prior map plus a deployment prior map.
- [ ] Data prerequisites B5 completed.

---

## 8. Change Log (rule 6)

| Date | Change | Reason |
|---|---|---|
| 2026-10-25 | Added Stage B (section 7): horizons 5, 10, 15, 30, 60, 120 d; no next-day forecast; metrics defined as skill over baselines; blob output specified | Land state is static at daily scale; long-lead raw R2 is dominated by seasonality; deliverable must be map and blob output |

**Stage A evidence log (NO2, 2019-2024 mean column, spatial block CV with buffer; run 2026-10-24, one seed, 5 folds):**

| Block size | geo_knn (location only) | lgbm_s2 (S2 only) | cnn_s2 (S2 only) | lgbm_s2 + location |
|---|---|---|---|---|
| ~20 km | 0.811 | 0.581 | 0.665 | 0.796 |
| ~40 km | 0.799 | 0.619 | 0.596 | 0.801 |
| ~80 km | 0.498 | 0.507 | 0.483 | 0.661 |

Reading: S2 alone carries signal on unseen blocks and degrades slowly with distance; it adds no information beyond location at 20-40 km, and a large amount at 80 km. Gain is concentrated in the 16-38 km neighbourhood features; compact industrial sources (power plants) are under-predicted at the ~600 m S2 resolution. The target map is highly reproducible (split-half r 0.987). Follow-up tests (stacking, scale ablation, 10 m point-source test) are logged in section 7.7 as prerequisites.
