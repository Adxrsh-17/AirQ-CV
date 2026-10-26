# AirQ-CV: FYP Presentation Guide

The panel's job is to scrutinise the work. Every claim on a slide needs a number, a file it comes from, and an answer to "compared to what?".

**Story:** NO₂ is the main result. CO and SO₂ appear only to show that the pipeline handles several pollutants.

---

## 1. Where everything is

| What | Where |
|---|---|
| Code (the reported model) | GitHub `antonioroger2/AirQ-CV`, branch **`feat/dual-clock-architecture`**, head **`14c129e`** (band-index fix) |
| Push first | `git push origin feat/dual-clock-architecture`. The commits are local only; GitHub still shows the old head. |
| Rules / spec | `Rule.md` §7 (Stage B) and §8 (Stage A evidence). Your uncommitted edit is still local. |
| Run notes | `Latest_run.md` |
| Committed run outputs | `results/run_2026-10-25_stageB/`: `dcnet_val_metrics_C.csv`, `dcnet_val_blobs_C.csv`, `train_log.csv`, `cache_audit.csv`, `summary_val_C.png`, `panels_{NO2,CO,SO2}_val_C.png`, `priors/prior_*.npy` |
| Newer run, same checkpoint (untracked) | `kaggle_job/xyz/runs/`. `eval_all/` holds the 54-row metrics with the Kongu box and the `*_box.png` panels. `attribution_all_C/` holds **`prior_hotspot_maps_val.png`** and `prior_hotspot_test_val.csv`. `dcnet_all_C_s0/best.pt` is the checkpoint. |
| Kaggle notebook | `antonioroger2/dcnet-dual-clock-training`. The notebook builder is `kaggle_job/build_notebook.py`. Push with `cd kaggle_job && kaggle kernels push -p .` |
| Download a run | `kaggle kernels output antonioroger2/dcnet-dual-clock-training -p kaggle_job/run_fixed` |
| Thesis | `FYP_THESIS-B6.pdf`. The section-by-section update list is in the chat. |

### Before the presentation: one rerun (~1 h)

Commit `14c129e` fixed the Sentinel-2 channel order. Before it, the inputs named NDBI, NDVI, MNDWI, NDRE and BSI were not those indices. Every number below comes from the run **before** that fix.

1. Push the notebook with `MODE='dcnet'`, which rebuilds the prior and retrains.
2. Also run `landuse_baseline.py --cnn 1 --blocks 8,4,16` (already in the notebook) to refresh the block-size table.
3. Replace the **[current]** numbers below with the new ones. If the rerun isn't done in time, present the current numbers and say: "band-order bug found in our audit, fixed in 14c129e, rerun pending." Panels respect a bug you found yourself. They don't respect one they find.

---

## 2. The story in one line

> Sentinel-2 land use alone recovers where NO₂ is high on unseen ground. DC-Net forecasts state-wide NO₂ column maps 5–120 days ahead, with calibrated uncertainty, against a strict climatology reference. The same pipeline runs for CO and SO₂.

---

## 3. Slide plan (about 13 slides)

| # | Slide | Content | Source file |
|---|---|---|---|
| 1 | Title | Name, team, guides | |
| 2 | Problem | Few ground monitors in Tamil Nadu. S5P sees gas columns; S2 sees land use. Goal: NO₂ column maps 5–120 d ahead, plus a land-use explanation of the hotspots. | |
| 3 | Data | S5P NO₂/CO/SO₂ 5-day composites, 2019–2024, 438 windows, 123×95 grid (~5 km). S2 600 m composites on an 8×8 sub-grid per cell. Weather history from 5 sites. NO₂ valid fraction 87–91% per window. | `cache_audit.csv`, thesis Table 5.1 |
| 4 | Pipeline | GEE export → aligned cache → **Stage A** (S2-only prior) → **Stage B** DC-Net → evaluation | thesis Fig 5.1 |
| 5 | **Stage A: S2 explains the NO₂ map** (key slide) | Spatial block CV with a buffer ring. NO₂ R² **0.624**, Spearman **0.781** [current]. On the unseen 2024 NO₂ map: Spearman **0.734**, top-10% hotspot precision **0.40** [current]. | log stage B-1; `prior_hotspot_test_val.csv`; `prior_hotspot_maps_val.png` |
| 6 | Honest control | At 20–40 km blocks, location alone beats S2; S2 catches up at 80 km. So S2 carries real but not unique information. | Rule.md §8 / thesis Table 5.5 (refresh after rerun) |
| 7 | DC-Net architecture | Slow clock (S2 + prior + context pyramid, SPADE) and fast clock (12×5-day anomalies, partial conv, TAU). Output = climatology + damped anomaly + learned residual, as quantiles q10/q50/q90. 3.08 M parameters. | thesis Fig 4.1 |
| 8 | Evaluation protocol | Train 2019–23, validate 2024. References are per-pixel climatology and persistence. Skill score with 95% block-bootstrap CI, held-out spatial blocks, hotspot blobs, interval coverage. | thesis Ch 3, 6 |
| 9 | **NO₂ results** | See §4 below. Lead with: R² 0.46 at 5 d, intervals calibrated (cov80 0.80), beats persistence by 40–60%, held-out blocks R² 0.51. | `dcnet_val_metrics_C.csv` |
| 10 | NO₂ maps | Observed vs forecast with hotspot contours, Kongu box panel | `panels_NO2_val_C.png`, `panels_NO2_val_C_box.png` |
| 11 | Region: Kongu belt | Coimbatore–Tiruppur–Erode–Salem–Namakkal box, scored inside the box with no retraining. NO₂ R² 0.351 vs climatology 0.325 at 5 d. Skill point estimates positive (+0.04 to +0.06); CIs include 0. | `kaggle_job/xyz/runs/eval_all/dcnet_val_metrics_C.csv` (region=bbox) |
| 12 | CO and SO₂ (side track) | CO: prior R² 0.683, forecast R² 0.76 with calibrated intervals. SO₂: prior 0.238, forecast weak because about half the pixels are missing and negatives are clamped. Shows the multi-pollutant pipeline and its data limits. | same CSV |
| 13 | Limits and next | Single validation year, single seed, no Config B ablation yet, 2025 test pending, BLH imputed for H1 2024, column ≠ surface. Next: 2025 frozen test, Config B, SO₂ re-export with negatives. | thesis §7.12, §8.4 |

---

## 4. NO₂ numbers to show [current, pre-fix; replace after rerun]

| Metric | 5 d | 30 d | 120 d | Compared to |
|---|---|---|---|---|
| R² (all pixels) | 0.455 | 0.434 | 0.367 | climatology 0.458 / 0.446 / 0.351 |
| Skill vs climatology [95% CI] | −0.005 [−0.06, 0.07] | −0.021 [−0.06, 0.03] | +0.024 [−0.03, 0.05] | 0 = equal to climatology |
| Skill vs persistence | 0.40 | 0.50 | 0.60 | |
| 80% interval coverage | 0.80 | 0.79 | 0.75 | nominal 0.80 |
| Held-out blocks R² | 0.512 | 0.503 | 0.432 | climatology 0.519 / 0.507 / 0.411 |
| Hotspot blob precision / IoU | 0.37 / 0.50 | 0.38 / 0.48 | 0.36 / 0.47 | centroid error 13–14 km |
| Kongu box R² | 0.351 | 0.299 | 0.269 | climatology 0.325 / 0.310 / 0.224 |

**How to say it:**
> "The NO₂ forecast matches a strong per-pixel climatology at every lead. At 120 days and in the Kongu belt its point estimates are above it, though not significantly. It is 40–60% better than persistence, and its 80% intervals are calibrated."

**Don't say:**
- "beats climatology"
- "R² 0.46 means 46% accurate"
- "we beat paper X"

---

## 5. Outdated or wrong: do not use

| Item | Why |
|---|---|
| Old DC-Net v1 "+33–47% skill vs climatology" | It was scored against a broken baseline (climatology R² ≈ 0). The true figure is about +1.5%. Mention it only as "why our protocol is strict". |
| ST-ResUNet / SSL4EO numbers (NO₂ 0.279, CO 0.448) | Different branch: Chennai tiles, 23 sequences, no climatology baseline. Not comparable. |
| "Next-day forecast" | Removed. Horizons are 5, 10, 15, 30, 60 and 120 days. |
| "ctx7_NDBI is the top feature" (Rule.md) | Came from the wrong band order. Invalid until the rerun. |
| Table 5.5 S2 columns (pre-fix) | Same bug. Refresh it or drop it. |
| `runs/` folder at repo root, `train_dual_clock.py`, `train_tn.py`, `src/` | Older pipelines, not the reported model. |
| "S2 at 10 m" | Inputs are 600 m composites aggregated to ~0.6 km sub-cells. |
| "Surface NO₂ / what people breathe" | Targets are column densities. |
| "Respiratory risk map" | Not implemented. It is future work (the title still names it, so expect a question). |
| "Climate / weather dominates land use" | Not established. Don't claim it. |
| Comparing our 0.4 with literature R² 0.54–0.89 | Different targets (ground NO₂), random splits and tasks. See §6. |
| "2025 test results" | No 2025 data in the cache. All numbers are validation-year (2024) numbers. |
| Distances in km | The code uses 5.5 km per cell; the grid step is about 5.0 km, so km values are ~10% high. |

---

## 6. Literature position (if asked "how do you compare?")

- **Khan et al. 2026, RSASE 41:101819 (Chicago):** S2 + S5P → ground NO₂. S2-only R² is 0.48–0.59 and fused is 0.61–0.66, on a random 70/15/15 split with the same sensor sites in train and test. Our S2-only prior reaches **0.62 under spatial block CV**: the same range, under a stricter split. We don't claim to beat it because the target and split differ.
- **Cao 2023, Front. Environ. Sci. (US national CNN):** ground NO₂ R² 0.89 with random CV falls to **0.59 with spatial hold-out**. That supports our use of spatial CV.
- **No published paper** that we found forecasts S5P column maps 5–120 days ahead with a climatology skill score. Our evaluation is the reference here.

---

## 7. Expected tough questions and honest answers

| Question | Answer |
|---|---|
| Your model doesn't beat climatology. What did DL add? | The climatology is a strong reference: it holds the fixed spatial pattern and the season. Our model matches it, produces calibrated 80% intervals, and is 40–60% better than persistence. For NO₂ at 5-day resolution, the anomaly autocorrelation is low (r ≈ 0.18), which limits any forecaster. |
| Does the forecaster use Sentinel-2? | Land use's role is demonstrated in Stage A: S2 alone predicts the NO₂ map on unseen blocks (R² 0.62, ρ 0.78) and on the unseen 2024 map (ρ 0.73). Its added value inside the forecaster needs the Config B ablation, which is our next experiment. |
| Why not a time-series model like LSTM? | The aim was land-use plus climatology structure, not pure temporal fitting. ConvLSTM and SimVP-style models were reviewed. With 354 training origins, a climatology anchor is safer than free recurrence. |
| Leakage? | Normalisation, climatology and thresholds are fitted on 2019–23 only. The prior is spatial out-of-fold with a buffer ring. No weather from the target date is used. Disclose: the prior's S2 composite includes Jan–Mar 2024 imagery (land surface only, no gas data). |
| Why is SO₂ so bad? | About half the pixels are invalid, and negative retrievals were clamped to zero, which biases the rest upward. It is a data-quality limit, not a model claim. The fix is a re-export keeping negatives. |
| Validation year is also the selection year? | Yes, so the numbers are slightly optimistic. With skill near 0 this can't create a positive claim. A frozen 2025 test is next. |
| What bug did you find? | The S2 channel order in the derived indices. We found it in our own audit and fixed it (14c129e), and rebuilt the prior and model [if rerun done]. |
| Why 5 km if S2 is 10 m? | The S5P target sets the grid. S2 enters at ~0.6 km sub-cells and is summarised per cell and by the 16–80 km context pyramid. |
| Respiratory risk? | Future work. Columns must first be related to surface monitors (CPCB) before any health claim. |

---

## 8. Checklist before the presentation

- [ ] Rerun on Kaggle with `MODE='dcnet'` after the band fix, and download it with `kaggle kernels output … -p kaggle_job/run_fixed`
- [ ] Replace every **[current]** number in this guide and the slides
- [ ] Copy the new `eval_all/` and the `attribution_all_C/prior_hotspot_*` files into `results/` and commit
- [ ] `git push origin feat/dual-clock-architecture`
- [ ] Revoke the old leaked Kaggle API key. Never commit `generate_colab_notebook.py`, which contains a token.
- [ ] Apply the thesis updates (band-fix text, commit hash, Jan–Mar 2024 disclosure)
