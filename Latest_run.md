# Latest run: DC-Net v2 (Stage B), Config C, Kaggle, 2026-10-25

Code: commit `0f2243e` (model/trainer/evaluator) on `feat/dual-clock-architecture`. Raw outputs: `results/run_2026-10-25_stageB/`.
Per Rule.md section 7. Validation year 2024 only (also the checkpoint-selection year; no 2025 data yet, so this is **validation, not a frozen test**, B1).

## Setup
| Item | Value |
|---|---|
| Model | DC-Net v2, 3,076,935 parameters, pollutants NO2+CO+SO2 jointly |
| Config | C = S2 land use + Stage A prior + S5P anomaly history + ERA5 history (no S2-off ablation run yet) |
| Horizons | 5, 10, 15, 30, 60, 120 days (steps 1, 2, 3, 6, 12, 24) |
| Split | train 2019-2023, validate 2024; 354 train origins, 72 val origins; 18% of the grid held out from the loss as 8-cell blocks |
| Training | 17 epochs (early stop, patience 10), best epoch 6, about 160 s/epoch on one GPU (DataParallel off), about 58 min total |
| Data checks | `check_cache.py`: no non-finite S2 values, no all-zero weather windows |
| Weather | history only; ERA5 `blh` for 2024-01-01..06-30 is imputed from training climatology (known gap) |

## Stage A priors (S2-only, spatial out-of-fold, 8-cell blocks, buffer 2, years 2019-2023)
| Pollutant | R² (OOF) | Spearman | Note |
|---|---|---|---|
| NO2 | 0.624 | 0.781 | log target |
| CO | 0.683 | 0.786 | log target |
| SO2 | 0.238 | 0.623 | raw target |

Inputs: S2 bands, NDBI/NDVI/MNDWI/NDRE, built/veg/water fractions, neighbourhood context. No coordinates, time, weather or S5P history. Stage A earlier showed location alone (geo kNN) scores higher at 20-40 km blocks (0.81/0.80 vs 0.58/0.62 for S2-only) and S2 only catches up at 80 km. The prior maps are blocky (fold seams) and smooth; industrial point sources are under-predicted.

## Forecast skill, 2024 validation, all valid pixels
SS_clim = 1 - MSE(model)/MSE(per-pixel calendar-month climatology of the training years), physical units; CI = 95% block bootstrap (blocks of 6 origins). SSIM and CSI at the training monthly p90.

| Pollutant | Lead (d) | SS_clim [95% CI] | R² model | R² climatology | Anomaly r | SSIM model / clim | CSI p90 | Cov80 |
|---|---|---|---|---|---|---|---|---|
| NO2 | 5 | -0.005 [-0.06, 0.07] | 0.455 | 0.458 | 0.184 | 0.249 / 0.252 | 0.306 | 0.80 |
| CO | 5 | -0.036 [-0.14, 0.09] | 0.758 | 0.766 | 0.252 | 0.223 / 0.260 | 0.131 | 0.83 |
| SO2 | 5 | -0.021 [-0.06, 0.02] | -0.026 | -0.006 | 0.171 | 0.052 / 0.045 | 0.081 | 0.82 |
| NO2 | 10 | -0.030 [-0.08, 0.03] | 0.438 | 0.455 | 0.113 | 0.250 / 0.250 | 0.291 | 0.79 |
| CO | 10 | -0.065 [-0.22, 0.10] | 0.751 | 0.766 | 0.192 | 0.243 / 0.260 | 0.111 | 0.82 |
| SO2 | 10 | -0.026 [-0.06, 0.01] | -0.031 | -0.005 | 0.154 | 0.054 / 0.045 | 0.084 | 0.81 |
| NO2 | 15 | -0.021 [-0.06, 0.04] | 0.440 | 0.452 | 0.116 | 0.250 / 0.248 | 0.292 | 0.79 |
| CO | 15 | -0.042 [-0.19, 0.10] | 0.756 | 0.766 | 0.187 | 0.264 / 0.259 | 0.090 | 0.83 |
| SO2 | 15 | -0.036 [-0.07, 0.00] | -0.050 | -0.013 | 0.137 | 0.054 / 0.045 | 0.085 | 0.81 |
| NO2 | 30 | -0.021 [-0.06, 0.03] | 0.434 | 0.446 | 0.110 | 0.241 / 0.241 | 0.285 | 0.79 |
| CO | 30 | -0.011 [-0.14, 0.14] | 0.761 | 0.764 | 0.208 | 0.259 / 0.257 | 0.092 | 0.82 |
| SO2 | 30 | -0.034 [-0.06, 0.01] | -0.049 | -0.014 | 0.132 | 0.055 / 0.046 | 0.082 | 0.80 |
| NO2 | 60 | -0.019 [-0.06, 0.03] | 0.415 | 0.426 | 0.105 | 0.226 / 0.228 | 0.276 | 0.78 |
| CO | 60 | 0.012 [-0.20, 0.22] | 0.757 | 0.754 | 0.249 | 0.250 / 0.250 | 0.116 | 0.82 |
| SO2 | 60 | -0.023 [-0.06, 0.02] | -0.041 | -0.018 | 0.142 | 0.054 / 0.046 | 0.075 | 0.78 |
| NO2 | 120 | 0.024 [-0.03, 0.05] | 0.367 | 0.351 | 0.154 | 0.196 / 0.193 | 0.264 | 0.75 |
| CO | 120 | 0.052 [-0.13, 0.23] | 0.737 | 0.723 | 0.276 | 0.238 / 0.232 | 0.114 | 0.82 |
| SO2 | 120 | 0.005 [-0.02, 0.04] | -0.034 | -0.039 | 0.172 | 0.055 / 0.045 | 0.072 | 0.78 |

Reading: the forecast equals climatology. Every CI includes 0 (best point estimates: CO 120 d +0.052, NO2 120 d +0.024). SS_pers is 0.3-0.9 but raw last-observation persistence is a very weak baseline, so it is not informative.

## Hotspot blobs (blobs >= 2 px on the 5.5 km grid, IoU >= 0.3 match)
| Pollutant | Lead (d) | F1 model | F1 clim | F1 pers | Precision | Recall | IoU (matched) | Centroid err (km) |
|---|---|---|---|---|---|---|---|---|
| CO | 5 | 0.032 | 0.012 | 0.027 | 0.05 | 0.02 | 0.47 | 8.3 |
| CO | 10 | 0.019 | 0.011 | 0.026 | 0.05 | 0.01 | 0.47 | 9.7 |
| CO | 15 | 0.022 | 0.011 | 0.036 | 0.08 | 0.01 | 0.47 | 10.3 |
| CO | 30 | 0.012 | 0.010 | 0.030 | 0.08 | 0.01 | 0.49 | 16.7 |
| CO | 60 | 0.015 | 0.007 | 0.031 | 0.08 | 0.01 | 0.49 | 18.0 |
| CO | 120 | 0.011 | 0.008 | 0.021 | 0.08 | 0.01 | 0.55 | 17.6 |
| NO2 | 5 | 0.110 | 0.110 | 0.081 | 0.37 | 0.06 | 0.50 | 13.0 |
| NO2 | 10 | 0.103 | 0.109 | 0.068 | 0.39 | 0.06 | 0.49 | 13.3 |
| NO2 | 15 | 0.102 | 0.106 | 0.066 | 0.40 | 0.06 | 0.49 | 13.7 |
| NO2 | 30 | 0.097 | 0.101 | 0.053 | 0.38 | 0.06 | 0.48 | 14.3 |
| NO2 | 60 | 0.092 | 0.092 | 0.047 | 0.36 | 0.05 | 0.47 | 14.0 |
| NO2 | 120 | 0.081 | 0.072 | 0.040 | 0.36 | 0.05 | 0.47 | 14.1 |
| SO2 | 5 | 0.014 | 0.011 | 0.053 | 0.17 | 0.01 | 0.42 | 12.8 |
| SO2 | 10 | 0.015 | 0.011 | 0.049 | 0.19 | 0.01 | 0.44 | 13.5 |
| SO2 | 15 | 0.013 | 0.011 | 0.051 | 0.16 | 0.01 | 0.45 | 13.0 |
| SO2 | 30 | 0.013 | 0.012 | 0.050 | 0.18 | 0.01 | 0.44 | 13.5 |
| SO2 | 60 | 0.015 | 0.012 | 0.052 | 0.21 | 0.01 | 0.48 | 10.8 |
| SO2 | 120 | 0.009 | 0.012 | 0.050 | 0.16 | 0.00 | 0.47 | 8.4 |

Reading: model blob F1 is about equal to the climatology map's, so the blobs in the panels come from the climatology/static pattern. Observed maps have about 60 speckle blobs per map (retrieval noise), which caps F1; matched blobs are well placed (IoU about 0.5, centroid error 8-18 km).

## Spatial hold-out blocks (excluded from the loss; climatology still includes their history)
| Pollutant | Lead (d) | SS_clim (held-out blocks) | R² |
|---|---|---|---|
| NO2 | 5 | -0.016 | 0.512 |
| CO | 5 | -0.039 | 0.765 |
| SO2 | 5 | -0.086 | -0.051 |
| NO2 | 10 | -0.037 | 0.499 |
| CO | 10 | -0.058 | 0.760 |
| SO2 | 10 | -0.086 | -0.050 |
| NO2 | 15 | -0.021 | 0.504 |
| CO | 15 | -0.029 | 0.766 |
| SO2 | 15 | -0.087 | -0.057 |
| NO2 | 30 | -0.008 | 0.503 |
| CO | 30 | -0.005 | 0.769 |
| SO2 | 30 | -0.068 | -0.041 |
| NO2 | 60 | -0.004 | 0.486 |
| CO | 60 | 0.015 | 0.765 |
| SO2 | 60 | -0.052 | -0.021 |
| NO2 | 120 | 0.035 | 0.432 |
| CO | 120 | 0.060 | 0.744 |
| SO2 | 120 | -0.017 | 0.008 |

## Comparison with earlier runs (same 2024 year; read the caveats)
| Run | NO2 R² 5 d / 120 d | CO R² 5 d / 120 d | Note |
|---|---|---|---|
| Old DC-Net (`runs/dcnet_test_metrics.csv`, commit `dddc41c`) | 0.466 / 0.354 | 0.765 / 0.698 | no long-memory dropout, per-pixel learned SeasonalPrior; its "skill vs climatology +46%" used a broken baseline (implied climatology R² about 0); against the correct baseline it is about +1.5% |
| DC-Net v2 (this run) | 0.455 / 0.367 | 0.758 / 0.737 | starts at climatology; skill vs climatology about 0 |
| ST-ResUNet champion (README `8ae7400`) | 0.279 | 0.448 | Chennai-corridor tiles, 23 sequences, no climatology baseline: not comparable |

## Verdict against Rule.md
- Passes: OOF prior with a buffer ring, temporal split, history-only weather, baselines and CIs reported, calibrated 80% intervals (0.75-0.83), no NaNs.
- Open: Config B (no S2/prior) ablation and an S2-shuffle check not yet run, so the forecaster's use of S2 is unproven (S2 is wired in; the prior carries the evidence); 2024 is both selection and report year (B1); `blh` 2024 H1 imputed (B4/B5); Stage A stacking and scale-ablation results not yet logged.
- Claim supported by this run: S2 alone recovers the NO2/CO hotspot map on unseen blocks (R² 0.62/0.68, log scale); the forecast reproduces the seasonal climatology with calibrated uncertainty and no significant gain over it.

## Known issues and next steps
1. Initial damped-persistence weight (exp(-h/15)) hurt at start (SS_clim -0.35 at 5 d, epoch 0); initialise from the measured anomaly autocorrelation instead.
2. q50 trained with weighted pinball loss but scored by MSE; use squared/Huber loss for q50 without hotspot weighting.
3. Early stop at epoch 16 fired before the cosine learning-rate decay; shorten the schedule, add regularisation.
4. Trainer scores against a harmonic climatology in z-space, evaluator against calendar-month means in physical units; unify.
5. Blob metric counts speckle; define hotspots on 3x3-smoothed maps with a minimum area of 4 px.
6. Run `ABLATE_B=True` and the S2-shuffle check; score the S2-only prior directly with the blob metric.
7. Region box (Kongu: Coimbatore, Palakkad, Tiruppur, Erode, Salem, Namakkal) scoring is implemented (`evaluate.py --bbox kongu`, commit `6c27a7e`) and not yet run; `MODE='eval'` re-evaluates this checkpoint without retraining (commit `f195481`).
