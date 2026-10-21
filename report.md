# Dual-Clock Land-Change / NO2 Architecture — Execution Report

Architecture doc: `IEEE_Dual_Clock_Land_Emission_Architecture_Proposal.txt` (attached in chat, not stored in repo).
Session: [`sessions/session_20261019_181942/`](sessions/session_20261019_181942/) (config, log, per-iteration artifacts).
Previous `report.md` (11 Oct) was replaced; copy kept at [`outputs/report_previous_20261011.md`](outputs/report_previous_20261011.md) and in git history.

**Headline.** Land change did **not** pass Gate G2, so the structural shift `B` is **0**. After iteration 4 (rolling memory added, user-requested), the frozen model scores **+4.7 pp on 2023 val** but only **+1.5 pp on 2024 test** (95% CI +0.1 to +3.1 pp), *below* damped persistence (+2.1 pp), and skill is slightly negative at 150 d. The val gain did not carry to test. The iteration-3 config (before memory) scored +2.3 pp on test; I did not pick it because choosing by test would use the test year for selection. 2024 is also **not clean** (see §8). Treat all skill gains as small and uncertain.

---

## 1. Objective and architecture summary

Test whether annual land change explains local NO2 change (C2), and whether that helps multi-horizon NO2 forecasts (fusion), while a fast anomaly model (C3) handles short leads.

```
Jan–Mar S2 composite -> C1 land features (NDVI,NDBI,BSI,MNDWI per 5km cell, dL vs 2019)
S5P NO2 -> regional common-mode normalisation -> annual Y[p,y] -> C2 two-way FE panel -> beta,theta -> gate G2 -> B
S5P NO2 (+past CO) -> anomalies vs climatology -> C3 Ridge/persistence -> A_hat
F[p,t+h] = Clim[p,s(t+h)] + B + g(h)*A_hat ;  10/50/90% = F + empirical out-of-fold residual quantiles
```
Leads 5,10,20,30,60,120,150 d = 1,2,4,6,12,24,30 five-day windows. `g(h)` is fitted on honest nested leave-one-year-out CV, clipped to [0,1], isotonic non-increasing in lead.
**Primary metric:** mean over 7 leads of `skill = 1 − MSE_model / MSE_ref`, ref = frozen climatology (smoothing k=2), valid Chennai pixels, target year 2023.

## 2. Data used (`data/processed` only, read-only)

| Item | Detail |
|---|---|
| S5P composites | 438 five-day windows 2019-01-01…2024-12-25, 3 bands (NO2, CO, SO2), 123×95 grid @0.0449° (whole Tamil Nadu). Exact 0 = missing → NaN (NO2 11.4% missing; 7.3% in crop). CO used only as a past-anomaly proxy. |
| S2 composites | 415 of 438 windows (23 cloud gaps), 12 bands @0.0054°. Used: Jan–Mar windows only (18–19 per year, 2019–2024), per-pixel median of cloud-free windows, then NDVI/NDBI/BSI/MNDWI, valid-pixel mean per S5P cell (valid fraction 99.9%). Collection per acquisition notebook: `S2_SR_HARMONIZED`. |
| Chennai crop | S5P grid rows 6–17, cols 83–94 (12×12, ≈12.76–13.30°N, 79.9–80.44°E). Inferred from `context.md` bbox; check it matches your intended 12×12. |
| Not used | `cached_dataset_256.pt`, `dataset.py`, `stream_utils.py`, SO2. |
| Missing vs spec | ERA5, VIIRS, AlphaEarth, Dynamic World, Open Buildings, any 2025 NO2, S5P processor-version tags (notebook shows `OFFL`, so mixed versions likely). |

Data hashes and package versions: `sessions/…/config.json`. Regression task, so no class balance.

## 3. Baseline vs final

Baseline = spec as written (k=2, Ridge α=10, B=0, nested-CV gate). Final = best config on val after iteration 4 (k=4, α=300, + past CO anomaly, + 30d/120d rolling anomaly memory).

| Metric (mean skill vs clim, 7 leads) | Baseline | Final (iter 4) | Damped persistence (same clim as final) | Iter-3 config, for reference |
|---|---|---|---|---|
| **Val 2023 (tuned)** | +0.0215 | **+0.0469** (CI +0.036…+0.058) | +0.0280 (CI +0.021…+0.036) | +0.0368 |
| **Test 2024 (frozen)** | +0.0031 (CI −0.005…+0.012) | **+0.0151** (CI +0.001…+0.031) | +0.0210 (CI +0.007…+0.034) | +0.0226 (CI +0.009…+0.039) |
| 80% interval coverage (mean) val / test | 0.800 / 0.834 | 0.793 / 0.827 | n/a | 0.793 / 0.829 |

Final skill by lead (days):

| Lead | 5 | 10 | 20 | 30 | 60 | 120 | 150 |
|---|---|---|---|---|---|---|---|
| Val 2023 | +.119 | +.045 | +.044 | +.043 | +.029 | +.034 | +.013 |
| Test 2024 | +.045 | +.016 | +.024 | +.016 | +.006 | +.001 | −.003 |
| Test persistence | +.050 | +.021 | +.022 | +.017 | +.012 | +.009 | +.016 |
| Test, iter-3 config | +.043 | +.017 | +.027 | +.025 | +.014 | +.013 | +.020 |

CIs: moving-block bootstrap (6 windows) over target windows, 1000 draws. Mean pinball (q10/q50/q90) test ≈ 0.377 (1e-5 mol/m² units). CRPS not computed. Part of the baseline→final gain is the smoother climatology, which also lifts persistence (0.0123→0.0280 on val), so fused-minus-persistence is the cleaner number: **+1.9 pp val, −0.6 pp test**. The learned gate on the test run is `g = [1.0, .78, .76, .76, .66, .66, .41]`, so long leads are still only partly shrunk even though the memory features did not help there.

Strata on test (final model): winter +0.059, pre-monsoon +0.030, NE monsoon +0.003, **SW monsoon −0.011**; dNDBI tercile low/mid/high +0.013 / +0.030 / +0.011. No sign that skill rises where land change is larger (H3 unsupported).

## 4. Iteration history (tuned on 2023 val only)

| # | Change | Val primary | Δ | Verdict |
|---|---|---|---|---|
| 0 | Baseline | +0.0215 | – | – |
| 1 | Climatology smoothing k 2→4 | +0.0341 | +0.0126 | kept |
| 2 | Ridge α 10→300 | +0.0342 | +0.00003 | kept (strictly higher, but noise) |
| 3 | Add past CO anomaly (pixel + city mean) | +0.0368 | +0.0027 | kept |
| – | Early stop triggered: gain < 0.5 pp twice in a row (iters 2, 3). | | | |
| 4 | Add 30d/120d rolling anomaly memory (**post-stop, user-requested**) | +0.0469 | +0.0100 | kept (on val) |

4 of 8 iterations used. Iteration 4 ran after the stop rule fired, because a val-only ablation had shown +1.0 pp.
Overfit/gap check: train out-of-fold skill is +0.008 vs val +0.047, so the val year is much easier than the training years, and the val gain of iteration 4 did not replicate on 2024 (test fell from +0.023 to +0.015). That points to val-selection noise: 73 val windows, a +1 pp gain, and a CI of roughly ±1 pp. Chart: [iteration_history.png](outputs/plots/iteration_history.png).

## 5. C2 land-to-NO2 attribution (Gate results)

- **G1 failed for Chennai alone**: 7 cells with persistent meaningful change (rule: |dNDVI| > 2×robust σ = 0.091 and dNDVI·dNDBI < 0, 2024 vs 2019; need ≥10). Panel expanded to the whole Tamil Nadu grid (11,562 cells, 6 years, 130 spatial blocks of 10×10 cells), not to other cities. Chennai stays the forecast target.
- Y = annual mean of NO2 divided by the same-window TN median (ERA5 substitute). Fixed effects: cell + year. Regressor dNDBI (+ queen-ring neighbour mean). NTL term dropped (no VIIRS).
- **β = 7.4e-7 mol/m² per unit dNDBI**, 95% block-bootstrap CI [2.6e-7, 1.2e-6]. One SD of dNDBI (0.051) ≈ 3.8e-8 mol/m², about 0.2% of mean NO2. θ (neighbours) CI includes 0. Chennai-only β = 1.3e-6, CI [−3.4e-6, 5.5e-6] (only 4 blocks, so the CI is not reliable).
- **Falsification:** placebo (permute cell trajectories, 300×) p = **0.07** → fails; lead-lag (future dNDBI vs past Y) β = 1.1e-7, CI includes 0 → passes; alt dNDVI (sign-flipped) and dBSI have CIs including 0; dropping 2020 Q2 gives β = 6.9e-7 (stable); raw NO2 β = 5.3e-7.
- **G2 = FAIL** (placebo p ≥ 0.05; alt features not significant) → **B = 0**. Detection limit (80% power) β ≈ 6.9e-7 for TN, 6.4e-6 for Chennai alone, so the TN estimate is at the detection limit. Result is "weak, fragile association; not accepted". It is an association, not causal.
- **Lockdown diagnostic flagged:** Chennai Apr–Jun 2020 vs mean of 2019/2021: raw −24.0%, normalised −19.4% (TN median itself −5.9%). Normalisation made the dip *smaller*, because the common-mode proxy removes real regional signal. Per spec this means the procedure needs inspection; it is a crude proxy, not a deweathering model.
- Not done: event studies (dates need public-source verification), processor-version sensitivity (no version metadata), Tier-2 Siamese U-Net, multi-city leave-one-city-out.

Charts: [c2_attribution.png](outputs/plots/c2_attribution.png), [c2_lockdown.png](outputs/plots/c2_lockdown.png), [chennai_maps.png](outputs/plots/chennai_maps.png).

## 6. Ablations (2023 val, mean skill)

| Variant | Skill |
|---|---|
| Best config (reference, iter 4) | +0.0469 |
| Without rolling 30d/120d memory | +0.0368 |
| Without CO proxy | +0.0442 |
| + latest land state L (NDVI, NDBI), static context | +0.0459 (no gain) |
| Level-2 = LightGBM instead of Ridge | +0.0260 (worse) |
| + structural shift B (in-sample β, optimistic) | +0.0446 (worse, consistent with G2 reject) |

Not run: with/without ERA5, forecast-met variant, CNN/ConvGRU (Level 2 barely beats persistence, so the spec rule says do not escalate), full vs no-2020Q2 for C3, Tier 1 vs Tier 2.

Charts: [skill_vs_lead.png](outputs/plots/skill_vs_lead.png), [calibration_gate.png](outputs/plots/calibration_gate.png), [skill_strata.png](outputs/plots/skill_strata.png), [example_forecast_manali.png](outputs/plots/example_forecast_manali.png).

## 7. Best configuration and reproduction

Best config ([`outputs/best_config.json`](outputs/best_config.json)): `k=4, alpha=300, model=ridge, co=true, hist=true, land=false, B=0, seed=42`. To get the iteration-3 config, set `hist=false` in that file and run `finalize.py` (its test numbers are in §3).
Outputs: [`outputs/best_model.pkl`](outputs/best_model.pkl) (trained on 2019–2023), [`outputs/best_output.csv`](outputs/best_output.csv) (val 2023 + test 2024 forecasts with q10/q50/q90, truth, climatology, persistence), `outputs/final_metrics.json`, `outputs/ablations_val2023.json`, `outputs/c2_results.json`, `outputs/iteration_history.csv`.

Run from anywhere (paths resolve from the script location; ~4 min total):
```bash
python outputs/code/build_cube.py        # processed tifs -> outputs/cube/*.npz
python outputs/code/c2_attribution.py    # C2 + G1/G2 -> outputs/c2_results.json
python outputs/code/run_iterations.py    # baseline + iterations -> session log.md, config.json
python outputs/code/iteration4.py        # post-stop iteration 4 (rolling memory); updates best_config.json
python outputs/code/finalize.py          # freeze, score 2024, ablations, plots, best_model/output
```
`run_iterations.py` reads the session folder from `outputs/code/.session_path`; write a new path there to start a fresh session. Packages and versions are in `config.json` (Python 3.12.7, numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, lightgbm 4.7.0, rasterio 1.5.0). Note: `finalize.py` has now scored 2024 three times: twice for the iteration-3 config (identical results; the rerun only fixed a NaN in a val-only ablation) and once for the iteration-4 config. The test year has therefore been looked at for two candidate configs.

## 8. Limitations, risks, next steps

**Limitations**
- **2024 is not a clean test.** It was used in earlier experiments and no 2025 data exists in `processed`. The spec's clean-2025 requirement is unmet.
- **No ERA5.** Mode B (past meteorology), H4 and H5 are untestable; CO anomalies are only a weak proxy. "Deweathering" is regional-median ratio normalisation, which removed part of the lockdown signal.
- **C1 is S2-index only**, from 6 annual composites (5 change years); no VIIRS, AlphaEarth, DW or OBT, so no true built-up fraction. Tier 2 not attempted.
- Effective sample for C2 is tens of independent blocks, not 69k observations; the effect is at the detection limit.
- NO2 processor versions are unknown (likely OFFL mix); a version artefact could mimic or hide a land effect.
- C3 uses time splits only; no spatial-block validation across pixels (only 144 correlated cells). Chennai crop choice is inferred.
- Quantile intervals are pooled, non-heteroscedastic residual quantiles: coverage ≈ 80% on average but not checked per season.
- ~11% of NO2 cells are missing (cloud). Skill is measured on valid pixels only, so it is biased toward clear-sky windows.

**Recommended next steps**
1. Decide between the iteration-3 and iteration-4 configs using multi-year rolling-origin validation (e.g. validate on each of 2021, 2022, 2023), not a single 73-window val year.
2. Obtain 2025 S5P NO2 (clean test) and ERA5 (past + forecast met); redo deweathering with LightGBM and re-check the lockdown dip.
3. Pull VIIRS, Dynamic World and Open Buildings for a real built-up measure; audit S5P processor versions or use PAL/RPRO.
4. Add verified event studies (Sriperumbudur–Oragadam, Ennore) and a leave-one-region-out check on the TN panel.
5. Skip CNN/ConvGRU until Level 2 clearly beats persistence on a clean test.
