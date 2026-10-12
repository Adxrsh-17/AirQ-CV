# AirQ-Astra — Classical ML / Statistical Models: Structure and Results

## 1. Purpose

This document summarizes the separate classical-ML / statistical forecasting branch created for AirQ-Astra.

The motivation was to test whether long-range 150-day forecasting can be handled more effectively using slow seasonal and structural signals rather than relying entirely on deep spatiotemporal dynamics.

The classical branch was developed separately under:

```text
classical_ml/
```

The intended 150-day forecasting task remains:

- History: H = 12 five-day windows
- Forecast horizon: K = 30 five-day windows
- Maximum lead: 150 days
- Train: 2019–2022
- Validation: 2023
- Test: 2024

---

## 2. Classical Modeling Strategy

Three pollutant-specific modeling ideas were implemented.

### 2.1 NO₂ branch

Concept:

```text
training climatology
      +
spatial coordinates
      +
seasonal calendar features
      +
land-use / S2-derived indices
      ↓
LightGBM / spatial regression
      ↓
NO₂ anomaly / field estimate
```

The intended idea is to model long-range NO₂ mainly from spatial structure and seasonality.

### 2.2 CO branch

Concept:

```text
historical CO raster sequence
      ↓
EOF / PCA decomposition
      ↓
small number of temporal coefficients
      ↓
SARIMA forecasting per mode
      ↓
reconstruction from EOF modes
      ↓
future CO spatial field
```

The motivation is that CO is more temporally persistent than the other channels and may retain more useful low-frequency temporal information.

### 2.3 SO₂ branch

Concept:

```text
training SO₂ historical maps
      ↓
persistent hotspot identification
      ↓
distance-to-hotspot / source features
      +
seasonal features
      +
spatial features
      ↓
LightGBM / distance-decay regression
      ↓
SO₂ field / risk estimate
```

This branch treats SO₂ as a sparse, point-source-dominated field.

---

## 3. Shared Feature Engineering

The classical pipeline includes or was designed around features such as:

- normalized spatial coordinates;
- cyclical day-of-year encoding:
  - `sin(2π × DOY / 365)`
  - `cos(2π × DOY / 365)`
- climatological mean;
- Sentinel-2-derived indices such as NDVI / NDBI / NDMI;
- CO temporal structure through EOF/PCA;
- SO₂ distance-to-hotspot features.

This is a useful modeling direction because long-range prediction may depend more on seasonal and structural information than on short-lived high-frequency dynamics.

---

## 4. Reported Climatology Baseline Results

The classical branch reported the following 2024 climatology R² values:

| Channel / intended pollutant label* | Reported climatology R² |
|---|---:|
| Band 0 / NO₂* | **0.4448** |
| Band 1 / CO* | **0.4154** |
| Band 2 / SO₂* | **0.0334** |

These numbers are substantially stronger than the normalized deep model's aggregate R² for Band 0 and Band 1.

However, they must currently be treated as **provisional classical-baseline results**, not final official 150-day benchmark numbers.

\*The pollutant names are the intended project mapping. V2 raster channel identity is not independently authenticated by the TIFF metadata.

---

## 5. What the Classical Pipeline Claimed

The classical execution reported that:

- monthly climatology was a strong predictor;
- the tested classical models did not outperform climatology;
- the long-range task is dominated by slow seasonal structure;
- CO was expected to be the strongest candidate for useful temporal forecasting;
- NO₂ and SO₂ were expected to be harder to forecast dynamically at 150 days.

It further claimed that climatology represented the "theoretical ceiling" for the 150-day problem.

That last conclusion is **not supported** by the current experiment.

The most defensible statement is:

> The tested classical models did not demonstrate a verified improvement over the reported climatology baseline under the current implementation.

---

## 6. Critical Review of the Current Classical Implementation

The current classical branch has several methodological issues that must be corrected before its results can be treated as official.

### 6.1 Future S2 leakage in NO₂

The feature-building code loads Sentinel-2 from the target date.

Conceptually, the current implementation behaves like:

```text
future target date
      ↓
same-date S2 features
      ↓
NO₂ prediction for that same date
```

For a genuine 150-day forecast, same-target-date S2 would not be available at the forecast origin.

Therefore the current NO₂ model is not a clean 150-day causal forecast.

### 6.2 Future S2 leakage in SO₂

The SO₂ regression also uses same-date S2-derived information such as NDBI.

That creates the same problem:

- target-period information is entering the model;
- the result cannot be treated as a strict forecast.

### 6.3 CO temporal horizon mismatch

The CO EOF+SARIMA implementation fits on the training block and then extrapolates over a much longer gap into 2024.

The evaluated temporal lead is therefore not consistently the intended K=30 / 150-day forecast from a rolling forecast origin.

This means the CO experiment is not yet directly comparable with the deep H=12/K=30 task.

### 6.4 Climatology support depends on S2 availability

The current tabular feature builder can skip dates when the corresponding S2 raster is unavailable.

That means even the climatology evaluation can be restricted to S2-available dates, despite climatology itself not requiring S2.

This changes evaluation support and makes model comparison unfair unless corrected.

### 6.5 Split-boundary issue

The current classical code uses start-date conditions rather than verifying complete interval containment.

For example, an interval starting near the end of 2022 but ending in 2023 can be incorrectly assigned to the training block.

The official V2 protocol requires both interval start and end to lie wholly inside the same split.

### 6.6 Random pixel-level validation

One RF helper uses random train/test splitting over pixel rows.

That can place pixels from the same atmospheric scenes in both training and validation.

For forecasting, chronological validation is required.

---

## 7. Why the Current Classical Numbers Are Provisional

The reported climatology values:

```text
Band 0: R² = 0.4448
Band 1: R² = 0.4154
Band 2: R² = 0.0334
```

should not yet be used as the final official 150-day comparison because:

1. target support may be restricted by S2 availability;
2. strict interval boundaries are not consistently enforced;
3. the NO₂/SO₂ feature models use future target-date S2;
4. the CO temporal experiment is not exactly a rolling 150-day forecast;
5. model results were not all saved in a common final metrics artifact;
6. the deep and classical branches have not yet been evaluated on exactly the same masks, target dates and horizons.

---

## 8. Correct Classical Structure for the Final Experiment

The corrected classical experiment should be:

```text
forecast origin t
│
├─ historical information available at or before t
│
├─ training-only climatology
│
├─ calendar features for target date
│
├─ static spatial coordinates
│
├─ source-distance features derived from TRAIN only
│
└─ causal historical S2 only
      where:
      S2 end + 20 days <= forecast origin

                    ↓

              model prediction

                    ↓

       next 30 five-day S5P maps
             up to t + 150d
```

Every model must be evaluated on exactly the same strict test sequences.

---

## 9. Recommended Corrected Pollutant-Specific Models

### 9.1 NO₂

Recommended corrected model:

```text
training climatology
      +
coordinates
      +
cyclical calendar features
      +
causally available historical/static spatial features
      ↓
LightGBM anomaly regression
```

Target:

```text
NO₂_target - training_climatology
```

Final prediction:

```text
climatology + predicted anomaly
```

No future S2 should be used.

### 9.2 CO

Recommended corrected model:

```text
rolling H=12 history
      ↓
train-fitted PCA / EOF
      ↓
temporal coefficient history
      ↓
SARIMA / regression
      ↓
30 future coefficient forecasts
      ↓
EOF reconstruction
```

The key correction is that forecasting must always begin from each valid forecast origin and produce exactly the next 30 steps.

### 9.3 SO₂

Recommended corrected model:

```text
TRAIN-only hotspot/source map
      +
distance-to-source
      +
calendar
      +
coordinates
      +
training climatology
      ↓
anomaly / risk model
```

No future S2 or future hotspot information should be used.

A classification formulation for high-SO₂ events may also be scientifically useful because the field is sparse.

---

## 10. Required Fair Comparison

The final comparison should use identical target support for:

```text
B0  Seasonal/monthly climatology
B1  Persistence
B2  Normalized deep model
C1  NO₂ classical anomaly model
C2  CO EOF/SARIMA
C3  SO₂ source-distance model
```

Report metrics separately at:

- 5 days
- 30 days
- 60 days
- 90 days
- 120 days
- 150 days

and aggregate across all 30 forecast horizons.

Recommended common metrics:

- MAE
- RMSE
- NRMSE
- R²
- Pearson r
- bias
- median absolute error
- q90 CSI / POD / FAR

Spatial metrics such as SSIM or FSS may be retained as secondary diagnostics.

---

## 11. Interpretation of the Current Classical Results

What can currently be stated:

- seasonality appears to contain useful long-range predictive information;
- the reported monthly climatology baseline is strong for Band 0 and Band 1;
- the current classical pipeline suggests long-range forecasting may benefit from simpler structural/seasonal models;
- current pollutant-specific classical models have not yet been demonstrated under a fully leakage-free and strictly comparable 150-day protocol.

What should **not** currently be stated:

- climatology is the theoretical ceiling;
- deterministic memory is exactly zero at 150 days;
- classical ML definitively fails;
- classical ML definitively beats deep learning;
- the reported R² values are already the final official 150-day benchmark.

---

## 12. Current Classical Model Status

```text
Classical branch:
Implemented and useful as an experimental research direction

NO₂:
LightGBM spatial/seasonal regression
Current issue: target-date S2 leakage

CO:
EOF/PCA + SARIMA
Current issue: forecast-horizon mismatch

SO₂:
Hotspot-distance / LightGBM regression
Current issue: target-date S2 leakage

Climatology:
Strong provisional baseline
Current issue: evaluation support must be made identical to all other models
```

The next classical step should be protocol correction, not additional tuning of the current leaking formulation.
