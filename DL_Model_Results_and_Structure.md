# AirQ-Astra — Deep Learning Model: Structure and Results

## 1. Purpose

This document summarizes the deep-learning forecasting pipeline implemented for the AirQ-Astra V2 dataset and the results obtained so far.

The V2 task is a long-range multi-horizon forecasting problem:

- Input history: **H = 12** five-day composites
- Historical coverage: **60 days**
- Forecast horizon: **K = 30** five-day composites
- Maximum forecast lead: **150 days**
- Spatial S5P grid: **3 × 123 × 95**
- Chronological split:
  - Train: 2019–2022
  - Validation: 2023
  - Test: 2024
- Strict sequence counts:
  - Train: **251**
  - Validation: **31**
  - Test: **31**

The current V2 archive is treated as a new dataset contract. The earlier V1 model trained on 13 × 13 S5P maps is not reused as the V2 model.

---

## 2. Deep-Learning Pipeline Structure

### 2.1 Input representation

The current deep model uses the V2 S5P raster sequence as its primary spatiotemporal input.

Conceptually:

```text
12 historical five-day S5P maps
        ↓
S5P tensor
[H=12, C=3, 123, 95]
        ↓
channel-wise normalization
        ↓
spatiotemporal neural encoder
        ↓
ConvLSTM / temporal-spatial latent representation
        ↓
multi-horizon decoder
        ↓
30 future S5P maps
[K=30, C=3, 123, 95]
```

The model is intended to forecast all future horizons up to 150 days.

The current pipeline is an S5P-based forecasting baseline. Sentinel-2 is not treated as direct pollution measurement. Any future multimodal version must use only causally legal S2 information.

---

## 3. Normalization

### 3.1 Initial problem

The first training run used raw S5P values directly.

The stored S5P values are numerically small for some channels. This caused poor optimization behavior and the first model produced very weak predictions.

Initial evaluation was approximately:

| Band | R² | Pearson r | Main observation |
|---|---:|---:|---|
| Band 0 | -615.85 | -0.0303 | Severe failure |
| Band 1 | -0.0756 | 0.1503 | Weak |
| Band 2 | -17.60 | 0.0161 | Severe failure |

This first run should be treated only as a failed unnormalized baseline.

### 3.2 Corrected normalization

The training pipeline was then changed to use channel-wise Z-score normalization:

```text
x_norm = (x - mean_train) / std_train
```

with inverse scaling during evaluation:

```text
x_physical = x_norm × std_train + mean_train
```

The normalization statistics were computed from the training data and then used consistently for training and evaluation.

This change substantially improved optimization and predictive skill.

---

## 4. Training Setup Used

The corrected model was trained using GPU acceleration.

Reported training configuration / execution characteristics:

- GPU: **NVIDIA GeForce RTX 4070 Laptop GPU**
- Mixed precision / GPU training used
- Parallel data loading and preloading were introduced
- Batch size was increased during optimization
- CuDNN autotuning was enabled
- Total epochs: **50**
- Best checkpoint:
  - `experiments/v2_baseline/best_model.pt`

The final accelerated run completed in approximately **17.5 minutes** after the data-loading and GPU optimizations were introduced.

The reported best validation checkpoint was saved near the end of the 50-epoch run.

---

## 5. Full Holdout Evaluation After Normalization

The following results were reported after retraining with normalized data and inverse-scaling predictions back to the stored physical scale.

### 5.1 Standard regression metrics

| Band | Intended pollutant label* | R² | RMSE | MAE | Pearson r | Mass error |
|---|---|---:|---:|---:|---:|---:|
| 0 | NO₂* | **0.2052** | 1.1538 × 10⁻⁵ | 8.2571 × 10⁻⁶ | **0.4532** | **+0.83%** |
| 1 | CO* | **-0.0307** | 1.1195 × 10⁻² | 6.5025 × 10⁻³ | **0.3312** | **+14.01%** |
| 2 | SO₂* | **-0.2190** | 1.4448 × 10⁻⁴ | 6.9984 × 10⁻⁵ | **0.1186** | **-90.05%** |

\*The pollutant names reflect the intended project mapping. The current V2 archive generation provenance is unauthenticated, so the stored raster channel identities and units are not independently authenticated by the raster files themselves.

### 5.2 Spatial metrics

| Band | Mean SSIM | FSS @ 9×9 | Interpretation |
|---|---:|---:|---|
| Band 0 | **0.3940** | **0.3500** | Some useful spatial structure retained |
| Band 1 | **0.2973** | **0.5622** | Strongest neighborhood-scale spatial score |
| Band 2 | **0.1661** | **0.0000** | Weak spatial reconstruction |

### 5.3 Extreme-event metrics

Using a 90th-percentile threshold from observed values:

| Band | q90 threshold | CSI | POD | FAR |
|---|---:|---:|---:|---:|
| Band 0 | 3.1314 × 10⁻⁵ | 0.1098 | 12.00% | 43.53% |
| Band 1 | 3.7792 × 10⁻² | **0.2699** | **47.09%** | 61.25% |
| Band 2 | 2.2249 × 10⁻⁴ | 0.0000 | 0.00% | 0.00% |

---

## 6. Five-Day Forecast Result

A separate 5-day forecast was reported for a holdout scene corresponding to an observation around early March 2024.

### 6.1 Five-day metrics

| Pollutant label* | R² | Pearson r | SSIM | RMSE | MAE | Mass bias |
|---|---:|---:|---:|---:|---:|---:|
| NO₂* | **0.3449** | **0.7263** | **0.7560** | 1.115 × 10⁻⁵ | 6.072 × 10⁻⁶ | -19.14% |
| CO* | **0.0923** | **0.3159** | **0.4721** | 7.109 × 10⁻³ | 3.110 × 10⁻³ | **+1.49%** |
| SO₂* | -0.2336 | 0.2106 | 0.2039 | 1.714 × 10⁻⁴ | 1.002 × 10⁻⁴ | -74.53% |

The short-horizon result is substantially better than the long-horizon aggregate result, especially for Band 0.

---

## 7. Interpretation of the Deep Model

### 7.1 Band 0

Band 0 is currently the strongest overall deep-learning result.

Evidence:

- Positive full-holdout R²: **0.2052**
- Pearson correlation: **0.4532**
- Five-day Pearson correlation: **0.7263**
- Five-day SSIM: **0.7560**

This suggests that the model can recover meaningful short-term spatial-temporal structure, but skill decreases substantially at longer forecast horizons.

### 7.2 Band 1

Band 1 shows:

- Full-holdout R² slightly below zero
- Pearson r = **0.3312**
- FSS = **0.5622**
- Stronger extreme-event recall than the other bands

The model appears better at reproducing broad spatial placement than exact pixel magnitudes for this channel.

### 7.3 Band 2

Band 2 remains the weakest channel.

Evidence:

- Negative R²
- Low Pearson correlation
- Very large negative mass bias
- Zero detected q90 events in the reported aggregate evaluation

The model tends to underpredict high-value sparse events.

---

## 8. Current Limitations

The deep-learning result should be interpreted as a baseline, not as a final production model.

Important limitations include:

1. **Long forecast horizon**
   - 150 days is much harder than the 5-day horizon.
   - Predictive skill decreases with forecast lead.

2. **Band 2 sparsity**
   - The third channel has a large fraction of exact zeros in the V2 archive.
   - Sparse extreme events are difficult to reproduce with a standard regression loss.

3. **Loss-function smoothing**
   - MAE/L1-style training can favor smooth predictions and suppress extreme peaks.

4. **Provenance limitation**
   - The new V2 archive has unauthenticated generation provenance.
   - Therefore exact physical band identity, units, QA rules and processing history should be described cautiously.

5. **Evaluation protocol**
   - Final reporting should compare the deep model against persistence and climatology on exactly the same strict H=12/K=30 target support.

---

## 9. Recommended Deep-Learning Improvements

The next deep-learning iterations should be evaluated without changing the locked test protocol.

Recommended directions:

- horizon-weighted multi-step loss;
- channel-specific loss weights;
- extreme-aware loss for sparse Band 2;
- residual prediction relative to climatology;
- seasonal/calendar embeddings;
- stronger direct multi-horizon decoder;
- delayed-causal Sentinel-2 branch;
- historical meteorology available at forecast origin;
- actual meteorological forecast fields if available;
- uncertainty estimates for long-range predictions.

A spatial loss such as SSIM may be explored, but it should be used as a secondary structural term rather than replacing physical-scale regression metrics.

---

## 10. Deep Model Status

Current status:

```text
Model family:
ConvLSTM-based / temporal-spatial deep forecasting baseline

Input:
12 historical 5-day S5P composites

Output:
30 future 5-day S5P composites

Maximum lead:
150 days

Best demonstrated behavior:
Short-range Band 0 forecasting

Major weakness:
Long-range Band 2 / sparse-extreme prediction
```

The normalized deep model is a meaningful V2 baseline, but it does not yet establish strong 150-day skill across all three channels.
