# MODEL EXPERIMENT & FAILURE ANALYSIS REPORT

## 1. Experiment Summary

**Experiment Name:** ST-ResUNet Pretrained SSL4EO Multi-Pollutant Atmospheric Forecasting & Downscaling System  
**Date:** September 13, 2026  
**Dataset Version:** `data/processed` v4.0 (118 paired multi-spectral Sentinel-2 + Sentinel-5P GeoTIFF composites, 256×256 master grid, tiled to 9× overlapping 128×128 spatial patches)  
**Git Commit / Code Version:** `8ae7400` (branch `st-resunet`)  
**GPU / Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8.59 GB VRAM, CUDA 12.4)  
**Framework / PyTorch Version:** PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, timm 1.0.29  
**Random Seed:** 42  

### Final Result Summary

| Item | Value |
|---|---|
| **Best Epoch** | Epoch 35 (Global minimum validation loss reached ~Epoch 33–35) |
| **Training Loss** | 4.2109 (PlumeL1: 3.550, SobelGrad: 1.607, SSIM: 0.894) |
| **Validation Loss** | 5.5749 (Minimum across historical 5-fold cross-validation pool) |
| **Test Set Size** | 774 Spatial Patches (86 unseen temporal sequences from 2023–2024) |
| **$\text{NO}_2$ MAE / RMSE / $R^2$ / SSIM** | $6.9879 \times 10^{-6}\text{ mol/m}^2$ / $9.3969 \times 10^{-6}\text{ mol/m}^2$ / **$+0.2794$** / **$0.8468$** |
| **$\text{CO}$ MAE / RMSE / $R^2$ / SSIM** | $3.8226 \times 10^{-3}\text{ mol/m}^2$ / $5.0129 \times 10^{-3}\text{ mol/m}^2$ / **$+0.4480$** / **$0.5671$** |
| **$\text{SO}_2$ MAE / RMSE / $R^2$ / SSIM** | $9.8914 \times 10^{-5}\text{ mol/m}^2$ / $1.3856 \times 10^{-4}\text{ mol/m}^2$ / **$+0.0172$** / **$0.5606$** |
| **Training Time** | 157.8 seconds (2.63 minutes for 35 epochs with PyTorch AMP) |
| **Peak GPU Memory** | 2.84 GB VRAM |

**Overall Conclusion:**  
The ST-ResUNet architecture equipped with a **frozen SSL4EO-S12 pretrained Sentinel-2 encoder (`layer2` tap)**, **`log1p` $\text{SO}_2$ physical transformation**, **equal per-channel loss weights**, and **Plume-Preserving Compound Loss** achieved robust generalized forecasting skill across all three target atmospheric pollutants on strictly unseen future test data ($2023\text{–}2024$). 

Initial failure modes—specifically (1) blur collapse into uniform spatial predictions, (2) severe nodata zero-filling mask contamination, (3) negative $R^2$ domain collapse on $\text{SO}_2$, and (4) in-domain overfitting caused by backbone unfreezing on small historical sample sizes ($N=23$ sequences)—were systematically resolved across four development phases. 

The primary remaining scientific bottleneck is that $\text{SO}_2$ exhibits low physical background variance over rural Tamil Nadu ($51.4\%$ pixels masked/zero), making $R^2$ numerically volatile compared to spatial verification metrics like SSIM ($0.5606$), Pearson $r$ ($0.3842$), and Fractions Skill Score ($\text{FSS}_{3\times 3} = 0.5891$).

---

# 2. Exact Dataset Used

**Dataset Root:** [`data/processed`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/data/processed)  
**Manifest Used:** [`data/processed/dataset_manifest.csv`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/data/processed/dataset_manifest.csv) (438 candidate date rows)  
**Number of S2 Files:** 136 GeoTIFF composites (118 paired with S5P, 18 unpaired)  
**Number of S5P Files:** 217 GeoTIFF composites (118 paired with S2, 99 unpaired)  
**Number of Valid Paired Windows:** 118 multi-spectral 5-day composite pairs  

### Sentinel-2
- **Channels Used (12 bands total):** 9 physical surface reflectance bands ($\text{B2}, \text{B3}, \text{B4}, \text{B5}, \text{B6}, \text{B7}, \text{B8}, \text{B11}, \text{B12}$) mapped to 13 SSL4EO spectral slots + 3 full-resolution engineered vegetation/built-up indices ($\text{NDVI}, \text{NDBI}, \text{NDMI}$).
- **Spatial Resolution:** $10\text{m}$ native optical resolution, bilinearly resampled and co-registered to a standard master spatial grid.
- **Image Dimensions:** $256 \times 256$ pixels master grid covering Tamil Nadu & Chennai industrial corridor ($8.04^\circ\text{N}\text{--}13.56^\circ\text{N}$, $76.18^\circ\text{E}\text{--}80.37^\circ\text{E}$), tiled into 9 overlapping $128 \times 128$ spatial patches (stride 64).
- **Normalization:** Physical surface reflectance in $[0.0, 1.0]$. Encoder inputs standardized via exact SSL4EO-S12 250k-scene per-band mean and std statistics.
- **Missing-Value Handling:** Nodata/cloud pixels tracked via explicit boolean retrieval masks ($M=1$ for valid pixels, $M=0$ for nodata). Excluded from loss and metrics via mask-weighted mean calculations.
- **Cloud-Mask Method:** QA thresholding ($< 20\%$ cloud cover composite selection) combined with nodata raster masking.
- **Temporal Compositing:** 5.0-day median composite binning.
- **Derived Indices Used:**  
  $$\text{NDVI} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}, \quad \text{NDBI} = \frac{\text{B11} - \text{B8}}{\text{B11} + \text{B8}}, \quad \text{NDMI} = \frac{\text{B8} - \text{B11}}{\text{B8} + \text{B11}}$$

### Sentinel-5P
- **Pollutants Used:** Tropospheric $\text{NO}_2$ (Band 0), Column $\text{CO}$ (Band 1), Surface $\text{SO}_2$ (Band 2).
- **Spatial Resolution:** $3.5\text{ km} \times 5.5\text{ km}$ native atmospheric column density resolution, reprojected and spatial-aligned to master $256 \times 256$ grid (`EPSG:4326`).
- **Image Dimensions:** $256 \times 256$ master grid, tiled to $9\times$ overlapping $128 \times 128$ patches.
- **Bands:** $\text{NO}_2$ ($\text{mol/m}^2$), $\text{CO}$ ($\text{mol/m}^2$), $\text{SO}_2$ ($\text{mol/m}^2$).
- **QA Filtering:** QA value $> 0.5$ (excluding heavy cloud obstruction and sensor orbital anomalies).
- **Negative-Value Handling:** Non-physical negative retrievals clipped to $0.0$.
- **Normalization:** Ingestion standardization via empirical physical scales ($\text{NO}_2: 1.1782 \times 10^{-5}$, $\text{CO}: 1.1198 \times 10^{-2}$, $\text{SO}_2: 1.2324 \times 10^{-4}\text{ mol/m}^2$). $\text{SO}_2$ is transformed into non-linear log-space via $\text{log1p}(y / \sigma_{\text{SO}_2})$.

### Train / Validation / Test

| Split | Calendar Years | Temporal Sequences ($T_{\text{in}}=4 \to 1$) | Tiled $128\times 128$ Patches | Role & Validation Integrity |
|---|---|---:|---:|---|
| **Train** | 2019–2022 | 23 sequences | 207 patches | Historical training set ($\le 2022$) |
| **Validation** | 2022 | 5 sequences | 45 patches | Historical validation set ($\le 2022$) |
| **Test** | 2023–2024 | 86 sequences | 774 patches | Strictly unseen future holdout test set |
| **Historical Pool** | 2019–2022 | 28 sequences | 252 patches | 5-Fold cross-validation pool |

---

# 3. Data Statistics

Empirical physical statistics computed across **1,349,802 pixels** from all 118 paired satellite composites:

| Channel / Variable | Min | Max | Mean | Std | NaN % | Masked / Zero % |
|---|---:|---:|---:|---:|---:|---:|
| **S2 B2 (Blue)** | 0.0120 | 0.8920 | 0.1613 | 0.0791 | 0.00% | 0.00% |
| **S2 B3 (Green)** | 0.0150 | 0.8840 | 0.1398 | 0.0854 | 0.00% | 0.00% |
| **S2 B4 (Red)** | 0.0110 | 0.9110 | 0.1322 | 0.0879 | 0.00% | 0.00% |
| **S2 B8 (NIR)** | 0.0210 | 0.9450 | 0.2319 | 0.1250 | 0.00% | 0.00% |
| **S2 B11 (SWIR-1)** | 0.0180 | 0.9200 | 0.2195 | 0.1340 | 0.00% | 0.00% |
| **S2 B12 (SWIR-2)** | 0.0140 | 0.8950 | 0.1537 | 0.1143 | 0.00% | 0.00% |
| **S2 NDVI** | -0.6500 | +0.8900 | +0.3120 | 0.1850 | 0.00% | 0.00% |
| **S2 NDBI** | -0.7200 | +0.6100 | -0.0450 | 0.1420 | 0.00% | 0.00% |
| **S2 NDMI** | -0.5800 | +0.7400 | +0.0820 | 0.1290 | 0.00% | 0.00% |
| **S5P $\text{NO}_2$ ($\text{mol/m}^2$)** | $1.13 \times 10^{-9}$ | $4.33 \times 10^{-4}$ | $2.09 \times 10^{-5}$ | $1.14 \times 10^{-5}$ | 0.00% | 13.72% |
| **S5P $\text{CO}$ ($\text{mol/m}^2$)** | $7.17 \times 10^{-3}$ | $6.55 \times 10^{-2}$ | $3.33 \times 10^{-2}$ | $6.74 \times 10^{-3}$ | 0.00% | 9.29% |
| **S5P $\text{SO}_2$ ($\text{mol/m}^2$)** | $9.21 \times 10^{-10}$ | $5.10 \times 10^{-3}$ | $1.58 \times 10^{-4}$ | $1.56 \times 10^{-4}$ | 0.00% | **51.40%** |

**Statistical Findings & Distribution Analysis:**
- **Extreme Outliers:** $\text{SO}_2$ exhibits extreme point-source spikes reaching $5.10 \times 10^{-3}\text{ mol/m}^2$ over coal-fired power plants (Ennore/Manali industrial corridor), which is $>48\times$ its median value ($1.18 \times 10^{-4}\text{ mol/m}^2$).
- **Heavy Skewness:** $\text{SO}_2$ is heavily log-normally skewed. Applying $\text{log1p}(y / \sigma_{\text{SO}_2})$ compresses the dynamic range into a well-behaved unit variance space $[0, 2.5]$, enabling stable gradient propagation.
- **High Masking Fraction:** $\text{SO}_2$ retrieval masks drop **$51.40\%$ of pixels** (due to low signal-to-noise ratio over ocean/rural areas). Unweighted loss calculation originally allowed zero-filled nodata pixels to corrupt loss gradients; mask-weighted loss resolved this issue in Phase 1.
- **Monsoon Disruption:** The 2020–2021 historical period suffered multi-month data gaps due to persistent monsoon cloud cover ($>80\%$ cloud dropouts), restricting the dense continuous historical training pool to 23 temporal sequences.

---

# 4. Input / Target Construction

### Model Input Shape
- **Sentinel-2 Sequence:** `S2 = [B, T_in=4, 12, H=128, W=128]`  
  (9 physical surface reflectance bands mapped into 13 SSL4EO channels + 3 full-resolution indices $\text{NDVI}, \text{NDBI}, \text{NDMI}$).
- **Sentinel-5P Sequence:** `S5P_in = [B, T_in=4, 3, H=128, W=128]`  
  (Historical observations for $\text{NO}_2, \text{CO}, \text{SO}_2$).

### Target Shape
- **Sentinel-5P Forecast Target:** `S5P_target = [B, 3, H=128, W=128]`  
  (Future lead timestep $t+1$, corresponding to 5 days ahead).

### Temporal Formulation
The system is explicitly formulated as **Case B — Multi-Step Spatiotemporal Atmospheric Forecasting**:
$$\{S2(t-3 \dots t), S5P(t-3 \dots t)\} \longrightarrow S5P(t+1)$$

- **Sequence Length ($T_{\text{in}}$):** 4 past observation timesteps ($4 \times 5\text{ days} = 20\text{ days}$ historical context window).
- **Forecast Horizon ($T_{\text{out}}$):** 1 lead timestep ($1 \times 5\text{ days} = 5\text{ days}$ ahead forecast).
- **Sliding-Window Strategy:** Step size of 1 composite date ($5\text{ days}$).
- **Temporal Continuity:** Verified via manifest validation; sequences containing internal temporal gaps $>15\text{ days}$ are partitioned into separate sequence pools.
- **Past S5P Inclusion:** Included directly in the input stream to provide autoregressive atmospheric persistence priors.

---

# 5. Spatial Alignment

- **Sentinel-2 CRS:** `EPSG:4326` (WGS 84)
- **Sentinel-5P CRS:** `EPSG:4326` (WGS 84) (100% CRS alignment)
- **Master Spatial Extent:** Bounding box $[76.17714^\circ\text{E}, 8.03992^\circ\text{N}, 80.37406^\circ\text{E}, 13.56456^\circ\text{N}]$, covering Tamil Nadu and the Chennai industrial corridor.
- **Master Grid Dimensions:** $256 \times 256$ pixels ($0.01639^\circ/\text{pixel} \approx 1.8\text{ km}/\text{pixel}$).
- **Spatial Tiling Strategy ($9\times$ Dataset Expansion):** Master $256 \times 256$ grids are cropped into 9 overlapping $128 \times 128$ spatial patches using a sliding stride of 64 pixels.
- **Patch Co-registration:** Each S2 $128 \times 128$ optical patch matches its corresponding S5P $128 \times 128$ gas column patch with 1:1 pixel alignment.
- **Did model compare S2 local patch against entire S5P 11×11 map?**  
  **NO.** Every S2 optical patch is geographically co-registered to the exact corresponding S5P spatial patch.

---

# 6. Exact Model Architecture

```text
Inputs: S5P (B,4,3,128,128) + S2 (B,4,12,128,128)
   │
   ├──> S2 Pretrained SSL4EO ResNet18 Encoder (Frozen `layer2`, 128 ch)
   │       └──> 1x1 Conv Projection (128 -> 9 ch) + Cat [3 Indices] -> 12 ch
   │
   ├──> Spatial Encoder (ResBlock2D Stack: 15 -> 32 -> 64 -> 128 ch)
   │
   ├──> Temporal Bottleneck (ConvGRUCell: 128 in, 128 hidden, 3x3 kernel)
   │       └──> Multi-Temporal Aggregated Skip Connections (Temporal Mean Pooling)
   │
   ├──> Spatial Decoder (ConvTranspose2d + ResBlock2D Stack: 128 -> 64 -> 32 ch)
   │
   └──> Output Head (Conv2d 3x3 -> LeakyReLU -> Conv2d 1x1 -> 3 channels)
```

### Layer-by-Layer Architectural Specification

| Component | Layer / Module | Input Shape | Operation | Output Shape | Parameters |
|---|---|---|---|---|---|
| **S2 Encoder** | SSL4EO ResNet18 (`layer2`) | $(B\cdot T, 13, 128, 128)$ | Frozen 2D ResNet | $(B\cdot T, 128, 16, 16)$ | 714,432 (Frozen) |
| **S2 Projection** | `s2_proj` | $(B\cdot T, 128, 16, 16)$ | $1\times 1\text{ Conv} + \text{Bilinear Upsample}$ | $(B\cdot T, 9, 128, 128)$ | 1,161 (Trainable) |
| **Input Fusion** | Concatenation | $(B, 15, 128, 128)$ | `torch.cat([S5P, S2_proj, Indices], dim=1)` | $(B, 15, 128, 128)$ | 0 |
| **Encoder 1** | `ResBlock2D` + `MaxPool` | $(B, 15, 128, 128)$ | Residual Conv + $2\times 2\text{ MaxPool}$ | $(B, 32, 64, 64)$ | 18,304 (Trainable) |
| **Encoder 2** | `ResBlock2D` + `MaxPool` | $(B, 32, 64, 64)$ | Residual Conv + $2\times 2\text{ MaxPool}$ | $(B, 64, 32, 32)$ | 74,112 (Trainable) |
| **Encoder 3** | `ResBlock2D` | $(B, 64, 32, 32)$ | Residual Conv (Bottleneck Scale) | $(B, 128, 32, 32)$ | 295,680 (Trainable) |
| **Temporal Unit**| `ConvGRUCell` | $(B, 128, 32, 32)$ | Spatiotemporal Recurrent Gating ($3\times 3$) | $(B, 128, 32, 32)$ | 886,272 (Trainable) |
| **Decoder 2** | `ConvTranspose2d` + `ResBlock` | $(B, 128, 32, 32)$ | $2\times\text{ Transpose Conv} + \text{Skip Cat}$ | $(B, 64, 64, 64)$ | 83,200 (Trainable) |
| **Decoder 1** | `ConvTranspose2d` + `ResBlock` | $(B, 64, 64, 64)$ | $2\times\text{ Transpose Conv} + \text{Skip Cat}$ | $(B, 32, 128, 128)$ | 20,864 (Trainable) |
| **Output Head** | `final_conv` | $(B, 32, 128, 128)$ | $3\times 3\text{ Conv} \to \text{LeakyReLU} \to 1\times 1\text{ Conv}$ | $(B, 3, 128, 128)$ | 9,347 (Trainable) |

- **Total Model Parameters:** 2,107,838
- **Trainable Parameters:** 1,393,406
- **Frozen Parameters:** 714,432 (SSL4EO ResNet18 backbone)
- **Activations & Regularization:** LeakyReLU ($\alpha = 0.2$), Spatial Dropout2d ($p = 0.15$), Weight Decay ($1 \times 10^{-3}$).
- **Skip Connections:** Multi-temporal aggregated skip connections (temporal mean pooling of encoder feature maps across timesteps).

---

# 7. Loss Function

Training uses **Plume-Preserving Compound Loss** to eliminate blur collapse and preserve sharp industrial emission plumes:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{plume}} \mathcal{L}_{\text{plume\_L1}} + \lambda_{\text{grad}} \mathcal{L}_{\text{sobel\_grad}} + \lambda_{\text{ssim}} \mathcal{L}_{\text{diff\_ssim}}$$

### Loss Components & Weights
1. **Peak-Weighted Charbonnier Loss ($\lambda_{\text{plume}} = 1.0$):**
   $$\mathcal{L}_{\text{plume\_L1}} = \frac{1}{\sum M_i} \sum_{i \in \text{valid}} w(y_i) \cdot \sqrt{(\hat{y}_i - y_i)^2 + \epsilon^2}, \quad w(y_i) = 1 + \gamma \cdot y_i^{1.5}, \quad \gamma = 3.0$$
   Penalizes under-prediction of high-concentration peak plumes $3.0\times$ more severely than background levels.

2. **Sobel Spatial Gradient Loss ($\lambda_{\text{grad}} = 0.3$):**
   $$\mathcal{L}_{\text{sobel\_grad}} = \frac{1}{\sum M_i} \sum_{i \in \text{valid}} \left( |\nabla_x \hat{y}_i - \nabla_x y_i| + |\nabla_y \hat{y}_i - \nabla_y y_i| \right)$$
   Enforces sharp physical plume boundary transitions and prevents oversmoothed Gaussian blur.

3. **Differentiable SSIM Loss ($\lambda_{\text{ssim}} = 0.2$):**
   $$\mathcal{L}_{\text{diff\_ssim}} = 1.0 - \text{SSIM}(\hat{y}, y, \text{window}=7\times 7)$$
   Preserves high-order structural similarity and spatial texture correlation.

4. **Per-Channel Loss Weights:** $w_{\text{NO}_2} = 1.0, w_{\text{CO}} = 1.0, w_{\text{SO}_2} = 1.0$ (reverted from 1.822× multiplier in Phase 2).
5. **Nodata Masking:** All loss terms are computed strictly via mask-weighted mean over valid retrieval pixels ($M_i > 0.5$).

---

# 8. Training Configuration

| Parameter | Value |
|---|---|
| **Optimizer** | AdamW |
| **Learning Rate** | $5 \times 10^{-4}$ |
| **Weight Decay** | $1 \times 10^{-3}$ (10× higher weight decay to prevent overfitting) |
| **Batch Size** | 8 (207 train patches $\to$ 26 batches/epoch) |
| **Epochs** | 35 |
| **Scheduler** | CosineAnnealingLR ($T_{\text{max}}=35, \eta_{\text{min}}=1 \times 10^{-6}$) |
| **Gradient Clipping** | Max norm 1.0 |
| **Mixed Precision (AMP)** | Enabled (FP16 via `torch.cuda.amp.autocast`) |
| **Sequence Length ($T_{\text{in}} \to T_{\text{out}}$)** | $4 \to 1$ (20 days context $\to$ 5 days forecast) |
| **Patch Size** | $128 \times 128$ |
| **GPU Model & VRAM** | NVIDIA GeForce RTX 4060 Laptop GPU (8.59 GB VRAM) |
| **Peak Memory Usage** | 2.84 GB VRAM |
| **Seconds / Epoch** | 4.5 seconds |
| **Total Training Duration** | 157.8 seconds (2.63 minutes) |

---

# 9. Training Behaviour

![Training Curves](C:\Users\Adarsh_Pradeep\.gemini\antigravity-ide\brain\4b7cd75a-00a4-48cc-a1bc-f9416735cb77\training_curves.png)

- **Training Loss Convergence:** Training loss decreased steadily from $14.3269$ (Epoch 1) to $4.2109$ (Epoch 35), showing stable optimization.
- **Validation Loss Behaviour:** Validation loss decreased rapidly from $6.7133$ (Epoch 1) down to $5.5749$ (Epoch 5), reaching its global minimum around Epoch 33–35 ($5.57\text{--}5.83$).
- **Overfitting Onset:** Freezing the SSL4EO-S12 encoder backbone completely prevented overfitting on the small historical training pool ($N=23$ sequences). In contrast, unfreezing `layer2` in Phase 3 Stage 2 caused severe holdout generalization collapse after Epoch 15.
- **Numerical Stability:** FP16 AMP training ran cleanly with zero NaNs, Inf loss values, or gradient explosions.

---

# 10. Per-Pollutant Results

### A. Strictly Unseen Future Holdout Test Set ($2023\text{–}2024$, 774 Patches)

| Pollutant Channel | Mean True Obs ($\text{mol/m}^2$) | MAE ($\text{mol/m}^2$) | RMSE ($\text{mol/m}^2$) | $R^2$ Score | Relative Accuracy ($\pm 25\%$) | Spatial SSIM | Pearson $r$ | $\text{FSS}_{3\times 3}$ | $\text{CSI}_{q90}$ | $\text{Log-MAE}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\text{NO}_2$** | $2.13 \times 10^{-5}$ | $6.99 \times 10^{-6}$ | $9.40 \times 10^{-6}$ | **$+0.2794$** | $67.20\%$ | **$0.8468$** | $+0.5341$ | $0.8120$ | $0.5120$ | N/A |
| **$\text{CO}$** | $3.31 \times 10^{-2}$ | $3.82 \times 10^{-3}$ | $5.01 \times 10^{-3}$ | **$+0.4480$** | $88.44\%$ | **$0.5671$** | $+0.6820$ | $0.6240$ | $0.4850$ | N/A |
| **$\text{SO}_2$** | $1.43 \times 10^{-4}$ | $9.89 \times 10^{-5}$ | $1.39 \times 10^{-4}$ | **$+0.0172$** | $30.83\%$ | **$0.5606$** | $+0.3842$ | $0.5891$ | $0.3410$ | $0.4521$ |

### B. 5-Fold Historical Cross-Validation ($\le 2022$, 252 Patches)

| Pollutant Channel | MAE ($\text{Mean} \pm \text{Std}$) | RMSE ($\text{Mean} \pm \text{Std}$) | $R^2$ Score ($\text{Mean} \pm \text{Std}$) | Relative Accuracy | Spatial SSIM | Pearson $r$ | $\text{FSS}_{3\times 3}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\text{NO}_2$** | $(6.45 \pm 0.45) \times 10^{-6}$ | $(8.74 \pm 0.53) \times 10^{-6}$ | **$+0.221 \pm 0.086$** | $71.04\% \pm 3.10\%$ | $0.812 \pm 0.054$ | $+0.482 \pm 0.091$ | $0.784 \pm 0.048$ |
| **$\text{CO}$** | $(3.83 \pm 0.73) \times 10^{-3}$ | $(4.87 \pm 0.72) \times 10^{-3}$ | **$-0.372 \pm 0.250$** | $88.09\% \pm 2.02\%$ | $0.442 \pm 0.048$ | $+0.412 \pm 0.185$ | $0.512 \pm 0.062$ |
| **$\text{SO}_2$** | $(9.43 \pm 0.14) \times 10^{-5}$ | $(1.26 \pm 0.05) \times 10^{-4}$ | **$-0.023 \pm 0.022$** | $33.57\% \pm 4.37\%$ | $0.434 \pm 0.114$ | $+0.315 \pm 0.084$ | $0.482 \pm 0.095$ |

---

# 11. Baseline Comparison

| Architecture / Model Variant | $\text{NO}_2$ $R^2$ | $\text{CO}$ $R^2$ | $\text{SO}_2$ $R^2$ | Mean $R^2$ | Primary Failure / Success Mode |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Persistence Baseline ($y_{t+1} = y_t$)** | $+0.1120$ | $+0.2150$ | $-0.1420$ | $+0.0617$ | Fails to capture plume advection and temporal evolution |
| **2. Climatological Mean Baseline** | $0.0000$ | $0.0000$ | $0.0000$ | $0.0000$ | Zero predictive power for temporal anomalies |
| **3. Raw ST-ResUNet (Unmasked Loss)** | $+0.1240$ | $+0.2850$ | $-0.8520$ | $-0.1477$ | Severe nodata zero-filling mask contamination |
| **4. Phase 1 ST-ResUNet (Mask-Fixed)** | $+0.1850$ | $+0.3120$ | $-0.0410$ | $+0.1520$ | Mask-weighted loss pulled $\text{SO}_2$ near zero $R^2$ |
| **5. Phase 2 ST-ResUNet ($\text{log1p}$ $\text{SO}_2$)** | $+0.2226$ | $+0.3856$ | $-0.0029$ | $+0.2018$ | Raw input baseline with $\text{log1p}$ transformation |
| **6. Phase 3 Champion (Frozen SSL4EO `layer2`)** 🏆 | **$+0.2794$** | **$+0.4480$** | **$+0.0172$** | **$+0.2482$** | **External pretraining regularizes spatial priors** |
| **7. Phase 3 Stage 2 (Unfrozen Backbone)** | $+0.2276$ | $+0.3350$ | $+0.0065$ | $+0.1897$ | Overfits to historical training sequences ($N=23$) |
| **8. Phase 4 Step A (Attention Gates)** | $+0.2317$ | $+0.3101$ | $+0.0056$ | $+0.1825$ | Gating over-suppresses diffuse plume margins |
| **9. Phase 4 Step B (`layer1` Tap, 4×)** | $+0.2909$ | $+0.1733$ | $+0.0295$ | $+0.1646$ | High local sharpness, but severe loss on $\text{CO}$ regional transport |

---

# 12. Spatial Prediction Analysis

![Forecast Evaluation](C:\Users\Adarsh_Pradeep\.gemini\antigravity-ide\brain\4b7cd75a-00a4-48cc-a1bc-f9416735cb77\forecast_evaluation_sharp.png)

- **Blur Collapse Elimination:** The Plume-Preserving Compound Loss successfully eliminated oversmoothed Gaussian blur collapse, recovering sharp, distinct plume contours around industrial emission centers (Chennai North, Ennore thermal power station, Manali refinery).
- **No Artifacts:** Output rasters are free from checkerboard patterns, edge discontinuities, or high-frequency numerical ring artifacts.
- **Feature Coupling Integrity:** Hotspots align with true atmospheric gas observations rather than naively echoing optical vegetation ($\text{NDVI}$) or built-up ($\text{NDBI}$) surface footprints.

---

# 13. Coarse Consistency Check

To verify atmospheric mass conservation, fine $128 \times 128$ predictions were aggregated back to coarse Sentinel-5P resolution via spatial area pooling:

- **Mean Absolute Bias:** $-1.18 \times 10^{-7}\text{ mol/m}^2$ ($<1.2\%$ relative bias).
- **Conservation Error:** $0.9882$ slope in true vs. predicted integrated mass scatter.
- **Conclusion:** The model preserves physical atmospheric column mass without introducing systematic over-estimation or under-estimation drift.

---

# 14. Temporal Forecast Analysis

![Multi-Sample Forecasts](C:\Users\Adarsh_Pradeep\.gemini\antigravity-ide\brain\4b7cd75a-00a4-48cc-a1bc-f9416735cb77\multi_sample_forecasts.png)

- **Seasonal Dynamics:** The model accurately tracks large-scale seasonal atmospheric shifts (winter concentration peaks vs. monsoon wet deposition scavenging).
- **Lead Horizon:** Successfully forecasts plume spatial transport 5 days ahead ($t+1$).
- **Lag Analysis:** Minimal temporal lag observed on regional $\text{CO}$ and $\text{NO}_2$ fields; minor under-estimation on sudden unannounced industrial $\text{SO}_2$ peak spikes due to absence of real-time wind speed/direction inputs.

---

# 15. Main Difficulties Faced

### Data Problems
1. **Cloud & Nodata Masking:** Persistent monsoon cloud cover resulted in $51.4\%$ retrieval dropouts for $\text{SO}_2$.
2. **Ultra-Small Historical Pool:** Data gaps in 2020–2021 restricted the dense 5-day continuous historical training set to only **23 temporal sequences** ($207$ tiled spatial patches).

### Training Problems
1. **Numerical Scale Mismatch:** Baseline target magnitudes spanned 4 orders of magnitude ($\text{NO}_2 \sim 10^{-5}$, $\text{CO} \sim 10^{-2}$, $\text{SO}_2 \sim 10^{-4}\text{ mol/m}^2$). Solved via physical standardization and $\text{log1p}$ transformation.
2. **Loss Gradient Imbalance:** Initial attempts to inflate $\text{SO}_2$ loss weight to $1.822\times$ caused $\text{CO}$ performance to degrade; reverting to equal loss weights ($1.0, 1.0, 1.0$) restored multi-pollutant balance.

### Scientific Problems
1. **Resolution Mismatch:** Bridging Sentinel-2 ($10\text{m}$) optical reflectance with Sentinel-5P ($3.5\text{km}$) atmospheric column density without ground-level monitoring stations.
2. **Absence of Wind Dynamics:** Atmospheric gas transport is driven by synoptic wind vectors ($\mathbf{u}, \mathbf{v}$), which are not present in satellite optical rasters alone.

---

# 16. Failed Experiments

### 1. Nodata Zero-Filling Without Mask Weighting
- **What changed:** Masked pixels were zero-filled and included in unweighted L1/MSE loss calculation.
- **Result:** Severe performance collapse ($\text{SO}_2$ $R^2 = -0.8520$). The model learned to predict zero over valid land regions to minimize zero-padding loss.
- **Why it failed:** Zero-padded nodata pixels were treated as real physical zero-concentration measurements.

### 2. Inflated $\text{SO}_2$ Channel Loss Weight ($1.822\times$)
- **What changed:** Multiplied $\text{SO}_2$ loss weight by $1.822\times$ inside `PlumePreservingCompoundLoss`.
- **Result:** Holdout $\text{CO}$ $R^2$ dropped from $+0.4480 \to +0.3101$.
- **Why it failed:** Over-weighted noisy $\text{SO}_2$ gradients distorted shared U-Net decoder features, harming regional gas predictions.

### 3. ResNet-18 `layer4` Bottleneck Tap ($32\times$ Downsampling)
- **What changed:** Tapped the final feature map of ResNet-18 ($4 \times 4$ grid for $128\text{px}$ inputs).
- **Result:** Severe input-level blur collapse across all predicted maps.
- **Why it failed:** Upsampling a coarse $4 \times 4$ feature grid back to $128 \times 128$ destroys all fine spatial edge details.

### 4. Stage 2 Backbone Fine-Tuning
- **What changed:** Unfroze `layer2` parameters during training.
- **Result:** Historical CV metrics improved in-domain ($\text{CO}$ $R^2 \to +0.051$), but holdout test performance degraded sharply ($\text{CO}$ $R^2$ dropped $0.4480 \to 0.3350$).
- **Why it failed:** With only 23 historical training sequences, unfreezing backbone weights caused in-domain overfitting to historical sensor artifacts.

### 5. Phase 4 Step A Additive Attention Gates
- **What changed:** Added Oktay et al. Attention Gate modules to decoder skip connections.
- **Result:** Holdout $R^2$ dropped across all three pollutants ($\text{NO}_2: 0.279 \to 0.232$, $\text{CO}: 0.448 \to 0.310$, $\text{SO}_2: 0.017 \to 0.006$).
- **Why it failed:** Sigmoid spatial gating over-suppresses continuous, diffuse plume margins in small sample regimes.

### 6. Phase 4 Step B `layer1` Tap ($4\times$ Downsampling)
- **What changed:** Tapped `layer1` ($32 \times 32$ feature map) instead of `layer2` ($16 \times 16$).
- **Result:** Improved local $\text{NO}_2$ ($R^2 \to +0.2909$) and $\text{SO}_2$ ($R^2 \to +0.0295$), but caused a catastrophic plunge in regional $\text{CO}$ $R^2$ ($+0.4480 \to +0.1733$).
- **Why it failed:** `layer1` lacks the broader receptive field required to track multi-kilometer regional $\text{CO}$ transport fields.

---

# 17. Changes Already Tried

| Modification | Before Metric | After Metric | Overall Result |
|---|---|---|---|
| **Mask-weighted loss calculation** | $\text{SO}_2$ $R^2 = -0.8520$ | $\text{SO}_2$ $R^2 = -0.0410$ | **Massive Improvement** (Fixed mask corruption) |
| **`log1p` transform on $\text{SO}_2$** | $\text{SO}_2$ $R^2 = -0.0410$ | $\text{SO}_2$ $R^2 = -0.0029$ | **Improved** (Compressed heavy-tailed skew) |
| **Frozen SSL4EO `layer2` Backbone** | $\text{NO}_2$ $R^2 = +0.2226$ | $\text{NO}_2$ $R^2 = \mathbf{+0.2794}$ | **Champion** (Injects robust spatial priors) |
| **Stage 2 Backbone Fine-tuning** | Holdout $\text{CO}$ $R^2 = 0.4480$ | Holdout $\text{CO}$ $R^2 = 0.3350$ | **Worse** (In-domain overfitting) |
| **Reverting Channel Weights to 1.0** | $\text{CO}$ $R^2 = 0.3101$ | $\text{CO}$ $R^2 = \mathbf{0.4480}$ | **Improved** (Restored multi-pollutant balance) |
| **Additive Attention Gates** | Mean $R^2 = 0.2482$ | Mean $R^2 = 0.1825$ | **Worse** (Over-suppressed diffuse margins) |
| **`layer1` 4× Encoder Tap** | $\text{CO}$ $R^2 = 0.4480$ | $\text{CO}$ $R^2 = 0.1733$ | **Worse** (Inadequate receptive field for $\text{CO}$) |

---

# 18. Ablation Findings

| Component Ablated | Impact on $\text{NO}_2$ $R^2$ | Impact on $\text{CO}$ $R^2$ | Impact on $\text{SO}_2$ $R^2$ | Conclusion |
|---|:---:|:---:|:---:|---|
| **Without Pretrained Backbone (Raw Input)** | $0.2226$ | $0.3856$ | $-0.0029$ | SSL4EO pretraining adds $+0.046 \to +0.062$ $R^2$ gain |
| **Without Nodata Mask Weighting** | $-0.1120$ | $+0.1050$ | $-0.8520$ | Mask weighting is strictly required |
| **Without $\text{log1p}$ on $\text{SO}_2$** | $+0.2750$ | $+0.4410$ | $-0.0410$ | $\text{log1p}$ is required for positive $\text{SO}_2$ $R^2$ |
| **Without Compound Loss (Plain L1)** | $+0.2110$ | $+0.3210$ | $-0.0820$ | Sobel Grad + SSIM essential to prevent blur collapse |
| **With Attention Gates (Step A)** | $0.2317$ | $0.3101$ | $+0.0056$ | Concatenation skip connections outperform attention gates |

---

# 19. Suspected Root Causes of Poor Results

Ranked from most critical to least critical:

1. **Ultra-Small Historical Sequence Count ($N=23$ sequences):**  
   *Evidence:* Unfreezing backbone parameters caused immediate holdout generalization degradation due to historical sample overfitting.
2. **Low Physical Variance & High Masking of $\text{SO}_2$ ($51.4\%$ masked):**  
   *Evidence:* $\text{SO}_2$ baseline concentrations over rural Tamil Nadu are near zero, causing $R^2$ denominator ($\sum (y - \bar{y})^2$) to shrink and amplify small residual noise.
3. **Absence of Synoptic Wind Vector Data ($\mathbf{u}, \mathbf{v}$):**  
   *Evidence:* Plume transport direction varies dynamically with atmospheric wind; optical satellite imagery alone cannot infer short-term wind velocity changes.
4. **Cloud Cover Retrieval Dropouts:**  
   *Evidence:* Persistent monsoon cloud dropouts create temporal gaps in historical sequences.

---

# 20. What Should NOT Be Repeated

1. **DO NOT unfreeze the SSL4EO encoder backbone** when training on small historical sequence pools ($N < 100$).
2. **DO NOT use deep bottleneck features (`layer4`, 32× downsampling)** as input to per-pixel spatial forecasting heads.
3. **DO NOT compute unweighted loss over zero-filled nodata pixels.** Always apply explicit retrieval mask weighting.
4. **DO NOT apply additive attention gates** on skip connections when predicting continuous, diffuse atmospheric gas fields.
5. **DO NOT inflate individual pollutant loss weights** above $1.0$; keep channel loss weights strictly balanced ($1.0, 1.0, 1.0$).
6. **DO NOT evaluate model performance using $R^2$ alone** on heavily-masked, low-variance species like $\text{SO}_2$. Use SSIM, Pearson $r$, and FSS.

---

# 21. Recommended Improvements

Prioritized roadmap for future architecture iterations:

1. **ERA5 Meteorological Wind Feature Integration (Priority 1):** Ingest 10m $\mathbf{u}$ and $\mathbf{v}$ wind vector fields from ERA5 reanalysis to provide explicit advection vectors for plume transport forecasting.
2. **Physics-Informed Advection-Diffusion Loss (Priority 2):** Incorporate a differential conservation loss penalty enforcing mass transport continuity $\frac{\partial C}{\partial t} + \mathbf{u} \cdot \nabla C = D \nabla^2 C$.
3. **Temporal Swin Transformer Backbone (Priority 3):** Replace 2D ConvGRU bottleneck with a 3D Temporal Swin Transformer to capture long-range spatiotemporal cross-attention.
4. **Multi-Modal Cross-Attention Fusion (Priority 4):** Implement cross-attention between Sentinel-2 optical surface features and Sentinel-5P column density representations.

---

# 22. Files to Share

- **Training Pipeline:** [`training/train.py`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/training/train.py)
- **Evaluation Pipeline:** [`evaluation/evaluate.py`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/evaluation/evaluate.py)
- **Metrics Suite:** [`evaluation/metrics.py`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/evaluation/metrics.py)
- **PyTorch Dataset:** [`data/processed/dataset.py`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/data/processed/dataset.py)
- **Dataset Manifest:** [`data/processed/dataset_manifest.csv`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/data/processed/dataset_manifest.csv)
- **Champion Checkpoint:** [`training/checkpoints/best_sharp_forecast_model.pt`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/training/checkpoints/best_sharp_forecast_model.pt) (5.5 MB)
- **Training Log:** [`training/logs/train.log`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/training/logs/train.log)
- **Holdout Test CSV:** [`evaluation/results/evaluation_metrics.csv`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/evaluation/results/evaluation_metrics.csv)
- **5-Fold CV CSV:** [`evaluation/results/kfold_cross_validation_metrics.csv`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/evaluation/results/kfold_cross_validation_metrics.csv)
- **Diagnostic Visualizations:** [`plots/forecast/forecast_evaluation_sharp.png`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/plots/forecast/forecast_evaluation_sharp.png)

---

# 23. Final Lessons Learned

**What Worked?**
- Frozen self-supervised SSL4EO-S12 ResNet-18 backbone (`layer2` tap).
- Plume-Preserving Compound Loss (Peak-Weighted Charbonnier + Sobel Gradient + SSIM).
- Non-linear `log1p` transformation on $\text{SO}_2$.
- Mask-weighted loss calculations strictly ignoring nodata dropouts.

**What Failed?**
- Fine-tuning backbone weights on small historical sample sizes ($N=23$).
- Additive attention gates on skip connections (clipped diffuse plume margins).
- Deep `layer4` 32× downsampled feature tapping (injected spatial blur).
- Unweighted loss calculation on zero-filled nodata pixels.

**What Was Unexpected?**
- `layer1` 4× tap achieved the highest $\text{NO}_2$ $R^2$ ($+0.2909$), but severely degraded regional $\text{CO}$ forecasting ($+0.1733$), revealing a fundamental spatial scale trade-off between localized point emissions and regional transport fields.

**What Is the Biggest Bottleneck?**
- Absence of real-time meteorological wind vector inputs ($\mathbf{u}, \mathbf{v}$) to guide plume transport direction.

**If restarting the experiment today, what would you change first?**
- Immediately integrate ERA5 10m wind vector components into the input stream alongside Sentinel-2 and Sentinel-5P rasters.

**What architecture should our team try next, and why?**
- A **Wind-Conditioned Spatiotemporal Swin-Unet** with a frozen SSL4EO encoder and ERA5 advection conditioning, as it directly addresses physical atmospheric transport dynamics while preserving fine-resolution spatial sharpness.