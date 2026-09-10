# Satellite-Driven Multi-Pollutant Forecasting & Downscaling for Industrial and Respiratory Disease Risk Mapping

---

## 1. Executive Summary & Project Identification

* **Project Title:** Satellite-Driven Multi-Pollutant Forecasting for Industrial and Respiratory Disease Risk Mapping
* **Academic Institution:** School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore
* **Academic Year / Phase:** 4th Year B.Tech (Semester 7) — Project Phase 1 & 2
* **Team ID:** Team No. B6
* **Team Members & Registration IDs:**
  * **Adarsh Pradeep** (`CB.SC.U4AIE23109`) — Research Lead & Environmental Health Risk Mapping
  * **Anto Rishath** (`CB.SC.U4AIE23103`) — Data Analysis, Seasonal Patterns & Hotspot Delineation
  * **Antonio Roger** (`CB.SC.U4AIE23104`) — AI Architecture, Downscaling & Spatiotemporal Deep Learning
  * **Naresh Kumar V** (`CB.SC.U4AIE23165`) — Data Engineering, Satellite Acquisition & GEE Pipelines
* **Faculty Mentorship:**
  * **Guide:** Dr. Iswarya K. V.
  * **Co-guide:** Dr. Sajith Variyar V. V.

---

## 2. Problem Statement & Research Motivation

### 2.1 The Core Scientific Challenge
1. **Ground Station Sparsity:** Conventional continuous ambient air quality monitoring stations (CAAQMS) operated by CPCB/TNPCB are geographically sparse (only $\sim 8\text{--}12$ stations across the entire Chennai metropolitan region) and prioritize historical logging rather than proactive spatiotemporal forecasting.
2. **Coarse Satellite Footprint:** Satellite-based atmospheric spectrometers like **Sentinel-5P (TROPOMI)** provide daily global gas columns, but at a coarse spatial resolution of **$5.5 \times 3.5\text{ km}$** (or $1.1\text{ km}$ gridded), which is unable to resolve localized industrial plumes (e.g., Manali Petrochemical cluster, Ennore Thermal Power Station) or neighborhood-level exposure gradients.
3. **The Multimodal Solution:** By fusing high-resolution **Sentinel-2 multispectral surface imagery ($10\text{–}20\text{ m}$)** with coarse **Sentinel-5P gas absorption measurements ($5.5\text{ km}$)** through a physics-guided deep learning downscaling architecture, we disaggregate coarse gas fields into high-resolution spatial maps and forecast multi-day exposure patterns.

```
+-------------------------------------------------------------------------------+
|                             MULTIMODAL PIPELINE                               |
|                                                                               |
|  [ Sentinel-2 MSI Optical ]      [ Sentinel-5P TROPOMI Gas Columns ]          |
|  12 Bands (10-20m) @ 558x558      NO2, CO, SO2 (5.5km) @ 13x13                |
|               |                                      |                        |
|               +------------------+-------------------+                        |
|                                  |                                            |
|                                  v                                            |
|                [ Cross-Modal Downscaling Neural Net ]                         |
|                 ResNet Feature Extractor + Gas Head                           |
|                                  |                                            |
|                                  v                                            |
|             [ Area-Weighted Spatial Consistency Loss ]                        |
|                                  |                                            |
|                                  v                                            |
|  [ High-Res Multi-Pollutant Map ] --> [ Population Hazard Index (PSI / EWI) ] |
+-------------------------------------------------------------------------------+
```

---

## 3. Dataset Specifications & Ingestion Architecture

### 3.1 Primary Multi-Year Dataset (2019 – 2024)

| Attribute | Sentinel-2 (High-Resolution Optical) | Sentinel-5P (Atmospheric Gas Columns) |
| :--- | :--- | :--- |
| **Satellite Sensor** | Multi-Spectral Instrument (MSI) Level-2A / Level-1C | TROPOMI Spectrometer (L3 Offline Products) |
| **Spectral Channels / Bands** | **12 Bands:** B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 | **3 Atmospheric Target Gases:** Tropospheric $\text{NO}_2$, Total $\text{CO}$, Total $\text{SO}_2$ |
| **Temporal Span** | **January 1, 2019 to December 30, 2024 (6 Full Years)** | **January 1, 2019 to December 30, 2024 (6 Full Years)** |
| **Temporal Cadence** | **5-Day Continuous Composites** (median reduction) | **5-Day Continuous Composites** (masked orbit mean) |
| **Spatial Resolution** | **High-Res Grid ($558 \times 558$ pixels, $\approx 100\text{ m}$)** | **Coarse Observation Grid ($13 \times 13$ pixels, $\approx 5.5\text{ km}$)** |
| **Bounding Box (Chennai ROI)** | Lat: $12.80^\circ\text{N} - 13.30^\circ\text{N}$, Lon: $79.95^\circ\text{E} - 80.45^\circ\text{E}$ | Lat: $12.75^\circ\text{N} - 13.34^\circ\text{N}$, Lon: $79.90^\circ\text{E} - 80.49^\circ\text{E}$ |
| **Coordinate Reference System** | `EPSG:4326` (WGS 84 Geographic Coordinates) | `EPSG:4326` (WGS 84 Geographic Coordinates) |
| **Total Ingested Rasters** | **291 Cloud-Screened GeoTIFFs** | **439 Continuous Gas GeoTIFFs** |
| **Paired Windows** | **438 Windows Total** (293 Train Windows for 2019–2022, 145 Holdout Test Windows for 2023–2024) |

### 3.2 Google Earth Engine Data Acquisition Pipeline (`Data_acquisition.js`)
* **Upstream Quality Filtering:** Leverages Google's L3 `harpconvert` automated QA thresholds ($0.75$ for $\text{NO}_2$, $0.50$ for $\text{CO}$, $0.50$ for $\text{SO}_2$).
* **Cloud Masking:** Applies `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` with a clear-sky probability threshold `cs_cdf >= 0.60`.
* **Outlier Floors:** Per-gas physical thresholds ($\text{NO}_2 \ge -0.0005$, $\text{CO} \ge -0.001$, $\text{SO}_2 \ge -0.005\text{ mol}/\text{m}^2$) to retain zero-centered retrieval noise without corrupting distributions.
* **Observation Weights (`n_obs`):** Exports contributing orbit pass counts per pixel ($2\text{--}5$ passes per 5-day window) as trust weights.

---

## 4. Key Scientific Insights & Technical Corrections ("Harsh Truths")

### 4.1 The Methane ($\text{CH}_4$) Exclusion Rationale
* **Issue:** Earlier proposal drafts listed $\text{CH}_4$ alongside $\text{NO}_2, \text{CO}, \text{SO}_2$.
* **Resolution:** Sentinel-5P Methane (`COPERNICUS/S5P/OFFL/L3_CH4`) enforces strict surface albedo and cloud fraction thresholds ($<0.05$). Over coastal, humid, and cloud-prone regions like Chennai, $>80\%$ of scenes are rejected by the satellite processor, creating unrecoverable spatial voids.
* **Scientific Stance:** The project strictly models the three physically robust pollutants: **$\text{NO}_2$ (traffic/combustion), $\text{CO}$ (incomplete burning/background), and $\text{SO}_2$ (industrial coal/petrochemical refining)**.

### 4.2 Physical Nature of Sentinel-2 Multispectral Proxy
* **Clarification:** Sentinel-2 optical bands do **not** measure molecular gas absorption.
* **Role in Model:** Sentinel-2 provides high-resolution **indirect spatial priors**:
  * Visible & NIR (B2, B3, B4, B8) $\rightarrow$ Road networks, urban density, vegetation stress ($\text{NDVI}$).
  * SWIR (B11, B12) $\rightarrow$ Industrial impervious roofs, built-up surfaces ($\text{NDBI}$), and thermal combustion anomalies.
  * Coastal Aerosol (B1) $\rightarrow$ Boundary-layer aerosol scattering.
* The neural network learns the structural mapping from these surface emission proxies to disaggregate coarse TROPOMI gas footprints.

### 4.3 Statistical Behavior of $\text{SO}_2$ over Chennai
* **Sensor Noise Floor:** TROPOMI's physical noise floor for single-pass $\text{SO}_2$ retrieval is $\approx 1.0 \times 10^{-4}\text{ mol}/\text{m}^2$.
* **Chennai Profile:** Ambient background $\text{SO}_2$ is low ($5.85 \times 10^{-5}\text{ mol}/\text{m}^2$), sitting near the noise floor, except for sharp episodic plumes over the **Manali / Ennore industrial zone**.
* **Recommendation:** Focus spatial error weighting on industrial cluster bounding boxes (rows 30–45, cols 380–400) rather than broad suburban background pixels.

### 4.4 Environmental Risk Mapping vs. Epidemiological Validation
* **Framing:** Without clinical patient-level health records across Chennai hospitals, the health module is properly defined as a **Geospatial Environmental Exposure & Health Hazard Index (HHI)** combining downscaled gas concentrations with WorldPop/LandScan population distribution grids.

---

## 5. Deep Learning Architecture & Mathematical Formulation

### 5.1 Physics-Guided Cross-Modal Downscaling Network

The model maps high-resolution optical features to fine-grained gas concentrations using a Residual Feature Extractor followed by an Area-Weighted Consistency Loss:

$$\hat{Y}_{\text{fine}} = \mathcal{G}_{\theta}(X_{\text{S2}}), \quad X_{\text{S2}} \in \mathbb{R}^{B \times 12 \times H_{\text{fine}} \times W_{\text{fine}}}$$

$$\hat{Y}_{\text{coarse}} = \mathcal{P}_{\text{avg}}(\hat{Y}_{\text{fine}}), \quad \hat{Y}_{\text{coarse}} \in \mathbb{R}^{B \times 3 \times H_{\text{coarse}} \times W_{\text{coarse}}}$$

### 5.2 Area-Weighted Spatial Consistency Loss

To train the high-resolution generator without needing ground-truth high-resolution gas maps, we enforce that the spatial average of the super-resolved field matches the observed Sentinel-5P coarse measurement:

$$\mathcal{L}_{\text{total}} = \frac{1}{B \cdot C \cdot H_c \cdot W_c} \sum_{b=1}^{B} \sum_{c=1}^{C} \sum_{i=1}^{H_c} \sum_{j=1}^{W_c} \left| \left( \frac{1}{|R_{i,j}|} \sum_{(u,v) \in R_{i,j}} \hat{Y}_{\text{fine}}(b, c, u, v) \right) - Y_{\text{coarse}}(b, c, i, j) \right|$$

Where $R_{i,j}$ represents the spatial pixel footprint of coarse pixel $(i,j)$.

---

## 6. Empirical Benchmarks & Validation Results

Trained on **293 paired training windows (2019–2022)** and evaluated strictly on **145 holdout test windows (2023–2024)**:

### 6.1 Training Loss Progression (25 Epochs)
* **Initial L1 Loss:** $0.3387$
* **Final Training L1 Loss:** $0.2911$ (**$26.8\%$ optimization drop**)
* **Test Holdout Consistency Loss:** $0.3110$

### 6.2 Holdout Test Evaluation Performance (2023 – 2024 Ground Truth)

| Target Pollutant | Mean Satellite Observation | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | $R^2$ Score | Relative Accuracy ($1 - \text{Rel Err}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\text{CO}$ (Carbon Monoxide)** | $3.12 \times 10^{-2}\text{ mol}/\text{m}^2$ | $6.76 \times 10^{-3}\text{ mol}/\text{m}^2$ | $1.01 \times 10^{-2}\text{ mol}/\text{m}^2$ | **$0.1647$** | **$78.32\%$** |
| **$\text{NO}_2$ (Nitrogen Dioxide)** | $1.89 \times 10^{-5}\text{ mol}/\text{m}^2$ | $9.16 \times 10^{-6}\text{ mol}/\text{m}^2$ | $1.30 \times 10^{-5}\text{ mol}/\text{m}^2$ | **$0.0003$** | **$51.64\%$** |
| **$\text{SO}_2$ (Sulfur Dioxide)** | $7.80 \times 10^{-5}\text{ mol}/\text{m}^2$ | $7.74 \times 10^{-5}\text{ mol}/\text{m}^2$ | $1.51 \times 10^{-4}\text{ mol}/\text{m}^2$ | $-0.2673$ | **$0.74\%$** |

---

## 7. Execution Guide: Kaggle & Local Environments

The pipeline script [`train_kaggle_downscaling.py`](file:///c:/Users/Adarsh_Pradeep/OneDrive/Desktop/4th%20Yr%20Sem%207/Project%20Phase%201/Projects/Dataset/train_kaggle_downscaling.py) supports direct in-memory streaming from Google Drive into RAM:

### Running on Kaggle GPU Notebook
```bash
# 1. Install dependencies in Kaggle notebook cell
!pip install gdown rasterio -q

# 2. Run in-memory streaming training on GPU
!python train_kaggle_downscaling.py \
    --drive_url "https://drive.google.com/drive/folders/1SQdU-L5SctKcsYXa4WTcto8_emaCurVQ" \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.003 \
    --patch_size 128 \
    --num_threads 16 \
    --output_dir ./outputs
```

### Command-Line Arguments & Flags

| Flag | Default | Description |
| :--- | :---: | :--- |
| `--drive_url` | Public Google Drive link | Folder URL containing paired S2 and S5P composites |
| `--epochs` | `25` | Number of optimization epochs |
| `--batch_size` | `8` | Training batch size |
| `--lr` | `0.003` | Initial learning rate (Cosine Annealed) |
| `--patch_size` | `128` | High-res spatial crop size ($128 \times 128$) |
| `--num_threads` | `16` | Parallel workers for Google Drive in-memory prefetching |
| `--max_train_samples` | `None` | Optional limit for rapid dry-run testing |

---

## 8. Project Phase 2 Roadmap & Next Milestones

1. **Spatiotemporal Sequence Integration:** Connect the high-resolution spatial feature maps into a **ConvLSTM / Swin-UNet Temporal Transformer** to forecast multi-step future horizons ($T+1$ to $T+4$, i.e., 5 to 20 days ahead).
2. **Industrial Hotspot Focused Loss:** Add spatial mask loss weighting over Manali and Ennore industrial coordinates to improve $\text{SO}_2$ and $\text{NO}_2$ peak localization.
3. **Population-Weighted Hazard Index (EWI):** Overlay predicted high-resolution pollutant concentration fields onto Gridded Population of the World (GPWv4) to compute neighborhood-level Early Warning Indicators for Chennai.
