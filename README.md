# 🛰️ AirQ-CV: Satellite-Driven Multi-Pollutant Forecasting & Downscaling

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for **spatiotemporal atmospheric forecasting and high-resolution air quality downscaling** using paired multi-spectral **Sentinel-2** (12 optical bands) and **Sentinel-5P** (Tropospheric $\text{NO}_2$, Column $\text{CO}$, Surface $\text{SO}_2$) satellite observations over Tamil Nadu and the Chennai industrial corridor.

---

## 🌟 Key Architecture & Highlights

- **ST-ResUNet Architecture:** Spatiotemporal U-Net featuring 2D Residual encoder-decoder blocks, **ConvGRU recurrent bottleneck**, and multi-temporal aggregated skip connections.
- **Multimodal Input-Level Standardization:** Standardizes 12 optical Sentinel-2 bands and 3 Sentinel-5P gas columns based on empirical physical atmospheric statistics.
- **Plume-Preserving Compound Loss:** Solves the blur-collapse problem using a combination of **Peak-Weighted Charbonnier Loss** ($1 + \gamma \cdot \text{target}^{1.5}$), **Spatial Gradient / Sobel Edge Loss**, and **Differentiable SSIM**.
- **Multi-Patch Spatial Tiling ($9\times$ Expansion):** Ingests $256 \times 256$ master composites and tiles into 9 overlapping $128 \times 128$ spatial patches (stride 64), multiplying dataset capacity while maintaining low inter-patch redundancy ($r \approx 0.05$).
- **Dihedral Data Augmentation:** Synchronized random horizontal/vertical flips and $90^\circ$ rotations applied identically to past inputs and future target fields.
- **Strict Temporal Splitting & K-Fold Validation:** Non-overlapping chronological partition ($\le 2022$ historical vs. $2023\text{–}2024$ holdout test), evaluated with 5-fold sequence-level cross-validation.

---

## 📊 Quantitative Benchmarks

### 1. 5-Fold Cross-Validation Across Historical Sequences ($\le 2022$)
*Grouped by temporal sequence (54 patches/fold, 252 total patches) to eliminate spatial leakage:*

| Target Pollutant | MAE ($\text{Mean} \pm \text{Std}$) | RMSE ($\text{Mean} \pm \text{Std}$) | $R^2$ Score ($\text{Mean} \pm \text{Std}$) | Relative Accuracy | Spatial SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\text{NO}_2$ (Nitrogen Dioxide)** | **$8.25 \times 10^{-6} \pm 1.28 \times 10^{-6}\text{ mol/m}^2$** | $1.12 \times 10^{-5} \pm 1.42 \times 10^{-6}$ | $+0.056 \pm 0.209$ | **$55.33\% \pm 13.71\%$** | **$0.723 \pm 0.113$** |
| **$\text{CO}$ (Carbon Monoxide)** | **$5.38 \times 10^{-3} \pm 1.08 \times 10^{-3}\text{ mol/m}^2$** | $8.89 \times 10^{-3} \pm 1.33 \times 10^{-3}$ | $-0.094 \pm 0.145$ | **$80.93\% \pm 7.55\%$** | **$0.689 \pm 0.076$** |
| **$\text{SO}_2$ (Sulfur Dioxide)** | **$1.39 \times 10^{-4} \pm 1.21 \times 10^{-5}\text{ mol/m}^2$** | $1.57 \times 10^{-4} \pm 9.23 \times 10^{-6}$ | $-1.136 \pm 0.510$ | $0.00\% \pm 0.00\%$ | **$0.292 \pm 0.092$** |

### 2. Strictly Unseen Future Test Set ($2023\text{–}2024$, 774 Spatial Patches)

| Target Pollutant | Mean True Observation | Mean Absolute Error ($\text{MAE}$) | Root Mean Squared Error ($\text{RMSE}$) | $R^2$ Score | Relative Accuracy | Spatial SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\text{NO}_2$ (Nitrogen Dioxide)** | $1.86 \times 10^{-5}\text{ mol/m}^2$ | **$8.24 \times 10^{-6}\text{ mol/m}^2$** | $1.12 \times 10^{-5}\text{ mol/m}^2$ | **$+0.1897$** | **$55.79\%$** | **$0.7652$** |
| **$\text{CO}$ (Carbon Monoxide)** | $3.06 \times 10^{-2}\text{ mol/m}^2$ | **$5.44 \times 10^{-3}\text{ mol/m}^2$** | $9.28 \times 10^{-3}\text{ mol/m}^2$ | **$+0.2476$** | **$82.20\%$** | **$0.7411$** |
| **$\text{SO}_2$ (Sulfur Dioxide)** | $8.00 \times 10^{-5}\text{ mol/m}^2$ | **$1.41 \times 10^{-4}\text{ mol/m}^2$** | $1.62 \times 10^{-4}\text{ mol/m}^2$ | $-0.8246$ | $0.00\%$ | **$0.3814$** |

---

## 📁 Repository Structure

```text
AirQ-CV/
├── data/
│   ├── processed/
│   │   ├── dataset.py                  # PyTorch Multimodal Dataset with spatial tiling
│   │   ├── dataset_manifest.csv        # Metadata and temporal pairing manifest
│   │   └── stream_utils.py             # Memory-mapped streaming utilities
│   └── raw/                            # Raw data folder (ignored by git)
│
├── training/
│   ├── checkpoints/
│   │   └── best_sharp_forecast_model.pt # Trained ST-ResUNet weights (5.3 MB)
│   ├── configs/
│   │   └── train_config.yaml           # Training parameters and loss weights
│   ├── logs/
│   │   └── train.log                   # Full 35-epoch training and validation loss log
│   └── train.py                        # Training pipeline with compound loss
│
├── evaluation/
│   ├── evaluate.py                     # Evaluation runner (5-Fold CV + holdout benchmark)
│   ├── metrics.py                      # Physical & spatial metrics (MAE, RMSE, R2, SSIM)
│   └── results/
│       ├── evaluation_metrics.csv      # Holdout test metrics
│       └── kfold_cross_validation_metrics.csv # 5-Fold cross-validation metrics
│
├── plots/
│   ├── analysis/
│   │   ├── parity_and_correlation_plots.png # Hexbin scatters & error distributions
│   │   └── plume_transect_profiles.png      # 1D plume cross-section transects
│   ├── environment/
│   │   └── environmental_hazard_index.png   # Multi-pollutant composite hazard map
│   ├── forecast/
│   │   ├── forecast_evaluation_sharp.png    # 3x3 Ground truth vs prediction comparison
│   │   └── multi_sample_forecasts.png       # Diverse multi-step forecasts
│   ├── training_curve_comparison.png        # Training & validation loss curves
│   └── generate_plots.py                    # Script to regenerate all diagnostic plots
│
├── docs/
│   ├── context.md                      # Detailed project context & scientific background
│   ├── dataset-details.md              # Band specifications & physical units
│   └── dataset_audit_report.md         # Full 8-question dataset audit report
│
├── utils/
│   ├── __init__.py
│   └── helper_functions.py             # Utilities for formatting and seeding
│
├── .gitignore                          # Excludes large raw GeoTIFF rasters (>4GB)
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Adxrsh-17/AirQ-CV.git
cd AirQ-CV
pip install torch torchvision rasterio numpy pandas matplotlib scipy pyyaml
```

### 2. Run Evaluation & 5-Fold Cross-Validation
```bash
python evaluation/evaluate.py
```

### 3. Generate Diagnostic Visualizations
```bash
python plots/generate_plots.py
```

### 4. Train Model
```bash
python training/train.py
```

---

## 📖 Scientific References & Documentation

For complete technical specifications, see:
- [`docs/dataset_audit_report.md`](docs/dataset_audit_report.md) — Complete 8-point audit of satellite rasters, distributions, masking, and alignment.
- [`docs/context.md`](docs/context.md) — Comprehensive atmospheric physics and project motivation.
- [`docs/dataset-details.md`](docs/dataset-details.md) — Sentinel-2 and Sentinel-5P sensor bands and spatial resolution details.
