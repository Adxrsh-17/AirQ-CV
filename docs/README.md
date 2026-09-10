# 🛰️ Satellite-Driven Multi-Pollutant Downscaling & Spatiotemporal Forecasting

> **Project:** Satellite-Driven Multi-Pollutant Forecasting for Industrial and Respiratory Disease Risk Mapping  
> **Team:** Team No. B6 | School of AI, Amrita Vishwa Vidyapeetham, Coimbatore  
> **Target Region:** Chennai Industrial & Port Corridor (Manali, Ennore, Ambattur, Guindy, Chennai Port)  

---

## 1. Project Directory Structure

```text
Projects/Dataset/
├── configs/
│   └── default_config.py           # Hyperparameters, paths, and training configs
│
├── data/
│   ├── dataset.py                  # PyTorch Dataset loader (T=6 sliding sequence windows)
│   ├── stream_utils.py             # High-speed in-memory streaming & caching engine
│   ├── dataset_manifest.csv        # 5-day window pairing & metadata tracking sheet
│   ├── s2_composites/              # Sentinel-2 12-channel high-res GeoTIFFs (100 m)
│   └── s5p_composites/             # Sentinel-5P 3-channel coarse supervision GeoTIFFs (5.5 km)
│
├── models/
│   ├── convlstm.py                 # Spatiotemporal ConvLSTM Cell & Downscaling Architecture
│   └── loss.py                     # Area-Weighted Spatial Consistency Loss Layer
│
├── training/
│   ├── trainer.py                  # PyTorch Trainer (AMP Mixed Precision, Cosine Annealing, Checkpoints)
│   └── metrics.py                  # Physical evaluation metrics (MAE, RMSE, R2, Rel-Accuracy)
│
├── checkpoints/                    # Saved best model weights (.pth)
├── logs/                           # Training history CSV logs & learning curves
├── results/                        # Holdout 2024 test metrics and evaluation summaries
├── outputs/                        # Kaggle/Local exported model checkpoints
│
├── run_sanity_check.py             # Sanity verification for shapes, loss, GPU/CPU & gradients
├── main.py                         # Main training, validation, and holdout test runner
├── train_kaggle_downscaling.py     # Standalone script for Kaggle GPU execution
├── context.md                      # Comprehensive project context & scientific documentation
└── dataset-details.md              # Dataset schema, band mapping, and physical specifications
```

---

## 2. Dataset & Splits (2019 – 2024)

* **Temporal Cadence:** Uniform 5-day median composites (**438 total windows**).
* **Sentinel-2 Input:** 12 spectral bands (B2, B3, B4, B5, B6, B7, B8, B11, B12, NDVI, NDBI, NDMI) at $100\,\text{m}$ resolution ($558 \times 558$ grid).
* **Sentinel-5P Target:** 3 atmospheric gases ($\text{NO}_2, \text{CO}, \text{SO}_2$) at $5.5\,\text{km}$ resolution ($13 \times 13$ coarse grid).
* **Chronological Splits:**
  * **Train Split ($2019\text{–}2022$):** $293$ windows ($year \le 2022$)
  * **Validation Split ($2023$):** $73$ windows ($year == 2023$)
  * **Holdout Test Split ($2024$):** $72$ windows ($year == 2024$)

---

## 3. Quick Start & Execution

### Step 1: Run Sanity Check
Verifies tensor shapes, forward/backward passes, consistency loss, and hardware execution:
```bash
python run_sanity_check.py
```

### Step 2: Run Spatiotemporal ConvLSTM Baseline
Trains the multi-layer ConvLSTM model across the training split and evaluates against the holdout 2024 test ground truth:
```bash
python main.py --epochs 25 --batch_size 4 --patch_size 64
```

### Step 3: Run on Kaggle GPU
```bash
python train_kaggle_downscaling.py --epochs 30 --batch_size 8 --patch_size 128
```
