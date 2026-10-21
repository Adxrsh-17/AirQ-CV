# DC-Net: Dual-Clock Multi-Horizon Atmospheric Deep Learning Forecaster

This directory implements **DC-Net**, a pure deep learning multi-pollutant spatiotemporal forecaster for Sentinel-5P (NO2, CO, SO2) and Sentinel-2 optical imagery across Tamil Nadu.

## Features
- **Fast Clock (5-day):** 12-lag ConvGRU unrolled across 3 spatial resolutions (1/2, 1/4, 1/8) with mask-aware partial convolutions.
- **Slow Clock (Monthly):** 26-channel causal Sentinel-2 composite at 8x native resolution (984x760) capturing static reflectance and year-over-year change (dNDBI).
- **Weather Conditioning:** Dynamic FiLM modulation incorporating ERA5 boundary layer height (BLH), wind speed, wind components (u10, v10), temperature, humidity, rainfall, and ventilation coefficients.
- **Hardware Safety Rails:**
  - Hard VRAM allocation cap (--vram_cap 0.60, ~4.8 GB max) for laptop RTX 4060 GPUs.
  - Automatic Mixed Precision (AMP FP16).
  - Safe Windows DataLoader (--workers 0).
  - Inter-epoch thermal pacing to prevent laptop GPU thermal throttling.

## Usage
1. Build aligned multimodal cache:
   `ash
   python make_cache.py --s2_dir ../data/processed/s2_composites --s5p_dir ../data/processed/s5p_composites --weather ../weather/era5_daily.csv --out cache
   `

2. Train DC-Net with hardware safety rails:
   `ash
   python train.py --cache cache --out runs/dcnet_main --batch 4 --vram_cap 0.60 --use_weather 1 --workers 0
   `

3. Multi-horizon evaluation:
   `ash
   python evaluate.py --ckpt runs/dcnet_main/best.pt --cache cache --split test --out runs/dcnet_eval
   `
