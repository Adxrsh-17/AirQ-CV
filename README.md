# AirQ-CV: Dual-Clock Architecture & Unified Dataset Workspace

This workspace unifies the **12GB satellite dataset (Sentinel-2 + Sentinel-5P)** with **ERA5 reanalysis weather** and the **IEEE Dual-Clock Land-Emission Architecture**.

---

## 1. Directory Structure

```
AirQ_DualClock/
├── data/
│   ├── raw/                  # Placeholders / raw ingestion logs
│   └── processed/            # Fast junction to 11.57 GB multi-spectral dataset
│       ├── s2_composites/    # 415 Sentinel-2 L2A optical GeoTIFFs (9.71 GB, 12 bands)
│       ├── s5p_composites/   # 438 Sentinel-5P TROPOMI gas GeoTIFFs (44.5 MB, NO2, CO, SO2)
│       └── cached_dataset_256.pt
├── weather/
│   ├── era5_daily.csv        # 6-year ECMWF ERA5 reanalysis (2019-2024, 2,192 records)
│   └── ingest_era5.py        # Automated ERA5 weather acquisition script
├── src/
│   ├── dual_clock_pipeline.py# Complete Dual-Clock pipeline (Slow Clock C1, Fast Clock C2, C3)
│   ├── preprocessing.py      # Feature preprocessor and window aggregator
│   └── feature_extractor.py  # Tabular feature generator
├── docs/
│   ├── IEEE_Dual_Clock_Land_Emission_Architecture_Proposal.txt
│   ├── ANALYSIS_OF_FAILURES_AND_KEY_FINDINGS.md
│   └── IEEE_DUAL_CLOCK_EXPERIMENT_REPORT.md
├── runs/
│   └── dual_clock_runs/
│       ├── c1_annual_land_change.csv
│       ├── c2_panel_attribution.json
│       └── multi_horizon_results.csv
├── train_dual_clock.py       # Top-level one-command execution runner
└── README.md
```

---

## 2. Quick Execution

To execute data alignment, Slow Clock C1 extraction, Fast Clock C2 deweathering, Gate G2 panel regression, and C3 multi-horizon forecasting:

```bash
python train_dual_clock.py
```

All experimental outputs will be automatically written to `runs/dual_clock_runs/`.
