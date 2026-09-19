# Classical ML Pivot Walkthrough (150-Day Horizon)

We have implemented a Classical ML approach.

## 1. Feature Engineering & The Climatology Baseline
We built `classical_ml/features/build_climatology.py` which:
1. Processed the entire train split (2019-2023) to compute Monthly Spatial Averages (Climatology).
2. Extracted exactly 117 valid SO2 hotspots.
3. Precomputed distance-to-hotspot maps for the spatial decay models.

We flattened the dataset using `tabular_extractor.py` to map (x, y, DOY, NDVI, NDBI, NDMI, Climatology) -> Anomaly.

## 2. Model Implementations
*   **NO2**: `no2_spatial_rf.py` (LightGBM Spatial Tree) to predict NO2 anomalies.
*   **SO2**: `so2_distance_decay.py` to predict SO2 utilizing the precomputed so2_dist from the static hotspots.
*   **CO**: `co_eof_sarima.py` to decompose the 2D grid into 5 principal spatial modes (EOF), then applied SARIMA.

## 3. Results and Scientific Validation
The classical ML models were evaluated against the Climatology Baseline.

> [!WARNING]
> **Protocol Limitations**:
> The existing review identified protocol limitations in the current classical experiment:
> - NO2 current target-date S2 leakage
> - SO2 current target-date S2 leakage
> - CO horizon mismatch
> - Climatology results are provisional
> - Classical results require corrected strict H12/K30 evaluation before becoming official.

Due to these limitations, the current results are provisional and we cannot definitively claim climatology as the theoretical ceiling until a strict H12/K30 evaluation is correctly completed.
