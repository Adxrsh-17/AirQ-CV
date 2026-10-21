#!/usr/bin/env python3
"""Root execution entrypoint for AirQ-CV Dual-Clock Architecture Pipeline.

Runs alignment across S2 optical, S5P NO2, and ERA5 weather, executes Slow Clock C1,
Fast Clock C2 (Deweathering + TWFE Panel Attribution), Gate G2 falsification, and
Component C3 Multi-Horizon Forecasting (5 to 150 days).
"""
import json
import sys
from pathlib import Path
import pandas as pd

# Add src to sys.path
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

import dual_clock_pipeline

def main():
    s2_dir = root / "data" / "processed" / "s2_composites"
    s5p_dir = root / "data" / "processed" / "s5p_composites"
    era5_path = root / "weather" / "era5_daily.csv"
    out_dir = root / "runs" / "dual_clock_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AIRQ-CV: DUAL-CLOCK ARCHITECTURE ALIGNMENT & TRAINING")
    print("=" * 80)
    print(f"Sentinel-2 Data : {s2_dir}")
    print(f"Sentinel-5P Data: {s5p_dir}")
    print(f"ERA5 Weather    : {era5_path}")
    print(f"Output Directory: {out_dir}")
    print("=" * 80)

    # 1. Load and Crop Chennai Corridor (121 cells)
    no2_cube, ndbi_cube, ndvi_cube, meta_df = dual_clock_pipeline.load_and_crop_dataset(s2_dir, s5p_dir)

    # 2. Slow Clock C1: Annual Land Dynamics
    df_land, annual_ndbi, annual_ndvi = dual_clock_pipeline.compute_annual_land_change(ndbi_cube, ndvi_cube, meta_df)
    df_land.to_csv(out_dir / "c1_annual_land_change.csv", index=False)

    # 3. Fast Clock C2: Deweathering + Two-Way Fixed Effects Panel Attribution
    era5_df = pd.read_csv(era5_path)
    deweathered_cube, m_df = dual_clock_pipeline.meteorological_normalisation(no2_cube, meta_df, era5_df)
    panel_res = dual_clock_pipeline.fit_two_way_fixed_effects(df_land, deweathered_cube, m_df)

    panel_summary = {k: v for k, v in panel_res.items() if k != "df_panel"}
    (out_dir / "c2_panel_attribution.json").write_text(json.dumps(panel_summary, indent=2))

    # 4. Component C3 & Fusion: Multi-Horizon Forecasting (5 to 150 days)
    forecast_df = dual_clock_pipeline.run_dual_clock_forecasting(no2_cube, df_land, panel_res, m_df, era5_df)
    forecast_df.to_csv(out_dir / "multi_horizon_results.csv", index=False)

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE! All outputs saved to: {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
