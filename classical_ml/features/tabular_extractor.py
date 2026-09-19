import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data == -np.inf] = np.nan
        data[data == np.inf] = np.nan
        return data

def extract_features(split='train'):
    base_dir = Path("data/raw_full438")
    s5p_dir = base_dir / "s5p_composites"
    s2_dir = Path("data/processed/v2_cache/s2_aligned")
    cml_data_dir = Path("classical_ml/data")
    
    # Load Climatology & SO2 distance map
    clim_mean = np.load(cml_data_dir / "climatology_mean.npy") # (12, 3, H, W)
    clim_std = np.load(cml_data_dir / "climatology_std.npy")   # (12, 3, H, W)
    so2_dist = np.load(cml_data_dir / "so2_dist_map.npy")      # (H, W)
    
    # Splits
    t_19 = pd.Timestamp('2019-01-01', tz='UTC')
    t_23 = pd.Timestamp('2023-01-01', tz='UTC')
    t_24 = pd.Timestamp('2024-01-01', tz='UTC')
    t_25 = pd.Timestamp('2025-01-01', tz='UTC')
    
    s5p_files = sorted(list(s5p_dir.glob("s5p_*.tif")))
    
    records = []
    
    Y, X = np.indices((123, 95))
    x_norm = X.flatten() / 95.0
    y_norm = Y.flatten() / 123.0
    so2_dist_flat = so2_dist.flatten()
    
    print(f"Extracting {split} tabular features...")
    for f in s5p_files:
        date_str = f.stem.split('_')[1]
        dt = pd.Timestamp(date_str, tz='UTC')
        
        # Check split
        if split == 'train' and not (t_19 <= dt <= t_23): continue
        if split == 'validation' and not (t_23 < dt <= t_24): continue
        if split == 'test' and not (t_24 < dt <= t_25): continue
        
        # We need S2 for the same date or closest past date
        s2_file = s2_dir / f"s2_{date_str}.tif"
        if not s2_file.exists():
            continue
            
        # Load S5P
        s5p_data = load_raster(f) # (3, H, W)
        # Load S2
        s2_data = load_raster(s2_file) # (12, H, W)
        
        # Calendar encoding
        doy = dt.dayofyear
        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        month_idx = dt.month - 1
        
        # Flatten
        s5p_flat = s5p_data.reshape(3, -1)
        s2_flat = s2_data.reshape(12, -1)
        
        clim_mean_flat = clim_mean[month_idx].reshape(3, -1)
        
        # NDVI, NDBI, NDMI are indices 9, 10, 11 (assuming S2 candidate order)
        ndvi = s2_flat[9]
        ndbi = s2_flat[10]
        ndmi = s2_flat[11]
        
        # For this date, we create a DataFrame where each row is a pixel
        df = pd.DataFrame({
            'date': [dt] * 11685,
            'x_norm': x_norm,
            'y_norm': y_norm,
            'sin_doy': [sin_doy] * 11685,
            'cos_doy': [cos_doy] * 11685,
            'ndvi': ndvi,
            'ndbi': ndbi,
            'ndmi': ndmi,
            'so2_dist': so2_dist_flat,
            'no2_clim_mean': clim_mean_flat[0],
            'co_clim_mean': clim_mean_flat[1],
            'so2_clim_mean': clim_mean_flat[2],
            'no2_target': s5p_flat[0],
            'co_target': s5p_flat[1],
            'so2_target': s5p_flat[2],
        })
        
        # Drop rows where target is NaN (missing S5P data)
        df = df.dropna(subset=['no2_target', 'co_target', 'so2_target'])
        records.append(df)
        
    full_df = pd.concat(records, ignore_index=True)
    out_path = cml_data_dir / f"{split}_tabular.parquet"
    full_df.to_parquet(out_path, index=False)
    print(f"Saved {len(full_df)} rows to {out_path}")

def main():
    extract_features('train')
    extract_features('test')

if __name__ == "__main__":
    main()

