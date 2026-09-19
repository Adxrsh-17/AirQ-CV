import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import json

def load_raster_as_nan(path):
    with rasterio.open(path) as src:
        data = src.read()
        # Convert nodata to nan to exclude from climatology
        nodata = src.nodata
        data = data.astype(np.float32)
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data == -np.inf] = np.nan
        data[data == np.inf] = np.nan
        return data

def main():
    base_dir = Path("data/raw_full438")
    s5p_dir = base_dir / "s5p_composites"
    out_dir = Path("classical_ml/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We use Train set for climatology (2019-01-01 to 2023-01-01)
    train_start = pd.Timestamp('2019-01-01', tz='UTC')
    train_end = pd.Timestamp('2023-01-01', tz='UTC')
    
    s5p_files = sorted(list(s5p_dir.glob("s5p_*.tif")))
    
    # Group by month (1 to 12)
    month_stacks = {m: [] for m in range(1, 13)}
    
    for f in s5p_files:
        date_str = f.stem.split('_')[1]
        dt = pd.Timestamp(date_str, tz='UTC')
        if train_start <= dt <= train_end:
            data = load_raster_as_nan(f)
            # data is (3, H, W)
            month_stacks[dt.month].append(data)
            
    # Calculate Monthly Mean and Std
    # Shape: (12, 3, 123, 95)
    climatology_mean = np.zeros((12, 3, 123, 95), dtype=np.float32)
    climatology_std = np.zeros((12, 3, 123, 95), dtype=np.float32)
    
    print("Building Climatology (Monthly Average Maps)...")
    for m in range(1, 13):
        stack = np.stack(month_stacks[m], axis=0) # (N, 3, H, W)
        with np.errstate(all='ignore'):
            # Some pixels might be entirely NaN in a month, handle warnings
            c_mean = np.nanmean(stack, axis=0)
            c_std = np.nanstd(stack, axis=0)
            
        # Fill completely NaN pixels with the global spatial mean for that month
        for c in range(3):
            nan_mask = np.isnan(c_mean[c])
            if np.any(nan_mask):
                global_c_mean = np.nanmean(c_mean[c])
                c_mean[c][nan_mask] = global_c_mean
                
            nan_mask_std = np.isnan(c_std[c])
            if np.any(nan_mask_std):
                global_c_std = np.nanmean(c_std[c])
                c_std[c][nan_mask_std] = global_c_std
                
        climatology_mean[m-1] = c_mean
        climatology_std[m-1] = c_std
        
    np.save(out_dir / "climatology_mean.npy", climatology_mean)
    np.save(out_dir / "climatology_std.npy", climatology_std)
    print(f"Saved climatology arrays to {out_dir}")
    
    # Identify SO2 Hotspots (Annual Mean)
    annual_mean = np.nanmean(climatology_mean, axis=0) # (3, H, W)
    so2_annual = annual_mean[2]
    # 99th percentile threshold for hotspot
    threshold = np.percentile(so2_annual, 99)
    hotspots = (so2_annual >= threshold).astype(int)
    np.save(out_dir / "so2_hotspots.npy", hotspots)
    
    hotspot_y, hotspot_x = np.where(hotspots == 1)
    print(f"Identified {len(hotspot_x)} SO2 hotspot pixels (threshold: {threshold:.2e}).")
    
    # Pre-calculate distance map for SO2
    # distance to nearest hotspot
    Y, X = np.indices((123, 95))
    dist_map = np.full((123, 95), np.inf)
    for hy, hx in zip(hotspot_y, hotspot_x):
        d = np.sqrt((Y - hy)**2 + (X - hx)**2)
        dist_map = np.minimum(dist_map, d)
        
    np.save(out_dir / "so2_dist_map.npy", dist_map)
    print(f"Saved SO2 distance map.")

if __name__ == "__main__":
    main()

