import os
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def get_s5p_profile(base_dir: Path):
    manifest = pd.read_csv(base_dir / "dataset_manifest.csv")
    first_s5p = base_dir / "s5p_composites" / manifest.iloc[0]['s5p_file']
    with rasterio.open(first_s5p) as src:
        return src.profile

def process_file(args):
    s2_path, out_path, s5p_profile = args
    if out_path.exists():
        return
        
    try:
        with rasterio.open(s2_path) as src:
            s2_data = src.read()
            s2_profile = src.profile
            
            # Prepare destination array
            out_data = np.zeros(
                (src.count, s5p_profile['height'], s5p_profile['width']), 
                dtype=s5p_profile['dtype']
            )
            
            reproject(
                source=s2_data,
                destination=out_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=s5p_profile['transform'],
                dst_crs=s5p_profile['crs'],
                resampling=Resampling.bilinear,
                src_nodata=-np.inf,
                dst_nodata=-np.inf
            )
            
            out_profile = s5p_profile.copy()
            out_profile.update({
                'count': src.count,
                'dtype': 'float32',
                'nodata': -np.inf
            })
            
            with rasterio.open(out_path, 'w', **out_profile) as dst:
                dst.write(out_data.astype(np.float32))
    except Exception as e:
        print(f"Failed to process {s2_path.name}: {e}")

def main():
    base_dir = Path("data/raw_full438")
    out_dir = Path("data/processed/v2_cache/s2_aligned")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = pd.read_csv(base_dir / "dataset_manifest.csv")
    s5p_profile = get_s5p_profile(base_dir)
    
    tasks = []
    for _, row in manifest.iterrows():
        if row['s2_status'] == 'ok':
            s2_path = base_dir / "s2_composites" / row['s2_file']
            out_path = out_dir / row['s2_file']
            tasks.append((s2_path, out_path, s5p_profile))
            
    print(f"Starting alignment for {len(tasks)} files...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(executor.map(process_file, tasks))
        
if __name__ == '__main__':
    main()
