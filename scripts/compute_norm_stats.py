import numpy as np
import json
from pathlib import Path
from airq.data.v2_dataset import V2Dataset
import argparse

def main():
    print("Loading training dataset to compute normalization statistics...")
    ds = V2Dataset('data/raw_full438', split='train', preload=True)
    
    s5p_sum = np.zeros(3)
    s5p_sq_sum = np.zeros(3)
    s5p_count = 0
    
    s2_sum = np.zeros(12)
    s2_sq_sum = np.zeros(12)
    s2_count = 0
    
    for i in range(len(ds)):
        item = ds[i]
        p1_x = item['p1_x'].numpy() # (H, 3, 123, 95)
        y = item['y'].numpy()       # (K, 3, 123, 95)
        s2_x = item['s2_x'].numpy() # (H, 12, 123, 95)
        
        s5p_all = np.concatenate([p1_x, y], axis=0) 
        
        # Aggregate per channel S5P
        s5p_sum += s5p_all.sum(axis=(0, 2, 3))
        s5p_sq_sum += (s5p_all ** 2).sum(axis=(0, 2, 3))
        s5p_count += s5p_all.shape[0] * s5p_all.shape[2] * s5p_all.shape[3]
        
        # Aggregate per channel S2
        s2_sum += s2_x.sum(axis=(0, 2, 3))
        s2_sq_sum += (s2_x ** 2).sum(axis=(0, 2, 3))
        s2_count += s2_x.shape[0] * s2_x.shape[2] * s2_x.shape[3]
        
    s5p_mean = s5p_sum / s5p_count
    s5p_std = np.sqrt((s5p_sq_sum / s5p_count) - (s5p_mean ** 2))
    
    s2_mean = s2_sum / s2_count
    s2_std = np.sqrt((s2_sq_sum / s2_count) - (s2_mean ** 2))
    
    stats = {
        's5p_mean': s5p_mean.tolist(),
        's5p_std': s5p_std.tolist(),
        's2_mean': s2_mean.tolist(),
        's2_std': s2_std.tolist(),
    }
    
    out_path = Path('data/processed/v2_cache/norm_stats.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
        
    print(f"\nSaved normalization stats to {out_path}")
    print("S5P Mean:", stats['s5p_mean'])
    print("S5P Std :", stats['s5p_std'])
    
if __name__ == "__main__":
    main()
