import pytest
import os
import json
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
from airq.data.v2_dataset import V2Dataset, get_strict_split

@pytest.fixture
def mock_dataset_env(tmp_path):
    base_dir = tmp_path / "data"
    s2_dir = base_dir / "s2_composites"
    s5p_dir = base_dir / "s5p_composites"
    cache_dir = tmp_path / "processed" / "v2_cache" / "s2_aligned"
    
    s2_dir.mkdir(parents=True)
    s5p_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    
    # Create 45 sequences of 5-day intervals
    # H=12, K=30. Sequence requires 42 frames. 45 frames give 4 valid sequences.
    dates = pd.date_range('2019-01-01', periods=45, freq='5D')
    manifest_data = []
    
    for i, d in enumerate(dates):
        d_end = d + pd.Timedelta(days=5)
        # Sequence logic: let's fail frame i=2 to test S2 masking
        status = 'failed' if i == 2 else 'ok'
        
        manifest_data.append({
            "window_id": i, 
            "start_date": d.strftime("%Y-%m-%dT00:00:00Z"), 
            "end_date": d_end.strftime("%Y-%m-%dT00:00:00Z"),
            "s2_file": f"s2_{i}.tif", 
            "s5p_file": f"s5p_{i}.tif", 
            "s2_status": status, 
            "s5p_status": "ok", 
            "status": "complete", "timestamp": ""
        })
        
        with rasterio.open(s5p_dir / f"s5p_{i}.tif", 'w', driver='GTiff', width=95, height=123, count=3, dtype=np.float32) as dst: 
            dst.write(np.ones((3,123,95), dtype=np.float32))
            
        if status == 'ok':
            with rasterio.open(cache_dir / f"s2_{i}.tif", 'w', driver='GTiff', width=95, height=123, count=12, dtype=np.float32) as dst: 
                dst.write(np.ones((12,123,95), dtype=np.float32) * 2.0)

    pd.DataFrame(manifest_data).to_csv(base_dir / "dataset_manifest.csv", index=False)
    return base_dir

def test_v2_dataset_sequence_counts(mock_dataset_env):
    dataset = V2Dataset(mock_dataset_env, split='train', h=12, k=30)
    # We generated 45 dates from 2019-01-01 -> all fall in 'train'
    # 45 - 12 - 30 + 1 = 4 sequences
    assert len(dataset) == 4

def test_v2_dataset_tensor_shapes(mock_dataset_env):
    dataset = V2Dataset(mock_dataset_env, split='train', h=12, k=30)
    item = dataset[0]
    
    p1_x = item["p1_x"]
    s2_x = item["s2_x"]
    y = item["y"]
    
    assert p1_x.shape == (12, 3, 123, 95)
    assert s2_x.shape == (12, 12, 123, 95)
    assert y.shape == (30, 3, 123, 95)
    
def test_v2_dataset_causal_safety(mock_dataset_env):
    dataset = V2Dataset(mock_dataset_env, split='train', h=12, k=30)
    
    # Check the first sequence (starts i=0)
    # Forecast origin is i=12 (which is 2019-03-02)
    # Safe S2 must end before or on 2019-03-02 - 20 days = 2019-02-10
    # i=0 ends 2019-01-06 (safe)
    # i=1 ends 2019-01-11 (safe)
    # i=2 ends 2019-01-16 (safe, but s2_status == 'failed' so it's masked!)
    # i=7 ends 2019-02-10 (safe, exactly on the limit)
    # i=8 ends 2019-02-15 (unsafe!)
    
    item = dataset[0]
    s2_x = item["s2_x"]
    
    # i=0 is present (value 2.0)
    assert float(s2_x[0].mean()) == 2.0
    
    # i=2 is missing (value 0.0)
    assert float(s2_x[2].mean()) == 0.0
    
    # i=7 is safe
    assert float(s2_x[7].mean()) == 2.0
    
    # i=8 to i=11 are unsafe (masked to 0.0)
    for i in range(8, 12):
        assert float(s2_x[i].mean()) == 0.0

