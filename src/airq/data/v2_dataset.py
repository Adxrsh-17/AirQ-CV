import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path

def get_strict_split(start: pd.Timestamp, end: pd.Timestamp) -> str:
    t_19 = pd.Timestamp('2019-01-01', tz='UTC')
    t_23 = pd.Timestamp('2023-01-01', tz='UTC')
    t_24 = pd.Timestamp('2024-01-01', tz='UTC')
    t_25 = pd.Timestamp('2025-01-01', tz='UTC')
    if start >= t_19 and end <= t_23: return 'train'
    if start >= t_23 and end <= t_24: return 'validation'
    if start >= t_24 and end <= t_25: return 'test'
    return None

import json
from concurrent.futures import ThreadPoolExecutor

class V2Dataset(Dataset):
    def __init__(self, base_dir: Path, split: str, h=12, k=30, preload: bool = True):
        self.base_dir = Path(base_dir)
        self.s5p_dir = self.base_dir / "s5p_composites"
        # We read aligned S2 cache instead of raw
        self.s2_cache_dir = self.base_dir.parent / "processed" / "v2_cache" / "s2_aligned"
        
        self.split = split
        self.h = h
        self.k = k
        self.sequences = self._build_sequences()
        self.cache = {}
        self.preload = preload
        if self.preload and len(self.sequences) > 0:
            self._preload_all()
            
        # Normalization
        self.stats_file = self.s2_cache_dir.parent / "norm_stats.json"
        self.normalize = False
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            self.s5p_mean = torch.tensor(stats['s5p_mean'], dtype=torch.float32).view(-1, 1, 1)
            self.s5p_std = torch.tensor(stats['s5p_std'], dtype=torch.float32).view(-1, 1, 1)
            self.s2_mean = torch.tensor(stats['s2_mean'], dtype=torch.float32).view(-1, 1, 1)
            self.s2_std = torch.tensor(stats['s2_std'], dtype=torch.float32).view(-1, 1, 1)
            self.normalize = True

    def _preload_all(self):
        s5p_files = {f for s in self.sequences for f in s["s5p_history"] + s["s5p_target"]}
        s2_files = {f for s in self.sequences for f in s["s2_history"] if f is not None}
        
        def load_s5p(fname):
            p = self.s5p_dir / fname
            if p.exists():
                return fname, self._load_raster(p)
            return fname, None
            
        def load_s2(fname):
            p = self.s2_cache_dir / fname
            if p.exists():
                return fname, self._load_raster(p)
            return fname, None

        with ThreadPoolExecutor(max_workers=16) as executor:
            for fname, arr in executor.map(load_s5p, s5p_files):
                if arr is not None:
                    self.cache[fname] = arr
            for fname, arr in executor.map(load_s2, s2_files):
                if arr is not None:
                    self.cache[fname] = arr
        
    def _build_sequences(self):
        manifest = pd.read_csv(self.base_dir / "dataset_manifest.csv")
        manifest['start_date_dt'] = pd.to_datetime(manifest['start_date'], utc=True)
        manifest['end_date_dt'] = pd.to_datetime(manifest['end_date'], utc=True)
        manifest = manifest.sort_values('start_date_dt').reset_index(drop=True)
        
        sequences = []
        for i in range(len(manifest) - self.h - self.k + 1):
            seq = manifest.iloc[i:i+self.h+self.k]
            
            # Check cadence
            is_valid_cadence = True
            for j in range(len(seq)-1):
                if (seq.iloc[j+1]['start_date_dt'] - seq.iloc[j]['start_date_dt']).days != 5:
                    is_valid_cadence = False
                    break
                if (seq.iloc[j]['end_date_dt'] - seq.iloc[j]['start_date_dt']).days != 5:
                    is_valid_cadence = False
                    break
                    
            if (seq.iloc[-1]['end_date_dt'] - seq.iloc[-1]['start_date_dt']).days != 5:
                is_valid_cadence = False
                
            if not is_valid_cadence:
                continue
                
            history = seq.iloc[:self.h]
            targets = seq.iloc[self.h:]
            
            h_split = get_strict_split(history.iloc[0]['start_date_dt'], history.iloc[-1]['end_date_dt'])
            t_split = get_strict_split(targets.iloc[0]['start_date_dt'], targets.iloc[-1]['end_date_dt'])
            
            if h_split and h_split == t_split and h_split == self.split:
                forecast_origin = targets.iloc[0]['start_date_dt']
                
                # Check causal safety for each S2 frame in history
                s2_files = []
                for _, hist_row in history.iterrows():
                    s2_nominal_end = hist_row['end_date_dt']
                    if s2_nominal_end + pd.Timedelta(days=20) <= forecast_origin:
                        if hist_row['s2_status'] == 'ok':
                            s2_files.append(hist_row['s2_file'])
                        else:
                            s2_files.append(None) # Missing
                    else:
                        s2_files.append(None) # Not causally safe
                        
                sequences.append({
                    "s5p_history": history['s5p_file'].tolist(),
                    "s2_history": s2_files,
                    "s5p_target": targets['s5p_file'].tolist(),
                    "start_date": str(history.iloc[0]['start_date']),
                    "target_start": str(forecast_origin)
                })
                
        return sequences
        
    def __len__(self):
        return len(self.sequences)
        
    def _load_raster(self, path: Path):
        with rasterio.open(path, 'r') as src:
            data = src.read()
            data[data == -np.inf] = 0.0
            data = np.nan_to_num(data, 0.0)
            return data.astype(np.float32)
            
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Load S5P history
        cache = getattr(self, 'cache', {})
        p1_x = []
        for f in seq["s5p_history"]:
            if f in cache:
                p1_x.append(cache[f])
            else:
                p1_x.append(self._load_raster(self.s5p_dir / f))
        p1_x = np.stack(p1_x) # (H, 3, 123, 95)
        
        # Load S2 history (padding missing/unsafe frames with 0)
        s2_x = []
        for f in seq["s2_history"]:
            if f is not None:
                if f in cache:
                    s2_x.append(cache[f])
                elif (self.s2_cache_dir / f).exists():
                    s2_x.append(self._load_raster(self.s2_cache_dir / f))
                else:
                    s2_x.append(np.zeros((12, 123, 95), dtype=np.float32))
            else:
                s2_x.append(np.zeros((12, 123, 95), dtype=np.float32))
        s2_x = np.stack(s2_x) # (H, 12, 123, 95)
        
        # Load S5P targets
        y = []
        for f in seq["s5p_target"]:
            if f in cache:
                y.append(cache[f])
            else:
                y.append(self._load_raster(self.s5p_dir / f))
        y = np.stack(y) # (K, 3, 123, 95)
        
        p1_x = torch.from_numpy(p1_x)
        s2_x = torch.from_numpy(s2_x)
        y = torch.from_numpy(y)

        if self.normalize:
            eps = 1e-8
            p1_x = (p1_x - self.s5p_mean) / (self.s5p_std + eps)
            s2_x = (s2_x - self.s2_mean) / (self.s2_std + eps)
            y = (y - self.s5p_mean) / (self.s5p_std + eps)

        return {
            "p1_x": p1_x,
            "s2_x": s2_x,
            "y": y,
        }

