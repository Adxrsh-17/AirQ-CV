import os
import io
import re
import requests
import rasterio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
try:
    from data.processed.stream_utils import scan_drive_folder, stream_geotiff_from_drive
except ImportError:
    try:
        from stream_utils import scan_drive_folder, stream_geotiff_from_drive
    except ImportError:
        from data.stream_utils import scan_drive_folder, stream_geotiff_from_drive

class ChennaiForecastingSatelliteDataset(Dataset):
    """
    PyTorch Dataset for Spatiotemporal Multi-Pollutant Forecasting:
    Inputs: Past T_in Sentinel-2 multi-spectral frames (t - T_in + 1 ... t)
    Target: Future Sentinel-5P coarse composite at t + 1 (NO2, CO, SO2)
    
    Temporal Splits:
      - 'train': Target date within 2019-2022 (first 80%)
      - 'val':   Target date within 2019-2022 (last 20%, strictly <= 2022)
      - 'test':  Target date within 2023-2024 (strictly unseen future test period)
    """
    def __init__(
        self,
        local_dir: str = "data",
        drive_url: str = None,
        in_seq_len: int = 4,
        forecast_horizon: int = 1,
        patch_size: int = 64,
        split: str = "train",
        val_ratio: float = 0.2,
        normalize: bool = True,
        max_sequences: int = None
    ):
        super().__init__()
        self.local_dir = local_dir
        self.in_seq_len = in_seq_len
        self.forecast_horizon = forecast_horizon
        self.patch_size = patch_size
        self.split = split
        self.normalize = normalize

        # 1. Load full chronological master pairing
        if local_dir and os.path.exists(local_dir):
            manifest_p = os.path.join(local_dir, "dataset_manifest.csv")
            s2_dir = os.path.join(local_dir, "s2_composites")
            s5p_dir = os.path.join(local_dir, "s5p_composites")

            if os.path.exists(manifest_p):
                df = pd.read_csv(manifest_p)
                s2_files = set(os.listdir(s2_dir)) if os.path.exists(s2_dir) else set()
                s5p_files = set(os.listdir(s5p_dir)) if os.path.exists(s5p_dir) else set()
                df = df[df['s2_file'].isin(s2_files) & df['s5p_file'].isin(s5p_files)].copy()
            else:
                s2_files = sorted([f for f in os.listdir(s2_dir) if f.endswith('.tif')])
                records = []
                for f in s2_files:
                    d_match = re.search(r'(\d{8})', f)
                    if d_match:
                        d = d_match.group(1)
                        s5p_f = f"s5p_{d}.tif"
                        if os.path.exists(os.path.join(s5p_dir, s5p_f)):
                            records.append({'s2_file': f, 's5p_file': s5p_f, 'start_date': f"{d[:4]}-{d[4:6]}-{d[6:]}", 'status': 'complete'})
                df = pd.DataFrame(records)
            self.id_map = None
        elif drive_url:
            self.id_map, manifest_id = scan_drive_folder(drive_url)
            if manifest_id:
                resp = requests.get(f"https://drive.google.com/uc?id={manifest_id}&export=download")
                df = pd.read_csv(io.StringIO(resp.text))
            else:
                raise ValueError("Could not find manifest on Drive")
        else:
            raise ValueError("Must provide either local_dir or drive_url")

        if 'status' in df.columns:
            df = df[df["status"] == "complete"].copy()
        
        # Sort master observations chronologically
        df = df.sort_values("start_date").reset_index(drop=True)
        df["year"] = pd.to_datetime(df["start_date"]).dt.year
        self.master_df = df

        # 2. Build temporal forecasting windows: input=[k - in_seq_len + 1 ... k], target=k + forecast_horizon
        all_sequences = []
        total_steps = len(df)
        
        for k in range(in_seq_len - 1, total_steps - forecast_horizon):
            hist_indices = list(range(k - in_seq_len + 1, k + 1))
            target_index = k + forecast_horizon
            target_year = df.iloc[target_index]["year"]
            target_date = df.iloc[target_index]["start_date"]

            seq_info = {
                "hist_indices": hist_indices,
                "target_index": target_index,
                "target_year": target_year,
                "target_date": target_date
            }
            all_sequences.append(seq_info)

        # 3. Partition into Train (<= 2022), Val (<= 2022), and Test (2023-2024)
        train_val_seqs = [s for s in all_sequences if s["target_year"] <= 2022]
        test_seqs = [s for s in all_sequences if s["target_year"] >= 2023]

        num_val = max(1, int(len(train_val_seqs) * val_ratio))
        num_train = len(train_val_seqs) - num_val

        if split == "train":
            self.sequences = train_val_seqs[:num_train]
        elif split == "val":
            self.sequences = train_val_seqs[num_train:]
        elif split == "test":
            self.sequences = test_seqs
        else:
            self.sequences = all_sequences

        if max_sequences and max_sequences > 0:
            self.sequences = self.sequences[:max_sequences]

        print(f"[{split.upper()} Split] Indexed {len(self.sequences)} forecasting sequences (Past T_in={in_seq_len} -> Future Target).")

    def __len__(self):
        return len(self.sequences)

    def _load_raster(self, fname):
        if self.id_map:
            fid = self.id_map.get(os.path.basename(fname))
            data = stream_geotiff_from_drive(fid)
        else:
            base = os.path.basename(fname)
            sub = 's2_composites' if base.startswith('s2_') else 's5p_composites'
            p = os.path.join(self.local_dir, sub, base)
            with rasterio.open(p) as src:
                data = src.read().astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        hist_indices = seq["hist_indices"]
        target_idx = seq["target_index"]

        # 1. Load historical Sentinel-2 frames
        s2_list = []
        for step_idx in hist_indices:
            row = self.master_df.iloc[step_idx]
            s2_data = self._load_raster(row["s2_file"]) # (12, H, W)

            # Spatial patch crop on S2
            _, H, W = s2_data.shape
            if self.patch_size is not None and H >= self.patch_size and W >= self.patch_size:
                top = (H - self.patch_size) // 2
                left = (W - self.patch_size) // 2
                s2_patch = s2_data[:, top:top + self.patch_size, left:left + self.patch_size]
            else:
                s2_patch = s2_data

            s2_list.append(s2_patch)

        # 2. Load future Sentinel-5P target frame
        target_row = self.master_df.iloc[target_idx]
        s5p_future = self._load_raster(target_row["s5p_file"]) # (3, H_c, W_c)

        s2_history_tensor = torch.from_numpy(np.stack(s2_list)) # (T_in, 12, H_patch, W_patch)
        s5p_future_tensor = torch.from_numpy(s5p_future)        # (3, H_c, W_c)
        raw_s5p_future = s5p_future_tensor.clone()

        # Normalization scaling for stable optimization
        if self.normalize:
            s5p_future_tensor[0, :, :] *= 10000.0 # NO2
            s5p_future_tensor[1, :, :] *= 10.0    # CO
            s5p_future_tensor[2, :, :] *= 10000.0 # SO2

        return {
            "s2_history": s2_history_tensor,
            "s5p_future": s5p_future_tensor,
            "raw_s5p_future": raw_s5p_future,
            "target_date": seq["target_date"]
        }


def prefetch_forecasting_dataset_to_tensors(dataset, max_workers=8):
    """
    Pre-fetches all unique GeoTIFF scenes for the forecasting dataset into RAM.
    Constructs high-speed in-memory PyTorch TensorDataset.
    """
    # Find all unique file paths needed
    needed_s2_files = set()
    needed_s5p_files = set()

    for seq in dataset.sequences:
        for idx in seq["hist_indices"]:
            needed_s2_files.add(dataset.master_df.iloc[idx]["s2_file"])
        needed_s5p_files.add(dataset.master_df.iloc[seq["target_index"]]["s5p_file"])

    all_needed = [(f, 's2') for f in needed_s2_files] + [(f, 's5p') for f in needed_s5p_files]
    print(f"\nPrefetching {len(all_needed)} unique scenes for [{dataset.split.upper()} Split] ({len(dataset)} forecasting sequences)...")

    raster_cache = {}

    def fetch_scene(item):
        fname, ftype = item
        data = dataset._load_raster(fname)
        return fname, data

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_scene, item) for item in all_needed]
        done = 0
        for fut in as_completed(futures):
            fname, data = fut.result()
            raster_cache[fname] = data
            done += 1
            if done % 25 == 0 or done == len(all_needed):
                print(f"  Loaded {done}/{len(all_needed)} unique scenes into RAM...")

    print(f"  Assembling {len(dataset)} forecasting tensors (Past T_in={dataset.in_seq_len} -> Future Target)...")
    s2_list = []
    s5p_list = []
    raw_list = []

    for seq in dataset.sequences:
        hist_s2 = []
        for idx in seq["hist_indices"]:
            s2_f = dataset.master_df.iloc[idx]["s2_file"]
            s2_d = raster_cache[s2_f]

            # Spatial crop
            _, H, W = s2_d.shape
            if dataset.patch_size is not None and H >= dataset.patch_size and W >= dataset.patch_size:
                top = (H - dataset.patch_size) // 2
                left = (W - dataset.patch_size) // 2
                s2_patch = s2_d[:, top:top + dataset.patch_size, left:left + dataset.patch_size]
            else:
                s2_patch = s2_d
            hist_s2.append(s2_patch)

        target_row = dataset.master_df.iloc[seq["target_index"]]
        s5p_d = raster_cache[target_row["s5p_file"]]

        s2_t = torch.from_numpy(np.stack(hist_s2))
        s5p_t = torch.from_numpy(s5p_d.copy())
        raw_t = s5p_t.clone()

        if dataset.normalize:
            s5p_t[0, :, :] *= 10000.0 # NO2
            s5p_t[1, :, :] *= 10.0    # CO
            s5p_t[2, :, :] *= 10000.0 # SO2

        s2_list.append(s2_t)
        s5p_list.append(s5p_t)
        raw_list.append(raw_t)

    s2_tensor = torch.stack(s2_list)
    s5p_tensor = torch.stack(s5p_list)
    raw_tensor = torch.stack(raw_list)

    print(f"  ✓ [{dataset.split.upper()}] Tensor Shapes: S2 History={s2_tensor.shape}, S5P Future={s5p_tensor.shape}")
    return torch.utils.data.TensorDataset(s2_tensor, s5p_tensor, raw_tensor)
