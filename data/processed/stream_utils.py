import os
import io
import re
import time
import requests
import rasterio
import numpy as np
import pandas as pd
import gdown

# Shared requests session
_session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=5)
_session.mount('https://', adapter)
_session.mount('http://', adapter)

_global_cache = {}

def scan_drive_folder(drive_url):
    """
    Scans a Google Drive folder and returns a map of filename -> Drive ID.
    """
    items = gdown.download_folder(drive_url, skip_download=True, quiet=True)
    if not items:
        raise ValueError(f"Could not retrieve files from Drive folder: {drive_url}")
    
    id_map = {}
    manifest_id = None
    for item in items:
        fname = os.path.basename(item.path if hasattr(item, 'path') else str(item))
        fid = item.id if hasattr(item, 'id') else ''
        id_map[fname] = fid
        if 'manifest' in fname.lower() and fname.endswith('.csv'):
            manifest_id = fid
            
    return id_map, manifest_id

def stream_geotiff_from_drive(fid, timeout=25):
    """
    Streams a single GeoTIFF file in-memory using its Google Drive ID with robust caching and retry backoff.
    """
    if fid in _global_cache:
        return _global_cache[fid]

    url = f"https://drive.google.com/uc?id={fid}&export=download"
    for attempt in range(6):
        try:
            resp = _session.get(url, timeout=timeout)
            if resp.status_code == 200 and len(resp.content) > 100:
                with rasterio.open(io.BytesIO(resp.content)) as src:
                    data = src.read().astype(np.float32)
                    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                    _global_cache[fid] = data
                    return data
            time.sleep(1.0 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))

    raise IOError(f"Failed to stream GeoTIFF (ID: {fid}) from Drive after 6 retry attempts.")
