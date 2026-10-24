"""Strict S2<->S5P alignment: S2 is area-averaged onto an exact high-res subgrid per S5P cell (valid-pixel weighted)."""
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine

R = 32   # S2 sub-pixels per S5P cell, per axis (~150m high-resolution; r=40 preserves full ~100m native grid)

def s5p_grid(s5p_path):
    with rasterio.open(s5p_path) as src:
        return src.transform, src.height, src.width, src.crs

def align_s2_to_s5p(s2_path, s5p_transform, H5, W5, crs, r=R):
    """returns S2 (12, r*H5, r*W5) float32 and valid (1, r*H5, r*W5) in [0,1] = fraction of valid sub-pixels."""
    dst_tf = s5p_transform * Affine.scale(1.0 / r, 1.0 / r)           # exact sub-grid of the S5P grid
    with rasterio.open(s2_path) as src:
        x = src.read().astype(np.float32)                              # (12,h,w)
        valid = (np.nan_to_num(x[:9]) != 0).any(0).astype(np.float32)  # all 9 reflectance bands == 0  -> NoData/ocean fill
        x = np.nan_to_num(x, nan=0.0) * valid[None]
        out = np.zeros((x.shape[0], H5 * r, W5 * r), np.float32); vout = np.zeros((1, H5 * r, W5 * r), np.float32)
        for b in range(x.shape[0]):
            reproject(x[b], out[b], src_transform=src.transform, src_crs=src.crs, dst_transform=dst_tf,
                      dst_crs=crs, resampling=Resampling.average)
        reproject(valid, vout[0], src_transform=src.transform, src_crs=src.crs, dst_transform=dst_tf,
                  dst_crs=crs, resampling=Resampling.average)
    out = out / np.clip(vout, 1e-6, None)                              # valid-weighted mean
    out[:, vout[0] < 0.05] = 0.0
    return out, vout
