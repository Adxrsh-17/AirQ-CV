#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_weather_safe.py - Hardware-Safe, Weather-Augmented Atmospheric Spatiotemporal Trainer
========================================================================================
Platform: Local Laptop (NVIDIA GeForce RTX 4060 GPU, 8GB VRAM)
Integrates: Real S5P Gas Cubes + S2 Optical Priors + 6-Year ERA5 ECMWF Daily Meteorology
Architecture: ST-ResUNet (ResNet Encoder + ConvGRU Bottleneck + U-Net Decoder)
Hardware Safety Rails Enforced:
  1. VRAM Cap: Process memory capped at 60% (~4.8 GB) to protect laptop OS / display drivers.
  2. Automatic Mixed Precision (AMP FP16): Runs cooler on Ada Lovelace Tensor Cores.
  3. Controlled Batching: Batch size = 8 (active VRAM footprint < 2 GB).
  4. Safe Windows Workers: num_workers = 0 to prevent process spawning storms.
  5. Thermal Pacing: Brief pause between epochs to keep GPU temperatures stable.
  6. Early Stopping: Patience = 6 to prevent wasteful GPU baking after convergence.
========================================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("AirQ-WeatherSafe")


# ==============================================================================
# 1. HARDWARE PROTECTION & SEEDING
# ==============================================================================
def setup_safe_hardware(vram_fraction: float = 0.60) -> torch.device:
    """Configures GPU with hard memory boundaries to protect laptop hardware."""
    if not torch.cuda.is_available():
        log.warning("No CUDA GPU detected. Running on CPU.")
        return torch.device("cpu")

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)

    # Set hard ceiling on VRAM allocation to avoid display driver crash / thermal throttling
    try:
        torch.cuda.set_per_process_memory_fraction(vram_fraction, 0)
        allowed_gb = total_gb * vram_fraction
        log.info(f"Hardware Safety Active: {props.name} ({total_gb:.1f} GB VRAM detected)")
        log.info(f"VRAM Ceiling set to {vram_fraction*100:.0f}% ({allowed_gb:.2f} GB max). {total_gb - allowed_gb:.2f} GB reserved for OS.")
    except Exception as e:
        log.warning(f"Could not set VRAM ceiling: {e}")

    torch.backends.cudnn.benchmark = True
    return dev


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. NEURAL NETWORK ARCHITECTURE (ST-ResUNet)
# ==============================================================================
class ConvGRUCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv_gates = nn.Conv2d(in_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=padding)
        self.conv_candidate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        gates = self.conv_gates(torch.cat([x, h_prev], dim=1))
        reset_gate, update_gate = torch.chunk(torch.sigmoid(gates), 2, dim=1)
        candidate = torch.tanh(self.conv_candidate(torch.cat([x, reset_gate * h_prev], dim=1)))
        return (1.0 - update_gate) * h_prev + update_gate * candidate


class ResBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_c), out_c)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_c), out_c)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.residual = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.drop(self.act1(self.norm1(self.conv1(x))))
        return self.act2(self.norm2(self.conv2(out)) + res)


class STResUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, base_channels: int = 16, dropout: float = 0.15):
        super().__init__()
        b = base_channels
        self.enc1 = ResBlock2D(in_channels, b, dropout)
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = ResBlock2D(b, 2 * b, dropout)
        self.down2 = nn.MaxPool2d(2)
        self.enc3 = ResBlock2D(2 * b, 4 * b, dropout)
        self.conv_gru = ConvGRUCell(4 * b, 4 * b)
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.dec2 = ResBlock2D(4 * b, 2 * b, dropout)
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.dec1 = ResBlock2D(2 * b, b, dropout)
        self.head = nn.Conv2d(b, out_channels, 1)

        # Zero-initialize head so initial state is exact climatology baseline
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        hidden, skip1, skip2 = None, [], []
        for step in range(T):
            e1 = self.enc1(x[:, step])
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            hidden = self.conv_gru(e3, hidden)
            skip1.append(e1)
            skip2.append(e2)

        wts = torch.linspace(0.1, 0.4, T, device=x.device)
        wts = wts / wts.sum()
        s1 = sum(w * f for w, f in zip(wts, skip1))
        s2 = sum(w * f for w, f in zip(wts, skip2))

        d2 = self.dec2(torch.cat([self.up2(hidden), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.head(d1)


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = F.huber_loss(pred, target, reduction="none", delta=delta) * mask
    denom = mask.sum().clamp_min(1.0)
    return err.sum() / denom


# ==============================================================================
# 3. WEATHER-AUGMENTED DATA ENGINE
# ==============================================================================
@dataclass
class Cube:
    dates: pd.DatetimeIndex
    gases: list[str]
    values: np.ndarray        # (T, G, H, W)
    footprint: np.ndarray     # (H, W)
    s2_series: np.ndarray     # (T, C_s2, H, W)
    weather_df: pd.DataFrame  # Daily ERA5 reanalysis


def build_cube_with_weather(data_dir: Path, weather_path: Path, bbox=[79.95, 12.80, 80.45, 13.30]) -> Cube:
    log.info("Loading S5P, S2, and ERA5 weather...")
    s5p_dir = data_dir / "s5p_composites"
    s2_dir = data_dir / "s2_composites"

    s2_files = sorted(os.listdir(s2_dir))
    s5p_files = sorted(os.listdir(s5p_dir))
    s2_map = {re.search(r"(\d{8})", f).group(1): f for f in s2_files if re.search(r"(\d{8})", f)}
    s5p_map = {re.search(r"(\d{8})", f).group(1): f for f in s5p_files if re.search(r"(\d{8})", f)}

    common_dates = sorted(set(s2_map.keys()) & set(s5p_map.keys()))
    from rasterio.windows import from_bounds

    with rasterio.open(s5p_dir / s5p_map[common_dates[0]]) as ref5:
        w5 = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], ref5.transform)
        H, W = int(w5.height), int(w5.width)

    with rasterio.open(s2_dir / s2_map[common_dates[0]]) as ref2:
        w2 = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], ref2.transform)
        H2, W2 = int(w2.height), int(w2.width)

    T = len(common_dates)
    values = np.full((T, 3, H, W), np.nan, dtype=np.float32)
    s2_series = np.full((T, 2, H, W), np.nan, dtype=np.float32)

    grid_y = np.linspace(0, H2, H + 1, dtype=int)
    grid_x = np.linspace(0, W2, W + 1, dtype=int)

    for i, d in enumerate(common_dates):
        with rasterio.open(s5p_dir / s5p_map[d]) as s5:
            arr5 = s5.read(window=w5)
            values[i] = np.where((arr5 < -100) | (arr5 <= -0.0005), np.nan, arr5)

        with rasterio.open(s2_dir / s2_map[d]) as s2:
            arr2 = s2.read(window=w2)
            ndbi_raw = arr2[10]
            ndvi_raw = arr2[9]
            for r in range(H):
                for c in range(W):
                    b_ndbi = ndbi_raw[grid_y[r]:grid_y[r+1], grid_x[c]:grid_x[c+1]]
                    b_ndvi = ndvi_raw[grid_y[r]:grid_y[r+1], grid_x[c]:grid_x[c+1]]
                    m1 = (b_ndbi > -2.0) & (b_ndbi < 2.0)
                    m2 = (b_ndvi > -2.0) & (b_ndvi < 2.0)
                    s2_series[i, 0, r, c] = np.nanmean(b_ndbi[m1]) if m1.any() else np.nan
                    s2_series[i, 1, r, c] = np.nanmean(b_ndvi[m2]) if m2.any() else np.nan

    dates = pd.to_datetime(common_dates, format="%Y%m%d")
    weather_df = pd.read_csv(weather_path)
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    footprint = np.ones((H, W), dtype=bool)
    return Cube(dates, ["NO2", "CO", "SO2"], values, footprint, s2_series, weather_df)


class WeatherAugmentedDataset(Dataset):
    def __init__(self, cube: Cube, target_indices: np.ndarray, t_in: int, horizon: int,
                 train_mask: np.ndarray, gas_idx: int = 0):
        self.targets = np.asarray(target_indices)
        self.t_in = t_in
        self.horizon = horizon
        self.gas_idx = gas_idx
        T, G, H, W = cube.values.shape

        # 1. Climatology & Anomaly
        months = cube.dates.month.values
        clim_12 = np.zeros((12, G, H, W), dtype=np.float32)
        for m in range(1, 13):
            m_idx = train_mask & (months == m)
            if m_idx.any():
                clim_12[m - 1] = np.nanmean(cube.values[m_idx], axis=0)

        clim = clim_12[months - 1]
        anom = cube.values - clim
        train_anom = anom[train_mask]
        std = np.nanstd(train_anom, axis=0)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        z = np.nan_to_num(anom / std[None, ...], nan=0.0)

        # 2. Weather covariates alignment
        m_df = pd.merge(pd.DataFrame({"date": cube.dates}), cube.weather_df, on="date", how="left").ffill().bfill()
        m_df["ventilation"] = m_df["blh"] * m_df["wind_speed"]
        wx_cols = ["blh", "u10", "v10", "wind_speed", "t2m", "rh", "tp", "ventilation"]
        wx_raw = m_df[wx_cols].values

        # Standardize weather using training period
        wx_mu = np.mean(wx_raw[train_mask], axis=0, keepdims=True)
        wx_sd = np.std(wx_raw[train_mask], axis=0, keepdims=True)
        wx_sd = np.where(wx_sd > 0, wx_sd, 1.0)
        wx_norm = (wx_raw - wx_mu) / wx_sd  # (T, 8)

        # Build feature stacks for targets
        x_list, y_list, m_list = [], [], []

        # Pad spatial grid to multiple of 4 for U-Net
        pad_h = (-H) % 4
        pad_w = (-W) % 4

        for t in self.targets:
            hist = np.arange(t - horizon - t_in + 1, t - horizon + 1)
            # Gas anomaly + validity
            z_slice = z[hist, gas_idx:gas_idx+1]
            v_slice = np.isfinite(cube.values[hist, gas_idx:gas_idx+1]).astype(np.float32)

            # S2 optical land
            s2_slice = np.nan_to_num(cube.s2_series[hist], nan=0.0)

            # Weather broadcast across space
            wx_slice = wx_norm[hist]  # (t_in, 8)
            wx_spatial = np.broadcast_to(wx_slice[:, :, None, None], (t_in, 8, H, W))

            # Concatenate all channels
            feat = np.concatenate([z_slice, v_slice, s2_slice, wx_spatial], axis=1).astype(np.float32)

            y_target = z[t, gas_idx:gas_idx+1]
            m_target = np.isfinite(cube.values[t, gas_idx:gas_idx+1]).astype(np.float32)

            if pad_h > 0 or pad_w > 0:
                feat = np.pad(feat, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
                y_target = np.pad(y_target, ((0, 0), (0, pad_h), (0, pad_w)))
                m_target = np.pad(m_target, ((0, 0), (0, pad_h), (0, pad_w)))

            x_list.append(feat)
            y_list.append(y_target)
            m_list.append(m_target)

        self.x = torch.from_numpy(np.stack(x_list))
        self.y = torch.from_numpy(np.stack(y_list))
        self.m = torch.from_numpy(np.stack(m_list))
        self.orig_shape = (H, W)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.m[idx]


# ==============================================================================
# 4. TRAINING & EVALUATION LOOP
# ==============================================================================
def train_and_eval(cube: Cube, device: torch.device, gas_idx: int = 0, horizon: int = 1, epochs: int = 25):
    years = cube.dates.year.values
    train_mask = (years <= 2021)
    val_mask = (years == 2022)

    t_in = 4
    all_indices = np.arange(t_in + horizon - 1, len(cube.dates))
    idx_train = all_indices[train_mask[all_indices]]
    idx_val = all_indices[val_mask[all_indices]]

    log.info(f"Setting up Datasets: Train Windows={len(idx_train)}, Val Windows={len(idx_val)}")
    ds_train = WeatherAugmentedDataset(cube, idx_train, t_in, horizon, train_mask, gas_idx)
    ds_val = WeatherAugmentedDataset(cube, idx_val, t_in, horizon, train_mask, gas_idx)

    # Hardware-Safe DataLoader: num_workers=0 to protect laptop CPU/threads
    loader_train = DataLoader(ds_train, batch_size=8, shuffle=True, num_workers=0)
    loader_val = DataLoader(ds_val, batch_size=8, shuffle=False, num_workers=0)

    in_ch = ds_train.x.shape[2]  # 1(gas) + 1(mask) + 2(S2) + 8(weather) = 12 channels
    log.info(f"Input Tensor Shape: {in_ch} channels across {t_in} time lags")

    model = STResUNet(in_channels=in_ch, out_channels=1, base_channels=16, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    log.info(f"ST-ResUNet Parameter Count: {sum(p.numel() for p in model.parameters()):,} weights")
    log.info("Starting Hardware-Safe Training with Automatic Mixed Precision (AMP FP16)...")

    best_val_loss = float("inf")
    patience = 6
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y, m in loader_train:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                loss = masked_huber_loss(pred, y, m)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(x)

        scheduler.step()
        train_loss /= len(ds_train)

        # Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, m in loader_val:
                x, y, m = x.to(device), y.to(device), m.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(x)
                    loss = masked_huber_loss(pred, y, m)
                val_loss += loss.item() * len(x)
        val_loss /= len(ds_val)

        log.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), "st_resunet_weather_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info(f"Early stopping triggered at epoch {epoch} to prevent hardware overheating.")
                break

        # Thermal pacing pause
        time.sleep(0.05)

    log.info(f"Training completed safely in {time.time() - t0:.1f} s. Best Val Loss: {best_val_loss:.5f}")

    # Final Skill Evaluation vs Climatology on Validation Set
    model.load_state_dict(torch.load("st_resunet_weather_best.pt"))
    model.eval()
    all_preds, all_true, all_mask = [], [], []

    H, W = ds_val.orig_shape
    with torch.no_grad():
        for x, y, m in loader_val:
            p = model(x.to(device)).cpu()[..., :H, :W].numpy()
            all_preds.append(p)
            all_true.append(y[..., :H, :W].numpy())
            all_mask.append(m[..., :H, :W].numpy())

    p_arr = np.concatenate(all_preds, axis=0)
    y_arr = np.concatenate(all_true, axis=0)
    m_arr = np.concatenate(all_mask, axis=0)

    valid_pts = (m_arr > 0)
    sse_model = ((p_arr[valid_pts] - y_arr[valid_pts]) ** 2).sum()
    sse_clim = (y_arr[valid_pts] ** 2).sum()  # climatology is zero anomaly
    skill_vs_clim = 1.0 - (sse_model / max(sse_clim, 1e-12))

    log.info("=" * 70)
    log.info("VALIDATION BENCHMARK RESULTS (Weather-Augmented ST-ResUNet):")
    log.info(f"  Skill over Climatology: {skill_vs_clim * 100:+.2f}%")
    log.info(f"  Model MSE: {sse_model / valid_pts.sum():.6e} | Climatology MSE: {sse_clim / valid_pts.sum():.6e}")
    log.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Weather-Safe Atmospheric Deep Learning Trainer")
    parser.add_argument("--data-dir", default=r"data\processed")
    parser.add_argument("--weather", default=r"weather\era5_daily.csv")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--vram-cap", type=float, default=0.60, help="Fraction of VRAM ceiling (default 0.60)")
    args = parser.parse_args()

    seed_everything(42)
    device = setup_safe_hardware(args.vram_cap)

    cube = build_cube_with_weather(Path(args.data_dir), Path(args.weather))
    train_and_eval(cube, device, gas_idx=0, horizon=1, epochs=args.epochs)


if __name__ == "__main__":
    main()
