#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scratch/train_multi_horizon.py
Multi-Horizon Spatiotemporal Training & Evaluation Suite for Phase 7
Supports Horizon=1 (5-day), Horizon=30 (150-day), Horizon=60 (300-day)
Using Phase 6 Decoupled Pollutant-Specific Decoder Heads Architecture
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add Projec-Code to sys.path
proj_dir = r"C:\Users\Adarsh_Pradeep\OneDrive\Desktop\4th Yr Sem 7\Project Phase 1\Projects\Projec-Code"
if proj_dir not in sys.path:
    sys.path.insert(0, proj_dir)

from training.train import (
    STResUNet, PlumePreservingCompoundLoss, CONFIG, DEVICE,
    invert_so2_prediction
)
from evaluation.metrics import compute_all_metrics

class MultiHorizonDataset(Dataset):
    """
    Multi-horizon spatiotemporal dataset supporting arbitrary lead time horizons.
    Horizon=1  -> t_target = t + 1  (5 days ahead)
    Horizon=30 -> t_target = t + 30 (150 days ahead)
    Horizon=60 -> t_target = t + 60 (300 days ahead)
    """
    _cached_data = None

    def __init__(self, data_dir=None, split="train", t_in=4, horizon=1, h=128, w=128, augment=True, custom_sequences=None):
        self.data_dir = data_dir or os.path.join(proj_dir, "data", "processed")
        self.split = split
        self.t_in = t_in
        self.horizon = horizon
        self.h = h
        self.w = w
        self.augment = augment and (split == "train")

        if MultiHorizonDataset._cached_data is None:
            cache_pt = os.path.join(self.data_dir, "cached_dataset_256.pt")
            if os.path.exists(cache_pt):
                print(f"[Dataset Cache] Loading pre-compiled tensors from {cache_pt} into RAM...")
                MultiHorizonDataset._cached_data = torch.load(cache_pt, map_location="cpu", weights_only=False)
                print(f"[Dataset Cache] Loaded {len(MultiHorizonDataset._cached_data['df'])} cached composites.")
            else:
                raise FileNotFoundError(f"Cache file {cache_pt} not found!")

        self.df = MultiHorizonDataset._cached_data["df"]
        self.s2_tensors = MultiHorizonDataset._cached_data["s2"]
        self.s5p_tensors = MultiHorizonDataset._cached_data["s5p"]
        self.s5p_mask_tensors = MultiHorizonDataset._cached_data["s5p_mask"]

        train_val_seqs = []
        test_seqs = []

        total_available = len(self.df) - (self.t_in - 1) - self.horizon
        for i in range(total_available):
            input_indices = list(range(i, i + self.t_in))
            target_idx = i + self.t_in - 1 + self.horizon
            target_year = self.df.iloc[target_idx]["year"]
            target_date = self.df.iloc[target_idx]["start_date"]
            seq_info = (input_indices, target_idx, target_date, target_year)
            if target_year <= 2022:
                train_val_seqs.append(seq_info)
            else:
                test_seqs.append(seq_info)

        self.train_val_seqs = train_val_seqs
        self.test_seqs = test_seqs

        if custom_sequences is not None:
            base_sequences = custom_sequences
        else:
            num_val = max(2, int(len(train_val_seqs) * 0.2))
            num_train = len(train_val_seqs) - num_val

            if split == "train":
                base_sequences = train_val_seqs[:num_train]
            elif split == "val":
                base_sequences = train_val_seqs[num_train:]
            elif split == "test":
                base_sequences = test_seqs
            elif split == "historical":
                base_sequences = train_val_seqs
            else:
                base_sequences = train_val_seqs + test_seqs

        patch_size = self.h
        stride = 64
        self.tile_offsets = []
        for ty in range(0, 256 - patch_size + 1, stride):
            for tx in range(0, 256 - patch_size + 1, stride):
                self.tile_offsets.append((ty, tx))

        self.samples = []
        for (hist_indices, target_idx, target_date, target_year) in base_sequences:
            for (ty, tx) in self.tile_offsets:
                self.samples.append((hist_indices, target_idx, target_date, target_year, ty, tx))

        print(f"[{split.upper()} Dataset | Horizon={horizon} ({horizon*5}d)] Sequences: {len(base_sequences)} x {len(self.tile_offsets)} tiles = {len(self.samples)} spatial patches.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hist_indices, target_idx, target_date, target_year, ty, tx = self.samples[idx]
        s5p_in = self.s5p_tensors[hist_indices, :, ty:ty+self.h, tx:tx+self.w].clone()
        s2_in  = self.s2_tensors[hist_indices, :, ty:ty+self.h, tx:tx+self.w].clone()
        s5p_target = self.s5p_tensors[target_idx, :, ty:ty+self.h, tx:tx+self.w].clone()
        target_mask = self.s5p_mask_tensors[target_idx, :, ty:ty+self.h, tx:tx+self.w].clone()

        if self.augment:
            if torch.rand(1).item() > 0.5:
                s5p_in = torch.flip(s5p_in, dims=[-1])
                s2_in  = torch.flip(s2_in, dims=[-1])
                s5p_target = torch.flip(s5p_target, dims=[-1])
                target_mask = torch.flip(target_mask, dims=[-1])
            if torch.rand(1).item() > 0.5:
                s5p_in = torch.flip(s5p_in, dims=[-2])
                s2_in  = torch.flip(s2_in, dims=[-2])
                s5p_target = torch.flip(s5p_target, dims=[-2])
                target_mask = torch.flip(target_mask, dims=[-2])
            rot_k = torch.randint(0, 4, (1,)).item()
            if rot_k > 0:
                s5p_in = torch.rot90(s5p_in, k=rot_k, dims=[-2, -1])
                s2_in  = torch.rot90(s2_in, k=rot_k, dims=[-2, -1])
                s5p_target = torch.rot90(s5p_target, k=rot_k, dims=[-2, -1])
                target_mask = torch.rot90(target_mask, k=rot_k, dims=[-2, -1])

        so2_scale = CONFIG["SCALES"]["SO2_SCALE"]
        s5p_in[:, 2] = torch.log1p(torch.clamp(s5p_in[:, 2], min=0.0) / so2_scale)
        s5p_target[2] = torch.log1p(torch.clamp(s5p_target[2], min=0.0) / so2_scale)
        return s5p_in, s2_in, s5p_target, target_mask


def train_horizon_model(horizon=30, num_epochs=35, batch_size=8, lr=5e-4):
    lead_days = horizon * 5
    ckpt_path = os.path.join(proj_dir, "training", "checkpoints", f"best_model_horizon{horizon}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TRAINING MULTI-HORIZON MODEL: HORIZON = {horizon} ({lead_days} DAYS AHEAD)")
    print("=" * 80)

    train_ds = MultiHorizonDataset(split="train", t_in=4, horizon=horizon)
    val_ds   = MultiHorizonDataset(split="val", t_in=4, horizon=horizon)
    test_ds  = MultiHorizonDataset(split="test", t_in=4, horizon=horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = STResUNet().to(DEVICE)
    criterion = PlumePreservingCompoundLoss(
        scales=CONFIG["SCALES"],
        gamma=CONFIG["PLUME_GAMMA"],
        lambda_plume=CONFIG["LAMBDA_PLUME"],
        lambda_grad=CONFIG["LAMBDA_GRAD"],
        lambda_ssim=CONFIG["LAMBDA_SSIM"],
        channel_weights=[1.0, 1.0, 1.0]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=CONFIG["USE_AMP"] and (device_type == "cuda"))

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        p_loss_total, g_loss_total, s_loss_total = 0.0, 0.0, 0.0

        for s5p_in, s2_in, target, target_mask in train_loader:
            s5p_in = s5p_in.to(DEVICE, non_blocking=True)
            s2_in  = s2_in.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            target_mask = target_mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, enabled=CONFIG["USE_AMP"] and (device_type == "cuda")):
                pred = model(s5p_in, s2_in)
                loss, loss_dict = criterion(pred, target, mask=target_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            p_loss_total += loss_dict.get("plume_l1", 0.0)
            g_loss_total += loss_dict.get("grad", 0.0)
            s_loss_total += loss_dict.get("ssim", 0.0)

        scheduler.step()
        train_loss /= len(train_loader)
        p_loss_total /= len(train_loader)
        g_loss_total /= len(train_loader)
        s_loss_total /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_mae_no2, val_mae_co, val_mae_so2 = 0.0, 0.0, 0.0

        with torch.no_grad():
            for s5p_in, s2_in, target, target_mask in val_loader:
                s5p_in = s5p_in.to(DEVICE, non_blocking=True)
                s2_in  = s2_in.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                target_mask = target_mask.to(DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type, enabled=CONFIG["USE_AMP"] and (device_type == "cuda")):
                    pred = model(s5p_in, s2_in)
                    loss, _ = criterion(pred, target, mask=target_mask)
                val_loss += loss.item()

                pred_phys = invert_so2_prediction(pred)
                target_phys = invert_so2_prediction(target)
                diff_no2 = torch.abs(pred_phys[:, 0] - target_phys[:, 0]) * target_mask[:, 0]
                diff_co  = torch.abs(pred_phys[:, 1] - target_phys[:, 1]) * target_mask[:, 1]
                diff_so2 = torch.abs(pred_phys[:, 2] - target_phys[:, 2]) * target_mask[:, 2]
                val_mae_no2 += (diff_no2.sum() / (target_mask[:, 0].sum() + 1e-8)).item()
                val_mae_co  += (diff_co.sum() / (target_mask[:, 1].sum() + 1e-8)).item()
                val_mae_so2 += (diff_so2.sum() / (target_mask[:, 2].sum() + 1e-8)).item()

        val_loss /= len(val_loader)
        val_mae_no2 /= len(val_loader)
        val_mae_co  /= len(val_loader)
        val_mae_so2 /= len(val_loader)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"Horizon {horizon:02d} | Epoch [{epoch:02d}/{num_epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} (Plume: {p_loss_total:.3f}, Grad: {g_loss_total:.3f}, SSIM: {s_loss_total:.3f}) | "
                f"Val Loss: {val_loss:.4f} | "
                f"MAE NO2: {val_mae_no2:.2e}, CO: {val_mae_co:.2e}, SO2: {val_mae_so2:.2e}",
                flush=True
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    total_time = time.time() - start_time
    print(f"Training finished for Horizon {horizon} in {total_time:.1f}s ({total_time/60:.2f} mins).")
    print(f"Saved Best Checkpoint: {ckpt_path}")
    return ckpt_path


def evaluate_horizon_model(horizon=30):
    lead_days = horizon * 5
    ckpt_path = os.path.join(proj_dir, "training", "checkpoints", f"best_model_horizon{horizon}.pt")
    
    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL FOR HORIZON = {horizon} ({lead_days} DAYS AHEAD)")
    print("=" * 80)

    test_ds = MultiHorizonDataset(split="test", t_in=4, horizon=horizon)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    model = STResUNet().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    preds_all = []
    trues_all = []
    masks_all = []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for s5p_in, s2_in, target, target_mask in test_loader:
            s5p_in = s5p_in.to(DEVICE)
            s2_in  = s2_in.to(DEVICE)
            with torch.amp.autocast(device_type, enabled=CONFIG["USE_AMP"] and (device_type == "cuda")):
                pred = model(s5p_in, s2_in)

            pred_phys = invert_so2_prediction(pred).cpu().numpy()
            target_phys = invert_so2_prediction(target).cpu().numpy()
            mask_np = target_mask.cpu().numpy()

            preds_all.append(pred_phys)
            trues_all.append(target_phys)
            masks_all.append(mask_np)

    preds_np = np.concatenate(preds_all, axis=0)
    trues_np = np.concatenate(trues_all, axis=0)
    masks_np = np.concatenate(masks_all, axis=0)

    metrics_df = compute_all_metrics(preds_np, trues_np, masks_np)
    metrics_csv = os.path.join(proj_dir, "evaluation", "results", f"evaluation_metrics_horizon{horizon}.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved Metrics to: {metrics_csv}")

    print(f"\n--- Holdout Test Metrics (Horizon={horizon} / {lead_days} Days Ahead, N={len(test_ds)} Patches) ---")
    cols = ["Target Pollutant", "MAE", "RMSE", "R² Score", "Pearson r", "Spatial SSIM", "FSS (9x9)", "CSI (q90)", "POD (q90)", "FAR (q90)", "Log-space MAE"]
    avail_cols = [c for c in cols if c in metrics_df.columns]
    print(metrics_df[avail_cols].to_string(index=False))
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=30, choices=[1, 30, 60], help="Forecast horizon steps (1, 30, 60)")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()

    if not args.eval_only:
        train_horizon_model(horizon=args.horizon, num_epochs=35)
    evaluate_horizon_model(horizon=args.horizon)
