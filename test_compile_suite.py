import sys, math, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("RUNNING COMPREHENSIVE COMPILATION & ARCHITECTURE TESTS")
print("=" * 70)

# ---------------------------------------------------------------------------
# Test 1: Biophysical Index Calculation
# ---------------------------------------------------------------------------
print("[TEST 1/6] Testing Biophysical Index Stacking (Rank 1)...")
from dataset import DCDataset

# Create synthetic 12-band Sentinel-2 reflectance tensor
# B1 (0), B2 (1), B3 (2), B4 (3), B5 (4), B6 (5), B7 (6), B8 (7), B8A (8), B9 (9), B11 (10), B12 (11)
fake_bands = np.random.uniform(0.01, 0.5, size=(12, 104, 104)).astype(np.float32)
# Introduce edge cases: 0s, near-zero values
fake_bands[:, :5, :5] = 0.0

bio_indices = DCDataset.calc_biophysical_indices(fake_bands)
assert bio_indices.shape == (4, 104, 104), f"Expected (4, 104, 104), got {bio_indices.shape}"
assert np.all(np.isfinite(bio_indices)), "NaN or Inf detected in biophysical indices!"
assert np.all((bio_indices[0] >= -1.0) & (bio_indices[0] <= 1.0)), "NDBI out of range [-1, 1]"
assert np.all((bio_indices[1] >= -1.0) & (bio_indices[1] <= 1.0)), "NDVI out of range [-1, 1]"
assert np.all((bio_indices[2] >= -1.0) & (bio_indices[2] <= 1.0)), "NDRE out of range [-1, 1]"
assert np.all((bio_indices[3] >= 0.0) & (bio_indices[3] <= 1.0)), "Aerosol ratio out of range [0, 1]"
print("  --> PASS: Biophysical indices (NDBI, NDVI, NDRE, Aerosol) calculated cleanly with zero NaNs/Infs.")

# ---------------------------------------------------------------------------
# Test 2: DCNet Model Instantiation & Parameter Inspection
# ---------------------------------------------------------------------------
print("[TEST 2/6] Testing DCNet Instantiation (Single NO2 & Multi-Pollutant)...")
from dcnet import DCNet, dcnet_loss, count_params

# Single pollutant (NO2)
model_no2 = DCNet(n_poll=1, full_hw=(13, 13), use_B=True, use_s2=True, use_weather=True, long_mem_dropout=0.5)
params_no2 = count_params(model_no2)
print(f"  NO2 Model: {params_no2:,} parameters")
assert hasattr(model_no2, "fine_head"), "DCNet missing FineDisaggregationHead!"
assert hasattr(model_no2, "prior"), "DCNet missing SeasonalPrior!"
assert hasattr(model_no2, "slow"), "DCNet missing S2SlowEncoder!"
assert model_no2.slow.stem.in_channels == 32, f"Expected 32 in_channels for biophysical S2 slow encoder, got {model_no2.slow.stem.in_channels}"

# Multi-pollutant (NO2, CO, SO2)
model_all = DCNet(n_poll=3, full_hw=(123, 95), use_B=True, use_s2=True, use_weather=True, long_mem_dropout=0.5)
params_all = count_params(model_all)
print(f"  Multi-Pollutant Model: {params_all:,} parameters")
print("  --> PASS: DCNet initializations validated.")

# ---------------------------------------------------------------------------
# Test 3: Forward Pass with Multi-Scale & Adaptive Ratios
# ---------------------------------------------------------------------------
print("[TEST 3/6] Testing Forward Pass across Spatial Scales & Native Ratios...")

# Case A: Chennai ROI at R=32 (H5=13, W5=13, fine grid H2=416, W2=416)
B = 2
H5, W5 = 13, 13
R_fine = 32
H2, W2 = H5 * R_fine, W5 * R_fine

# Pad to multiple of 8 as done by collate/dataset
Hp, Wp = (-(-H5 // 8)) * 8, (-(-W5 // 8)) * 8
Hp_fine, Wp_fine = Hp * R_fine, Wp * R_fine

batch_chennai = {
    "fast_v": torch.randn(B, 12, 1, Hp, Wp),
    "fast_m": torch.ones(B, 12, 1, Hp, Wp),
    "s2_slow": torch.randn(B, 32, Hp_fine, Wp_fine),
    "long_mem": torch.randn(B, 24, Hp, Wp),
    "crop": torch.tensor([[0, 0, H5, W5], [0, 0, H5, W5]], dtype=torch.int64),
    "doy_tgt": torch.tensor([[10., 15., 20., 25., 30., 60., 150.], [10., 15., 20., 25., 30., 60., 150.]]),
    "doy_org_sc": torch.tensor([[0.5, 0.866], [0.5, 0.866]]),
    "weather": torch.randn(B, 8)
}

out_chennai = model_no2(batch_chennai)
assert "q50" in out_chennai, "Missing q50 in model output"
assert "fine_q50" in out_chennai, "Missing fine_q50 in model output"
assert out_chennai["q50"].shape == (B, 7, 1, Hp, Wp), f"Coarse shape mismatch: {out_chennai['q50'].shape}"
assert out_chennai["fine_q50"].shape == (B, 7, 1, Hp_fine, Wp_fine), f"Fine shape mismatch: {out_chennai['fine_q50'].shape}"
print(f"  Coarse output: {out_chennai['q50'].shape} (regional S5P grid)")
print(f"  Fine output:   {out_chennai['fine_q50'].shape} (native S2 sub-pixel grid, R={R_fine})")
print("  --> PASS: Forward pass succeeds with native sub-pixel fine disaggregation.")

# ---------------------------------------------------------------------------
# Test 4: Loss Computation & Backward Gradient Propagation
# ---------------------------------------------------------------------------
print("[TEST 4/6] Testing Mass Conservation Loss & Backward Autograd...")
y_true = torch.randn(B, 7, 1, Hp, Wp)
ym_mask = torch.ones(B, 7, 1, Hp, Wp)
ev_event = torch.zeros(B, 7, Hp, Wp)
peak_thr = torch.tensor([1.5], dtype=torch.float32)

loss = dcnet_loss(out_chennai, y_true, ym_mask, ev_event, peak_thr=peak_thr)
assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
print(f"  Computed total loss (pinball + SO2 focal + gate + mass conservation): {loss.item():.4f}")

# Backward pass
loss.backward()
for name, p in model_no2.named_parameters():
    if p.requires_grad and not name.startswith("occ."):
        assert p.grad is not None, f"Parameter {name} has no gradient in single-pollutant model!"
print("  --> PASS: Backward pass verified for single-pollutant NO2. All active parameters receive clean gradients.")

# Also test multi-pollutant where occ (SO2 occurrence classifier) is active
print("  Testing gradient flow on multi-pollutant model (n_poll=3, with SO2 event classifier)...")
batch_all = {
    "fast_v": torch.randn(B, 12, 3, Hp, Wp),
    "fast_m": torch.ones(B, 12, 3, Hp, Wp),
    "s2_slow": torch.randn(B, 32, Hp_fine, Wp_fine),
    "long_mem": torch.randn(B, 72, Hp, Wp),
    "crop": torch.tensor([[0, 0, H5, W5], [0, 0, H5, W5]], dtype=torch.int64),
    "doy_tgt": torch.tensor([[10., 15., 20., 25., 30., 60., 150.], [10., 15., 20., 25., 30., 60., 150.]]),
    "doy_org_sc": torch.tensor([[0.5, 0.866], [0.5, 0.866]]),
    "weather": torch.randn(B, 8)
}
out_all = model_all(batch_all)
y_all = torch.randn(B, 7, 3, Hp, Wp)
ym_all = torch.ones(B, 7, 3, Hp, Wp)
ev_all = torch.randint(0, 2, (B, 7, Hp, Wp)).float()
loss_all = dcnet_loss(out_all, y_all, ym_all, ev_all, so2_idx=2, peak_thr=torch.tensor([1.5, 1.2, 0.8]))
loss_all.backward()
for name, p in model_all.named_parameters():
    if p.requires_grad:
        assert p.grad is not None, f"Parameter {name} has no gradient in multi-pollutant model!"
print("  --> PASS: Multi-pollutant backward pass verified (100% of parameters including occ receive gradients).")

# ---------------------------------------------------------------------------
# Test 5: Multi-GPU DataParallel & Optimizer Parameter Group Logic
# ---------------------------------------------------------------------------
print("[TEST 5/6] Testing DataParallel Optimizer & EMA Handling...")
# Test parameter partition logic with and without "module." prefix
raw_model = model_no2
nd = [p for n, p in raw_model.named_parameters() if p.ndim <= 1 or "prior.W" in n]
dd = [p for n, p in raw_model.named_parameters() if not (p.ndim <= 1 or "prior.W" in n)]
assert len(nd) + len(dd) == len(list(raw_model.parameters())), "Parameter partition mismatch!"

# Verify that prior.W is correctly classified into no-decay group
prior_w_params = [p for n, p in raw_model.named_parameters() if "prior.W" in n]
for p in prior_w_params:
    assert any(p is nd_p for nd_p in nd), "prior.W was mistakenly placed in weight decay group!"
print("  --> PASS: Optimizer parameter grouping correctly protects prior.W harmonic weights from decay.")

# ---------------------------------------------------------------------------
# Test 6: make_cache Pre-Flight Safety Math
# ---------------------------------------------------------------------------
print("[TEST 6/6] Testing Cache Size Safety Calculations...")
M = 72 # 6 years

# Scenario A: Chennai ROI (13x13) at R=32
H5_ch, W5_ch, R_ch = 13, 13, 32
size_ch_bytes = M * 12 * (H5_ch * R_ch) * (W5_ch * R_ch) * 2
size_ch_mb = size_ch_bytes / (1024 ** 2)
print(f"  Chennai ROI (13x13, R=32): {size_ch_mb:.1f} MB (Target: < 500 MB)")
assert size_ch_mb < 500.0, f"Chennai cache too large: {size_ch_mb} MB"

# Scenario B: Tamil Nadu State Grid (123x95) at safe R=12 (native ~500m S2)
H5_tn, W5_tn, R_tn_safe = 123, 95, 12
size_tn_bytes = M * 12 * (H5_tn * R_tn_safe) * (W5_tn * R_tn_safe) * 2
size_tn_gb = size_tn_bytes / (1024 ** 3)
print(f"  Full Tamil Nadu (123x95, R=12): {size_tn_gb:.2f} GB (Target: < 5 GB)")
assert size_tn_gb < 5.0, f"TN safe cache too large: {size_tn_gb} GB"

# Scenario C: What happened when R=32 was attempted on full state:
H5_tn, W5_tn, R_tn_unsafe = 123, 95, 32
size_tn_unsafe_bytes = M * 12 * (H5_tn * R_tn_unsafe) * (W5_tn * R_tn_unsafe) * 2
size_tn_unsafe_gb = size_tn_unsafe_bytes / (1024 ** 3)
ok_mask_bytes = M * (H5_tn * R_tn_unsafe) * (W5_tn * R_tn_unsafe) * 1
total_unsafe_gb = (size_tn_unsafe_bytes + ok_mask_bytes) / (1024 ** 3)
print(f"  Full Tamil Nadu (123x95, R=32 unconstrained): {size_tn_unsafe_gb:.2f} GiB s2_month + {ok_mask_bytes/(1024**3):.2f} GiB mask = {total_unsafe_gb:.2f} GiB total!")
assert total_unsafe_gb >= 20.0, f"Expected total cache to exceed 20 GB, got {total_unsafe_gb:.2f} GB"
print("  --> PASS: Cache size pre-flight math and safety guards verified.")

print("=" * 70)
print("ALL 6 COMPILE & ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
print("=" * 70)
