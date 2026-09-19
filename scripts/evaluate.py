import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    precision_recall_curve,
    auc
)
from skimage.metrics import structural_similarity as ssim

# Import our custom modules
from airq.data.v2_dataset import V2Dataset
from airq.models.v2_forecaster import V2Forecaster

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate V2Forecaster performance on holdout test set.")
    parser.add_argument('--data_dir', type=str, default='data/raw_full438')
    parser.add_argument('--checkpoint', type=str, default='experiments/v2_baseline/best_model.pt')
    parser.add_argument('--out_dir', type=str, default='evaluation')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--hidden_dim', type=int, default=64)
    return parser.parse_args()

def calculate_fss(y_true, y_pred, threshold, window_size=9):
    """
    Calculate Fractions Skill Score (FSS) at a given spatial window scale.
    Expects y_true, y_pred of shape (B, H, W)
    """
    # Binarize based on threshold
    obs_binary = (y_true >= threshold).float()
    pred_binary = (y_pred >= threshold).float()
    
    # 2D Average pooling to get fractions in the neighborhood
    # Add channel dimension (B, 1, H, W)
    obs_binary = obs_binary.unsqueeze(1)
    pred_binary = pred_binary.unsqueeze(1)
    
    padding = window_size // 2
    obs_fraction = F.avg_pool2d(obs_binary, kernel_size=window_size, stride=1, padding=padding)
    pred_fraction = F.avg_pool2d(pred_binary, kernel_size=window_size, stride=1, padding=padding)
    
    # Calculate Mean Squared Error of fractions
    mse_fractions = torch.mean((obs_fraction - pred_fraction) ** 2)
    
    # Calculate reference MSE
    mse_ref = torch.mean(obs_fraction ** 2 + pred_fraction ** 2)
    
    if mse_ref == 0:
        return 1.0 # Perfect match if both are completely empty
        
    fss = 1.0 - (mse_fractions / mse_ref)
    return fss.item()

def compute_spatial_metrics(y_true, y_pred, q90):
    """
    Compute SSIM and FSS across all frames in the dataset.
    y_true, y_pred: (N, C, H, W) numpy arrays
    """
    N, C, H, W = y_true.shape
    ssim_scores = []
    
    # Calculate SSIM (frame by frame, channel by channel)
    for i in range(N):
        for c in range(C):
            # Dynamic range for SSIM based on the specific frame/channel
            drange = y_true[i, c].max() - y_true[i, c].min()
            if drange == 0: drange = 1e-5
            
            s = ssim(y_true[i, c], y_pred[i, c], data_range=drange)
            ssim_scores.append(s)
            
    avg_ssim = np.mean(ssim_scores)
    
    # Calculate FSS per channel
    y_t = torch.from_numpy(y_true)
    y_p = torch.from_numpy(y_pred)
    
    fss_scores = []
    for c in range(C):
        fss = calculate_fss(y_t[:, c], y_p[:, c], threshold=q90[c], window_size=9)
        fss_scores.append(fss)
        
    return avg_ssim, fss_scores

def plot_parity(y_true, y_pred, channel_idx, out_path):
    plt.figure(figsize=(8, 8))
    # Hexbin plot for dense scatter data
    hb = plt.hexbin(y_true.flatten(), y_pred.flatten(), gridsize=50, cmap='magma', bins='log', mincnt=1)
    plt.colorbar(hb, label='log10(count)')
    
    # 1:1 reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Perfect Fit')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Parity Plot (Band {channel_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_comparison(y_true, y_pred, channel_idx, out_path):
    # Select a random scene
    idx = np.random.randint(0, y_true.shape[0])
    
    true_frame = y_true[idx, channel_idx]
    pred_frame = y_pred[idx, channel_idx]
    error_frame = np.abs(true_frame - pred_frame)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    vmin = min(true_frame.min(), pred_frame.min())
    vmax = max(true_frame.max(), pred_frame.max())
    
    sns.heatmap(true_frame, ax=axes[0], cmap='viridis', vmin=vmin, vmax=vmax, cbar_kws={'label': 'Concentration'})
    axes[0].set_title(f'True Field (Scene {idx}, Band {channel_idx})')
    axes[0].axis('off')
    
    sns.heatmap(pred_frame, ax=axes[1], cmap='viridis', vmin=vmin, vmax=vmax, cbar_kws={'label': 'Concentration'})
    axes[1].set_title(f'Predicted Field (Scene {idx}, Band {channel_idx})')
    axes[1].axis('off')
    
    sns.heatmap(error_frame, ax=axes[2], cmap='magma', cbar_kws={'label': 'Absolute Error'})
    axes[2].set_title(f'Absolute Error (Scene {idx}, Band {channel_idx})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(y_true, y_pred, channel_idx, out_path):
    errors = (y_true - y_pred).flatten()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=100, kde=True, color='indigo')
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.xlabel('Residual Error (True - Predicted)')
    plt.ylabel('Density / Count')
    plt.title(f'Error Distribution (Band {channel_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curve(y_true, y_pred, channel_idx, q90, out_path):
    true_binary = (y_true >= q90).flatten().astype(int)
    pred_probs = y_pred.flatten()
    
    precision, recall, thresholds = precision_recall_curve(true_binary, pred_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall (Probability of Detection)')
    plt.ylabel('Precision (1 - False Alarm Ratio)')
    plt.title(f'Precision-Recall Curve for Extreme Events (> {q90:.3f}) (Band {channel_idx})')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = get_args()
    
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / 'plots'
    results_dir = out_dir / 'results'
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # 1. Load Data
    print("Loading test dataset...")
    test_ds = V2Dataset(args.data_dir, split='test', preload=True)
    if len(test_ds) == 0:
        print("Error: No test sequences found! Ensure there is data strictly between 2024-01-01 and 2025-01-01.")
        return
        
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0)
    )
    print(f"Test sequences: {len(test_ds)}")
    
    # 2. Load Model
    model = V2Forecaster(hidden_dim=args.hidden_dim).to(device)
    if Path(args.checkpoint).exists():
        print(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    else:
        print(f"WARNING: Checkpoint {args.checkpoint} not found. Using untrained weights.")
    
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    # 3. Run Inference
    print("Running inference on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            p1_x = batch['p1_x'].to(device, non_blocking=True)
            s2_x = batch['s2_x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True) # (B, K, C, H, W)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                preds = model(p1_x, s2_x)
            
            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(preds.cpu().numpy())
            
    # Concatenate all batches (Total_N, K, C, H, W) -> flatten temporal dim to (N_frames, C, H, W)
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    
    B, K, C, H, W = y_true.shape
    y_true_flat = y_true.reshape(-1, C, H, W)
    y_pred_flat = y_pred.reshape(-1, C, H, W)
    
    # Inverse scaling from Z-score to physical units
    stats_file = Path('data/processed/v2_cache/norm_stats.json')
    if stats_file.exists():
        import json
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        s5p_mean = np.array(stats['s5p_mean']).reshape(1, C, 1, 1)
        s5p_std = np.array(stats['s5p_std']).reshape(1, C, 1, 1)
        
        y_true_flat = (y_true_flat * s5p_std) + s5p_mean
        y_pred_flat = (y_pred_flat * s5p_std) + s5p_mean
        print("\nDataset inverse scaling applied (Z-score -> Physical Units).")
    else:
        print("\nNo norm_stats.json found, skipping inverse scaling.")
    
    # Generate q90 per channel
    q90 = [np.percentile(y_true_flat[:, c], 90) for c in range(C)]
    
    metrics_records = []
    
    # 4. Calculate Metrics Per Band
    for c in range(C):
        print(f"\n--- Calculating Metrics for Band {c} ---")
        
        y_t_c = y_true_flat[:, c].flatten()
        y_p_c = y_pred_flat[:, c].flatten()
        
        # valid pixel masking (if dataset contains NaN or specific nodata value, filter them)
        valid_mask = ~np.isnan(y_t_c)
        y_t_c = y_t_c[valid_mask]
        y_p_c = y_p_c[valid_mask]
        
        r2 = r2_score(y_t_c, y_p_c)
        rmse = np.sqrt(mean_squared_error(y_t_c, y_p_c))
        mae = mean_absolute_error(y_t_c, y_p_c)
        pearson_r = np.corrcoef(y_t_c, y_p_c)[0, 1]
        
        # Mass Conservation
        total_true_mass = np.sum(y_t_c)
        total_pred_mass = np.sum(y_p_c)
        mass_error_pct = ((total_pred_mass - total_true_mass) / (total_true_mass + 1e-8)) * 100
        
        # Contingency Metrics (Extremes)
        true_extremes = (y_t_c >= q90[c])
        pred_extremes = (y_p_c >= q90[c])
        
        hits = np.sum(true_extremes & pred_extremes)
        misses = np.sum(true_extremes & ~pred_extremes)
        false_alarms = np.sum(~true_extremes & pred_extremes)
        
        csi = hits / (hits + misses + false_alarms + 1e-8)
        pod = hits / (hits + misses + 1e-8) # Recall
        far = false_alarms / (hits + false_alarms + 1e-8)
        
        print(f"R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Pearson: {pearson_r:.4f}")
        print(f"Mass Conservation Error: {mass_error_pct:.2f}%")
        print(f"Extreme Events (> {q90[c]:.4f}) -> CSI: {csi:.4f} | POD: {pod:.4f} | FAR: {far:.4f}")
        
        # SSIM & FSS
        # Spatial metrics require spatial shape (N, H, W)
        avg_ssim, fss_list = compute_spatial_metrics(
            y_true_flat[:, c:c+1], 
            y_pred_flat[:, c:c+1], 
            [q90[c]]
        )
        print(f"Spatial SSIM: {avg_ssim:.4f} | Spatial FSS (9x9): {fss_list[0]:.4f}")
        
        metrics_records.append({
            'Band': c,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'Pearson_r': pearson_r,
            'Mass_Conservation_Error_Pct': mass_error_pct,
            'q90_Threshold': q90[c],
            'CSI': csi,
            'POD': pod,
            'FAR': far,
            'SSIM': avg_ssim,
            'FSS_9x9': fss_list[0]
        })
        
        # 5. Generate Visualizations
        print("Generating visual plots...")
        plot_parity(y_t_c, y_p_c, c, plots_dir / f'parity_band_{c}.png')
        plot_spatial_comparison(y_true_flat, y_pred_flat, c, plots_dir / f'spatial_band_{c}.png')
        plot_error_distribution(y_t_c, y_p_c, c, plots_dir / f'error_dist_band_{c}.png')
        plot_pr_curve(y_t_c, y_p_c, c, q90[c], plots_dir / f'pr_curve_band_{c}.png')
        
    # Save CSV
    df_metrics = pd.DataFrame(metrics_records)
    csv_path = results_dir / 'evaluation_metrics.csv'
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nEvaluation complete. Metrics saved to {csv_path} and plots in {plots_dir}")

if __name__ == '__main__':
    main()
