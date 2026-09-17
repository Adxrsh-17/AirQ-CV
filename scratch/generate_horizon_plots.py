import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

art_dir = r"C:\Users\Adarsh_Pradeep\.gemini\antigravity-ide\brain\4b7cd75a-00a4-48cc-a1bc-f9416735cb77"
plots_dir = r"C:\Users\Adarsh_Pradeep\OneDrive\Desktop\4th Yr Sem 7\Project Phase 1\Projects\Projec-Code\plots\analysis"
os.makedirs(plots_dir, exist_ok=True)

# Compile Multi-Horizon Metrics
horizons = [5, 150, 300]
horizons_str = ["5-Day (Hz=1)", "150-Day (Hz=30)", "300-Day (Hz=60)"]

# Data from evaluations
no2_r2 = [0.5085, -1.7974, -0.6423]
no2_r  = [0.7186, 0.5276, 0.5469]
no2_ssim = [0.8936, 0.4067, 0.5157]

co_r2  = [0.5859, -0.0210, -0.1487]
co_r   = [0.7745, 0.6802, 0.5445]
co_ssim = [0.5998, 0.2790, 0.3315]

so2_r2 = [0.0775, -17.5568, -11.8409]
so2_r  = [0.2903, 0.1966, 0.2193]
so2_ssim = [0.5847, 0.0000, 0.0028]

# Climatology reference bars
clim_r2 = {"NO2": -0.0913, "CO": 0.6374, "SO2": -4.4257}
clim_r  = {"NO2": 0.7243, "CO": 0.8402, "SO2": 0.3055}

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=250)
fig.suptitle("Phase 7: Multi-Horizon Spatiotemporal Forecasting Skill vs. Lead Time (Tamil Nadu)", fontsize=14, fontweight="bold", y=1.02)

x = np.arange(len(horizons))
width = 0.25

# Panel 1: Pearson Correlation (r) Decay Across Horizons
axes[0].plot(horizons, no2_r, marker='o', linewidth=2.2, color='#2ca02c', label=r'$\mathrm{NO}_2$ (Short Lifetime $\sim$1d)')
axes[0].plot(horizons, co_r, marker='s', linewidth=2.2, color='#ff7f0e', label=r'$\mathrm{CO}$ (Long Lifetime $\sim$60d)')
axes[0].plot(horizons, so2_r, marker='^', linewidth=2.2, color='#d62728', label=r'$\mathrm{SO}_2$ (Episodic Stacks $\sim$2d)')
axes[0].axhline(clim_r["NO2"], color='#2ca02c', linestyle='--', alpha=0.5, label=r'$\mathrm{NO}_2$ Climatology ($r=0.72$)')
axes[0].axhline(clim_r["CO"], color='#ff7f0e', linestyle='--', alpha=0.5, label=r'$\mathrm{CO}$ Climatology ($r=0.84$)')
axes[0].set_title("A. Linear Spatial Correlation (Pearson r)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Forecast Lead Time (Days Ahead)", fontsize=10)
axes[0].set_ylabel("Pearson Correlation ($r$)", fontsize=10)
axes[0].set_xticks(horizons)
axes[0].set_xticklabels(["5d (Hz=1)", "150d (Hz=30)", "300d (Hz=60)"])
axes[0].set_ylim(0.0, 1.0)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(fontsize=8, loc="upper right")

# Panel 2: Spatial SSIM Across Horizons
axes[1].plot(horizons, no2_ssim, marker='o', linewidth=2.2, color='#2ca02c', label=r'$\mathrm{NO}_2$')
axes[1].plot(horizons, co_ssim, marker='s', linewidth=2.2, color='#ff7f0e', label=r'$\mathrm{CO}$')
axes[1].plot(horizons, so2_ssim, marker='^', linewidth=2.2, color='#d62728', label=r'$\mathrm{SO}_2$')
axes[1].set_title("B. Spatial Structural Similarity (SSIM)", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Forecast Lead Time (Days Ahead)", fontsize=10)
axes[1].set_ylabel("Spatial SSIM", fontsize=10)
axes[1].set_xticks(horizons)
axes[1].set_xticklabels(["5d (Hz=1)", "150d (Hz=30)", "300d (Hz=60)"])
axes[1].set_ylim(0.0, 1.0)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(fontsize=9, loc="upper right")

# Panel 3: R2 Score Comparison
bar_no2 = axes[2].bar(x - width, [max(v, -2.0) for v in no2_r2], width, label=r'$\mathrm{NO}_2$', color='#2ca02c', alpha=0.85)
bar_co  = axes[2].bar(x, [max(v, -2.0) for v in co_r2], width, label=r'$\mathrm{CO}$', color='#ff7f0e', alpha=0.85)
bar_so2 = axes[2].bar(x + width, [max(v, -2.0) for v in so2_r2], width, label=r'$\mathrm{SO}_2$ (clipped at -2.0)', color='#d62728', alpha=0.85)
axes[2].axhline(0.0, color='black', linewidth=1.0)
axes[2].set_title("C. Explained Variance ($R^2$ Score)", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Forecast Horizon", fontsize=10)
axes[2].set_ylabel(r"Holdout $R^2$ Score", fontsize=10)
axes[2].set_xticks(x)
axes[2].set_xticklabels(["5d (Hz=1)", "150d (Hz=30)", "300d (Hz=60)"])
axes[2].set_ylim(-2.2, 0.8)
axes[2].grid(True, linestyle=':', alpha=0.6)
axes[2].legend(fontsize=9, loc="upper right")

plt.tight_layout()
save_p1 = os.path.join(plots_dir, "multi_horizon_comparison.png")
save_p2 = os.path.join(art_dir, "multi_horizon_comparison.png")
plt.savefig(save_p1, bbox_inches="tight", dpi=250)
plt.savefig(save_p2, bbox_inches="tight", dpi=250)
plt.close()
print(f"Saved Multi-Horizon Plot to:\n  1. {save_p1}\n  2. {save_p2}")
