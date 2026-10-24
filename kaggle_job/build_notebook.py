import json, os, gzip, base64

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Read code files
    code_files = ["align_tn.py", "dataset.py", "dcnet.py", "train.py", "evaluate.py", "make_cache.py"]
    code_contents = {}
    for cf in code_files:
        with open(os.path.join(root, cf), "r") as f:
            code_contents[cf] = f.read()

    # Read weather data and compress
    with open(os.path.join(root, "weather", "era5_tn_clusters.csv"), "rb") as f:
        wx_b64 = base64.b64encode(gzip.compress(f.read())).decode("utf-8")

    cells = []

    # Cell 1: Markdown title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# AirQ-CV: Dual-Clock Architecture & DC-Net GPU Training\n",
            "### Automated Run Pipeline (Option B: Cloud GDrive Streaming -> GPU Cache -> Training -> Empirical Evaluation)\n",
            "- **Target Pollutant**: $\\text{NO}_2$ focused (`n_poll=1`)\n",
            "- **Historical Shortcut Prevention**: 50% Spatial Dropout on `long_mem`\n",
            "- **Spatial Prior**: Per-sample batch slicing across harmonic lookup tables\n",
            "- **Temporal Split**: Stage 1 (2019–2023 Train, 2024 Held-Out Validation)\n",
            "- **Benchmark Metric**: Physical empirical training pixel climatology (external yardstick)\n"
        ]
    })

    # Cell 2: GPU check & environment
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, torch\n",
            "print('=== GPU ENVIRONMENT CHECK ===')\n",
            "print(f'PyTorch Version: {torch.__version__}')\n",
            "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'Device Count: {torch.cuda.device_count()}')\n",
            "    for d in range(torch.cuda.device_count()):\n",
            "        print(f'  GPU {d}: {torch.cuda.get_device_name(d)} ({torch.cuda.get_device_properties(d).total_memory / (1024**3):.2f} GB)')\n",
            "\n",
            "!pip install -q rasterio gdown lightgbm\n"
        ]
    })

    # Cell 3: Write code files
    deploy_code_lines = [
        "import os\n",
        "print('=== DEPLOYING UPDATED DUAL-CLOCK CODE MODULES ===')\n"
    ]
    for cf, content in code_contents.items():
        escaped_content = repr(content)
        deploy_code_lines.append(f"with open('{cf}', 'w') as f:\n    f.write({escaped_content})\n")
        deploy_code_lines.append(f"print('Deployed: {cf}')\n")

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": deploy_code_lines
    })

    # Cell 4: Deploy weather data
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, gzip, base64\n",
            "os.makedirs('weather', exist_ok=True)\n",
            f"wx_b64 = '{wx_b64}'\n",
            "with open('weather/era5_tn_clusters.csv', 'wb') as f:\n",
            "    f.write(gzip.decompress(base64.b64decode(wx_b64)))\n",
            "print('Extracted: weather/era5_tn_clusters.csv (1.6 MB, 5 regional clusters, 2019-2024)')\n"
        ]
    })

    # Cell 5: Download from Google Drive
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, shutil, glob\n",
            "\n",
            "os.makedirs('data2/s2_composites', exist_ok=True)\n",
            "os.makedirs('data2/s5p_composites', exist_ok=True)\n",
            "\n",
            "# 1. Check if dataset is mounted from Kaggle Datasets (Instant & 100% Reliable)\n",
            "kaggle_input = '/kaggle/input'\n",
            "s2_src, s5p_src = None, None\n",
            "for root, dirs, files in os.walk(kaggle_input):\n",
            "    if 's2_composites' in dirs: s2_src = os.path.join(root, 's2_composites')\n",
            "    if 's5p_composites' in dirs: s5p_src = os.path.join(root, 's5p_composites')\n",
            "\n",
            "if s2_src and s5p_src:\n",
            "    print(f'Mounted Kaggle Dataset detected!\\nS2: {s2_src}\\nS5P: {s5p_src}')\n",
            "    for f in glob.glob(os.path.join(s2_src, '*.tif')):\n",
            "        dst = os.path.join('data2/s2_composites', os.path.basename(f))\n",
            "        if not os.path.exists(dst): os.symlink(f, dst)\n",
            "    for f in glob.glob(os.path.join(s5p_src, '*.tif')):\n",
            "        dst = os.path.join('data2/s5p_composites', os.path.basename(f))\n",
            "        if not os.path.exists(dst): os.symlink(f, dst)\n",
            "else:\n",
            "    print('Dataset not mounted in /kaggle/input. Downloading via upgraded gdown...')\n",
            "    !pip install -q --upgrade gdown\n",
            "    import gdown\n",
            "    DRIVE_URL = 'https://drive.google.com/drive/folders/1SQdU-L5SctKcsYXa4WTcto8_emaCurVQ'\n",
            "    try:\n",
            "        !gdown \"{DRIVE_URL}\" --folder -O data2/raw/\n",
            "    except Exception as e:\n",
            "        print('Download warning:', e)\n",
            "    for f in glob.glob('data2/raw/**/s2_*.tif', recursive=True):\n",
            "        shutil.move(f, os.path.join('data2/s2_composites', os.path.basename(f)))\n",
            "    for f in glob.glob('data2/raw/**/s5p_*.tif', recursive=True):\n",
            "        shutil.move(f, os.path.join('data2/s5p_composites', os.path.basename(f)))\n",
            "\n",
            "s2_count = len(os.listdir('data2/s2_composites'))\n",
            "s5p_count = len(os.listdir('data2/s5p_composites'))\n",
            "print(f'Files Ready: S2 composites={s2_count}, S5P composites={s5p_count}')\n"
        ]
    })

    # Cell 6: Run make_cache.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=== STEP 1: BUILDING SATELLITE & WEATHER CACHE ===')\n",
            "!python make_cache.py --s2_dir data2/s2_composites --s5p_dir data2/s5p_composites --out cache --weather weather/era5_tn_clusters.csv --start 2019-01-01 --end 2024-12-31 --zero_is_invalid 1\n"
        ]
    })

    # Cell 7: Run train.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=== STEP 2: TRAINING DC-NET ON GPU (STAGE 1, NO2 FOCUS) ===')\n",
            "!python train.py --cache cache --out /kaggle/working/runs/dcnet_main --stage 1 --poll NO2 --long_mem_dropout 0.50 --batch 8 --epochs 120 --vram_cap 0.90\n"
        ]
    })

    # Cell 8: Run evaluate.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=== STEP 3: RUNNING EMPIRICAL CLIMATOLOGY EVALUATION (2024 HELD-OUT) ===')\n",
            "!python evaluate.py --ckpt /kaggle/working/runs/dcnet_main/best.pt --cache cache --split val --out /kaggle/working/runs/dcnet_main\n"
        ]
    })

    # Cell 9: Display Metrics Table
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "metrics_path = '/kaggle/working/runs/dcnet_main/dcnet_val_metrics.csv'\n",
            "if os.path.exists(metrics_path):\n",
            "    df = pd.read_csv(metrics_path)\n",
            "    print('\\n' + '='*80)\n",
            "    print('FINAL EVALUATION METRICS (2024 HELD-OUT SPLIT):')\n",
            "    print('='*80)\n",
            "    print(df[['pollutant', 'horizon_days', 'R2', 'pearson_r', 'skill_vs_clim_pct', 'coverage_80', 'csi_p90']].to_string())\n",
            "else:\n",
            "    print('Metrics file not found.')\n"
        ]
    })

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = os.path.join(root, "kaggle_job", "kaggle_dcnet_run.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully generated self-contained notebook: {out_path}")

if __name__ == "__main__":
    main()
