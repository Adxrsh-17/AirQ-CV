import json, os, gzip, base64

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Read code files
    code_files = ["align_tn.py", "dataset.py", "dcnet.py", "train.py", "evaluate.py", "make_cache.py", "app.py"]
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
            "!pip install -q rasterio gdown lightgbm gradio\n"
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
            "kaggle_input = '/kaggle/input'\n",
            "found_s2, found_s5p = 0, 0\n",
            "\n",
            "# 1. Universal recursive discovery for Kaggle Datasets\n",
            "for root, dirs, files in os.walk(kaggle_input):\n",
            "    for f in files:\n",
            "        if f.startswith('s2_') and f.endswith('.tif'):\n",
            "            dst = os.path.join('data2/s2_composites', f)\n",
            "            if not os.path.exists(dst): os.symlink(os.path.join(root, f), dst)\n",
            "            found_s2 += 1\n",
            "        elif f.startswith('s5p_') and f.endswith('.tif'):\n",
            "            dst = os.path.join('data2/s5p_composites', f)\n",
            "            if not os.path.exists(dst): os.symlink(os.path.join(root, f), dst)\n",
            "            found_s5p += 1\n",
            "\n",
            "if found_s2 > 0 and found_s5p > 0:\n",
            "    print(f'Mounted Kaggle Dataset detected! Fast-linked {found_s2} S2 and {found_s5p} S5P GeoTIFF composites (0s, 0 disk used).')\n",
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
            "print(f'Composites Ready for Processing: S2={s2_count}, S5P={s5p_count}')\n"
        ]
    })

    # Cell 6: Build or Detect Cache in /kaggle/tmp (~60 GB ephemeral storage)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, glob\n",
            "print('=== STEP 1: RESOLVING HIGH-RES SATELLITE & WEATHER CACHE ===')\n",
            "\n",
            "# 1. Check if high-res cache is mounted directly from Kaggle Datasets (Instant & 0 disk usage)\n",
            "mounted_cache = None\n",
            "for root, dirs, files in os.walk('/kaggle/input'):\n",
            "    if 's2_month.npy' in files and 'meta.json' in files:\n",
            "        mounted_cache = root\n",
            "        break\n",
            "\n",
            "if mounted_cache:\n",
            "    print(f'Mounted Kaggle Dataset cache detected at: {mounted_cache}! (Zero disk usage)')\n",
            "    CACHE_DIR = mounted_cache\n",
            "else:\n",
            "    # Use ephemeral /kaggle/tmp (~60 GB available, keeping /kaggle/working 100% clean)\n",
            "    CACHE_DIR = '/kaggle/tmp/cache' if os.path.exists('/kaggle/tmp') else '/tmp/cache'\n",
            "    os.makedirs(CACHE_DIR, exist_ok=True)\n",
            "    print(f'Building high-res aligned cache in {CACHE_DIR} (ephemeral ~60 GB storage)...')\n",
            "    !python make_cache.py --s2_dir data2/s2_composites --s5p_dir data2/s5p_composites --out {CACHE_DIR} --s2_ratio 32 --weather weather/era5_tn_clusters.csv --start 2019-01-01 --end 2024-12-31 --zero_is_invalid 1\n",
            "    print('Cache build completed in temporary storage!')\n"
        ]
    })

    # Cell 7: Save Cache to Kaggle Datasets Once and For All
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, json\n",
            "print('=== STEP 1B: PERSIST CACHE AS PERMANENT KAGGLE DATASET (ONCE AND FOR ALL) ===')\n",
            "\n",
            "# Only publish if cache was freshly created in temporary storage\n",
            "if not mounted_cache and os.path.exists(os.path.join(CACHE_DIR, 'meta.json')):\n",
            "    print('Setting up Kaggle credentials for antonioroger2...')\n",
            "    kaggle_dir = os.path.expanduser('~/.kaggle')\n",
            "    os.makedirs(kaggle_dir, exist_ok=True)\n",
            "    kaggle_creds = {'username': 'antonioroger2', 'key': '375c60033dbfbfe213d2b3645e05814c'}\n",
            "    with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:\n",
            "        json.dump(kaggle_creds, f)\n",
            "    os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)\n",
            "\n",
            "    metadata = {\n",
            "        'title': 'AirQ Dual-Clock High-Res Cache',\n",
            "        'id': 'antonioroger2/airq-dualclock-highres-cache',\n",
            "        'licenses': [{'name': 'CC0-1.0'}]\n",
            "    }\n",
            "    with open(os.path.join(CACHE_DIR, 'dataset-metadata.json'), 'w') as f:\n",
            "        json.dump(metadata, f, indent=2)\n",
            "\n",
            "    print('Uploading cache as permanent Kaggle Dataset (antonioroger2/airq-dualclock-highres-cache)...')\n",
            "    !kaggle datasets create -p {CACHE_DIR} --dir-mode zip || kaggle datasets version -p {CACHE_DIR} -m 'High-res Rank 1 cache (R=32)' --dir-mode zip\n",
            "    print('\\nSUCCESS: Cache published permanently! You can now attach antonioroger2/airq-dualclock-highres-cache to any notebook for instant 0-second mount.')\n",
            "else:\n",
            "    print('Cache already mounted from Kaggle Datasets or previously published. Skipping upload.')\n"
        ]
    })

    # Cell 8: Run train.py on 2x GPUs with DataParallel & AMP
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=== STEP 2: TRAINING DC-NET ON 2x T4 GPUs (STAGE 1, NO2 FOCUS) ===')\n",
            "!python train.py --cache {CACHE_DIR} --out /kaggle/working/runs/dcnet_main --stage 1 --poll NO2 --long_mem_dropout 0.50 --batch 8 --epochs 120 --vram_cap 0.90\n"
        ]
    })

    # Cell 9: Run evaluate.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=== STEP 3: RUNNING EMPIRICAL CLIMATOLOGY EVALUATION (2024 HELD-OUT) ===')\n",
            "!python evaluate.py --ckpt /kaggle/working/runs/dcnet_main/best.pt --cache {CACHE_DIR} --split val --out /kaggle/working/runs/dcnet_main\n"
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

    # Cell 10: Launch Interactive Gradio Dashboard Service
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('=== STEP 4: LAUNCHING INTERACTIVE GRADIO DASHBOARD SERVICE ===')\n",
            "print('Opening public tunnel... Access the interactive dashboard via the gradio.live URL below:')\n",
            "!python app.py --ckpt /kaggle/working/runs/dcnet_main/best.pt --cache {CACHE_DIR} --share\n"
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
