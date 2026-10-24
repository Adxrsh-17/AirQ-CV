import json, os, gzip, base64

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Read code files
    code_files = ["align_tn.py", "dataset.py", "dcnet.py", "train.py", "evaluate.py", "make_cache.py", "app.py", "check_cache.py", "landuse_baseline.py"]
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
            "# AirQ-CV Stage B: land-use-prior forecaster (DC-Net v2)\n",
            "- S2 land use (annual + seasonal bands, NDBI/NDVI/MNDWI/NDRE/BSI, context pyramid) + Stage A OOF prior + S5P anomaly history + ERA5 history\n",
            "- Forecast of future S5P composites at **5, 10, 15, 30, 60, 120 days** (Rule.md section 7)\n",
            "- Output = harmonic climatology + damped anomaly persistence + network residual; quantiles q10/q50/q90\n",
            "- Split: train 2019-2023, validate 2024; metrics = skill vs climatology/persistence, SSIM, hotspot blobs\n"
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
            "# 1. Check if cache is mounted directly from Kaggle Datasets (Instant & 0 disk usage)\n",
            "mounted_cache = None\n",
            "for root, dirs, files in os.walk('/kaggle/input'):\n",
            "    if 's2_month.npy' in files and 'meta.json' in files:\n",
            "        mounted_cache = root\n",
            "        break\n",
            "\n",
            "if mounted_cache:\n",
            "    import numpy as _np\n",
            "    _s2 = _np.load(os.path.join(mounted_cache, 's2_month.npy'), mmap_mode='r')\n",
            "    _bad = sum(int((~_np.isfinite(_s2[m])).sum()) for m in range(_s2.shape[0]))\n",
            "    print(f'Mounted cache {mounted_cache}: s2_month {_s2.shape} non-finite values = {_bad:,}')\n",
            "    if _bad > 0:\n",
            "        print('STALE CACHE (contains NaN/inf) -> rebuilding with current make_cache.py')\n",
            "        mounted_cache = None\n",
            "if mounted_cache:\n",
            "    print(f'Mounted Kaggle Dataset cache detected at: {mounted_cache}! (Zero disk usage)')\n",
            "    CACHE_DIR = mounted_cache\n",
            "else:\n",
            "    # Use ephemeral /kaggle/tmp (~60 GB available, keeping /kaggle/working 100% clean)\n",
            "    CACHE_DIR = '/kaggle/tmp/cache' if os.path.exists('/kaggle/tmp') else '/tmp/cache'\n",
            "    os.makedirs(CACHE_DIR, exist_ok=True)\n",
            "    print('Downloading verified native R=8 cache from https://www.kaggle.com/datasets/adarshpradeep17/airq-dualclock-cache (~15s)...')\n",
            "    pass  # skip re-downloading the published cache (it is the stale one)\n",
            "    if os.path.exists(os.path.join(CACHE_DIR, 'meta.json')):\n",
            "        print(f'SUCCESS: Cache downloaded and unzipped into {CACHE_DIR}!')\n",
            "    else:\n",
            "        print('Building aligned cache in temporary storage (RATIO=8, ~6-8 mins)...')\n",
            "        !python make_cache.py --s2_dir data2/s2_composites --s5p_dir data2/s5p_composites --out {CACHE_DIR} --s2_ratio 8 --weather weather/era5_tn_clusters.csv --start 2019-01-01 --end 2024-12-31 --zero_is_invalid 1\n",
            "    print('Cache ready for multi-GPU training!')\n"
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
            "# Cache is already live at https://www.kaggle.com/datasets/adarshpradeep17/airq-dualclock-cache\n",
            "PUBLISH_TO_OWN_ACCOUNT = False\n",
            "if PUBLISH_TO_OWN_ACCOUNT and not mounted_cache and os.path.exists(os.path.join(CACHE_DIR, 'meta.json')):\n",
            "    print('Setting up Kaggle credentials for antonioroger2...')\n",
            "    kaggle_dir = os.path.expanduser('~/.kaggle')\n",
            "    os.makedirs(kaggle_dir, exist_ok=True)\n",
            "    # Credentials: add KAGGLE_USERNAME / KAGGLE_KEY as Kaggle Secrets; never hardcode them.\n",
            "    from kaggle_secrets import UserSecretsClient\n",
            "    _sec = UserSecretsClient(); os.environ['KAGGLE_USERNAME'] = _sec.get_secret('KAGGLE_USERNAME'); os.environ['KAGGLE_KEY'] = _sec.get_secret('KAGGLE_KEY')\n",
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
            "    !kaggle datasets create -p {CACHE_DIR} --dir-mode zip || kaggle datasets version -p {CACHE_DIR} -m 'Native 1:1 high-fidelity cache (R=8, 600m)' --dir-mode zip\n",
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
            "# MODE: 'landuse' = Rule.md Stage A (spatial land-use baseline); 'dcnet' = Stage B forecaster (Rule.md section 7)\n",
            "MODE = 'dcnet'\n",
            "EPOCHS = 40          # early stopping (patience 10) on mean validation skill vs climatology\n",
            "POLL = 'all'         # 'all' = NO2+CO+SO2 joint model, or 'NO2' / 'CO' / 'SO2'\n",
            "ABLATE_B = False     # True: also train Config B (no S2, no prior) to measure what land use adds (doubles run time)\n",
            "SEEDS = [0]          # e.g. [0, 1, 2] -> seed ensemble in evaluate.py (better metrics, 3x run time)\n",
            "RUN = '/kaggle/working/runs'\n",
            "PRIOR_DIR = '/kaggle/working/priors'\n",
            "AUX_DIR = CACHE_DIR if os.access(CACHE_DIR, os.W_OK) else '/kaggle/tmp/aux'\n",
            "CONFIGS = ['C'] + (['B'] if ABLATE_B else [])\n",
            "OUT = f'{RUN}/dcnet_{POLL}_C_s{SEEDS[0]}'\n",
            "print('=== CACHE AUDIT (valid-pixel coverage, finiteness) ===')\n",
            "!python check_cache.py --cache {CACHE_DIR} --csv /kaggle/working/cache_audit.csv\n",
            "if MODE == 'landuse':\n",
            "    print('=== STAGE A: LAND-USE-INFERRED POLLUTION (S2 only, no time/weather/S5P inputs) ===')\n",
            "    !python -u landuse_baseline.py --cache {CACHE_DIR} --out /kaggle/working/runs/landuse --poll NO2 --cnn 1 --blocks 8,4,16\n",
            "if MODE == 'dcnet':\n",
            "    print('=== STAGE B-1: OUT-OF-FOLD LAND-USE PRIOR MAPS (S2 only, target years 2019-2023 = training period) ===')\n",
            "    for _p, _lt in (('NO2', 1), ('CO', 1), ('SO2', 0)):\n",
            "        if POLL in ('all', _p):\n",
            "            !python -u landuse_baseline.py --cache {CACHE_DIR} --out {PRIOR_DIR} --poll {_p} --y0 2019 --y1 2023 --prior_only 1 --prior_block 8 --log_target {_lt}\n",
            "    for _cfg in CONFIGS:\n",
            "        for _sd in SEEDS:\n",
            "            print(f'=== STAGE B-2: TRAIN config {_cfg} seed {_sd} ({EPOCHS} epochs max) ===')\n",
            "            _out = f'{RUN}/dcnet_{POLL}_{_cfg}_s{_sd}'\n",
            "            !python -u train.py --cache {CACHE_DIR} --out {_out} --config {_cfg} --poll {POLL} --prior_dir {PRIOR_DIR} --aux_dir {AUX_DIR} --epochs {EPOCHS} --seed {_sd} --workers 4 --no_dp\n"
        ]
    })

    # Cell 9: Run evaluate.py
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if MODE == 'dcnet':\n",
            "    for _cfg in CONFIGS:\n",
            "        _ck = ','.join(f'{RUN}/dcnet_{POLL}_{_cfg}_s{_sd}/best.pt' for _sd in SEEDS)\n",
            "        print(f'=== STAGE B-3: EVALUATE config {_cfg} on the 2024 validation year ===')\n",
            "        !python -u evaluate.py --ckpt {_ck} --cache {CACHE_DIR} --split val --out {RUN}/eval_{POLL} --tag {_cfg}\n"
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
            "if MODE == 'landuse':\n",
            "    for f_ in sorted(glob.glob('/kaggle/working/runs/landuse/*.csv')):\n",
            "        print('\\n' + '='*80 + '\\n' + f_ + '\\n' + '='*80); print(pd.read_csv(f_).to_string(index=False))\n",
            "if MODE == 'dcnet':\n",
            "    from IPython.display import Image, display\n",
            "    _ev = f'{RUN}/eval_{POLL}'\n",
            "    _m = {c: pd.read_csv(f'{_ev}/dcnet_val_metrics_{c}.csv') for c in CONFIGS if os.path.exists(f'{_ev}/dcnet_val_metrics_{c}.csv')}\n",
            "    for c, d in _m.items():\n",
            "        print('\\n' + '='*100 + f'\\nCONFIG {c} - 2024 validation (all valid pixels)\\n' + '='*100)\n",
            "        print(d[d.region == 'all'][['pollutant', 'horizon_days', 'SS_clim', 'SS_clim_lo', 'SS_clim_hi', 'SS_pers', 'anomaly_r', 'R2', 'R2_climatology', 'SSIM', 'CSI_p90', 'coverage_80']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))\n",
            "        _b = pd.read_csv(f'{_ev}/dcnet_val_blobs_{c}.csv')\n",
            "        print('\\nhotspot blobs:'); print(_b.to_string(index=False, float_format=lambda v: f'{v:.3f}'))\n",
            "        for _f in [f'{_ev}/summary_val_{c}.png'] + sorted(glob.glob(f'{_ev}/panels_*_val_{c}.png')):\n",
            "            if os.path.exists(_f): display(Image(_f))\n",
            "    if 'C' in _m and 'B' in _m:\n",
            "        _j = _m['C'].merge(_m['B'], on=['pollutant', 'horizon_days', 'region'], suffixes=('_C', '_B'))\n",
            "        _j['land_use_gain'] = _j['SS_clim_C'] - _j['SS_clim_B']\n",
            "        print('\\n=== LAND-USE CONTRIBUTION (Config C - Config B, SS_clim) ===')\n",
            "        print(_j[['pollutant', 'horizon_days', 'region', 'SS_clim_C', 'SS_clim_B', 'land_use_gain']].to_string(index=False, float_format=lambda v: f'{v:+.3f}'))\n"
        ]
    })

    # Cell 10: Launch Interactive Gradio Dashboard Service
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "LAUNCH_APP = False  # the Gradio server never exits, so a batch (push) run would hang until the Kaggle time limit\n",
            "if LAUNCH_APP:\n",
            "    print('=== STEP 4: LAUNCHING INTERACTIVE GRADIO DASHBOARD SERVICE ===')\n",
            "    !python app.py --ckpt {OUT}/best.pt --cache {CACHE_DIR} --share\n",
            "else:\n",
            "    print('Skipping dashboard (LAUNCH_APP=False). Outputs are in /kaggle/working/runs/')\n"
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
