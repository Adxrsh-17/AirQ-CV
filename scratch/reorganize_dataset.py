import os
import sys
import glob
import shutil
import zipfile
import stat
import pandas as pd
import rasterio

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("📦 REORGANIZING NEW COMPLETE SATELLITE DATASET INTO data/processed/")
print("=" * 80)

# Target directories
data_dir = os.path.abspath("data")
processed_dir = os.path.join(data_dir, "processed")
s2_target_dir = os.path.join(processed_dir, "s2_composites")
s5p_target_dir = os.path.join(processed_dir, "s5p_composites")
manifest_target = os.path.join(processed_dir, "dataset_manifest.csv")

os.makedirs(s2_target_dir, exist_ok=True)
os.makedirs(s5p_target_dir, exist_ok=True)

# 1. Extract Part 002 if zip exists
zip_002 = os.path.join(data_dir, "Satellite_Downscaling_Project-20260917T020746Z-1-002.zip")
extract_002_dir = os.path.join(data_dir, "Satellite_Downscaling_Project-20260917T020746Z-1-002")

if os.path.exists(zip_002) and not os.path.exists(extract_002_dir):
    print(f"Extracting Part 002 zip: {zip_002}...")
    with zipfile.ZipFile(zip_002, 'r') as zf:
        zf.extractall(extract_002_dir)
    print("✓ Part 002 extracted successfully.")

# 2. Collect all files across all unzipped folders
part_folders = sorted(glob.glob(os.path.join(data_dir, "Satellite_Downscaling_Project*")))
part_folders = [p for p in part_folders if os.path.isdir(p)]
print(f"\nFound {len(part_folders)} extracted part directories to merge.")

s2_moved = 0
s5p_moved = 0
manifest_found = None

for p_dir in part_folders:
    for root, dirs, files in os.walk(p_dir):
        for f in files:
            src_path = os.path.join(root, f)
            if f.startswith("s2_") and f.endswith(".tif"):
                dest_path = os.path.join(s2_target_dir, f)
                shutil.move(src_path, dest_path)
                s2_moved += 1
            elif f.startswith("s5p_") and f.endswith(".tif"):
                dest_path = os.path.join(s5p_target_dir, f)
                shutil.move(src_path, dest_path)
                s5p_moved += 1
            elif f == "dataset_manifest.csv":
                manifest_found = src_path
                shutil.copy2(src_path, manifest_target)

print(f"✓ Consolidated {s2_moved} Sentinel-2 GeoTIFFs into {s2_target_dir}")
print(f"✓ Consolidated {s5p_moved} Sentinel-5P GeoTIFFs into {s5p_target_dir}")
if manifest_found:
    print(f"✓ Copied latest manifest to {manifest_target}")

# 3. Clean up temporary extracted folders and zip files
print("\n--- Cleaning up temporary extraction folders and zip archives ---")
for p_dir in part_folders:
    try:
        shutil.rmtree(p_dir, onexc=lambda func, path, exc: (os.chmod(path, stat.S_IWRITE), func(path)))
        print(f"✓ Removed temporary folder: {os.path.basename(p_dir)}")
    except Exception as e:
        print(f"Could not remove {p_dir}: {e}")

zip_files = sorted(glob.glob(os.path.join(data_dir, "Satellite_Downscaling_Project*.zip")))
for z in zip_files:
    try:
        os.chmod(z, stat.S_IWRITE)
        os.remove(z)
        print(f"✓ Removed zip archive: {os.path.basename(z)}")
    except Exception as e:
        print(f"Could not remove {z}: {e}")

# 4. Final Verification and Manifest Alignment
s2_final = sorted(glob.glob(os.path.join(s2_target_dir, "*.tif")))
s5p_final = sorted(glob.glob(os.path.join(s5p_target_dir, "*.tif")))
print(f"\nFinal Consolidated File Counts in data/processed/:")
print(f"  Sentinel-2 GeoTIFFs:   {len(s2_final)}")
print(f"  Sentinel-5P GeoTIFFs:  {len(s5p_final)}")

s2_set = set(os.path.basename(f) for f in s2_final)
s5p_set = set(os.path.basename(f) for f in s5p_final)

paired_count = 0
if os.path.exists(manifest_target):
    df_manifest = pd.read_csv(manifest_target)
    # Update actual on-disk status
    statuses = []
    for _, r in df_manifest.iterrows():
        s2_present = r["s2_file"] in s2_set
        s5p_present = r["s5p_file"] in s5p_set
        if s2_present and s5p_present:
            statuses.append("complete")
            paired_count += 1
        elif s2_present:
            statuses.append("s5p_missing")
        elif s5p_present:
            statuses.append("s2_cloud_gap")
        else:
            statuses.append("missing")
    df_manifest["status"] = statuses
    df_manifest.to_csv(manifest_target, index=False)
    print(f"  Total Rows in Manifest: {len(df_manifest)}")
    print(f"  Valid Paired Windows (Both S2 + S5P Present): {paired_count} ({paired_count/len(df_manifest)*100:.1f}%)")
    print(f"  Manifest Status Breakdown:\n{df_manifest['status'].value_counts()}")

# Check sample raster properties
if len(s2_final) > 0:
    with rasterio.open(s2_final[0]) as src:
        print(f"\nSample S2 ({os.path.basename(s2_final[0])}): shape={src.shape}, bands={src.count}, res={src.res}")
if len(s5p_final) > 0:
    with rasterio.open(s5p_final[0]) as src:
        print(f"Sample S5P ({os.path.basename(s5p_final[0])}): shape={src.shape}, bands={src.count}, res={src.res}")

print("=" * 80)
print("🎉 REORGANIZATION COMPLETED SUCCESSFULLY!")
print("=" * 80)
