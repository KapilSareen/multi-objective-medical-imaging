"""
Phase 0 - Step 1.5: Download Full NIH ChestX-ray14 Dataset
Uses kagglehub to download all 12 image zip files (~42GB total)
"""

import sys
from pathlib import Path
import subprocess
import zipfile
import time

def main():
    print("="*80)
    print("PHASE 0 - STEP 1.5: DOWNLOAD NIH IMAGES VIA KAGGLEHUB")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading 12 NIH ChestX-ray14 image archives (~42GB)")
    print(f"   Using Kaggle API (kagglehub) - more reliable than NIH Box")
    print(f"   This will take 1-3 hours depending on network speed")
    print(f"   Target location: {data_dir}\n")
    
    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed. Install with:")
        print("   pip install kagglehub")
        sys.exit(1)
    
    start_time = time.time()
    
    # Download full dataset to kagglehub cache
    print("📦 Downloading dataset via Kaggle API...")
    print("   (First download will be slow, subsequent runs use cache)")
    try:
        cache_path = kagglehub.dataset_download("nih-chest-xrays/data")
        print(f"✅ Dataset cached at: {cache_path}")
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        print("\n💡 Make sure you're authenticated with Kaggle:")
        print("   1. Go to https://www.kaggle.com/settings/account")
        print("   2. Create API token (downloads kaggle.json)")
        print("   3. Place at ~/.kaggle/kaggle.json")
        sys.exit(1)
    
    cache_path = Path(cache_path)
    
    # Extract all 12 image zip files to data/raw/
    zip_files = [
        "images_001.zip", "images_002.zip", "images_003.zip", 
        "images_004.zip", "images_005.zip", "images_006.zip",
        "images_007.zip", "images_008.zip", "images_009.zip",
        "images_010.zip", "images_011.zip", "images_012.zip"
    ]
    
    print(f"\n📦 Extracting images from {len(zip_files)} zip files...")
    for i, zip_name in enumerate(zip_files, 1):
        zip_path = cache_path / zip_name
        extract_dir = data_dir / zip_name.replace('.zip', '')
        
        # Skip if already extracted
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"✅ [{i}/12] {zip_name} - Already extracted")
            continue
        
        if not zip_path.exists():
            print(f"⚠️  [{i}/12] {zip_name} - Not found in cache, skipping")
            continue
        
        print(f"📦 [{i}/12] Extracting {zip_name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"✅ [{i}/12] Extracted {zip_name}")
        except Exception as e:
            print(f"❌ [{i}/12] Failed to extract {zip_name}: {e}")
            sys.exit(1)
    
    # Verify extraction
    image_dirs = list(data_dir.glob("images_*"))
    image_count = sum(
        1 for d in image_dirs 
        for img in (d / "images").glob("*.png") if (d / "images").exists()
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Download & extraction complete!")
    print(f"   Total images extracted: {image_count:,}")
    print(f"   Image directories: {len(image_dirs)}")
    print(f"   Time elapsed: {elapsed/60:.1f} minutes")
    print(f"   Storage used: ~45GB")
    print("="*80)

if __name__ == "__main__":
    main()
