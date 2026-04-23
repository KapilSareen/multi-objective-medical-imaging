"""
Phase 0 - Step 4: Keep Only Coreset Images
Deletes non-selected images, keeps only the 22K coreset
"""

import sys
from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

def cleanup_to_coreset(coreset_csv, image_base_dir, output_dir):
    """Keep only coreset images, delete the rest"""
    print(f"\n🗂️  Cleaning up to coreset only")
    
    # Load coreset
    df_coreset = pd.read_csv(coreset_csv)
    coreset_ids = set(df_coreset['Image Index'].values)
    print(f"   Coreset size: {len(coreset_ids):,} images")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image directories
    image_dirs = list(image_base_dir.glob("images_*/images"))
    if not image_dirs:
        image_dirs = [image_base_dir / "images"]
    
    copied_count = 0
    
    print(f"\n📦 Copying coreset images...")
    with tqdm(total=len(coreset_ids), desc="Copying", unit="img") as pbar:
        for image_id in coreset_ids:
            # Find image
            found = False
            for img_dir in image_dirs:
                src_path = img_dir / image_id
                if src_path.exists():
                    dst_path = output_dir / image_id
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                    found = True
                    break
            
            if not found:
                print(f"\n⚠️  Image not found: {image_id}")
            
            pbar.update(1)
    
    print(f"\n✅ Copied {copied_count:,} images to {output_dir}")
    
    # Optionally delete original extracted directories to save space
    print(f"\n🗑️  Cleaning up full dataset directories...")
    deleted_size = 0
    for img_dir_parent in image_base_dir.glob("images_*"):
        if img_dir_parent.is_dir():
            # Get size before deletion
            size = sum(f.stat().st_size for f in img_dir_parent.rglob('*') if f.is_file())
            deleted_size += size
            shutil.rmtree(img_dir_parent)
    
    print(f"   Freed up: {deleted_size / (1024**3):.1f} GB")
    
    return copied_count


def main():
    print("="*80)
    print("PHASE 0 - STEP 4: CREATE CORESET IMAGE DIRECTORY")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    coreset_csv = project_root / "data" / "processed" / "nih_coreset.csv"
    image_base_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "raw" / "nih_coreset_images"
    
    if not coreset_csv.exists():
        print(f"❌ Coreset CSV not found: {coreset_csv}")
        print("   Run 03_run_zcore.py first!")
        sys.exit(1)
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"⚠️  Coreset directory already exists: {output_dir}")
        print(f"   Contains {len(list(output_dir.iterdir()))} files")
        print("   Skipping...")
        sys.exit(0)
    
    count = cleanup_to_coreset(coreset_csv, image_base_dir, output_dir)
    
    print("\n" + "="*80)
    print("STEP 4 COMPLETE")
    print("="*80)
    print(f"📦 Coreset images: {output_dir}")
    print(f"📊 Total images: {count:,}")
    print("\n📌 Phase 0 complete! Ready for Phase 1 (Model Training)")
    print("="*80)

if __name__ == "__main__":
    main()
