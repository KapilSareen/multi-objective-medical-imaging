"""
Phase 0 - Step 1: Download NIH ChestX-ray14 Metadata
Uses kagglehub API to download metadata CSV file
"""

import kagglehub
import shutil
from pathlib import Path
import sys
from utils import Timer, log_system_info

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def main():
    print("="*80)
    print("PHASE 0 - STEP 1: DOWNLOAD NIH METADATA")
    print("="*80)
    
    # Log system info
    log_system_info()
    
    # Create output directory
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_file = data_dir / "Data_Entry_2017_v2020.csv"
    
    # Skip if already exists
    if metadata_file.exists():
        print(f"\n⏭️  Skipping metadata download (already exists)")
        print(f"   Location: {metadata_file}")
        print(f"   Size: {metadata_file.stat().st_size / (1024**2):.2f} MB")
        return
    
    print("\n📥 Downloading NIH ChestX-ray14 metadata via Kaggle API...")
    print("   Dataset: nih-chest-xrays/data")
    print("   File: Data_Entry_2017.csv")
    
    with Timer("Metadata download"):
        try:
            # Download just the CSV using kagglehub pandas adapter (avoids downloading full 42GB dataset)
            from kagglehub import KaggleDatasetAdapter
            import pandas as pd
            
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "nih-chest-xrays/data",
                "Data_Entry_2017.csv"
            )
            
            print(f"\n✅ Loaded DataFrame: {len(df):,} rows, {len(df.columns)} columns")
            
            # Save to CSV
            print(f"\n📁 Saving to: {metadata_file}")
            df.to_csv(metadata_file, index=False)
            
            print(f"✅ Metadata ready at: {metadata_file}")
            print(f"   Size: {metadata_file.stat().st_size / (1024**2):.2f} MB")
            
        except Exception as e:
            print(f"\n❌ Error downloading metadata: {e}")
            print("\n⚠️  FALLBACK: You can manually download from:")
            print("   https://www.kaggle.com/datasets/nih-chest-xrays/data")
            print(f"   Save as: {metadata_file}")
            sys.exit(1)
    
    print("\n" + "="*80)
    print("METADATA DOWNLOAD COMPLETE")
    print("="*80)
    print(f"📁 Location: {metadata_file}")
    print(f"📊 Size: {metadata_file.stat().st_size / (1024**2):.2f} MB")
    print("\n📌 Next step: Run 02_generate_embeddings.py")
    print("="*80)


if __name__ == "__main__":
    main()
