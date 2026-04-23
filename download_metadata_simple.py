"""
Simple script to download ONLY the NIH metadata CSV (8MB)
"""
import requests
from pathlib import Path

# Create directory
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

output_file = data_dir / "Data_Entry_2017_v2020.csv"

if output_file.exists():
    print(f"✅ File already exists: {output_file}")
    print(f"   Size: {output_file.stat().st_size / (1024**2):.2f} MB")
else:
    print("📥 Downloading NIH metadata CSV (8MB)...")
    
    # Try direct URL to compressed metadata
    url = "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz"
    
    try:
        response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
        response.raise_for_status()
        
        # Save and decompress
        import gzip
        with open(output_file, 'wb') as f:
            f.write(gzip.decompress(response.content))
        
        print(f"✅ Downloaded: {output_file}")
        print(f"   Size: {output_file.stat().st_size / (1024**2):.2f} MB")
        
        # Verify it's a CSV
        with open(output_file, 'r') as f:
            first_line = f.readline()
            print(f"   Header: {first_line.strip()[:80]}...")
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n📝 Manual download instructions:")
        print("   1. Visit: https://www.kaggle.com/datasets/nih-chest-xrays/data")
        print("   2. Download Data_Entry_2017.csv")
        print(f"   3. Save as: {output_file}")
