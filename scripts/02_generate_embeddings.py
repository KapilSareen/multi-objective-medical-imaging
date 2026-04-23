"""
Phase 0 - Step 2: Generate CLIP Embeddings for NIH Dataset
Downloads and processes NIH images, generates CLIP embeddings
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import clip
from PIL import Image
import time

sys.path.append(str(Path(__file__).parent.parent))

try:
    from scripts.utils import Timer
except:
    class Timer:
        def __init__(self, name="", verbose=True):
            self.name = name
            self.verbose = verbose
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            if self.verbose:
                print(f"✅ {self.name}: {time.time() - self.start:.1f}s")


def load_and_filter_metadata(csv_path, target_disease="Pleural Effusion"):
    """Load NIH metadata and filter for target disease + healthy controls"""
    print(f"\n📂 Loading metadata from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Total images in dataset: {len(df):,}")
    
    mask_disease = df['Finding Labels'].str.contains(target_disease, na=False)
    mask_healthy = df['Finding Labels'] == 'No Finding'
    df_filtered = df[mask_disease | mask_healthy].copy()
    
    df_filtered['target_label'] = df_filtered['Finding Labels'].str.contains(target_disease, na=False).astype(int)
    
    print(f"   Images with {target_disease}: {mask_disease.sum():,}")
    print(f"   Healthy images: {mask_healthy.sum():,}")
    print(f"   Total filtered: {len(df_filtered):,}")
    
    return df_filtered


def find_image_path(image_id, image_dirs):
    """Find image file in extracted NIH directories"""
    for img_dir in image_dirs:
        img_path = img_dir / image_id
        if img_path.exists():
            return img_path
    return None


def generate_clip_embeddings(df, output_path, image_base_dir, device='cuda', batch_size=32, checkpoint_interval=1000):
    """Generate CLIP embeddings for images"""
    print(f"\n🧠 Generating CLIP embeddings")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Checkpoint every: {checkpoint_interval} images")
    
    # Find all image directories
    # Check for resized dataset first (nih_images/)
    if (image_base_dir / "nih_images").exists():
        image_dirs = [image_base_dir / "nih_images"]
    else:
        # Fallback to original structure (images_*/images/)
        image_dirs = list(image_base_dir.glob("images_*/images"))
        if not image_dirs:
            image_dirs = [image_base_dir / "images"]
    
    print(f"   Found {len(image_dirs)} image directories")
    print(f"   Image paths: {[str(d) for d in image_dirs]}")
    
    # Check for existing checkpoint
    checkpoint_dir = output_path.parent
    checkpoint_files = sorted(checkpoint_dir.glob("embeddings_checkpoint_*.npy"))
    
    start_idx = 0
    embeddings = []
    valid_indices = []
    
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print(f"\n♻️  Found checkpoint: {latest_checkpoint.name}")
        try:
            existing_data = np.load(latest_checkpoint, allow_pickle=True).item()
            embeddings = list(existing_data['embeddings'])
            valid_indices = list(existing_data['indices'])
            start_idx = len(embeddings)
            print(f"   Resuming from image {start_idx:,}/{len(df):,}")
        except:
            print(f"   Could not load checkpoint, starting from scratch")
            start_idx = 0
    
    # Load CLIP model
    with Timer("Load CLIP model"):
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
    
    print(f"\n📊 Processing images...")
    failed_count = 0
    
    with torch.no_grad():
        with tqdm(total=len(df) - start_idx, desc="Generating embeddings", unit="img") as pbar:
            for idx in range(start_idx, len(df)):
                row = df.iloc[idx]
                image_id = row['Image Index']
                
                try:
                    # Find image file
                    image_path = find_image_path(image_id, image_dirs)
                    
                    if image_path is None:
                        failed_count += 1
                        if failed_count <= 10:
                            print(f"\n⚠️  Image not found: {image_id}")
                        continue
                    
                    # Load and preprocess image
                    image = Image.open(image_path).convert('RGB')
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    
                    # Generate embedding
                    embedding = model.encode_image(image_input)
                    embeddings.append(embedding.cpu().numpy().squeeze())
                    valid_indices.append(idx)
                    
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 10:
                        print(f"\n⚠️  Failed for {image_id}: {str(e)[:80]}")
                
                pbar.update(1)
                
                # Save checkpoint
                if (idx + 1) % checkpoint_interval == 0:
                    checkpoint_path = checkpoint_dir / f"embeddings_checkpoint_{idx+1}.npy"
                    np.save(checkpoint_path, {
                        'embeddings': np.vstack(embeddings),
                        'indices': valid_indices
                    })
                    print(f"\n💾 Checkpoint saved: {checkpoint_path.name}")
    
    print(f"\n📊 Successfully processed: {len(embeddings):,}/{len(df):,}")
    print(f"   Failed: {failed_count:,}")
    
    # Save final embeddings
    embeddings_array = np.vstack(embeddings)
    np.save(output_path, embeddings_array)
    
    # Save valid metadata
    df_valid = df.iloc[valid_indices].copy()
    df_valid.to_csv(output_path.parent / "nih_filtered_with_embeddings.csv", index=False)
    
    print(f"\n✅ Final embeddings saved: {output_path}")
    print(f"   Shape: {embeddings_array.shape}")
    
    # Cleanup checkpoints
    for checkpoint_file in checkpoint_dir.glob("embeddings_checkpoint_*.npy"):
        checkpoint_file.unlink()
    
    return embeddings_array, df_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--non-interactive', action='store_true')
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 0 - STEP 2: GENERATE CLIP EMBEDDINGS")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    metadata_csv = project_root / "data" / "raw" / "Data_Entry_2017_v2020.csv"
    output_path = project_root / "data" / "cache" / "nih_embeddings.npy"
    image_base_dir = project_root / "data" / "raw"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not metadata_csv.exists():
        print(f"❌ Metadata not found: {metadata_csv}")
        sys.exit(1)
    
    # Check for NIH images (support both directory structures)
    if (image_base_dir / "nih_images").exists():
        image_dirs = [image_base_dir / "nih_images"]
    else:
        image_dirs = list(image_base_dir.glob("images_*/images"))
    if not image_dirs:
        print(f"❌ NIH images not found in {image_base_dir}")
        print("   Expected: data/raw/nih_images/ or data/raw/images_*/images/")
        sys.exit(1)
    
    if output_path.exists() and not args.non_interactive:
        response = input(f"\n⚠️  Embeddings exist. Regenerate? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    elif output_path.exists():
        print(f"\n⚠️  Embeddings exist, skipping")
        sys.exit(0)
    
    df_filtered = load_and_filter_metadata(metadata_csv)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")
    
    with Timer("Total embedding generation"):
        embeddings, df_valid = generate_clip_embeddings(
            df_filtered,
            output_path,
            image_base_dir,
            device=device,
            batch_size=32,
            checkpoint_interval=1000
        )
    
    print("\n" + "="*80)
    print("STEP 2 COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
