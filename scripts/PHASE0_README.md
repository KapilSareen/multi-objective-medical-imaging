# Phase 0: ZCore Coreset Selection - Complete Guide

## Overview

Phase 0 runs on your **LOCAL MACHINE** (not the cluster). It reduces the NIH dataset from 112K images (50GB) to 22K images (10GB) using ZCore.

## The 4 Steps

```
Step 1: Download metadata (1-2 min)     → 01_download_metadata.py
Step 2: Generate embeddings (2-4 hours) → 02_generate_embeddings.py
Step 3: Run ZCore selection (5-10 min)  → 03_run_zcore.py
Step 4: Download coreset (2-4 hours)    → 04_download_coreset.py
```

**Total time:** 6-10 hours (mostly automated)

---

## Prerequisites

### System Requirements
- Python 3.8+
- 30GB free storage
- Internet connection
- **Recommended:** GPU (CUDA-capable) for Step 2
  - With GPU: 2-4 hours
  - With CPU: 6-8 hours

### Install Dependencies

```bash
cd /home/viper/Desktop/multi-object/nsga2_medical_ensemble
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step-by-Step Execution

### Step 1: Download Metadata (~2 minutes)

```bash
python scripts/01_download_metadata.py
```

**What it does:**
- Downloads `Data_Entry_2017.csv` (~1MB)
- Contains image IDs, labels, and demographics
- No images downloaded yet (just metadata)

**Output:**
- `data/raw/Data_Entry_2017.csv`

**Verify:**
```bash
wc -l data/raw/Data_Entry_2017.csv  # Should show ~112,000 lines
```

---

### Step 2: Generate CLIP Embeddings (2-4 hours with GPU)

```bash
python scripts/02_generate_embeddings.py
```

**What it does:**
- Loads CLIP ViT-B/32 model
- Filters metadata for Pleural Effusion cases
- Generates 512-dim embeddings for each image
- **Note:** Requires images to be available (see note below)

**IMPORTANT - Image Access Strategy:**

The script needs actual images to generate embeddings. You have 2 options:

**Option A: Download full dataset first (Recommended)**
```bash
# This takes 2-4 hours for ~50GB
bash scripts/download_nih_full.sh

# Then run embedding generation
python scripts/02_generate_embeddings.py
```

**Option B: Use pre-extracted features (if available)**
- Some researchers share pre-computed CLIP embeddings
- Check NIH dataset page or contact previous researchers

**Output:**
- `data/raw/nih_embeddings.npy` (~500MB)
- `data/raw/nih_valid_metadata.csv`
- Checkpoints every 1000 images: `embeddings_checkpoint_*.npy`

**Timing metrics:**
- Shows progress bar with ETA
- Logs GPU utilization
- Reports images/second processing rate

---

### Step 3: Run ZCore Selection (~5-10 minutes)

```bash
python scripts/03_run_zcore.py
```

**What it does:**
- Loads embeddings from Step 2
- Runs ZCore algorithm (818,560 subspaces)
- Selects 20% most representative samples
- Verifies demographic balance

**Output:**
- `data/raw/nih_coreset.csv` (22,000 selected image IDs)
- `logs/zcore_timing.json` (benchmark data)

**What to check:**
- Coreset size: Should be ~22,000 (20% of filtered dataset)
- Demographic balance: F/M ratio difference < 10%
- Selection time: ~5-10 minutes on CPU

**If demographic imbalance detected:**
The script will warn you. Consider:
- Running stratified ZCore (separate selection per gender)
- Adjusting tolerance threshold
- Documenting as limitation for paper

---

### Step 4: Download Coreset Images (2-4 hours)

```bash
python scripts/04_download_coreset.py
```

**What it does:**
- Copies only the 22,000 selected images from full dataset
- Creates compressed tarball for cluster transfer
- Verifies all images copied successfully

**Output:**
- `data/raw/nih_coreset_images/` (22,000 images, ~10GB)
- `nih_coreset.tar.gz` (~8GB compressed)

**Timing:**
- Shows progress bar with ETA
- Reports images/second copy rate
- Logs total time and tarball creation time

---

## Phase 0 Complete! What's Next?

### Transfer to Cluster

```bash
# Upload tarball to cluster
scp nih_coreset.tar.gz dhish_s@10.13.1.162:/mnt/home2/home/dhish_s/nsga2_medical_ensemble/data/raw/

# Or use rsync for resumable transfer
rsync -avz --progress nih_coreset.tar.gz dhish_s@10.13.1.162:/mnt/home2/home/dhish_s/nsga2_medical_ensemble/data/raw/
```

**Transfer time:** ~30-60 minutes for 8GB

### Verify on Cluster

```bash
# SSH to cluster
ssh dhish_s@10.13.1.162

# Extract
cd /mnt/home2/home/dhish_s/nsga2_medical_ensemble/data/raw/
tar -xzf nih_coreset.tar.gz

# Verify
ls nih_coreset_images/ | wc -l  # Should show 22000
head -5 nih_coreset.csv
```

---

## Troubleshooting

### Issue: "CUDA out of memory" in Step 2
**Solution:**
```bash
# Use CPU instead (slower but works)
CUDA_VISIBLE_DEVICES="" python scripts/02_generate_embeddings.py
```

### Issue: "ZCore not installed" in Step 3
**Solution:**
```bash
pip install fiftyone
git clone https://github.com/voxel51/zcore.git
cd zcore && pip install -e . && cd ..
```

### Issue: Downloads are very slow
**Solution:**
```bash
# Use aria2c for parallel downloads (much faster)
sudo apt install aria2
# Edit download_nih_full.sh to use aria2c instead of wget
```

### Issue: Demographic imbalance in coreset
**Solution:**
- Document in paper as limitation
- Or implement stratified ZCore (run separately on M/F subsets)

---

## Timing Benchmarks

All timing data is logged to:
- `logs/embeddings_timing.json`
- `logs/zcore_timing.json`
- `logs/download_timing.json`

View summary:
```python
from scripts.utils import BenchmarkLogger
logger = BenchmarkLogger("logs/embeddings_timing.json")
logger.print_summary()
```

---

## Files Created by Phase 0

```
data/raw/
├── Data_Entry_2017.csv          # Metadata (~1MB)
├── nih_embeddings.npy           # CLIP embeddings (~500MB)
├── nih_valid_metadata.csv       # Filtered metadata
├── nih_coreset.csv              # Selected 22K image IDs
└── nih_coreset_images/          # 22K images (~10GB)

nih_coreset.tar.gz               # Transfer package (~8GB)

logs/
├── embeddings_timing.json
├── zcore_timing.json
└── download_timing.json
```

---

## Estimated Storage Usage

| Item | Size |
|------|------|
| Metadata CSV | 1 MB |
| Full NIH dataset (if downloaded) | 50 GB |
| CLIP embeddings | 500 MB |
| Coreset images | 10 GB |
| Tarball | 8 GB |
| **Total (with full dataset)** | **68.5 GB** |
| **Total (without full dataset)** | **18.5 GB** |

**Available on your machine:** 34 GB

**Recommendation:** If using Option A (download full dataset first), you'll need more space. Consider:
- External drive
- Delete full dataset after copying coreset
- Or use Option B (API access) if available

---

## Next Phase

After Phase 0 completes and coreset is transferred to cluster:
- **Phase 1-2:** Train 7 CNN models on cluster (15 hours)
- See main plan for details
