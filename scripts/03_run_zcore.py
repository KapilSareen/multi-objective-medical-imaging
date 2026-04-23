"""
Phase 0 - Step 3: Run ZCore Coreset Selection
Selects 20% representative subset from embeddings
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from utils import Timer, BenchmarkLogger, log_system_info

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def run_zcore_selection(embeddings, prune_rate=0.8, num_subspaces=818560, m=2, alpha=1000, beta=4):
    """
    Run ZCore coreset selection
    
    Args:
        embeddings: numpy array of shape (N, D)
        prune_rate: Fraction to prune (0.8 = keep 20%)
        num_subspaces: Number of subspaces to sample (default from paper)
        m: Subspace dimensionality (default 2)
        alpha: Number of nearest neighbors for redundancy (default 1000)
        beta: Redundancy penalty exponent (default 4)
    
    Returns:
        Array of selected indices
    """
    print(f"\n🔬 Running ZCore Selection")
    print(f"   Input shape: {embeddings.shape}")
    print(f"   Prune rate: {prune_rate} (keeping {1-prune_rate:.0%})")
    print(f"   Target coreset size: {int(len(embeddings) * (1-prune_rate)):,}")
    
    print(f"\n   Hyperparameters:")
    print(f"      num_subspaces: {num_subspaces:,}")
    print(f"      m (subspace dim): {m}")
    print(f"      alpha (neighbors): {alpha}")
    print(f"      beta (penalty): {beta}")
    
    try:
        # Try importing ZCore
        from zcore import select_coreset
        
        with Timer("ZCore selection algorithm"):
            coreset_indices = select_coreset(
                embeddings=embeddings,
                prune_rate=prune_rate,
                num_subspaces=num_subspaces,
                m=m,
                alpha=alpha,
                beta=beta
            )
        
        return coreset_indices
        
    except ImportError:
        print("\n❌ ZCore not installed!")
        print("   Install with: pip install fiftyone && git clone https://github.com/voxel51/zcore && cd zcore && pip install -e .")
        print("\n   For now, using RANDOM SELECTION as fallback (not optimal!)")
        
        # Fallback: random selection
        np.random.seed(42)
        n_select = int(len(embeddings) * (1 - prune_rate))
        coreset_indices = np.random.choice(len(embeddings), size=n_select, replace=False)
        print(f"   Selected {len(coreset_indices):,} random samples")
        
        return coreset_indices


def verify_demographic_balance(df_full, df_coreset, tolerance=0.1):
    """
    Verify that coreset maintains demographic balance
    
    Args:
        df_full: Full dataset metadata
        df_coreset: Coreset metadata
        tolerance: Maximum acceptable ratio difference (default 0.1 = 10%)
    
    Returns:
        True if balanced, False otherwise
    """
    print(f"\n⚖️  Verifying Demographic Balance")
    print(f"   Tolerance: {tolerance*100:.0f}%")
    
    # Check gender balance
    if 'Patient Gender' in df_full.columns:
        # Full dataset ratios
        full_gender = df_full['Patient Gender'].value_counts()
        full_f = full_gender.get('F', 0)
        full_m = full_gender.get('M', 0)
        full_ratio = full_f / full_m if full_m > 0 else 0
        
        # Coreset ratios
        coreset_gender = df_coreset['Patient Gender'].value_counts()
        coreset_f = coreset_gender.get('F', 0)
        coreset_m = coreset_gender.get('M', 0)
        coreset_ratio = coreset_f / coreset_m if coreset_m > 0 else 0
        
        # Calculate difference
        ratio_diff = abs(full_ratio - coreset_ratio)
        is_balanced = ratio_diff < tolerance
        
        print(f"\n   Gender Distribution:")
        print(f"      Full dataset  - F: {full_f:,} ({full_f/len(df_full)*100:.1f}%), M: {full_m:,} ({full_m/len(df_full)*100:.1f}%)")
        print(f"      Coreset       - F: {coreset_f:,} ({coreset_f/len(df_coreset)*100:.1f}%), M: {coreset_m:,} ({coreset_m/len(df_coreset)*100:.1f}%)")
        print(f"\n   F/M Ratio:")
        print(f"      Full dataset: {full_ratio:.3f}")
        print(f"      Coreset:      {coreset_ratio:.3f}")
        print(f"      Difference:   {ratio_diff:.3f}")
        
        if is_balanced:
            print(f"   ✅ Balance preserved (within {tolerance*100:.0f}% tolerance)")
        else:
            print(f"   ⚠️  WARNING: Imbalance detected! (>{tolerance*100:.0f}% difference)")
            print(f"      This may introduce fairness bias!")
        
        return is_balanced
    else:
        print("   ⚠️  Gender column not found - cannot verify balance")
        return None


def main():
    print("="*80)
    print("PHASE 0 - STEP 3: RUN ZCORE CORESET SELECTION")
    print("="*80)
    
    # Log system info
    log_system_info()
    
    # Initialize benchmark logger
    logger = BenchmarkLogger("logs/zcore_timing.json")
    
    # Paths
    project_root = Path(__file__).parent.parent
    embeddings_path = project_root / "data" / "cache" / "nih_embeddings.npy"
    metadata_path = project_root / "data" / "cache" / "nih_filtered_with_embeddings.csv"
    output_csv = project_root / "data" / "processed" / "nih_coreset.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if inputs exist
    if not embeddings_path.exists():
        print(f"❌ Embeddings not found: {embeddings_path}")
        print("   Run 02_generate_embeddings.py first!")
        sys.exit(1)
    
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        print("   Run 02_generate_embeddings.py first!")
        sys.exit(1)
    
    # Load embeddings
    with Timer("Load embeddings"):
        embeddings = np.load(embeddings_path)
        metadata = pd.read_csv(metadata_path)
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Metadata rows: {len(metadata):,}")
        assert len(embeddings) == len(metadata), "Embeddings and metadata size mismatch!"
    
    # Run ZCore selection
    with Timer("ZCore coreset selection") as timer:
        coreset_indices = run_zcore_selection(
            embeddings,
            prune_rate=0.8,  # Keep 20%
            num_subspaces=818560,  # Default from paper
            m=2,
            alpha=1000,
            beta=4
        )
    
    # Log timing
    logger.log("zcore_selection", timer.elapsed, metadata={
        'n_samples': len(embeddings),
        'coreset_size': len(coreset_indices),
        'prune_rate': 0.8
    })
    
    # Extract coreset metadata
    print(f"\n📊 Extracting coreset metadata...")
    coreset_metadata = metadata.iloc[coreset_indices].copy()
    
    # Verify demographic balance
    is_balanced = verify_demographic_balance(metadata, coreset_metadata, tolerance=0.1)
    
    # Save coreset metadata
    with Timer("Save coreset metadata"):
        coreset_metadata.to_csv(output_csv, index=False)
        print(f"   Saved to: {output_csv}")
    
    # Print summary
    print("\n" + "="*80)
    print("ZCORE SELECTION COMPLETE")
    print("="*80)
    print(f"📁 Output: {output_csv}")
    print(f"📊 Original size: {len(metadata):,} images")
    print(f"📊 Coreset size: {len(coreset_indices):,} images ({len(coreset_indices)/len(metadata)*100:.1f}%)")
    print(f"⚖️  Demographic balance: {'✅ Preserved' if is_balanced else '⚠️  Check required'}")
    print(f"⏱️  Selection time: {Timer.format_time(timer.elapsed)}")
    print("\n📌 Next step: Run 04_download_coreset.py")
    print("="*80)
    
    # Save benchmark summary
    logger.print_summary()


if __name__ == "__main__":
    main()
