"""
Test script for Phase 0 - Verifies all components work
Run this before executing the full Phase 0 pipeline
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import Timer, ProgressTracker, BenchmarkLogger, log_system_info


def test_imports():
    """Test that all required libraries are installed"""
    print("\n🔍 Testing imports...")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'clip': 'CLIP',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'requests': 'Requests',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All imports successful")
    return True


def test_gpu():
    """Test GPU availability"""
    print("\n🖥️  Testing GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   ✅ GPU available: {device_name}")
            print(f"   💾 VRAM: {memory_gb:.1f} GB")
            
            # Test CUDA operation
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x.T)
            print(f"   ✅ CUDA operations work")
            
            return True
        else:
            print(f"   ⚠️  No GPU available - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        return False


def test_clip_model():
    """Test CLIP model loading"""
    print("\n🧠 Testing CLIP model...")
    
    try:
        import torch
        import clip
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with Timer("Load CLIP ViT-B/32"):
            model, preprocess = clip.load("ViT-B/32", device=device)
        
        print(f"   ✅ CLIP model loaded on {device}")
        
        # Test embedding generation
        fake_image = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            embedding = model.encode_image(fake_image)
        
        print(f"   ✅ Embedding shape: {embedding.shape}")
        print(f"   ✅ CLIP works correctly")
        
        return True
    except Exception as e:
        print(f"   ❌ CLIP test failed: {e}")
        print("   Install with: pip install git+https://github.com/openai/CLIP.git")
        return False


def test_zcore():
    """Test ZCore installation"""
    print("\n🔬 Testing ZCore...")
    
    try:
        from zcore import select_coreset
        
        # Test with tiny fake data
        fake_embeddings = np.random.randn(100, 512)
        
        with Timer("ZCore test selection"):
            indices = select_coreset(
                embeddings=fake_embeddings,
                prune_rate=0.8,
                num_subspaces=100,
                m=2
            )
        
        print(f"   ✅ ZCore installed")
        print(f"   ✅ Selected {len(indices)} from 100 samples")
        
        return True
    except ImportError:
        print(f"   ⚠️  ZCore not installed (optional)")
        print("   Install with:")
        print("      pip install fiftyone")
        print("      git clone https://github.com/voxel51/zcore.git")
        print("      cd zcore && pip install -e .")
        print("\n   Scripts will use random selection as fallback")
        return False
    except Exception as e:
        print(f"   ❌ ZCore test failed: {e}")
        return False


def test_utils():
    """Test timing utilities"""
    print("\n⏱️  Testing utilities...")
    
    try:
        # Test Timer
        with Timer("Test timer", verbose=False) as t:
            time.sleep(0.1)
        assert t.elapsed >= 0.1
        print(f"   ✅ Timer works")
        
        # Test ProgressTracker
        with ProgressTracker(10, desc="Test", unit="items") as tracker:
            for i in range(10):
                time.sleep(0.01)
                tracker.update(1)
        print(f"   ✅ ProgressTracker works")
        
        # Test BenchmarkLogger
        logger = BenchmarkLogger("test_benchmark.json")
        logger.log("test_op", 1.5, metadata={'test': True})
        print(f"   ✅ BenchmarkLogger works")
        
        # Cleanup
        Path("test_benchmark.json").unlink(missing_ok=True)
        
        return True
    except Exception as e:
        print(f"   ❌ Utils test failed: {e}")
        return False


def main():
    print("="*80)
    print("PHASE 0 - COMPONENT TESTING")
    print("="*80)
    
    # Log system info
    log_system_info()
    
    # Run tests
    tests = {
        'Imports': test_imports(),
        'GPU': test_gpu(),
        'CLIP': test_clip_model(),
        'ZCore': test_zcore(),
        'Utils': test_utils()
    }
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, result in tests.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    required_tests = ['Imports', 'Utils']
    optional_tests = ['GPU', 'CLIP', 'ZCore']
    
    required_pass = all(tests[t] for t in required_tests)
    
    print("\n" + "="*80)
    if required_pass:
        print("✅ ALL REQUIRED TESTS PASSED")
        print("\nYou can proceed with Phase 0 execution:")
        print("   bash scripts/run_phase0.sh")
        
        if not tests['GPU']:
            print("\n⚠️  Note: No GPU detected - embedding generation will be slow (6-8h)")
        
        if not tests['CLIP']:
            print("\n⚠️  Note: CLIP not available - install before running Step 2")
        
        if not tests['ZCore']:
            print("\n⚠️  Note: ZCore not available - random selection will be used")
    else:
        print("❌ REQUIRED TESTS FAILED")
        print("\nFix the failures above before proceeding")
        sys.exit(1)
    
    print("="*80)


if __name__ == "__main__":
    main()
