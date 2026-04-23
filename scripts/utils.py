"""
Utility functions for timing, progress tracking, and logging
Used across all scripts to monitor performance
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import functools


class Timer:
    """Context manager and decorator for timing code execution"""
    
    def __init__(self, name="Operation", verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"⏱️  Starting: {self.name}")
            print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.verbose:
            print(f"✅ Completed: {self.name}")
            print(f"   Duration: {self.format_time(self.elapsed)}")
            print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    @staticmethod
    def format_time(seconds):
        """Format seconds into human-readable time"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes ({seconds:.0f}s)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.2f} hours ({minutes:.0f}m {seconds % 60:.0f}s)"
    
    def __call__(self, func):
        """Use as decorator"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(f"{func.__name__}", verbose=self.verbose):
                return func(*args, **kwargs)
        return wrapper


class ProgressTracker:
    """Track progress with ETA for long-running operations"""
    
    def __init__(self, total, desc="Processing", unit="items"):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.pbar = tqdm(total=total, desc=desc, unit=unit)
        self.start_time = time.time()
    
    def update(self, n=1):
        """Update progress by n items"""
        self.pbar.update(n)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.pbar.n > 0:
            rate = self.pbar.n / elapsed
            remaining = self.total - self.pbar.n
            eta_seconds = remaining / rate if rate > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))
            
            # Update progress bar postfix with ETA
            self.pbar.set_postfix({
                'ETA': str(eta),
                'Rate': f'{rate:.2f} {self.unit}/s'
            })
    
    def close(self):
        """Close progress bar"""
        self.pbar.close()
        elapsed = time.time() - self.start_time
        print(f"\n✅ Completed {self.total} {self.unit} in {Timer.format_time(elapsed)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class BenchmarkLogger:
    """Log timing benchmarks to file for analysis"""
    
    def __init__(self, log_file="timing_report.json"):
        self.log_file = Path(log_file)
        self.benchmarks = {}
        self.load()
    
    def load(self):
        """Load existing benchmarks"""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.benchmarks = json.load(f)
    
    def save(self):
        """Save benchmarks to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.benchmarks, f, indent=2)
    
    def log(self, operation, duration, metadata=None):
        """Log a timing benchmark"""
        timestamp = datetime.now().isoformat()
        
        if operation not in self.benchmarks:
            self.benchmarks[operation] = []
        
        entry = {
            'timestamp': timestamp,
            'duration_seconds': duration,
            'duration_formatted': Timer.format_time(duration),
            'metadata': metadata or {}
        }
        
        self.benchmarks[operation].append(entry)
        self.save()
        
        print(f"📊 Benchmark logged: {operation} took {Timer.format_time(duration)}")
    
    def get_average(self, operation):
        """Get average time for an operation"""
        if operation not in self.benchmarks or len(self.benchmarks[operation]) == 0:
            return None
        
        durations = [entry['duration_seconds'] for entry in self.benchmarks[operation]]
        avg = sum(durations) / len(durations)
        return Timer.format_time(avg)
    
    def print_summary(self):
        """Print summary of all benchmarks"""
        print("\n" + "="*60)
        print("TIMING BENCHMARK SUMMARY")
        print("="*60)
        
        for operation, entries in self.benchmarks.items():
            if entries:
                durations = [e['duration_seconds'] for e in entries]
                avg = sum(durations) / len(durations)
                min_dur = min(durations)
                max_dur = max(durations)
                
                print(f"\n{operation}:")
                print(f"  Runs: {len(entries)}")
                print(f"  Average: {Timer.format_time(avg)}")
                print(f"  Min: {Timer.format_time(min_dur)}")
                print(f"  Max: {Timer.format_time(max_dur)}")
        
        print("\n" + "="*60)


def estimate_remaining_time(current, total, elapsed):
    """
    Estimate remaining time based on current progress
    
    Args:
        current: Number of items processed
        total: Total number of items
        elapsed: Time elapsed so far (seconds)
    
    Returns:
        Formatted string with ETA
    """
    if current == 0:
        return "Calculating..."
    
    rate = current / elapsed
    remaining_items = total - current
    eta_seconds = remaining_items / rate if rate > 0 else 0
    
    return Timer.format_time(eta_seconds)


def log_system_info():
    """Log system information for debugging"""
    import platform
    import psutil
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"CPU Count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    except ImportError:
        print("PyTorch: Not installed")
    
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Test Timer
    print("Testing Timer...")
    with Timer("Test operation"):
        time.sleep(2)
    
    # Test ProgressTracker
    print("\nTesting ProgressTracker...")
    with ProgressTracker(100, desc="Processing items") as tracker:
        for i in range(100):
            time.sleep(0.01)
            tracker.update(1)
    
    # Test BenchmarkLogger
    print("\nTesting BenchmarkLogger...")
    logger = BenchmarkLogger("test_timing.json")
    logger.log("test_operation", 2.5, metadata={'items': 100})
    logger.print_summary()
    
    # Test system info
    log_system_info()
