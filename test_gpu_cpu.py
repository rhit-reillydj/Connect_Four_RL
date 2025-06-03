#!/usr/bin/env python3
"""
Test GPU detection and CPU calculation functionality.
"""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the functions we want to test
try:
    from main import check_gpu_availability, calculate_cpu_workers
    from utils import dotdict
    
    print("Testing GPU Detection:")
    print("=" * 40)
    gpu_available = check_gpu_availability()
    
    print("\nTesting CPU Worker Calculation:")
    print("=" * 40)
    
    # Test with different configurations
    test_configs = [
        {'cpu_usage_fraction': 0.5, 'max_cpu_cores': None},
        {'cpu_usage_fraction': 0.75, 'max_cpu_cores': None},
        {'cpu_usage_fraction': 1.0, 'max_cpu_cores': None},
        {'cpu_usage_fraction': 0.5, 'max_cpu_cores': 2},
        {'cpu_usage_fraction': 0.5, 'max_cpu_cores': 999},  # Should be capped
    ]
    
    for config in test_configs:
        args = dotdict(config)
        workers = calculate_cpu_workers(args)
        print(f"Config {config} → {workers} workers")
    
    print(f"\nSummary:")
    print(f"- GPU Available: {gpu_available}")
    print(f"- Total CPU cores: {os.cpu_count()}")
    print("✅ All tests completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required modules are available.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 