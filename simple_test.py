#!/usr/bin/env python3
"""
Simple test to verify everything works
"""

print("Testing Optiver setup...")

try:
    import pandas as pd
    print("Pandas imported successfully")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")
    exit(1)

try:
    from public_timeseries_testing_util import make_env
    print("Optiver API imported successfully")
except ImportError as e:
    print(f"✗ Optiver API import failed: {e}")
    exit(1)

# Test the API
print("\nTesting API initialization...")
env = make_env()
print("API initialized successfully!")

# Test data loading
print("\nTesting data loading...")
try:
    for batch in env.iter_test():
        if batch is not None:
            test_data, submission_data = batch
            print(f"Test data shape: {test_data.shape}")
            print(f"Submission data shape: {submission_data.shape}")
            print(f"First time_id: {test_data['time_id'].iloc[0]}")
            print(f"Columns: {list(test_data.columns)}")
            break
    print("Data loading test completed!")
except Exception as e:
    print(f"✗ Data loading failed: {e}")

    print("\nAll tests passed! You're ready to start building your time-series analysis.")
