"""
Data loading utilities
"""

import pandas as pd
import sys


def load_training_data(data_path='data/train.csv'):
    """
    Load training data with error handling.
    
    Args:
        data_path: Path to training CSV file
        
    Returns:
        DataFrame with training data
    """
    try:
        train_df = pd.read_csv(data_path)
        print(f"Training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
        print(f"Column check: {list(train_df.columns[:5])}...")
        return train_df
    except Exception as e:
        print(f"ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

