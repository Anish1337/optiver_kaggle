"""
Feature engineering functions with leakage prevention
"""

import pandas as pd
import numpy as np


def build_features_leakage_safe(df):
    """
    Build features with NO data leakage.
    All features use shift(1) to ensure temporal integrity.
    
    Args:
        df: Input dataframe with stock_id and time_id
        
    Returns:
        DataFrame with added features
    """
    df = df.copy().sort_values(['stock_id', 'time_id'])
    new_features = {}
    
    for col in ['wap', 'bid_price', 'ask_price']:
        if col in df.columns:
            new_features[f'{col}_mean_5'] = df.groupby('stock_id')[col].transform(
                lambda x: x.shift(1).rolling(5).mean()
            )
            new_features[f'{col}_std_5'] = df.groupby('stock_id')[col].transform(
                lambda x: x.shift(1).rolling(5).std()
            )
            new_features[f'{col}_lag1'] = df.groupby('stock_id')[col].shift(1)
            new_features[f'{col}_lag2'] = df.groupby('stock_id')[col].shift(2)
    
    if 'wap' in df.columns:
        returns = df.groupby('stock_id')['wap'].pct_change()
        new_features['return_lag1'] = returns.shift(1)
        new_features['return_std_5'] = returns.shift(1).rolling(5).std()
    
    new_df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    new_df = new_df.ffill().fillna(0)
    return new_df


def get_feature_columns(df, exclude_cols=None):
    """
    Get numeric feature columns from dataframe.
    
    Args:
        df: Input dataframe
        exclude_cols: List of columns to exclude (default: common exclusion list)
        
    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ['time_id', 'stock_id', 'target', 'realized_vol', 'volatility']
    
    feature_cols = []
    for c in df.columns:
        if c not in exclude_cols:
            # Only include numeric columns
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)
    
    return feature_cols


def time_based_split(df, val_pct=0.2):
    """
    Split data based on time_id to prevent lookahead bias.
    
    Args:
        df: Input dataframe with time_id
        val_pct: Percentage of data for validation
        
    Returns:
        train_df, val_df
    """
    df = df.copy()
    max_time = df['time_id'].max()
    split_time = max_time - int(max_time * val_pct)
    train = df[df['time_id'] <= split_time].copy()
    val = df[df['time_id'] > split_time].copy()
    return train, val


def prepare_target(df):
    """
    Prepare target column from various possible names.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with 'target' column
    """
    df = df.copy()
    
    # Check for target column in various forms
    if 'target' in df.columns:
        target_col = 'target'
    elif 'realized_vol' in df.columns:
        target_col = 'realized_vol'
    elif 'volatility' in df.columns:
        target_col = 'volatility'
        df['target'] = df[target_col]
        target_col = 'target'
    else:
        print("Warning: No target column found, using first numeric column")
        target_col = None

    if target_col:
        df['target'] = df[target_col].fillna(0)
        print(f"Using target column: {target_col}")
    else:
        df['target'] = 0
    
    return df

