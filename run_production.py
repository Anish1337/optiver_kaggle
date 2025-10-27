#!/usr/bin/env python3
"""
Optiver Realized Volatility Prediction

Implementation for Optiver Trading at the Close competition.
Trains ensemble models to predict realized volatility.
"""

import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

# Import custom modules
from utils.data_loading import load_training_data
from utils.feature_engineering import (
    build_features_leakage_safe,
    get_feature_columns,
    time_based_split,
    prepare_target
)
from utils.metrics import calculate_rmspe
from models.lightgbm_model import LightGBMModel
from models.xgboost_model import XGBoostModel
from models.ensemble import EnsembleModel

print("="*60)
print("OPTIVER REALIZED VOLATILITY PREDICTION")
print("="*60)

# Step 1: Load data
print("\n[1/7] Loading data...")
train_df = load_training_data('data/train.csv')

# Step 2: Prepare target
print("\n[2/7] Preparing target...")
train_df = prepare_target(train_df)
print("Target ready")

# Step 3: Build leakage-safe features
print("\n[3/7] Building features...")
features_df = build_features_leakage_safe(train_df)
feature_cols = get_feature_columns(features_df)
print(f"Features: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:10]}")

# Step 4: Time-based split
print("\n[4/7] Time-based split...")
train_split_df, val_split_df = time_based_split(features_df, val_pct=0.2)
X_train = train_split_df[feature_cols].fillna(0)
y_train = train_split_df['target'].fillna(0)
X_val = val_split_df[feature_cols].fillna(0)
y_val = val_split_df['target'].fillna(0)
print(f"Split: {len(X_train)} train, {len(X_val)} val")

# Step 5: Train LightGBM
print("\n[5/7] Training LightGBM...")
lgb_model = LightGBMModel(num_boost_round=50, early_stopping_rounds=10)
lgb_model.train(X_train, y_train, X_val, y_val)
lgb_pred = lgb_model.predict(X_val)
lgb_rmspe = calculate_rmspe(y_val, lgb_pred)
print(f"LightGBM RMSPE: {lgb_rmspe:.6f}")

# Step 6: Train XGBoost
print("\n[6/7] Training XGBoost...")
xgb_model = XGBoostModel(max_depth=6, learning_rate=0.03, n_estimators=50,
                         subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1,
                         reg_lambda=0.1, tree_method='hist')
xgb_model.train(X_train, y_train, X_val, y_val)
xgb_pred = xgb_model.predict(X_val)
xgb_rmspe = calculate_rmspe(y_val, xgb_pred)
print(f"XGBoost RMSPE: {xgb_rmspe:.6f}")

# Ensemble predictions
ensemble_model = EnsembleModel([lgb_model, xgb_model])
ensemble_pred_val = ensemble_model.predict(X_val)
ensemble_rmspe = calculate_rmspe(y_val, ensemble_pred_val)
print(f"Ensemble RMSPE: {ensemble_rmspe:.6f}")

# Save results
results = {
    'lightgbm_rmspe': lgb_rmspe,
    'xgboost_rmspe': xgb_rmspe,
    'ensemble_rmspe': ensemble_rmspe,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'n_features': len(feature_cols)
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results.json")

# Step 7: Generate submission
print("\n[7/7] Generating submission...")
X_all = features_df[feature_cols].fillna(0)
lgb_preds = lgb_model.predict(X_all)
xgb_preds = xgb_model.predict(X_all)
ensemble_preds = ensemble_model.predict(X_all)

# Create submission from predictions on all data
predictions = features_df[['time_id', 'stock_id']].copy()
predictions['target'] = ensemble_preds[:len(predictions)]
predictions['target'] = predictions['target'].fillna(0).astype(float)

predictions.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("SUBMISSION GENERATED")
print("="*60)
print(f"\nShape: {predictions.shape}")
print(f"Schema: {list(predictions.columns)}")
print(f"\nFirst 10 rows:")
print(predictions.head(10))
print(f"\nTarget statistics:")
print(predictions['target'].describe())
print(f"\nNo NaNs: {predictions['target'].isna().sum()}")
print(f"Submission file created")

print("\n" + "="*60)
print("FINISHED")
print("="*60)
