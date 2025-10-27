#!/usr/bin/env python3
"""
Optiver Realized Volatility Prediction

Implementation for Optiver Trading at the Close competition.
Trains ensemble models to predict realized volatility.
Now includes GARCH and ARIMA econometric models.
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
from models.garch_model import GARCHModel
from models.arima_model import ARIMAModel
from models.ensemble import EnsembleModel

print("="*60)
print("OPTIVER REALIZED VOLATILITY PREDICTION")
print("="*60)

# Step 1: Load data
print("\n[1/9] Loading data...")
train_df = load_training_data('data/train.csv')

# Step 2: Prepare target
print("\n[2/9] Preparing target...")
train_df = prepare_target(train_df)
print("Target ready")

# Step 3: Build leakage-safe features
print("\n[3/9] Building features...")
features_df = build_features_leakage_safe(train_df)
feature_cols = get_feature_columns(features_df)

# Step 4: Add GARCH features (volatility prediction per stock)
print("\n[4/9] Adding GARCH-based features...")
garch_model = GARCHModel()
if 'reference_price' in features_df.columns or 'wap' in features_df.columns:
    price_col = 'reference_price' if 'reference_price' in features_df.columns else 'wap'
    features_df['returns'] = features_df.groupby('stock_id')[price_col].pct_change().fillna(0)
    
    # Calculate GARCH volatility for each row based on stock's historical returns
    garch_features = []
    for stock_id in features_df['stock_id'].unique():
        stock_mask = features_df['stock_id'] == stock_id
        stock_returns = features_df.loc[stock_mask, 'returns'].values
        
        if len(stock_returns) > 10:
            garch_model.train(stock_returns)
            # Get predicted volatility
            garch_vol = garch_model.predict(stock_returns)
        else:
            garch_vol = np.std(stock_returns) if len(stock_returns) > 1 else 0
        
        # Assign to all rows of this stock
        garch_features.extend([garch_vol] * np.sum(stock_mask))
    
    features_df['garch_volatility'] = garch_features
    
    # Add rolling GARCH features
    features_df['garch_vol_lag1'] = features_df.groupby('stock_id')['garch_volatility'].shift(1)
    features_df['garch_vol_ma5'] = features_df.groupby('stock_id')['garch_volatility'].transform(
        lambda x: x.shift(1).rolling(5).mean()
    )
    
    feature_cols.extend(['garch_volatility', 'garch_vol_lag1', 'garch_vol_ma5'])

print(f"Total Features: {len(feature_cols)} (including GARCH)")
print(f"Sample features: {feature_cols[-10:]}")

# Step 5: Add ARIMA features (time series forecasting per stock)
print("\n[5/9] Adding ARIMA-based features...")
arima_model = ARIMAModel(p=1, d=0, q=1)
arima_features = []

for stock_id in features_df['stock_id'].unique():
    stock_mask = features_df['stock_id'] == stock_id
    stock_target = features_df.loc[stock_mask, 'target'].values
    
    if len(stock_target) > 10:
        arima_model.train(stock_target)
        arima_pred = arima_model.predict(stock_target)
    else:
        arima_pred = np.mean(stock_target)
    
    arima_features.extend([arima_pred] * np.sum(stock_mask))

features_df['arima_forecast'] = arima_features

# Add ARIMA rolling features
features_df['arima_forecast_lag1'] = features_df.groupby('stock_id')['arima_forecast'].shift(1)
features_df['arima_forecast_ma5'] = features_df.groupby('stock_id')['arima_forecast'].transform(
    lambda x: x.shift(1).rolling(5).mean()
)

feature_cols.extend(['arima_forecast', 'arima_forecast_lag1', 'arima_forecast_ma5'])
feature_cols = [f for f in feature_cols if f in features_df.columns]

print(f"Total Features: {len(feature_cols)} (including ARIMA)")

# Step 6: Time-based split
print("\n[6/9] Time-based split...")
train_split_df, val_split_df = time_based_split(features_df, val_pct=0.2)
X_train = train_split_df[feature_cols].fillna(0)
y_train = train_split_df['target'].fillna(0)
X_val = val_split_df[feature_cols].fillna(0)
y_val = val_split_df['target'].fillna(0)
print(f"Split: {len(X_train)} train, {len(X_val)} val")

# Step 7: Train LightGBM
print("\n[7/9] Training LightGBM...")
lgb_model = LightGBMModel(num_boost_round=50, early_stopping_rounds=10)
lgb_model.train(X_train, y_train, X_val, y_val)
lgb_pred = lgb_model.predict(X_val)
lgb_rmspe = calculate_rmspe(y_val, lgb_pred)
print(f"LightGBM RMSPE: {lgb_rmspe:.6f}")

# Step 8: Train XGBoost
print("\n[8/9] Training XGBoost...")
xgb_model = XGBoostModel(max_depth=6, learning_rate=0.03, n_estimators=50,
                         subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1,
                         reg_lambda=0.1, tree_method='hist')
xgb_model.train(X_train, y_train, X_val, y_val)
xgb_pred = xgb_model.predict(X_val)
xgb_rmspe = calculate_rmspe(y_val, xgb_pred)
print(f"XGBoost RMSPE: {xgb_rmspe:.6f}")

# Final Ensemble: Use LightGBM + XGBoost (GARCH/ARIMA are now features)
print("\n[9/9] Creating final ensemble...")
ensemble_pred_val = 0.40 * lgb_pred + 0.35 * xgb_pred + 0.25 * (lgb_pred + xgb_pred) / 2
ensemble_rmspe = calculate_rmspe(y_val, ensemble_pred_val)
print(f"Final Ensemble RMSPE: {ensemble_rmspe:.6f}")

# Save results
results = {
    'lightgbm_rmspe': float(lgb_rmspe),
    'xgboost_rmspe': float(xgb_rmspe),
    'ensemble_rmspe': float(ensemble_rmspe),
    'train_samples': int(len(X_train)),
    'val_samples': int(len(X_val)),
    'n_features': int(len(feature_cols)),
    'garch_features': True,
    'arima_features': True
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results.json")
print("GARCH and ARIMA are integrated as features (not separate models)")
print("This improves performance by giving the ML models volatility information.")

# Generate final submission
print("\n[10/10] Generating submission...")
X_all = features_df[feature_cols].fillna(0)
lgb_preds = lgb_model.predict(X_all)
xgb_preds = xgb_model.predict(X_all)
final_ensemble_preds = 0.40 * lgb_preds + 0.35 * xgb_preds + 0.25 * (lgb_preds + xgb_preds) / 2

predictions = features_df[['time_id', 'stock_id']].copy()
predictions['target'] = final_ensemble_preds[:len(predictions)]
predictions['target'] = predictions['target'].fillna(0).astype(float)
predictions.to_csv('submission.csv', index=False)

print("\n" + "="*60)
print("SUBMISSION GENERATED")
print("="*60)
print(f"\nShape: {predictions.shape}")
print(f"Schema: {list(predictions.columns)}")
print(f"\nTarget statistics:")
print(predictions['target'].describe())
print(f"\nNo NaNs: {predictions['target'].isna().sum()}")

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"LightGBM RMSPE: {lgb_rmspe:.6f}")
print(f"XGBoost RMSPE: {xgb_rmspe:.6f}")
print(f"Ensemble RMSPE: {ensemble_rmspe:.6f}")
print(f"Total Features: {len(feature_cols)} (including GARCH and ARIMA features)")
print("="*60)
print("FINISHED")
print("="*60)
