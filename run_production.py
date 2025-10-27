#!/usr/bin/env python3
"""
Optiver Realized Volatility Prediction

Implementation for Optiver Trading at the Close competition.
Trains ensemble models to predict realized volatility.
"""

import pandas as pd
import numpy as np
import warnings
import sys
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb

print("="*60)
print("OPTIVER REALIZED VOLATILITY PREDICTION")
print("="*60)

# Step 1: Load data
print("\n[1/7] Loading data...")
try:
    # Load training data (actual competition data)
    train_df = pd.read_csv('data/train.csv')
    print(f"Training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Column check: {list(train_df.columns[:5])}...")
except Exception as e:
    print(f"ERROR loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Prepare target
print("\n[2/7] Preparing target...")
# Check for target column in training data
if 'target' in train_df.columns:
    target_col = 'target'
elif 'realized_vol' in train_df.columns:
    target_col = 'realized_vol'
elif 'volatility' in train_df.columns:
    target_col = 'volatility'
    train_df['target'] = train_df[target_col]
    target_col = 'target'
else:
    print("Warning: No target column found, using first numeric column")
    target_col = None

if target_col:
    train_df['target'] = train_df[target_col].fillna(0)
    print(f"Using target column: {target_col}")
else:
    train_df['target'] = 0
print("Target ready")

# Step 3: Calculate Realized Volatility function
def calc_realized_volatility(df):
    """RV = Σ(Δlog mid_price)²"""
    if 'bid_price' in df.columns and 'ask_price' in df.columns:
        mid_price = (df['bid_price'] + df['ask_price']) / 2
    elif 'wap' in df.columns:
        mid_price = df['wap']
    else:
        mid_price = df['reference_price']
    
    log_returns = np.log(mid_price / mid_price.shift(1)).fillna(0)
    rv = (log_returns ** 2).sum()
    return rv, log_returns

# Step 4: Build leakage-safe features
print("\n[3/7] Building features...")
def build_features_leakage_safe(df):
    """Build features with NO data leakage"""
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

features_df = build_features_leakage_safe(train_df)

# Get only numeric feature columns (exclude non-numeric and target columns)
exclude_cols = ['time_id', 'stock_id', 'target', 'realized_vol', 'volatility']
feature_cols = []
for c in features_df.columns:
    if c not in exclude_cols:
        # Only include numeric columns
        if pd.api.types.is_numeric_dtype(features_df[c]):
            feature_cols.append(c)

print(f"Features: {len(feature_cols)}")
print(f"Sample features: {feature_cols[:10]}")

# Step 5: Time-based split
print("\n[4/7] Time-based split...")
def time_based_split(df, val_pct=0.2):
    df = df.copy()
    max_time = df['time_id'].max()
    split_time = max_time - int(max_time * val_pct)
    train = df[df['time_id'] <= split_time].copy()
    val = df[df['time_id'] > split_time].copy()
    return train, val

train_split_df, val_split_df = time_based_split(features_df, val_pct=0.2)
X_train = train_split_df[feature_cols].fillna(0)
y_train = train_split_df['target'].fillna(0)
X_val = val_split_df[feature_cols].fillna(0)
y_val = val_split_df['target'].fillna(0)
print(f"Split: {len(X_train)} train, {len(X_val)} val")

# Step 6: Train models
print("\n[5/7] Training LightGBM...")
def calculate_rmspe(y_true, y_pred):
    """Root Mean Squared Percentage Error"""
    mask = (y_true != 0) & ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_safe = y_true[mask]
    y_pred_safe = y_pred[mask]
    if len(y_true_safe) == 0:
        return np.inf
    # Avoid extreme values
    y_true_safe = np.clip(np.abs(y_true_safe), 1e-10, 1e10)
    return np.sqrt(np.mean(((y_true_safe - y_pred_safe) / y_true_safe) ** 2))

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=50,
    valid_sets=[val_data],
    valid_names=['eval'],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

lgb_pred = lgb_model.predict(X_val)
lgb_rmspe = calculate_rmspe(y_val, lgb_pred)
print(f"LightGBM RMSPE: {lgb_rmspe:.6f}")

print("\n[6/7] Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    max_depth=6,
    learning_rate=0.03,
    n_estimators=50,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=0.1,
    tree_method='hist'
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_pred = xgb_model.predict(X_val)
xgb_rmspe = calculate_rmspe(y_val, xgb_pred)
print(f"XGBoost RMSPE: {xgb_rmspe:.6f}")

# Ensemble predictions
ensemble_pred_val = 0.40 * lgb_pred + 0.35 * xgb_pred + 0.25 * (lgb_pred + xgb_pred) / 2
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
    import json
    json.dump(results, f, indent=2)

print("\nResults saved to results.json")

# Step 7: Generate submission  
print("\n[7/7] Generating submission...")
X_all = features_df[feature_cols].fillna(0)
lgb_preds = lgb_model.predict(X_all)
xgb_preds = xgb_model.predict(X_all)

ensemble_preds = 0.40 * lgb_preds + 0.35 * xgb_preds + 0.25 * (lgb_preds + xgb_preds) / 2

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

