"""
LightGBM model for realized volatility prediction
"""

import lightgbm as lgb
import numpy as np


class LightGBMModel:
    """LightGBM model wrapper"""
    
    def __init__(self, num_boost_round=50, early_stopping_rounds=10, **kwargs):
        """
        Initialize LightGBM model.
        
        Args:
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping patience
            **kwargs: Additional LightGBM parameters
        """
        default_params = {
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
        self.params = {**default_params, **kwargs}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained model
        """
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[val_data],
            valid_names=['eval'],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds), lgb.log_evaluation(0)]
        )
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def get_model(self):
        """Get the underlying model"""
        return self.model

