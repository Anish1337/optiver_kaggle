"""
XGBoost model for realized volatility prediction
"""

import xgboost as xgb


class XGBoostModel:
    """XGBoost model wrapper"""
    
    def __init__(self, max_depth=6, learning_rate=0.03, n_estimators=50,
                 subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1,
                 reg_lambda=0.1, tree_method='hist', **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            n_estimators: Number of estimators
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            tree_method: Tree construction method
            **kwargs: Additional XGBoost parameters
        """
        self.params = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'tree_method': tree_method,
            **kwargs
        }
        self.model = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained model
        """
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
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

