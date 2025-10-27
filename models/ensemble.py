"""
Ensemble model combining multiple models
"""

import numpy as np


class EnsembleModel:
    """Ensemble of multiple models with weighted averaging"""
    
    def __init__(self, models, weights=None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of model objects
            weights: List of weights for each model (default: equal weights)
        """
        self.models = models
        if weights is None:
            # Default weights: 40% first model, 35% second model, 25% average
            weights = [0.40, 0.35, 0.25]
        self.weights = weights
    
    def predict(self, X):
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted combination
        if len(self.models) == 2:
            # Special case for two models with default weights
            lgb_pred = predictions[0]
            xgb_pred = predictions[1]
            ensemble_pred = (0.40 * lgb_pred + 
                           0.35 * xgb_pred + 
                           0.25 * (lgb_pred + xgb_pred) / 2)
            return ensemble_pred
        else:
            # General case: weighted average
            ensemble_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                ensemble_pred += weight * pred
            return ensemble_pred

