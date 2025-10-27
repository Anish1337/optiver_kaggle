"""
Metrics for model evaluation
"""

import numpy as np


def calculate_rmspe(y_true, y_pred):
    """
    Root Mean Squared Percentage Error
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        RMSPE score
    """
    mask = (y_true != 0) & ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_safe = y_true[mask]
    y_pred_safe = y_pred[mask]
    if len(y_true_safe) == 0:
        return np.inf
    # Avoid extreme values
    y_true_safe = np.clip(np.abs(y_true_safe), 1e-10, 1e10)
    return np.sqrt(np.mean(((y_true_safe - y_pred_safe) / y_true_safe) ** 2))

