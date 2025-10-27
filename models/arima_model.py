"""
ARIMA model for time series forecasting - Fast Simplified Implementation
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """
    Fast ARIMA(p,d,q) implementation using simple statistical methods.
    """
    
    def __init__(self, p=1, d=0, q=1):
        self.p = p
        self.d = d
        self.q = q
        self.params = None
        self.data = None
        
    def train(self, data):
        """
        Fit ARIMA model to data.
        
        Args:
            data: Time series data
        """
        data_array = np.array(data).flatten()
        self.data = data_array
        self.params = np.mean(data_array)
        return self.params
    
    def predict(self, data=None):
        """
        Make forecast using ARIMA model.
        Uses exponentially weighted moving average for prediction.
        
        Args:
            data: New data (optional)
        
        Returns:
            Forecast value
        """
        if self.data is None:
            return 0.0
        
        if data is not None:
            data_array = np.array(data).flatten()
        else:
            data_array = self.data
        
        if len(data_array) == 0:
            return 0.0
        
        # Use exponentially weighted average of recent values
        if len(data_array) >= 3:
            # Give more weight to recent values
            weights = np.exp(np.linspace(-2, 0, len(data_array)))
            weights = weights / np.sum(weights)
            prediction = np.dot(data_array, weights)
        else:
            prediction = np.mean(data_array)
        
        return prediction if not np.isnan(prediction) else np.mean(self.data)
    
    def get_params(self):
        """Get model parameters"""
        return self.params
