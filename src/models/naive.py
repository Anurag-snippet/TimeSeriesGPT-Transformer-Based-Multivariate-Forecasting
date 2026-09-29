"""Classical baseline forecasting models.

1. NaiveLastForecaster: Repeats the last observed timestep across all forecast horizons.
2. MovingAverageForecaster: Predicts future horizon using the moving average of the input window.
3. LinearRegressionForecaster: Direct multi-output linear baseline.
"""

from typing import Optional
import numpy as np
from sklearn.linear_model import Ridge


class NaiveLastForecaster:
    """Predicts future horizons by holding the last historical value constant."""

    def __init__(self, forecast_horizon: int = 10):
        self.forecast_horizon = forecast_horizon

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveLastForecaster":
        """No parameters to train for persistence model."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Repeat last timestep X[:, -1, :] across the forecast horizon.

        Args:
            X: shape (N, input_window, features)

        Returns:
            y_pred: shape (N, forecast_horizon, features)
        """
        last_step = X[:, -1:, :]  # shape: (N, 1, features)
        return np.repeat(last_step, self.forecast_horizon, axis=1)


class MovingAverageForecaster:
    """Predicts future horizons using the mean of the historical window."""

    def __init__(self, forecast_horizon: int = 10, window_subset: Optional[int] = None):
        self.forecast_horizon = forecast_horizon
        self.window_subset = window_subset

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MovingAverageForecaster":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Compute average across historical window and tile across horizon."""
        if self.window_subset is not None:
            sub_window = X[:, -self.window_subset:, :]
        else:
            sub_window = X

        mean_val = np.mean(sub_window, axis=1, keepdims=True)  # (N, 1, features)
        return np.repeat(mean_val, self.forecast_horizon, axis=1)


class RidgeBaselineForecaster:
    """Direct multi-output linear Ridge baseline."""

    def __init__(self, forecast_horizon: int = 10, alpha: float = 1.0):
        self.forecast_horizon = forecast_horizon
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)
        self.num_features = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaselineForecaster":
        N, win, feats = X.shape
        self.num_features = feats
        X_flat = X.reshape(N, win * feats)
        y_flat = y.reshape(N, self.forecast_horizon * feats)
        self.model.fit(X_flat, y_flat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N, win, feats = X.shape
        X_flat = X.reshape(N, win * feats)
        y_flat_pred = self.model.predict(X_flat)
        return y_flat_pred.reshape(N, self.forecast_horizon, feats)
