"""Reusable evaluation metrics for multi-step multivariate time-series forecasting.

Functions:
    mean_absolute_error (MAE)
    root_mean_squared_error (RMSE)
    mean_absolute_percentage_error (MAPE, zero-safe)
    symmetric_mean_absolute_percentage_error (sMAPE)
    compute_all_metrics: returns dictionary of metrics.
"""

from typing import Dict
import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Compute Mean Absolute Percentage Error safely avoiding division by zero.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        epsilon: Small positive constant added to denominator.

    Returns:
        Percentage error as float (0 - 100%).
    """
    denominator = np.clip(np.abs(y_true), epsilon, None)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (sMAPE).

    Args:
        y_true: Ground truth array.
        y_pred: Predicted array.
        epsilon: Small constant to avoid zero division.

    Returns:
        sMAPE as percentage float.
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    return float(np.mean(numerator / denominator) * 100.0)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive regression metrics for multivariate predictions.

    Args:
        y_true: True array of shape (N, horizon, features) or (N, features).
        y_pred: Predicted array with identical shape.

    Returns:
        Dictionary containing MAE, RMSE, MAPE, sMAPE.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    return {
        "MAE": round(mean_absolute_error(y_t, y_p), 4),
        "RMSE": round(root_mean_squared_error(y_t, y_p), 4),
        "MAPE": round(mean_absolute_percentage_error(y_t, y_p), 4),
        "SMAPE": round(symmetric_mean_absolute_percentage_error(y_t, y_p), 4),
    }
