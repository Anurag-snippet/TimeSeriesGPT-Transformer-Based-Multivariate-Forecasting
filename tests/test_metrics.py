"""Unit tests for metric computations."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)


def test_metrics_perfect_prediction():
    y_true = np.array([[10.0, 20.0], [30.0, 40.0]])
    y_pred = np.array([[10.0, 20.0], [30.0, 40.0]])

    metrics = compute_all_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE"] == 0.0
    assert metrics["SMAPE"] == 0.0


def test_metrics_nonzero_error():
    y_true = np.array([[10.0, 20.0], [30.0, 40.0]])
    y_pred = np.array([[12.0, 18.0], [33.0, 36.0]])

    assert mean_absolute_error(y_true, y_pred) > 0.0
    assert root_mean_squared_error(y_true, y_pred) > 0.0
    assert mean_absolute_percentage_error(y_true, y_pred) > 0.0
    assert symmetric_mean_absolute_percentage_error(y_true, y_pred) > 0.0


def test_mape_zero_handling():
    # denominator has zero, must not raise ZeroDivisionError
    y_true = np.array([[0.0, 10.0]])
    y_pred = np.array([[1.0, 10.0]])
    mape = mean_absolute_percentage_error(y_true, y_pred)
    assert np.isfinite(mape)
