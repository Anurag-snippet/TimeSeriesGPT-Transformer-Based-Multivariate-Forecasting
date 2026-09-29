"""Unit and pipeline tests for dataset generation, sliding windows, and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import create_sliding_windows
from src.data.loader import generate_industrial_telemetry
from src.data.preprocessing import DataPreprocessor


def test_data_generation_shape_and_features():
    df = generate_industrial_telemetry(num_timesteps=100, random_seed=42)
    assert len(df) == 100
    expected_cols = [
        "timestamp",
        "T24_combustor_inlet_temp",
        "T30_hpt_coolant_temp",
        "T50_lpt_outlet_temp",
        "P30_hpc_outlet_pressure",
        "Nf_fan_speed_rpm",
        "Nc_core_speed_rpm",
        "vib_vibration_amplitude",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_chronological_split_no_leakage():
    df = generate_industrial_telemetry(num_timesteps=1000, random_seed=42)
    feature_cols = ["T24_combustor_inlet_temp", "P30_hpc_outlet_pressure"]
    preprocessor = DataPreprocessor(
        feature_cols=feature_cols,
        train_split=0.70,
        val_split=0.15,
        test_split=0.15,
    )

    clean_df = preprocessor.clean_dataframe(df)
    train_df, val_df, test_df = preprocessor.split_chronological(clean_df)

    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150

    # Ensure chronological continuity (no shuffling leakage)
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()


def test_sliding_window_dimensions():
    data = np.random.randn(100, 5).astype(np.float32)
    input_window = 20
    forecast_horizon = 5

    X, y = create_sliding_windows(data, input_window=input_window, forecast_horizon=forecast_horizon)

    expected_windows = 100 - (input_window + forecast_horizon) + 1
    assert X.shape == (expected_windows, input_window, 5)
    assert y.shape == (expected_windows, forecast_horizon, 5)
