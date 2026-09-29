"""Sliding-window sequence dataset generator and tf.data.Dataset pipeline.

Implements:
- Chronological multivariate sliding window construction:
    X: shape (N, input_window, num_features)
    y: shape (N, forecast_horizon, num_features)
- High-performance tf.data.Dataset creation with .batch(), .prefetch(tf.data.AUTOTUNE)
- Deterministic test/validation pipelines and optional shuffle buffer on training pipeline only.
"""

from typing import Tuple
import numpy as np
import tensorflow as tf

from src.utils.logger import setup_logger

logger = setup_logger("DatasetPipeline")


def create_sliding_windows(
    data: np.ndarray,
    input_window: int = 60,
    forecast_horizon: int = 10,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multi-step multivariate sequences using a sliding window.

    Args:
        data: 2D array of shape (timesteps, num_features).
        input_window: Number of historical time steps (N).
        forecast_horizon: Number of future time steps to predict (H).
        stride: Step size between consecutive windows.

    Returns:
        X: shape (num_windows, input_window, num_features)
        y: shape (num_windows, forecast_horizon, num_features)
    """
    total_len = len(data)
    window_total = input_window + forecast_horizon

    if total_len < window_total:
        raise ValueError(
            f"Data length ({total_len}) is smaller than input_window + forecast_horizon ({window_total})"
        )

    X_list = []
    y_list = []

    for start_idx in range(0, total_len - window_total + 1, stride):
        input_end = start_idx + input_window
        target_end = input_end + forecast_horizon

        X_list.append(data[start_idx:input_end])
        y_list.append(data[input_end:target_end])

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    return X, y


def build_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build an efficient tf.data pipeline with batching and prefetching.

    Args:
        X: Historical sequences array (N, input_window, features).
        y: Target forecast array (N, forecast_horizon, features).
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle windows (ONLY recommended for training, never test).
        shuffle_buffer: Size of shuffle buffer.
        seed: Random seed for deterministic shuffling.

    Returns:
        tf.data.Dataset prefetching batches to AUTOTUNE.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), shuffle_buffer), seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
