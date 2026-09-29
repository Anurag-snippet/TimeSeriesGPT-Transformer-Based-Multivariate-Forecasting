"""Seed utility for reproducibility across Python, NumPy, and TensorFlow."""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python standard library, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
