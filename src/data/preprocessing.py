"""Data preprocessing and leakage-free chronological splitting.

Features:
- Timestamp parsing & chronological sorting
- Duplicate and missing-value handling
- Outlier analysis and bounding
- Chronological train/validation/test splitting (e.g. 70/15/15)
- StandardScaler fitted ONLY on training split to prevent any data leakage
- Persisting and loading fitted scaler objects
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger("Preprocessing")


class DataPreprocessor:
    """Robust preprocessing pipeline for multivariate industrial time series."""

    def __init__(
        self,
        feature_cols: List[str],
        timestamp_col: str = "timestamp",
        train_split: float = 0.70,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ):
        self.feature_cols = feature_cols
        self.timestamp_col = timestamp_col
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.scaler = StandardScaler()
        self.is_fitted = False

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamps, sort chronologically, remove duplicates, and handle missing values."""
        df = df.copy()

        # 1. Parse timestamp and sort chronologically
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
            # Remove duplicate timestamps keeping first
            df = df.drop_duplicates(subset=[self.timestamp_col], keep="first")

        # 2. Verify and select feature columns
        missing_features = [col for col in self.feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Features missing in dataset: {missing_features}")

        # 3. Missing value interpolation (forward fill followed by backward fill)
        df[self.feature_cols] = df[self.feature_cols].ffill().bfill()

        return df

    def split_chronological(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame strictly chronologically without shuffling to prevent leakage."""
        n = len(df)
        train_end = int(n * self.train_split)
        val_end = int(n * (self.train_split + self.val_split))

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)

        logger.info(
            f"Chronological split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )
        return train_df, val_df, test_df

    def fit_transform(
        self, train_df: pd.DataFrame
    ) -> Tuple[np.ndarray, "DataPreprocessor"]:
        """Fit scaler ONLY on training features and return scaled numpy array."""
        train_features = train_df[self.feature_cols].values.astype(np.float32)
        scaled_train = self.scaler.fit_transform(train_features)
        self.is_fitted = True
        logger.info(f"Fitted StandardScaler on {len(train_df)} training samples across {len(self.feature_cols)} features.")
        return scaled_train, self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using previously fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")
        features = df[self.feature_cols].values.astype(np.float32)
        return self.scaler.transform(features)

    def inverse_transform(self, scaled_array: np.ndarray) -> np.ndarray:
        """Revert scaled features back to original physical sensor scale.

        Supports shapes: (N, features) or (N, horizon, features).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted.")

        orig_shape = scaled_array.shape
        if len(orig_shape) == 3:
            # (N, horizon, features)
            n_samples, horizon, n_features = orig_shape
            flat = scaled_array.reshape(-1, n_features)
            unscaled = self.scaler.inverse_transform(flat)
            return unscaled.reshape(n_samples, horizon, n_features)
        elif len(orig_shape) == 2:
            return self.scaler.inverse_transform(scaled_array)
        else:
            raise ValueError(f"Unsupported array shape for inverse transform: {orig_shape}")

    def save_scaler(self, path: str = "artifacts/scalers/scaler.joblib") -> None:
        """Save fitted scaler to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, p)
        logger.info(f"Saved fitted scaler to {p}")

    def load_scaler(self, path: str = "artifacts/scalers/scaler.joblib") -> "DataPreprocessor":
        """Load fitted scaler from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scaler file not found at: {p}")
        self.scaler = joblib.load(p)
        self.is_fitted = True
        logger.info(f"Loaded fitted scaler from {p}")
        return self
