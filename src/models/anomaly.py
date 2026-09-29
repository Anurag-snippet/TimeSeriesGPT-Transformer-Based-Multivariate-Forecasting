"""Anomaly Detection Module for Industrial Time Series.

Implements:
1. LSTM Autoencoder (TensorFlow / Keras):
   Sequence reconstruction with encoder-decoder bottleneck.
   Reconstruction error serves as continuous anomaly score.
2. Isolation Forest (Classical Machine Learning Baseline):
   Tree-based ensemble isolation scoring on statistical sequence representations.
3. Rolling Z-Score Baseline:
   Statistical moving window thresholding for abrupt spikes / dropouts.
4. Evaluation & Synthetic Anomaly injection for unsupervised verification.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras import layers, Model

from src.utils.logger import setup_logger

logger = setup_logger("AnomalyDetector")


def build_lstm_autoencoder(
    sequence_length: int = 60,
    num_features: int = 7,
    latent_dim: int = 16,
    encoder_units: int = 32,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
) -> Model:
    """Build an LSTM Autoencoder for time-series reconstruction."""
    # Encoder
    inputs = layers.Input(shape=(sequence_length, num_features), name="autoencoder_input")
    x = layers.LSTM(encoder_units, return_sequences=True, name="enc_lstm_1")(inputs)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    latent = layers.LSTM(latent_dim, return_sequences=False, name="latent_representation")(x)

    # Decoder
    x = layers.RepeatVector(sequence_length, name="repeat_vector")(latent)
    x = layers.LSTM(encoder_units, return_sequences=True, name="dec_lstm_1")(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    reconstructed = layers.TimeDistributed(layers.Dense(num_features), name="reconstruction_output")(x)

    autoencoder = Model(inputs=inputs, outputs=reconstructed, name="LSTM_Autoencoder")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return autoencoder


class LSTMAutoencoderDetector:
    """Unsupervised anomaly detector based on LSTM reconstruction error."""

    def __init__(
        self,
        sequence_length: int = 60,
        num_features: int = 7,
        latent_dim: int = 16,
        encoder_units: int = 32,
        threshold_percentile: float = 97.5,
    ):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.threshold_percentile = threshold_percentile
        self.threshold: Optional[float] = None
        self.model = build_lstm_autoencoder(
            sequence_length=sequence_length,
            num_features=num_features,
            latent_dim=latent_dim,
            encoder_units=encoder_units,
        )

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "LSTMAutoencoderDetector":
        """Train autoencoder on normal operational sequences to establish baseline reconstruction."""
        val_tuple = (validation_data[0], validation_data[0]) if validation_data is not None else None
        self.model.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_tuple,
            verbose=0,
        )

        # Compute reconstruction error on training data to establish baseline threshold
        train_recon = self.model.predict(X_train, verbose=0)
        train_errors = np.mean(np.square(X_train - train_recon), axis=(1, 2))
        self.threshold = float(np.percentile(train_errors, self.threshold_percentile))
        logger.info(f"Fitted LSTM Autoencoder. Threshold at {self.threshold_percentile}th percentile = {self.threshold:.5f}")
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute Mean Squared Reconstruction Error per sequence."""
        reconstructed = self.model.predict(X, verbose=0)
        mse_per_sample = np.mean(np.square(X - reconstructed), axis=(1, 2))
        return mse_per_sample

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary anomaly classification: 1 for anomaly, 0 for normal."""
        scores = self.score_samples(X)
        if self.threshold is None:
            raise RuntimeError("Model must be fitted before predicting anomalies.")
        return (scores > self.threshold).astype(int)


class IsolationForestDetector:
    """Classical baseline anomaly detector using Isolation Forest on sequence summaries."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Summarize sequences with mean, std, min, max per sensor feature."""
        N, win, feats = X.shape
        mean = np.mean(X, axis=1)
        std = np.std(X, axis=1)
        min_v = np.min(X, axis=1)
        max_v = np.max(X, axis=1)
        return np.hstack([mean, std, min_v, max_v])

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        feats = self._extract_features(X)
        self.model.fit(feats)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Inverted anomaly score where higher values indicate greater anomaly."""
        feats = self._extract_features(X)
        return -self.model.score_samples(feats)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self._extract_features(X)
        preds = self.model.predict(feats)  # -1 for anomaly, 1 for normal
        return (preds == -1).astype(int)


def inject_synthetic_anomalies(
    X: np.ndarray,
    anomaly_rate: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject controlled synthetic anomalies (spikes, thermal runaways, sensor freeze) for validation."""
    rng = np.random.default_rng(seed)
    X_corrupted = X.copy()
    n_samples, win_len, n_feats = X.shape
    labels = np.zeros(n_samples, dtype=int)

    num_anomalies = max(1, int(n_samples * anomaly_rate))
    anomaly_indices = rng.choice(n_samples, size=num_anomalies, replace=False)

    for idx in anomaly_indices:
        labels[idx] = 1
        anomaly_type = rng.choice(["spike", "drift", "freeze"])
        target_feat = rng.integers(0, n_feats)

        if anomaly_type == "spike":
            t_loc = rng.integers(win_len // 2, win_len)
            X_corrupted[idx, t_loc, target_feat] += rng.choice([-1.0, 1.0]) * rng.uniform(3.5, 6.0)
        elif anomaly_type == "drift":
            drift = np.linspace(0, rng.uniform(2.5, 5.0), win_len)
            X_corrupted[idx, :, target_feat] += drift
        elif anomaly_type == "freeze":
            X_corrupted[idx, win_len // 2:, target_feat] = X_corrupted[idx, win_len // 2, target_feat]

    return X_corrupted, labels
