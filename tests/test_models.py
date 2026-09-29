"""Unit tests verifying deep-learning model architectures and output tensor dimensions."""

import numpy as np
import pytest

from src.models.anomaly import IsolationForestDetector, build_lstm_autoencoder
from src.models.gru import build_gru_forecaster
from src.models.lstm import build_lstm_forecaster
from src.models.transformer import build_transformer_forecaster


def test_lstm_output_shape():
    batch = 8
    input_window = 30
    horizon = 5
    num_features = 4

    model = build_lstm_forecaster(
        input_window=input_window,
        forecast_horizon=horizon,
        num_features=num_features,
        units=[16, 8],
    )
    dummy_input = np.random.randn(batch, input_window, num_features).astype(np.float32)
    pred = model.predict(dummy_input, verbose=0)
    assert pred.shape == (batch, horizon, num_features)


def test_gru_output_shape():
    batch = 8
    input_window = 30
    horizon = 5
    num_features = 4

    model = build_gru_forecaster(
        input_window=input_window,
        forecast_horizon=horizon,
        num_features=num_features,
        units=[16, 8],
    )
    dummy_input = np.random.randn(batch, input_window, num_features).astype(np.float32)
    pred = model.predict(dummy_input, verbose=0)
    assert pred.shape == (batch, horizon, num_features)


def test_transformer_output_shape():
    batch = 8
    input_window = 30
    horizon = 5
    num_features = 4

    model = build_transformer_forecaster(
        input_window=input_window,
        forecast_horizon=horizon,
        num_features=num_features,
        d_model=16,
        num_heads=2,
        ff_dim=32,
        num_layers=1,
    )
    dummy_input = np.random.randn(batch, input_window, num_features).astype(np.float32)
    pred = model.predict(dummy_input, verbose=0)
    assert pred.shape == (batch, horizon, num_features)


def test_autoencoder_reconstruction_shape():
    batch = 8
    seq_len = 20
    num_features = 4

    ae = build_lstm_autoencoder(sequence_length=seq_len, num_features=num_features, latent_dim=8)
    dummy_input = np.random.randn(batch, seq_len, num_features).astype(np.float32)
    recon = ae.predict(dummy_input, verbose=0)
    assert recon.shape == (batch, seq_len, num_features)
