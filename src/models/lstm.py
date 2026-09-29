"""LSTM multi-step multivariate forecasting architecture in TensorFlow / Keras.

Architecture:
Input (input_window, num_features)
    ↓
LSTM Layer 1 (return_sequences=True)
    ↓
Dropout
    ↓
LSTM Layer 2 (return_sequences=False)
    ↓
Dropout
    ↓
Dense representation
    ↓
Projection Head (Dense -> forecast_horizon * num_features)
    ↓
Reshape (forecast_horizon, num_features)
"""

from typing import List, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_lstm_forecaster(
    input_window: int = 60,
    forecast_horizon: int = 10,
    num_features: int = 7,
    units: Optional[List[int]] = None,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    loss: str = "mse",
) -> Model:
    """Build and compile a multi-layer stacked LSTM network for multi-step forecasting."""
    if units is None:
        units = [64, 32]

    inputs = layers.Input(shape=(input_window, num_features), name="sensor_sequence_input")
    x = inputs

    # Stacked recurrent layers
    for i, u in enumerate(units):
        return_sequences = (i < len(units) - 1)
        x = layers.LSTM(
            units=u,
            return_sequences=return_sequences,
            name=f"lstm_layer_{i+1}",
        )(x)
        if dropout > 0.0:
            x = layers.Dropout(dropout, name=f"lstm_dropout_{i+1}")(x)

    # Dense bottleneck representation
    x = layers.Dense(units[-1], activation="relu", name="latent_dense")(x)

    # Forecasting output projection
    total_output_dims = forecast_horizon * num_features
    outputs = layers.Dense(total_output_dims, name="forecast_projection")(x)
    outputs = layers.Reshape((forecast_horizon, num_features), name="forecast_reshape")(outputs)

    model = Model(inputs=inputs, outputs=outputs, name="Multivariate_LSTM_Forecaster")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])

    return model
