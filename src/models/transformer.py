"""Transformer Architecture for Multivariate Time-Series Forecasting.

Components implemented from scratch using Keras layers & functional API:
1. PositionalEncoding: Deterministic sinusoidal encoding injecting sequential order
2. TransformerEncoder:
     x -> LayerNorm -> MultiHeadAttention -> Dropout -> Residual Add
       -> LayerNorm -> FeedForward (Dense -> ReLU -> Dense) -> Dropout -> Residual Add
3. MultivariateTransformerForecaster:
     Input -> Feature Projection to d_model -> Positional Encoding
           -> N x TransformerEncoder blocks -> Global Pooling / Flatten
           -> Multi-step forecasting projection head -> Reshape to (horizon, features)
"""

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for sequential temporal data.

    Since standard self-attention operations are permutation-equivariant,
    positional encodings inject order information into token representations.
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, max_len: int = 500, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Precompute encoding table
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        # Register non-trainable constant tensor
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class TransformerEncoder(layers.Layer):
    """Custom Transformer Encoder Block implemented with LayerNorm and MultiHeadAttention."""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        # Pre-LN architecture (Pre-Layer Normalization for stable training)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.dropout1 = layers.Dropout(dropout)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Pre-LN Self-Attention sublayer
        norm_x = self.norm1(x)
        attn_out = self.mha(query=norm_x, value=norm_x, key=norm_x, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        x = x + attn_out

        # Pre-LN Feed-Forward sublayer
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        x = x + ffn_out
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


def build_transformer_forecaster(
    input_window: int = 60,
    forecast_horizon: int = 10,
    num_features: int = 7,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    loss: str = "mse",
) -> Model:
    """Build and compile the multivariate Transformer forecasting model.

    Args:
        input_window: Length of historical context (N).
        forecast_horizon: Number of future timesteps to predict (H).
        num_features: Sensor dimensions (e.g. 7).
        d_model: Hidden embedding projection dimension.
        num_heads: Number of attention heads.
        ff_dim: Dimension of feed-forward expansion.
        num_layers: Number of stacked TransformerEncoder layers.
        dropout: Regularization dropout probability.
        learning_rate: Adam optimizer learning rate.
        loss: Loss function ('mse').

    Returns:
        Compiled Keras Model producing shape (batch, forecast_horizon, num_features).
    """
    inputs = layers.Input(shape=(input_window, num_features), name="sensor_sequence_input")

    # 1. Feature Projection: maps raw physical features to d_model latent dimension
    x = layers.Dense(d_model, name="feature_projection")(inputs)

    # 2. Positional Encoding
    pos_encoding = PositionalEncoding(max_len=max(500, input_window * 2), d_model=d_model, name="sinusoidal_pos_encoding")
    x = pos_encoding(x)

    # 3. Stacked Transformer Encoder Blocks
    for i in range(num_layers):
        x = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            name=f"transformer_encoder_{i+1}",
        )(x)

    # 4. Temporal Representation: Global Average Pooling + Sequence flattening context
    gap = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    last_step = layers.Lambda(lambda t: t[:, -1, :], name="last_step_slice")(x)
    context = layers.Concatenate(name="concat_context")([gap, last_step])

    # 5. Multi-Step Projection Head
    dense_rep = layers.Dense(ff_dim, activation="relu", name="representation_dense")(context)
    dense_rep = layers.Dropout(dropout, name="head_dropout")(dense_rep)

    total_outputs = forecast_horizon * num_features
    proj = layers.Dense(total_outputs, name="forecast_projection")(dense_rep)
    outputs = layers.Reshape((forecast_horizon, num_features), name="forecast_reshape")(proj)

    model = Model(inputs=inputs, outputs=outputs, name="Multivariate_Transformer_Forecaster")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])

    return model
