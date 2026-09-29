"""Keras Callbacks factory for robust, automated deep-learning training."""

import os
from pathlib import Path
from typing import List, Optional
import tensorflow as tf


def get_callbacks(
    checkpoint_filepath: Optional[str] = None,
    monitor: str = "val_loss",
    early_stopping_patience: int = 7,
    reduce_lr_patience: int = 3,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-5,
) -> List[tf.keras.callbacks.Callback]:
    """Instantiate standard production callbacks.

    - EarlyStopping: prevents overfitting and restores best model weights
    - ReduceLROnPlateau: decreases learning rate when loss plateaus
    - ModelCheckpoint: persists best model state
    """
    callbacks = []

    # Early stopping
    es = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(es)

    # Learning rate reduction
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1,
    )
    callbacks.append(lr_reducer)

    # Model checkpointing
    if checkpoint_filepath is not None:
        p = Path(checkpoint_filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(p),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        )
        callbacks.append(ckpt)

    return callbacks
