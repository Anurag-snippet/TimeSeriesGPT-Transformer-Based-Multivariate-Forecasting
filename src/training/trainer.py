"""Unified Trainer interface for training and evaluating time series models."""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from src.training.callbacks import get_callbacks
from src.utils.logger import setup_logger

logger = setup_logger("Trainer")


class ModelTrainer:
    """Orchestrates model compilation, fitting, checkpointing, and execution timing."""

    def __init__(
        self,
        model: Model,
        model_name: str = "model",
        checkpoint_dir: str = "artifacts/checkpoints",
    ):
        self.model = model
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = str(self.checkpoint_dir / f"{model_name}_best.weights.h5")
        self.training_time_sec: float = 0.0
        self.history: Optional[tf.keras.callbacks.History] = None

    def train(
        self,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        epochs: int = 25,
        early_stopping_patience: int = 7,
        reduce_lr_patience: int = 3,
        reduce_lr_factor: float = 0.5,
        min_lr: float = 1e-5,
    ) -> Dict[str, Any]:
        """Fit model with callbacks and log precise timing."""
        logger.info(f"Initiating training for {self.model_name} (Max Epochs: {epochs})...")

        callbacks = get_callbacks(
            checkpoint_filepath=self.checkpoint_path,
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_factor=reduce_lr_factor,
            min_lr=min_lr,
        )

        start_time = time.perf_counter()
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        self.training_time_sec = round(time.perf_counter() - start_time, 2)
        logger.info(f"Training completed in {self.training_time_sec:.2f} seconds.")

        # Save complete model architecture & weights
        full_model_path = str(self.checkpoint_dir / f"{self.model_name}_final.keras")
        self.model.save(full_model_path)
        logger.info(f"Saved complete Keras model to {full_model_path}")

        return {
            "training_time": self.training_time_sec,
            "epochs_completed": len(self.history.epoch),
            "final_loss": float(self.history.history["loss"][-1]),
            "final_val_loss": float(self.history.history["val_loss"][-1]),
        }
