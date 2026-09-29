"""Inference and multi-step prediction script for single sequences or test instances."""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_sliding_windows
from src.data.preprocessing import DataPreprocessor
from src.models.transformer import PositionalEncoding, TransformerEncoder
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("PredictScript")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict future sensor values using trained model.")
    default_model = (
        "artifacts/checkpoints/transformer_full_final.keras"
        if Path("artifacts/checkpoints/transformer_full_final.keras").exists()
        else "artifacts/checkpoints/transformer_quick_final.keras"
    )
    parser.add_argument("--model_path", type=str, default=default_model)
    parser.add_argument("--sample_index", type=int, default=0, help="Index of sample in test set to predict")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("configs/config.yaml")

    data_cfg = config["data"]
    feature_cols = data_cfg["feature_cols"]
    timestamp_col = data_cfg["timestamp_col"]

    df = pd.read_csv(data_cfg["processed_path"])

    preprocessor = DataPreprocessor(
        feature_cols=feature_cols,
        timestamp_col=timestamp_col,
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
    )
    preprocessor.load_scaler("artifacts/scalers/scaler.joblib")

    _, _, test_df = preprocessor.split_chronological(df)
    scaled_test = preprocessor.transform(test_df)

    input_window = config["quick_training"]["input_window"] if "quick" in args.model_path else data_cfg["input_window"]
    forecast_horizon = config["quick_training"]["forecast_horizon"] if "quick" in args.model_path else data_cfg["forecast_horizon"]

    X_test, y_test = create_sliding_windows(scaled_test, input_window, forecast_horizon)

    logger.info(f"Loading trained model from {args.model_path}...")
    custom_objects = {
        "PositionalEncoding": PositionalEncoding,
        "TransformerEncoder": TransformerEncoder,
    }
    model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects, safe_mode=False)

    sample_x = X_test[args.sample_index : args.sample_index + 1]
    sample_y = y_test[args.sample_index : args.sample_index + 1]

    pred_scaled = model.predict(sample_x)

    pred_phys = preprocessor.inverse_transform(pred_scaled)[0]
    true_phys = preprocessor.inverse_transform(sample_y)[0]

    logger.info(f"\n--- Multi-step Forecast for Sample {args.sample_index} ---")
    logger.info(f"Horizon steps: {forecast_horizon} | Features: {len(feature_cols)}")
    for step in range(forecast_horizon):
        logger.info(f"Step +{step+1}:")
        for f_idx, col in enumerate(feature_cols):
            logger.info(f"  {col}: True = {true_phys[step, f_idx]:.2f}, Pred = {pred_phys[step, f_idx]:.2f}")


if __name__ == "__main__":
    main()
