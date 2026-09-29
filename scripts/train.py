"""Training script for multivariate time series forecasting models.

Supports:
- --model [naive, ridge, lstm, gru, transformer]
- --quick (fast validation on CPU / local laptops)
- Dynamic device detection (reports CPU vs GPU execution)
- Checkpoint persistence and experimental metric recording
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure workspace root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_tf_dataset, create_sliding_windows
from src.data.preprocessing import DataPreprocessor
from src.evaluation.evaluator import ModelEvaluator
from src.models.gru import build_gru_forecaster
from src.models.lstm import build_lstm_forecaster
from src.models.naive import MovingAverageForecaster, NaiveLastForecaster, RidgeBaselineForecaster
from src.models.transformer import build_transformer_forecaster
from src.training.trainer import ModelTrainer
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("TrainScript")


def parse_args():
    parser = argparse.ArgumentParser(description="Train time series forecasting models.")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["naive", "moving_average", "ridge", "lstm", "gru", "transformer"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick execution mode for testing on laptop CPU",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("configs/config.yaml")
    set_seed(config.get("seed", 42))

    # Hardware detection
    gpus = tf.config.list_physical_devices("GPU")
    device_name = f"GPU ({len(gpus)} device(s))" if gpus else "CPU (Standard compute)"
    logger.info(f"Target Execution Device: {device_name}")

    data_cfg = config["data"]
    feature_cols = data_cfg["feature_cols"]
    timestamp_col = data_cfg["timestamp_col"]

    # Load clean data
    data_path = Path(data_cfg["processed_path"])
    if not data_path.exists():
        logger.warning("Processed data not found. Running prepare_data first...")
        from scripts.prepare_data import main as prep_main
        prep_main()

    df = pd.read_csv(data_path)

    # Preprocessor & chronological split
    preprocessor = DataPreprocessor(
        feature_cols=feature_cols,
        timestamp_col=timestamp_col,
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
    )
    preprocessor.load_scaler("artifacts/scalers/scaler.joblib")

    train_df, val_df, test_df = preprocessor.split_chronological(df)

    if args.quick:
        logger.info(">>> QUICK MODE ACTIVATED (Optimized for CPU & Rapid Validation) <<<")
        quick_cfg = config.get("quick_training", {})
        max_samples = quick_cfg.get("subset_samples", 2500)
        train_df = train_df.iloc[:max_samples]
        val_df = val_df.iloc[:max_samples // 3]
        test_df = test_df.iloc[:max_samples // 3]
        input_window = quick_cfg.get("input_window", 30)
        forecast_horizon = quick_cfg.get("forecast_horizon", 5)
        epochs = args.epochs or quick_cfg.get("epochs", 3)
        batch_size = args.batch_size or quick_cfg.get("batch_size", 16)
    else:
        input_window = data_cfg["input_window"]
        forecast_horizon = data_cfg["forecast_horizon"]
        epochs = args.epochs or config["training"]["epochs"]
        batch_size = args.batch_size or config["training"]["batch_size"]

    # Transform splits
    scaled_train = preprocessor.transform(train_df)
    scaled_val = preprocessor.transform(val_df)
    scaled_test = preprocessor.transform(test_df)

    # Generate sliding windows
    X_train, y_train = create_sliding_windows(scaled_train, input_window, forecast_horizon)
    X_val, y_val = create_sliding_windows(scaled_val, input_window, forecast_horizon)
    X_test, y_test = create_sliding_windows(scaled_test, input_window, forecast_horizon)

    logger.info(f"Sliding Windows Generated:")
    logger.info(f"  Train: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    logger.info(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    evaluator = ModelEvaluator(config["artifacts"]["results_csv"])
    model_name = args.model.lower()
    num_features = len(feature_cols)

    # Handle Baselines (No gradient descent needed)
    if model_name in ["naive", "moving_average", "ridge"]:
        if model_name == "naive":
            model = NaiveLastForecaster(forecast_horizon=forecast_horizon)
        elif model_name == "moving_average":
            model = MovingAverageForecaster(forecast_horizon=forecast_horizon)
        else:
            model = RidgeBaselineForecaster(forecast_horizon=forecast_horizon)

        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)

        # Invert scaling to physical units for honest, realistic metric calculation
        y_test_phys = preprocessor.inverse_transform(y_test)
        y_pred_phys = preprocessor.inverse_transform(y_test_pred)

        evaluator.evaluate(
            y_true=y_test_phys,
            y_pred=y_pred_phys,
            model_name=model_name.upper(),
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            num_features=num_features,
            training_time=0.05,
            parameter_count=0 if model_name != "ridge" else (input_window * num_features * forecast_horizon * num_features),
            batch_size=batch_size,
            epochs=0,
            learning_rate=0.0,
        )
        return

    # Deep Learning Pipelines (tf.data)
    train_ds = build_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = build_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_ds = build_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    learning_rate = config["training"]["learning_rate"]
    train_cfg = config["training"]

    if model_name == "lstm":
        m_cfg = config["models"]["lstm"]
        model = build_lstm_forecaster(
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            num_features=num_features,
            units=m_cfg["units"] if not args.quick else [32, 16],
            dropout=m_cfg["dropout"],
            learning_rate=learning_rate,
        )
    elif model_name == "gru":
        m_cfg = config["models"]["gru"]
        model = build_gru_forecaster(
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            num_features=num_features,
            units=m_cfg["units"] if not args.quick else [32, 16],
            dropout=m_cfg["dropout"],
            learning_rate=learning_rate,
        )
    elif model_name == "transformer":
        m_cfg = config["models"]["transformer"]
        model = build_transformer_forecaster(
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            num_features=num_features,
            d_model=m_cfg["d_model"] if not args.quick else 32,
            num_heads=m_cfg["num_heads"] if not args.quick else 2,
            ff_dim=m_cfg["ff_dim"] if not args.quick else 64,
            num_layers=m_cfg["num_layers"] if not args.quick else 1,
            dropout=m_cfg["dropout"],
            learning_rate=learning_rate,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    param_count = int(model.count_params())
    logger.info(f"Initialized {model.name} with {param_count:,} trainable parameters.")

    trainer = ModelTrainer(
        model=model,
        model_name=f"{model_name}_{'quick' if args.quick else 'full'}",
        checkpoint_dir=config["artifacts"]["checkpoints_dir"],
    )

    train_res = trainer.train(
        train_data=train_ds,
        val_data=val_ds,
        epochs=epochs,
        early_stopping_patience=train_cfg["early_stopping_patience"],
        reduce_lr_patience=train_cfg["reduce_lr_patience"],
        reduce_lr_factor=train_cfg["reduce_lr_factor"],
        min_lr=train_cfg["min_lr"],
    )

    # Evaluate on test set
    y_test_pred_scaled = model.predict(test_ds)
    y_test_phys = preprocessor.inverse_transform(y_test)
    y_pred_phys = preprocessor.inverse_transform(y_test_pred_scaled)

    evaluator.evaluate(
        y_true=y_test_phys,
        y_pred=y_pred_phys,
        model_name=model_name.upper(),
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        num_features=num_features,
        training_time=train_res["training_time"],
        parameter_count=param_count,
        batch_size=batch_size,
        epochs=train_res["epochs_completed"],
        learning_rate=learning_rate,
    )

    # Save test sample prediction artifact for plotting in UI
    pred_path = Path("artifacts/predictions") / f"{model_name}_sample_prediction.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        pred_path,
        history=preprocessor.inverse_transform(X_test[:10]),
        ground_truth=y_test_phys[:10],
        prediction=y_pred_phys[:10],
        feature_names=np.array(feature_cols),
    )
    logger.info(f"Saved sample prediction artifact to {pred_path}")


if __name__ == "__main__":
    main()
