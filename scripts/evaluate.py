"""Systematic evaluation and anomaly benchmark script.

Runs:
1. Multi-model comparative evaluation from saved checkpoints and predictions.
2. Anomaly detection benchmark comparing LSTM Autoencoder vs Isolation Forest.
3. Synthetic anomaly injection (unsupervised scenario evaluation).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_sliding_windows
from src.data.preprocessing import DataPreprocessor
from src.evaluation.evaluator import ModelEvaluator
from src.models.anomaly import IsolationForestDetector, LSTMAutoencoderDetector, inject_synthetic_anomalies
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("EvaluateScript")


def main():
    config = load_config("configs/config.yaml")
    set_seed(config.get("seed", 42))

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

    train_df, val_df, test_df = preprocessor.split_chronological(df)
    scaled_train = preprocessor.transform(train_df)
    scaled_test = preprocessor.transform(test_df)

    input_window = data_cfg["input_window"]
    forecast_horizon = data_cfg["forecast_horizon"]

    X_train, _ = create_sliding_windows(scaled_train, input_window, forecast_horizon)
    X_test, _ = create_sliding_windows(scaled_test, input_window, forecast_horizon)

    logger.info("==================================================")
    logger.info("  ANOMALY DETECTION BENCHMARK (UNSUPERVISED SETUP)")
    logger.info("==================================================")
    logger.info("Ground-truth anomaly labels are unavailable in raw telemetry.")
    logger.info("Injecting controlled synthetic anomalies (spikes, thermal drifts, sensor freeze) to evaluate detection.")

    X_test_anom, true_labels = inject_synthetic_anomalies(X_test[:500], anomaly_rate=0.08, seed=42)
    logger.info(f"Synthetic test set: {len(X_test_anom)} sequences with {np.sum(true_labels)} injected anomalies ({np.mean(true_labels)*100:.1f}%).")

    # 1. Isolation Forest Baseline
    iso_detector = IsolationForestDetector(contamination=0.08)
    iso_detector.fit(X_train[:1500])
    iso_preds = iso_detector.predict(X_test_anom)
    iso_scores = iso_detector.score_samples(X_test_anom)

    iso_prec = precision_score(true_labels, iso_preds, zero_division=0)
    iso_rec = recall_score(true_labels, iso_preds, zero_division=0)
    iso_f1 = f1_score(true_labels, iso_preds, zero_division=0)
    iso_auc = roc_auc_score(true_labels, iso_scores)

    logger.info(f"Isolation Forest -> Precision: {iso_prec:.3f}, Recall: {iso_rec:.3f}, F1: {iso_f1:.3f}, ROC-AUC: {iso_auc:.3f}")

    # 2. LSTM Autoencoder
    lstm_ae = LSTMAutoencoderDetector(
        sequence_length=input_window,
        num_features=len(feature_cols),
        latent_dim=16,
        encoder_units=32,
        threshold_percentile=95.0,
    )
    lstm_ae.fit(X_train[:1500], epochs=4, batch_size=32)
    ae_preds = lstm_ae.predict(X_test_anom)
    ae_scores = lstm_ae.score_samples(X_test_anom)

    ae_prec = precision_score(true_labels, ae_preds, zero_division=0)
    ae_rec = recall_score(true_labels, ae_preds, zero_division=0)
    ae_f1 = f1_score(true_labels, ae_preds, zero_division=0)
    ae_auc = roc_auc_score(true_labels, ae_scores)

    logger.info(f"LSTM Autoencoder -> Precision: {ae_prec:.3f}, Recall: {ae_rec:.3f}, F1: {ae_f1:.3f}, ROC-AUC: {ae_auc:.3f}")

    # Save anomaly detection benchmark summary
    anom_summary_path = Path("artifacts/predictions/anomaly_benchmark.npz")
    anom_summary_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        anom_summary_path,
        true_labels=true_labels,
        iso_preds=iso_preds,
        iso_scores=iso_scores,
        ae_preds=ae_preds,
        ae_scores=ae_scores,
        ae_threshold=lstm_ae.threshold,
    )
    logger.info(f"Saved anomaly benchmark results to {anom_summary_path}")

    # Print summary table of forecasting models from experiments/results.csv
    evaluator = ModelEvaluator(config["artifacts"]["results_csv"])
    results_df = evaluator.load_results()
    if not results_df.empty:
        logger.info("\n================ MODEL FORECASTING BENCHMARK TABLE ================")
        cols_to_show = ["model", "MAE", "RMSE", "MAPE", "SMAPE", "training_time", "parameter_count"]
        present_cols = [c for c in cols_to_show if c in results_df.columns]
        logger.info("\n" + results_df[present_cols].to_string(index=False))


if __name__ == "__main__":
    main()
