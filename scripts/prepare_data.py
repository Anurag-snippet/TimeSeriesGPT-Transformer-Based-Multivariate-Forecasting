"""Data preparation script.

Executes:
1. Loads/generates raw industrial turbine telemetry.
2. Applies DataPreprocessor (parsing, sorting, deduping, imputation).
3. Performs strict chronological train/val/test split.
4. Fits StandardScaler ONLY on training data to prevent leakage.
5. Saves clean processed data and fitted scaler artifact.
"""

from pathlib import Path
import pandas as pd

from src.data.loader import load_raw_dataset
from src.data.preprocessing import DataPreprocessor
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("PrepareData")


def main():
    config = load_config("configs/config.yaml")
    set_seed(config.get("seed", 42))

    data_cfg = config["data"]
    feature_cols = data_cfg["feature_cols"]
    timestamp_col = data_cfg["timestamp_col"]

    logger.info("Starting Industrial Telemetry Data Preparation...")

    # 1. Load or generate raw data
    raw_df = load_raw_dataset(
        raw_path=data_cfg["raw_path"],
        auto_generate_if_missing=True,
        num_timesteps=12000,
    )
    logger.info(f"Loaded raw dataset with shape: {raw_df.shape}")

    # 2. Preprocess data
    preprocessor = DataPreprocessor(
        feature_cols=feature_cols,
        timestamp_col=timestamp_col,
        train_split=data_cfg["train_split"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
    )

    clean_df = preprocessor.clean_dataframe(raw_df)
    train_df, val_df, test_df = preprocessor.split_chronological(clean_df)

    # 3. Fit scaler ONLY on train data (strictly prevents data leakage)
    _, fitted_prep = preprocessor.fit_transform(train_df)

    # 4. Save scaler artifact
    scaler_path = Path("artifacts/scalers/scaler.joblib")
    fitted_prep.save_scaler(str(scaler_path))

    # 5. Save processed data
    proc_path = Path(data_cfg["processed_path"])
    proc_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(proc_path, index=False)
    logger.info(f"Saved clean processed telemetry to {proc_path}")

    # Log summary statistics
    logger.info("Data preparation completed successfully.")
    logger.info(f"Sensors: {feature_cols}")
    logger.info(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")


if __name__ == "__main__":
    main()
