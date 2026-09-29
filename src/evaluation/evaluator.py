"""Model evaluation engine and experiment results recorder."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics
from src.utils.logger import setup_logger

logger = setup_logger("Evaluator")


class ModelEvaluator:
    """Evaluates multi-step forecasting models on test sets and persists experiments."""

    def __init__(self, results_csv: str = "experiments/results.csv"):
        self.results_csv = Path(results_csv)
        self.results_csv.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        input_window: int,
        forecast_horizon: int,
        num_features: int,
        training_time: float = 0.0,
        parameter_count: int = 0,
        batch_size: int = 32,
        epochs: int = 0,
        learning_rate: float = 0.001,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute metrics and record to experiments/results.csv."""
        metrics = compute_all_metrics(y_true, y_pred)
        logger.info(f"Evaluation for {model_name}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, MAPE={metrics['MAPE']}%, sMAPE={metrics['SMAPE']}%")

        if experiment_id is None:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name.lower()}"

        record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "input_window": input_window,
            "forecast_horizon": forecast_horizon,
            "num_features": num_features,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "parameter_count": parameter_count,
            "training_time": training_time,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"],
            "SMAPE": metrics["SMAPE"],
        }

        self._append_to_csv(record)
        return record

    def _append_to_csv(self, record: Dict[str, Any]) -> None:
        """Atomically append experiment record to CSV."""
        df_new = pd.DataFrame([record])
        if not self.results_csv.exists() or self.results_csv.stat().st_size == 0:
            df_new.to_csv(self.results_csv, index=False)
        else:
            df_new.to_csv(self.results_csv, mode="a", header=False, index=False)
        logger.info(f"Recorded experiment {record['experiment_id']} to {self.results_csv}")

    def load_results(self) -> pd.DataFrame:
        """Load experiment results into DataFrame."""
        if not self.results_csv.exists() or self.results_csv.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(self.results_csv)
