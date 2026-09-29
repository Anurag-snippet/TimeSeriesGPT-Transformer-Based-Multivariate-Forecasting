"""Industrial Time-Series Intelligence Platform - Interactive Research Dashboard.

Built with Streamlit, Plotly, TensorFlow, and Scikit-Learn.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.loader import load_raw_dataset
from src.evaluation.evaluator import ModelEvaluator
from src.utils.config import load_config
from src.visualization.plots import plot_anomaly_detection_results, plot_forecast_sample, plot_multivariate_series

st.set_page_config(
    page_title="Industrial Time-Series Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for research styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F52BA;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #0F52BA;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_dataset():
    config = load_config("configs/config.yaml")
    df = load_raw_dataset(config["data"]["raw_path"])
    return config, df


@st.cache_data
def get_experiment_results():
    evaluator = ModelEvaluator("experiments/results.csv")
    return evaluator.load_results()


config, df = get_dataset()
results_df = get_experiment_results()
sensor_cols = config["data"]["feature_cols"]

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/engine.png", width=64)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Platform Module:",
    [
        "1. Executive Overview",
        "2. Data Explorer & Telemetry",
        "3. Multi-Step Forecasting",
        "4. Model Benchmarks & Comparison",
        "5. Anomaly Detection Studio",
        "6. Google Colab & Scaling Guide",
        "7. Research Report Summary",
    ],
)

# 1. Executive Overview
if menu == "1. Executive Overview":
    st.markdown('<div class="main-title">Industrial Time-Series Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Transformer-Based Multivariate Sensor Forecasting and Unsupervised Anomaly Detection</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Monitored Sensors", len(sensor_cols), "Coupled Turbofan")
    with col2:
        st.metric("Total Telemetry Steps", f"{len(df):,}", "60s Resolution")
    with col3:
        st.metric("Deep Learning Framework", "TensorFlow & Keras", "tf.data & MHA")
    with col4:
        st.metric("Target Execution Mode", "CPU & Colab GPU Ready", "Zero Hardware Burden")

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown("""
    This platform provides an end-to-end industrial intelligence pipeline:
    1. **Physical Turbine Telemetry**: Dynamically coupled thermal, pressure, rotational, and vibration telemetry.
    2. **Leakage-Free Preprocessing**: Chronological 70/15/15 train-val-test partitioning with scaler fitted strictly on train data.
    3. **Multi-Step Forecasting**: Self-attention Transformer compared directly with Naive, Moving Average, Ridge, LSTM, and GRU baselines.
    4. **Unsupervised Anomaly Detection**: Dual-paradigm detection via LSTM Autoencoder (reconstruction error) and Isolation Forest.
    5. **Colab-Ready Design**: Full support for quick local CPU validation as well as single-command Colab training.
    """)

# 2. Data Explorer & Telemetry
elif menu == "2. Data Explorer & Telemetry":
    st.header("Industrial Turbine Sensor Telemetry Explorer")
    st.write("Examine chronological sensor dynamics, pairwise correlations, and operational load cycles.")

    st.subheader("Interactive Multi-Sensor Telemetry")
    fig_telemetry = plot_multivariate_series(df.iloc[:600], sensor_cols, title="Turbine Telemetry (Initial 600 Steps)")
    st.plotly_chart(fig_telemetry, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Correlation Matrix")
        corr = df[sensor_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Blues", title="Physical Sensor Coupling")
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_b:
        st.subheader("Descriptive Statistics")
        st.dataframe(df[sensor_cols].describe().T[["mean", "std", "min", "50%", "max"]].round(2))

# 3. Multi-Step Forecasting
elif menu == "3. Multi-Step Forecasting":
    st.header("Multi-Step Multivariate Forecasting")
    st.write("Inspect past historical sequences (input window) against multi-step future ground-truth and predictions.")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Select Model Architecture", ["TRANSFORMER", "LSTM", "GRU"])
    with col2:
        sensor_choice = st.selectbox("Select Sensor Variable", sensor_cols)

    sensor_idx = sensor_cols.index(sensor_choice)
    pred_file = Path("artifacts/predictions") / f"{model_choice.lower()}_sample_prediction.npz"

    if pred_file.exists():
        data = np.load(pred_file)
        sample_idx = st.slider("Select Evaluation Sequence Window", 0, len(data["prediction"]) - 1, 0)
        hist = data["history"][sample_idx]
        true_fut = data["ground_truth"][sample_idx]
        pred_fut = data["prediction"][sample_idx]

        fig_pred = plot_forecast_sample(
            history=hist,
            ground_truth=true_fut,
            prediction=pred_fut,
            feature_idx=sensor_idx,
            feature_name=sensor_choice,
            model_name=model_choice,
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Run model training to generate prediction artifacts.")

# 4. Model Benchmarks & Comparison
elif menu == "4. Model Benchmarks & Comparison":
    st.header("Empirical Model Benchmark & Experimental Comparison")
    st.write("All performance metrics are generated from actual test evaluation on physical scale values (no fabricated data).")

    if not results_df.empty:
        st.dataframe(results_df[["model", "MAE", "RMSE", "MAPE", "SMAPE", "training_time", "parameter_count"]], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_mae = px.bar(results_df, x="model", y="MAE", color="model", title="Mean Absolute Error (Lower is Better)")
            st.plotly_chart(fig_mae, use_container_width=True)
        with col2:
            fig_time = px.bar(results_df, x="model", y="training_time", color="model", title="Training Time (Seconds)")
            st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.warning("No experiment records found in experiments/results.csv.")

# 5. Anomaly Detection Studio
elif menu == "5. Anomaly Detection Studio":
    st.header("Unsupervised Anomaly Detection Studio")
    st.markdown("""
    In raw industrial settings, ground-truth anomaly labels are typically unavailable.
    We evaluate detection sensitivity by injecting controlled physical faults (thermal drift, rotor vibration spikes, and sensor freeze).
    """)

    anom_file = Path("artifacts/predictions/anomaly_benchmark.npz")
    if anom_file.exists():
        anom_data = np.load(anom_file)
        method = st.radio("Detection Paradigm", ["LSTM Autoencoder (Reconstruction Error)", "Isolation Forest (Statistical Ensemble)"])

        if "LSTM" in method:
            scores = anom_data["ae_scores"]
            preds = anom_data["ae_preds"]
            thresh = float(anom_data["ae_threshold"])
        else:
            scores = anom_data["iso_scores"]
            preds = anom_data["iso_preds"]
            thresh = float(np.percentile(scores, 92.0))

        # Sample preview
        timesteps = np.arange(len(scores))
        fig_anom = plot_anomaly_detection_results(
            timestamps=timesteps[:150],
            sensor_values=scores[:150],
            anomaly_scores=scores[:150],
            threshold=thresh,
            predicted_anomalies=preds[:150],
            sensor_name="Anomaly Score Trajectory",
        )
        st.plotly_chart(fig_anom, use_container_width=True)
        st.success(f"Detected {np.sum(preds)} anomalous sequences out of {len(preds)} evaluated windows.")
    else:
        st.info("Run `python scripts/evaluate.py` to produce anomaly detection artifacts.")

# 6. Google Colab & Scaling Guide
elif menu == "6. Google Colab & Scaling Guide":
    st.header("Google Colab Execution & GPU Scaling Guide")
    st.markdown("""
    To avoid local hardware burden or GPU consumption on your machine, this project includes a complete Google Colab workflow.

    ### 3-Step Execution in Google Colab:

    1. **Upload or Clone Repository**:
    ```bash
    git clone https://github.com/<your-username>/TimeSeriesGPT.git
    cd TimeSeriesGPT
    ```

    2. **Install Dependencies in Colab (with free T4 GPU)**:
    ```bash
    !pip install -r requirements.txt
    ```

    3. **Run Full Deep-Learning Training**:
    ```bash
    # Prepare data
    !python scripts/prepare_data.py

    # Train baselines
    !python scripts/train.py --model naive
    !python scripts/train.py --model ridge

    # Train deep learning models (Full mode with GPU acceleration)
    !python scripts/train.py --model lstm
    !python scripts/train.py --model gru
    !python scripts/train.py --model transformer

    # Run complete evaluation and anomaly benchmark
    !python scripts/evaluate.py
    ```

    ### Key Colab Benefits:
    - **Zero GPU stress on your PC**: Training executes entirely on Google Cloud infrastructure.
    - **Native GPU Acceleration**: TensorFlow automatically detects NVIDIA T4/V100/A100 instances in Colab.
    - **Artifact Download**: You can easily zip and download `artifacts/` and `experiments/results.csv` back to your local environment.
    """)

# 7. Research Report Summary
elif menu == "7. Research Report Summary":
    st.header("Research Report & Internship Portfolio Insights")
    st.markdown("""
    ### Key Theoretical Insights for Siemens Interview:
    - **Self-Attention vs Recurrence**: LSTMs and GRUs process sequential steps recurrently ($O(N)$ sequential operations), whereas Transformer self-attention computes pairwise token affinities in parallel ($O(N^2 \cdot d)$ computation with $O(1)$ sequential operations).
    - **Why Sinusoidal Positional Encoding**: Since self-attention is permutation-equivariant, adding sinusoidal signals:
      $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
      informs the model of chronological progression without introducing learnable temporal biases.
    - **Pre-LN vs Post-LN**: Pre-Layer Normalization stabilizes gradient flow through deep encoder blocks, allowing immediate convergence with standard Adam optimization without requiring extensive warm-up schedules.
    - **Multi-Step Direct Projection**: Directly projecting the latent representation into a reshaped tensor $(H, \text{features})$ prevents error accumulation typical of autoregressive rolling-step forecasting.
    """)

st.sidebar.markdown("---")
st.sidebar.info("Industrial Time-Series Intelligence Platform\nDeveloped for Siemens AI/ML Research Internship")
