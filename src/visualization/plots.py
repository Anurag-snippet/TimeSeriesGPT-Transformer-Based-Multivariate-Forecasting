"""Visualization utility producing publication-quality Matplotlib and interactive Plotly figures."""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_multivariate_series(
    df,
    feature_cols: List[str],
    timestamp_col: str = "timestamp",
    title: str = "Multivariate Industrial Telemetry",
) -> go.Figure:
    """Create interactive multi-sensor line plot."""
    fig = make_subplots(
        rows=len(feature_cols),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[col.replace("_", " ").title() for col in feature_cols],
        vertical_spacing=0.03,
    )

    for i, col in enumerate(feature_cols):
        fig.add_trace(
            go.Scatter(
                x=df[timestamp_col],
                y=df[col],
                name=col,
                mode="lines",
                line=dict(width=1.5),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        height=220 * len(feature_cols),
        title_text=title,
        showlegend=False,
        template="plotly_white",
    )
    return fig


def plot_forecast_sample(
    history: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    feature_idx: int = 0,
    feature_name: str = "Sensor",
    model_name: str = "Model",
) -> go.Figure:
    """Plot past input window against future ground truth and forecast."""
    input_len = len(history)
    horizon = len(ground_truth)

    t_hist = np.arange(-input_len + 1, 1)
    t_fut = np.arange(1, horizon + 1)

    fig = go.Figure()

    # Historical context
    fig.add_trace(
        go.Scatter(
            x=t_hist,
            y=history[:, feature_idx],
            mode="lines+markers",
            name="Observed History",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # Ground truth future
    fig.add_trace(
        go.Scatter(
            x=t_fut,
            y=ground_truth[:, feature_idx],
            mode="lines+markers",
            name="Ground Truth (Future)",
            line=dict(color="#2ca02c", width=2.5),
        )
    )

    # Predicted future
    fig.add_trace(
        go.Scatter(
            x=t_fut,
            y=prediction[:, feature_idx],
            mode="lines+markers",
            name=f"{model_name} Forecast",
            line=dict(color="#d62728", width=2.5, dash="dash"),
        )
    )

    # Shaded forecast horizon region
    fig.add_vrect(
        x0=0.5,
        x1=horizon + 0.5,
        fillcolor="rgba(255, 165, 0, 0.12)",
        layer="below",
        line_width=0,
        annotation_text="Forecast Horizon",
        annotation_position="top left",
    )

    fig.update_layout(
        title=f"Multi-Step Forecast Comparison: {feature_name} ({model_name})",
        xaxis_title="Relative Time Steps (t)",
        yaxis_title=f"Value ({feature_name})",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )

    return fig


def plot_anomaly_detection_results(
    timestamps,
    sensor_values: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    predicted_anomalies: np.ndarray,
    sensor_name: str = "Vibration",
) -> go.Figure:
    """Plot sensor readings alongside anomaly scores and flagged anomalous events."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Physical Sensor Readings: {sensor_name}", "Anomaly Score & Decision Threshold"],
        vertical_spacing=0.1,
    )

    # Top: Sensor values
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=sensor_values,
            mode="lines",
            name=sensor_name,
            line=dict(color="#2c3e50", width=1.5),
        ),
        row=1,
        col=1,
    )

    # Highlight anomalies on top plot
    anomaly_idx = np.where(predicted_anomalies == 1)[0]
    if len(anomaly_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=timestamps[anomaly_idx],
                y=sensor_values[anomaly_idx],
                mode="markers",
                name="Detected Anomaly",
                marker=dict(color="#e74c3c", size=7, symbol="x"),
            ),
            row=1,
            col=1,
        )

    # Bottom: Anomaly scores
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=anomaly_scores,
            mode="lines",
            name="Anomaly Score",
            line=dict(color="#8e44ad", width=1.5),
        ),
        row=2,
        col=1,
    )

    # Threshold horizontal line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Threshold ({threshold:.4f})",
        annotation_position="top right",
        row=2,
        col=1,
    )

    fig.update_layout(
        height=600,
        title="Industrial Anomaly Detection & Telemetry Monitoring",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig
