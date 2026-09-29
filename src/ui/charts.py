"""Industrial Plotly Charts Module.

Provides compact, information-dense, dark-themed charts matching Siemens-inspired enterprise styling:
- Palette: #0B0F14 / #111820, #0066FF, #00A6A6, #22C55E, #F59E0B, #EF4444
- Subtle grid lines, uncluttered axes, unified hovermode
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.ui.theme import (
    COLOR_BG,
    COLOR_BORDER,
    COLOR_BORDER_SUBTLE,
    COLOR_DANGER,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_SURFACE,
    COLOR_SURFACE_ELEVATED,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    COLOR_WARNING,
)

# Standard Plotly Layout Tokens
PLOT_LAYOUT_BASE = dict(
    paper_bgcolor=COLOR_SURFACE,
    plot_bgcolor=COLOR_SURFACE,
    font=dict(family="Inter, sans-serif", color=COLOR_TEXT_MUTED, size=11),
    margin=dict(l=48, r=24, t=36, b=36),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=COLOR_SURFACE_ELEVATED,
        bordercolor=COLOR_BORDER,
        font=dict(family="Inter, sans-serif", color=COLOR_TEXT_PRIMARY, size=11),
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#1B2430",
        gridwidth=1,
        zeroline=False,
        linecolor="#26313D",
        tickfont=dict(size=10, color=COLOR_TEXT_MUTED),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#1B2430",
        gridwidth=1,
        zeroline=False,
        linecolor="#26313D",
        tickfont=dict(size=10, color=COLOR_TEXT_MUTED),
    ),
)


def create_telemetry_primary_chart(
    df: pd.DataFrame,
    sensor_col: str,
    sensor_label: str,
    unit: str = "",
    height: int = 330,
) -> go.Figure:
    """Create a sleek, high-density line chart for selected telemetry stream."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"] if "timestamp" in df.columns else np.arange(len(df)),
            y=df[sensor_col],
            mode="lines",
            name=sensor_label,
            line=dict(color=COLOR_SECONDARY, width=1.7),
            fill="tozeroy",
            fillcolor="rgba(0, 166, 166, 0.04)",
        )
    )

    # Compute rolling 30-step trendline
    if len(df) > 30:
        roll_mean = df[sensor_col].rolling(30, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"] if "timestamp" in df.columns else np.arange(len(df)),
                y=roll_mean,
                mode="lines",
                name="Moving Trend (30s)",
                line=dict(color=COLOR_PRIMARY, width=1.4, dash="dot"),
            )
        )

    layout = dict(PLOT_LAYOUT_BASE)
    layout.update(
        height=height,
        title=dict(
            text=f"<b>{sensor_label.upper()}</b> — Telemetry Dynamics {f'({unit})' if unit else ''}",
            font=dict(size=12.5, color=COLOR_TEXT_PRIMARY),
            x=0.01,
            y=0.98,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10.5),
        ),
        showlegend=True,
    )
    fig.update_layout(**layout)
    return fig


def create_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str], height: int = 340) -> go.Figure:
    """Create compact physical sensor coupling heatmap."""
    corr = df[feature_cols].corr()
    clean_labels = [c.split("_")[0] + " " + " ".join(c.split("_")[1:3]) for c in feature_cols]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=clean_labels,
            y=clean_labels,
            colorscale=[
                [0.0, "#0B1526"],
                [0.3, "#0E2A54"],
                [0.7, "#0066FF"],
                [1.0, "#00D2D2"],
            ],
            zmin=-1.0,
            zmax=1.0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(family="JetBrains Mono, monospace", size=10, color="#FFFFFF"),
            colorbar=dict(
                thickness=12,
                tickfont=dict(size=9, color=COLOR_TEXT_MUTED),
                outlinecolor=COLOR_BORDER,
            ),
        )
    )

    layout = dict(PLOT_LAYOUT_BASE)
    layout.update(
        height=height,
        title=dict(
            text="<b>PHYSICAL SENSOR COUPLING MATRIX</b>",
            font=dict(size=12, color=COLOR_TEXT_PRIMARY),
            x=0.01,
            y=0.98,
        ),
        margin=dict(l=60, r=20, t=36, b=50),
    )
    fig.update_layout(**layout)
    return fig


def create_forecast_comparison_chart(
    history: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    feature_idx: int = 0,
    feature_name: str = "Combustor Inlet Temp",
    model_name: str = "Transformer",
    unit: str = "",
    height: int = 340,
) -> go.Figure:
    """Create multi-step forecast visualization with separated past and future regions."""
    input_len = len(history)
    horizon = len(ground_truth)

    t_hist = np.arange(-input_len + 1, 1)
    t_fut = np.arange(1, horizon + 1)

    fig = go.Figure()

    # Historical context (Observed Past)
    fig.add_trace(
        go.Scatter(
            x=t_hist,
            y=history[:, feature_idx],
            mode="lines",
            name="Observed History (W=60)",
            line=dict(color=COLOR_PRIMARY, width=2.0),
        )
    )

    # Ground truth future
    fig.add_trace(
        go.Scatter(
            x=t_fut,
            y=ground_truth[:, feature_idx],
            mode="lines+markers",
            name="Ground Truth Future (H=10)",
            line=dict(color=COLOR_SUCCESS, width=2.2),
            marker=dict(size=5, symbol="circle"),
        )
    )

    # Predicted future
    fig.add_trace(
        go.Scatter(
            x=t_fut,
            y=prediction[:, feature_idx],
            mode="lines+markers",
            name=f"{model_name} Prediction",
            line=dict(color="#38D9D9", width=2.2, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
        )
    )

    # Shaded forecast horizon region
    fig.add_vrect(
        x0=0.5,
        x1=horizon + 0.5,
        fillcolor="rgba(0, 166, 166, 0.08)",
        layer="below",
        line_width=1,
        line_color="rgba(0, 166, 166, 0.3)",
        annotation_text="Forecast Horizon (H=10)",
        annotation_position="top left",
        annotation_font=dict(size=10, color=COLOR_SECONDARY),
    )

    # Vertical split line at t=0
    fig.add_vline(
        x=0.5,
        line_width=1.5,
        line_dash="dot",
        line_color="#48596E",
    )

    layout = dict(PLOT_LAYOUT_BASE)
    layout.update(
        height=height,
        title=dict(
            text=f"<b>MULTI-STEP TRAJECTORY:</b> {feature_name.upper()} · {model_name}",
            font=dict(size=12.5, color=COLOR_TEXT_PRIMARY),
            x=0.01,
            y=0.98,
        ),
        xaxis_title=dict(text="Chronological Relative Steps (t)", font=dict(size=10.5, color=COLOR_TEXT_MUTED)),
        yaxis_title=dict(text=f"Sensor Value {f'({unit})' if unit else ''}", font=dict(size=10.5, color=COLOR_TEXT_MUTED)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10.5),
        ),
        showlegend=True,
    )
    fig.update_layout(**layout)
    return fig


def create_model_benchmark_barchart(
    df: pd.DataFrame,
    metric: str = "MAE",
    height: int = 280,
) -> go.Figure:
    """Create clean horizontal bar chart for model benchmark comparison."""
    if df.empty or metric not in df.columns:
        return go.Figure()

    df_sorted = df.sort_values(by=metric, ascending=False).copy()
    
    # Identify best (minimum value for error metrics, or maximum if applicable)
    min_idx = df_sorted[metric].idxmin()
    colors = [COLOR_PRIMARY if idx == min_idx else "#26313D" for idx in df_sorted.index]

    fig = go.Figure(
        go.Bar(
            y=df_sorted["model"],
            x=df_sorted[metric],
            orientation="h",
            marker=dict(color=colors, line=dict(color=COLOR_BORDER, width=1)),
            text=[f"{val:.3f}" if isinstance(val, float) else str(val) for val in df_sorted[metric]],
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=10, color=COLOR_TEXT_PRIMARY),
        )
    )

    layout = dict(PLOT_LAYOUT_BASE)
    layout.update(
        height=height,
        title=dict(
            text=f"<b>COMPARATIVE {metric.upper()} ON PHYSICAL TEST DATA</b> (Lower is Better)",
            font=dict(size=12, color=COLOR_TEXT_PRIMARY),
            x=0.01,
            y=0.98,
        ),
        margin=dict(l=110, r=50, t=36, b=30),
        xaxis_title=dict(text=metric, font=dict(size=10, color=COLOR_TEXT_MUTED)),
    )
    fig.update_layout(**layout)
    return fig


def create_anomaly_timeline_chart(
    timestamps,
    sensor_values: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: float,
    predicted_anomalies: np.ndarray,
    sensor_name: str = "Vibration Trajectory",
    height: int = 420,
) -> go.Figure:
    """Create dual-stage industrial anomaly monitoring timeline."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"<b>{sensor_name.upper()} READINGS & DETECTED EVENTS</b>",
            "<b>ANOMALY RECONSTRUCTION SCORE & THRESHOLD</b>",
        ],
        vertical_spacing=0.12,
    )

    # 1. Sensor Trajectory
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=sensor_values,
            mode="lines",
            name=sensor_name,
            line=dict(color=COLOR_SECONDARY, width=1.5),
        ),
        row=1,
        col=1,
    )

    # Highlight Anomalies
    anomaly_idx = np.where(predicted_anomalies == 1)[0]
    if len(anomaly_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=timestamps[anomaly_idx],
                y=sensor_values[anomaly_idx],
                mode="markers",
                name="Flagged Anomaly",
                marker=dict(color=COLOR_DANGER, size=7, symbol="x", line=dict(width=1.5)),
            ),
            row=1,
            col=1,
        )

    # 2. Anomaly Score
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=anomaly_scores,
            mode="lines",
            name="Reconstruction Score",
            line=dict(color="#A78BFA", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(167, 139, 250, 0.05)",
        ),
        row=2,
        col=1,
    )

    # Horizontal Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=COLOR_DANGER,
        line_width=1.5,
        annotation_text=f"Threshold ({threshold:.3f})",
        annotation_position="top right",
        annotation_font=dict(size=10, color=COLOR_DANGER),
        row=2,
        col=1,
    )

    layout = dict(PLOT_LAYOUT_BASE)
    layout.update(
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, font=dict(size=10)),
    )
    fig.update_layout(**layout)
    return fig
