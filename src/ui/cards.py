"""Industrial Reusable Cards Components.

Includes:
- Compact KPI Cards
- Sensor Metric Cards with Inline SVG Sparklines
- System Insights Cards
- Anomaly Event Cards
"""

import textwrap
from typing import List, Optional
import numpy as np


def generate_sparkline_svg(values: List[float], width: int = 140, height: int = 32, color: str = "#00A6A6") -> str:
    """Generate lightweight, responsive inline SVG polyline for sensor sparkline trends."""
    if not values or len(values) < 2:
        return f'<svg width="{width}" height="{height}"></svg>'

    arr = np.array(values, dtype=float)
    min_v, max_v = np.min(arr), np.max(arr)
    diff = max_v - min_v if max_v != min_v else 1.0

    points = []
    step = width / (len(arr) - 1)
    for i, val in enumerate(arr):
        x = i * step
        y = height - ((val - min_v) / diff * (height - 6) + 3)
        points.append(f"{x:.1f},{y:.1f}")

    points_str = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" style="overflow: visible;">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" points="{points_str}" />'
        f'</svg>'
    )


def render_kpi_card_html(label: str, value: str, subtext: str, badge_text: Optional[str] = None, badge_type: str = "primary") -> str:
    """Generate HTML for a compact, high-density KPI card."""
    badge_html = ""
    if badge_text:
        badge_class = f"ind-badge-{badge_type}" if badge_type in ["primary", "secondary", "success"] else ""
        badge_html = f'<span class="ind-badge {badge_class}" style="padding: 2px 7px; font-size: 10px;">{badge_text}</span>'

    raw_html = f"""
<div class="kpi-card">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
        <div class="kpi-label">{label}</div>
        {badge_html}
    </div>
    <div class="kpi-value">{value}</div>
    <div class="kpi-subtext">{subtext}</div>
</div>
"""
    return textwrap.dedent(raw_html).strip()


def render_sensor_card_html(
    sensor_name: str,
    sensor_code: str,
    current_val: float,
    unit: str,
    min_val: float,
    max_val: float,
    trend_str: str = "+1.2%",
    status: str = "NORMAL",
    sparkline_data: Optional[List[float]] = None,
) -> str:
    """Generate HTML for an industrial sensor telemetry card."""
    sparkline_svg = generate_sparkline_svg(sparkline_data or [current_val, current_val], width=110, height=28)
    
    is_normal = status.upper() == "NORMAL"
    status_dot = '<span class="status-dot"></span>' if is_normal else '<span class="status-dot-red"></span>'
    status_color = "#22C55E" if is_normal else "#EF4444"

    raw_html = f"""
<div class="ind-card" style="margin-bottom: 10px; padding: 14px 16px;">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <div style="font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; color: #8B98A8; text-transform: uppercase;">
                {sensor_name}
            </div>
            <div style="font-size: 9.5px; font-family: monospace; color: #505F70;">
                {sensor_code}
            </div>
        </div>
        <div style="text-align: right;">
            {sparkline_svg}
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 10px;">
        <div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: #F5F7FA;">
                {current_val:,.1f}
            </span>
            <span style="font-size: 12px; color: #8B98A8; margin-left: 3px;">{unit}</span>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; font-family: monospace; font-weight: 600; color: #00A6A6;">
                {trend_str}
            </div>
            <div style="font-size: 10px; font-weight: 600; color: {status_color}; display: flex; align-items: center; justify-content: flex-end; gap: 4px; margin-top: 2px;">
                {status_dot} {status}
            </div>
        </div>
    </div>
    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #1E2833; display: flex; justify-content: space-between; font-size: 10.5px; color: #8B98A8; font-family: monospace;">
        <span>MIN: <b style="color: #D1D9E0;">{min_val:,.1f}</b></span>
        <span>MAX: <b style="color: #D1D9E0;">{max_val:,.1f}</b></span>
        <span>DELTA: <b style="color: #D1D9E0;">{(max_val - min_val):,.1f}</b></span>
    </div>
</div>
"""
    return textwrap.dedent(raw_html).strip()


def render_system_insights_html(
    state: str = "NORMAL",
    horizon_steps: int = 10,
    anomaly_score: float = 0.081,
    anomaly_threshold: float = 0.501,
    last_step: str = "12,000",
    model_name: str = "Self-Attention Transformer",
) -> str:
    """Generate HTML for the Executive System Insights panel."""
    is_normal = state.upper() == "NORMAL"
    status_dot = '<span class="status-dot"></span>' if is_normal else '<span class="status-dot-red"></span>'
    status_color = "#22C55E" if is_normal else "#EF4444"

    raw_html = f"""
<div class="ind-card" style="height: 100%;">
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.1em; color: #00A6A6; text-transform: uppercase; margin-bottom: 12px; display: flex; justify-content: space-between;">
        <span>System Insights</span>
        <span style="font-size: 10px; color: #8B98A8;">DIAGNOSTICS</span>
    </div>
    <div style="display: flex; flex-direction: column; gap: 12px;">
        <div style="padding: 10px; background-color: #111820; border: 1px solid #1E2833; border-radius: 6px;">
            <div style="font-size: 10px; color: #8B98A8; text-transform: uppercase; letter-spacing: 0.06em;">Operating State</div>
            <div style="font-size: 15px; font-weight: 700; color: {status_color}; display: flex; align-items: center; gap: 7px; margin-top: 3px;">
                {status_dot} {state}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="padding: 10px; background-color: #111820; border: 1px solid #1E2833; border-radius: 6px;">
                <div style="font-size: 10px; color: #8B98A8; text-transform: uppercase; letter-spacing: 0.06em;">Forecast Horizon</div>
                <div style="font-family: monospace; font-size: 16px; font-weight: 700; color: #F5F7FA; margin-top: 3px;">
                    {horizon_steps} <span style="font-size: 11px; color: #8B98A8;">steps (10m)</span>
                </div>
            </div>
            <div style="padding: 10px; background-color: #111820; border: 1px solid #1E2833; border-radius: 6px;">
                <div style="font-size: 10px; color: #8B98A8; text-transform: uppercase; letter-spacing: 0.06em;">Anomaly Score</div>
                <div style="font-family: monospace; font-size: 16px; font-weight: 700; color: #38D9D9; margin-top: 3px;">
                    {anomaly_score:.3f}
                </div>
            </div>
        </div>
        <div style="padding: 10px; background-color: #111820; border: 1px solid #1E2833; border-radius: 6px;">
            <div style="font-size: 10px; color: #8B98A8; text-transform: uppercase; letter-spacing: 0.06em;">Decision Boundary Threshold</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                <span style="font-family: monospace; font-size: 13px; color: #F59E0B; font-weight: 600;">{anomaly_threshold:.3f}</span>
                <span style="font-size: 10.5px; color: #8B98A8;">95th Percentile Reconstruction</span>
            </div>
        </div>
        <div style="padding: 10px; background-color: #111820; border: 1px solid #1E2833; border-radius: 6px;">
            <div style="font-size: 10px; color: #8B98A8; text-transform: uppercase; letter-spacing: 0.06em;">Active Core Architecture</div>
            <div style="font-size: 13px; font-weight: 600; color: #70A9FF; margin-top: 3px;">
                {model_name}
            </div>
            <div style="font-size: 10px; color: #8B98A8; margin-top: 2px;">
                4 Heads · 2 Layers · d_model=64
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #505F70; padding: 0 4px;">
            <span>Chronological Step: <b style="color: #8B98A8; font-family: monospace;">t={last_step}</b></span>
            <span>Latency: <b style="color: #22C55E; font-family: monospace;">&lt; 15ms</b></span>
        </div>
    </div>
</div>
"""
    return textwrap.dedent(raw_html).strip()


def render_event_card_html(timestamp: str, sensor_name: str, score: float, threshold: float, severity: str = "HIGH") -> str:
    """Generate HTML for an anomaly event record."""
    severity_colors = {
        "HIGH": ("#EF4444", "rgba(239, 68, 68, 0.15)"),
        "MEDIUM": ("#F59E0B", "rgba(245, 158, 11, 0.15)"),
        "LOW": ("#3B82F6", "rgba(59, 130, 246, 0.15)"),
    }
    color, bg = severity_colors.get(severity.upper(), ("#EF4444", "rgba(239, 68, 68, 0.15)"))

    raw_html = f"""
<div style="background-color: #111820; border: 1px solid #26313D; border-left: 3px solid {color}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
    <div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-family: monospace; font-size: 11px; font-weight: 700; color: #F5F7FA;">{timestamp}</span>
            <span style="background-color: {bg}; color: {color}; border-radius: 4px; padding: 2px 6px; font-size: 9.5px; font-weight: 700;">{severity}</span>
        </div>
        <div style="font-size: 11.5px; color: #8B98A8; margin-top: 3px;">
            Sensor anomaly detected in <b>{sensor_name}</b>
        </div>
    </div>
    <div style="text-align: right; font-family: monospace;">
        <div style="font-size: 13px; font-weight: 700; color: {color};">Score: {score:.3f}</div>
        <div style="font-size: 10px; color: #505F70;">Threshold: {threshold:.3f}</div>
    </div>
</div>
"""
    return textwrap.dedent(raw_html).strip()
