"""Visual components including SVG pipeline diagrams, architecture schemas, and industrial states."""

import textwrap
import streamlit as st


def render_pipeline_diagram_html() -> str:
    """Render a modern industrial pipeline diagram using inline SVG and CSS cards."""
    raw_html = """
<div style="background-color: #111820; border: 1px solid #26313D; border-radius: 8px; padding: 18px 20px; margin-top: 12px;">
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.1em; color: #8B98A8; text-transform: uppercase; margin-bottom: 14px; display: flex; justify-content: space-between;">
        <span>End-to-End System Pipeline</span>
        <span style="font-size: 10px; color: #00A6A6; font-family: monospace;">CHRONOLOGICAL ZERO-LEAKAGE FLOW</span>
    </div>
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <!-- Step 1 -->
        <div style="flex: 1; min-width: 140px; background-color: #161E27; border: 1px solid #26313D; border-radius: 6px; padding: 10px 12px; text-align: center;">
            <div style="font-size: 9.5px; font-weight: 700; color: #00A6A6; letter-spacing: 0.08em; text-transform: uppercase;">01 / INGESTION</div>
            <div style="font-size: 12px; font-weight: 700; color: #F5F7FA; margin-top: 3px;">Turbine Telemetry</div>
            <div style="font-size: 10px; color: #8B98A8; margin-top: 2px;">7 Sensors · 60s Rate</div>
        </div>
        <!-- Arrow -->
        <div style="color: #0066FF; font-weight: bold; font-size: 14px;">→</div>
        <!-- Step 2 -->
        <div style="flex: 1; min-width: 140px; background-color: #161E27; border: 1px solid #26313D; border-radius: 6px; padding: 10px 12px; text-align: center;">
            <div style="font-size: 9.5px; font-weight: 700; color: #00A6A6; letter-spacing: 0.08em; text-transform: uppercase;">02 / PREPROCESS</div>
            <div style="font-size: 12px; font-weight: 700; color: #F5F7FA; margin-top: 3px;">Split & Scale</div>
            <div style="font-size: 10px; color: #8B98A8; margin-top: 2px;">70/15/15 · Train-Fit</div>
        </div>
        <!-- Arrow -->
        <div style="color: #0066FF; font-weight: bold; font-size: 14px;">→</div>
        <!-- Step 3 -->
        <div style="flex: 1; min-width: 140px; background-color: #161E27; border: 1px solid #26313D; border-radius: 6px; padding: 10px 12px; text-align: center;">
            <div style="font-size: 9.5px; font-weight: 700; color: #00A6A6; letter-spacing: 0.08em; text-transform: uppercase;">03 / WINDOWING</div>
            <div style="font-size: 12px; font-weight: 700; color: #F5F7FA; margin-top: 3px;">Sliding Window</div>
            <div style="font-size: 10px; color: #8B98A8; margin-top: 2px;">W=60 · Horizon=10</div>
        </div>
        <!-- Arrow -->
        <div style="color: #0066FF; font-weight: bold; font-size: 14px;">→</div>
        <!-- Step 4 -->
        <div style="flex: 1.1; min-width: 150px; background-color: rgba(0, 102, 255, 0.08); border: 1px solid #0066FF; border-radius: 6px; padding: 10px 12px; text-align: center;">
            <div style="font-size: 9.5px; font-weight: 700; color: #70A9FF; letter-spacing: 0.08em; text-transform: uppercase;">04 / DEEP LEARNING</div>
            <div style="font-size: 12px; font-weight: 700; color: #FFFFFF; margin-top: 3px;">Transformer Core</div>
            <div style="font-size: 10px; color: #70A9FF; margin-top: 2px;">MHA + Pos Encoding</div>
        </div>
        <!-- Arrow -->
        <div style="color: #0066FF; font-weight: bold; font-size: 14px;">→</div>
        <!-- Step 5 Dual Heads -->
        <div style="flex: 1.2; min-width: 160px; display: flex; flex-direction: column; gap: 6px;">
            <div style="background-color: #161E27; border: 1px solid #22C55E; border-radius: 4px; padding: 6px 10px; text-align: left;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; font-weight: 700; color: #4ADE80;">Forecasting Head</span>
                    <span style="font-size: 9px; color: #8B98A8; font-family: monospace;">(10, 7)</span>
                </div>
            </div>
            <div style="background-color: #161E27; border: 1px solid #F59E0B; border-radius: 4px; padding: 6px 10px; text-align: left;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; font-weight: 700; color: #FBBF24;">Anomaly Detector</span>
                    <span style="font-size: 9px; color: #8B98A8; font-family: monospace;">LSTM-AE</span>
                </div>
            </div>
        </div>
    </div>
</div>
"""
    return textwrap.dedent(raw_html).strip()


def render_transformer_architecture_html() -> str:
    """Render the Transformer Encoder visual architecture flow diagram."""
    raw_html = """
<div style="background-color: #111820; border: 1px solid #26313D; border-radius: 8px; padding: 18px; margin-top: 10px;">
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.1em; color: #00A6A6; text-transform: uppercase; margin-bottom: 14px;">
        Transformer Encoder Computational Graph
    </div>
    <div style="display: flex; flex-direction: column; gap: 8px; max-width: 480px; margin: 0 auto;">
        <div style="background-color: #161E27; border: 1px solid #26313D; border-radius: 6px; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 12px; font-weight: 600; color: #F5F7FA;">Multivariate Sequence Input</span>
            <span style="font-family: monospace; font-size: 11px; color: #8B98A8;">(Batch, 60, 7)</span>
        </div>
        <div style="text-align: center; color: #0066FF; font-size: 12px;">↓</div>
        <div style="background-color: #161E27; border: 1px solid #26313D; border-radius: 6px; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 12px; font-weight: 600; color: #F5F7FA;">Linear Feature Projection</span>
                <span style="font-size: 10px; color: #8B98A8; display: block;">Dense(d_model=64)</span>
            </div>
            <span style="font-family: monospace; font-size: 11px; color: #8B98A8;">(Batch, 60, 64)</span>
        </div>
        <div style="text-align: center; color: #0066FF; font-size: 12px;">↓ + Sinusoidal PE</div>
        <div style="background-color: rgba(0, 102, 255, 0.08); border: 1px solid #0066FF; border-radius: 6px; padding: 12px 14px;">
            <div style="font-size: 10.5px; font-weight: 700; color: #70A9FF; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">
                Encoder Block (x2 Stacked Layers)
            </div>
            <div style="display: flex; flex-direction: column; gap: 6px;">
                <div style="background-color: #111820; border: 1px solid #26313D; border-radius: 4px; padding: 6px 10px; font-size: 11.5px; display: flex; justify-content: space-between;">
                    <span>Multi-Head Self-Attention</span>
                    <span style="font-family: monospace; color: #00A6A6;">4 Heads · d_k=16</span>
                </div>
                <div style="background-color: #111820; border: 1px solid #26313D; border-radius: 4px; padding: 6px 10px; font-size: 11.5px; display: flex; justify-content: space-between;">
                    <span>Pre-Layer Normalization & Residual</span>
                    <span style="font-family: monospace; color: #8B98A8;">LayerNorm(x + MHA)</span>
                </div>
                <div style="background-color: #111820; border: 1px solid #26313D; border-radius: 4px; padding: 6px 10px; font-size: 11.5px; display: flex; justify-content: space-between;">
                    <span>Feed-Forward Network (FFN)</span>
                    <span style="font-family: monospace; color: #00A6A6;">Dense(128) → ReLU → Dense(64)</span>
                </div>
                <div style="background-color: #111820; border: 1px solid #26313D; border-radius: 4px; padding: 6px 10px; font-size: 11.5px; display: flex; justify-content: space-between;">
                    <span>Dropout & LayerNorm Residual</span>
                    <span style="font-family: monospace; color: #8B98A8;">p=0.1 · LayerNorm</span>
                </div>
            </div>
        </div>
        <div style="text-align: center; color: #0066FF; font-size: 12px;">↓ Global Temporal Pooling / Flatten</div>
        <div style="background-color: #161E27; border: 1px solid #26313D; border-radius: 6px; padding: 8px 12px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 12px; font-weight: 600; color: #F5F7FA;">Direct Multi-Step Projection Head</span>
                <span style="font-size: 10px; color: #8B98A8; display: block;">Dense(10 × 7 = 70) → Reshape</span>
            </div>
            <span style="font-family: monospace; font-size: 11px; color: #22C55E;">(Batch, 10, 7)</span>
        </div>
    </div>
</div>
"""
    return textwrap.dedent(raw_html).strip()


def render_empty_state(title: str, message: str, command: str = "python scripts/train.py --model transformer"):
    """Render an industrial empty state box with CLI remediation instructions."""
    raw_html = f"""
<div style="background-color: #111820; border: 1px dashed #26313D; border-radius: 8px; padding: 32px 24px; text-align: center; margin: 16px 0;">
    <div style="font-size: 24px; color: #F59E0B; margin-bottom: 8px;">☵</div>
    <div style="font-size: 15px; font-weight: 700; color: #F5F7FA; text-transform: uppercase; letter-spacing: 0.06em;">
        {title}
    </div>
    <div style="font-size: 12.5px; color: #8B98A8; margin-top: 6px; max-width: 500px; margin-left: auto; margin-right: auto;">
        {message}
    </div>
    <div style="margin-top: 14px;">
        <code style="background-color: #161E27; border: 1px solid #26313D; color: #38D9D9; padding: 6px 14px; border-radius: 4px; font-size: 12px; font-family: monospace;">
            {command}
        </code>
    </div>
</div>
"""
    st.markdown(textwrap.dedent(raw_html).strip(), unsafe_allow_html=True)
