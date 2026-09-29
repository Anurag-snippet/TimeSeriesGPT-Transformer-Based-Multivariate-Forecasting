"""Industrial Sidebar Component.

Includes:
- Enterprise Brand Identity (SVG turbine/neural logo, INDUSTRIAL AI wordmark)
- Compact Navigation Menu with active states
- System Status Telemetry indicators (Pipeline Online, TensorFlow Ready, Dataset Loaded)
- Industrial Portfolio metadata
"""

import textwrap
import streamlit as st

from src.ui.navigation import NAV_ITEMS, NavItem


# Industrial Turbine / Neural Network SVG Logo
LOGO_SVG = """
<svg width="34" height="34" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="36" height="36" rx="8" fill="#161E27" stroke="#26313D" stroke-width="1.2"/>
  <circle cx="18" cy="18" r="11" stroke="#0066FF" stroke-width="1.8" stroke-dasharray="3 2"/>
  <circle cx="18" cy="18" r="6" stroke="#00A6A6" stroke-width="1.6"/>
  <circle cx="18" cy="18" r="2.5" fill="#0066FF"/>
  <line x1="18" y1="3" x2="18" y2="7" stroke="#0066FF" stroke-width="2" stroke-linecap="round"/>
  <line x1="18" y1="29" x2="18" y2="33" stroke="#0066FF" stroke-width="2" stroke-linecap="round"/>
  <line x1="3" y1="18" x2="7" y2="18" stroke="#00A6A6" stroke-width="2" stroke-linecap="round"/>
  <line x1="29" y1="18" x2="33" y2="18" stroke="#00A6A6" stroke-width="2" stroke-linecap="round"/>
</svg>
"""


def render_sidebar() -> NavItem:
    """Render the industrial application sidebar and return the active NavItem."""
    with st.sidebar:
        # Brand Header
        brand_html = f"""
<div style="display: flex; align-items: center; gap: 12px; padding: 4px 6px 16px 6px; border-bottom: 1px solid #1E2833; margin-bottom: 16px;">
    <div>{LOGO_SVG}</div>
    <div>
        <div style="font-size: 13.5px; font-weight: 700; letter-spacing: 0.08em; color: #F5F7FA; text-transform: uppercase; line-height: 1.15;">
            Industrial AI
        </div>
        <div style="font-size: 9.5px; font-weight: 600; letter-spacing: 0.12em; color: #00A6A6; text-transform: uppercase; margin-top: 2px;">
            Time-Series Intelligence
        </div>
    </div>
</div>
"""
        st.markdown(textwrap.dedent(brand_html).strip(), unsafe_allow_html=True)

        # Section Label
        section_label_html = """
<div style="font-size: 10px; font-weight: 700; letter-spacing: 0.12em; color: #505F70; text-transform: uppercase; padding: 0 8px 8px 8px;">
    Platform Modules
</div>
"""
        st.markdown(textwrap.dedent(section_label_html).strip(), unsafe_allow_html=True)

        nav_labels = [item.label for item in NAV_ITEMS]
        nav_icons = {
            "Overview": "◈",
            "Data & Telemetry": "▥",
            "Forecasting": "↗",
            "Model Benchmarks": "☵",
            "Anomaly Detection": "◎",
            "Experiments": "◫",
            "Research": "▤",
        }

        # Formatted display with crisp technical symbols
        display_options = [f"{nav_icons.get(item.label, '•')}  {item.label}" for item in NAV_ITEMS]

        selected_display = st.radio(
            "Navigation",
            options=display_options,
            index=0,
            label_visibility="collapsed",
        )

        # Map back to NavItem
        selected_index = display_options.index(selected_display)
        selected_item = NAV_ITEMS[selected_index]

        # Spacer before bottom status
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

        # System Status Section
        status_html = """
<div style="background-color: #111820; border: 1px solid #1E2833; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
    <div style="font-size: 10px; font-weight: 700; letter-spacing: 0.1em; color: #8B98A8; text-transform: uppercase; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
        <span>System Status</span>
        <span style="font-size: 9px; color: #22C55E; font-weight: 600;">ACTIVE</span>
    </div>
    <div style="display: flex; flex-direction: column; gap: 7px;">
        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 11.5px; color: #D1D9E0;">
            <span style="display: flex; align-items: center; gap: 7px;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background-color: #22C55E; box-shadow: 0 0 5px #22C55E;"></span>
                Pipeline Online
            </span>
            <span style="font-size: 10px; color: #8B98A8; font-family: monospace;">v1.2</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 11.5px; color: #D1D9E0;">
            <span style="display: flex; align-items: center; gap: 7px;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background-color: #22C55E; box-shadow: 0 0 5px #22C55E;"></span>
                TensorFlow Ready
            </span>
            <span style="font-size: 10px; color: #8B98A8; font-family: monospace;">tf.data</span>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 11.5px; color: #D1D9E0;">
            <span style="display: flex; align-items: center; gap: 7px;">
                <span style="width: 6px; height: 6px; border-radius: 50%; background-color: #22C55E; box-shadow: 0 0 5px #22C55E;"></span>
                Dataset Loaded
            </span>
            <span style="font-size: 10px; color: #8B98A8; font-family: monospace;">12K</span>
        </div>
    </div>
    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #1A232E; font-size: 10.5px; color: #8B98A8; display: flex; justify-content: space-between;">
        <span>Execution Device</span>
        <span style="color: #00A6A6; font-family: monospace; font-weight: 600;">CPU / Colab GPU</span>
    </div>
</div>
"""
        st.markdown(textwrap.dedent(status_html).strip(), unsafe_allow_html=True)

    return selected_item
