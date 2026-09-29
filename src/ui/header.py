"""Industrial Header Component.

Provides a compact, high-density top bar displaying:
- Left: Module Eyebrow, Main Title (24-28px), and brief technical description
- Right: Industrial status badges (SYSTEM ONLINE, TensorFlow 2.x, CPU/GPU, Unit #TF-04A)
"""

import textwrap
import streamlit as st
from src.ui.navigation import NavItem


def render_header(item: NavItem, custom_status: str = "SYSTEM ONLINE"):
    """Render top enterprise header."""
    raw_html = f"""
<div class="top-header-bar">
    <div>
        <div class="header-eyebrow">{item.eyebrow}</div>
        <div class="header-title">{item.title}</div>
        <div style="font-size: 12px; color: #8B98A8; margin-top: 3px;">
            {item.description}
        </div>
    </div>
    <div class="header-badges">
        <span class="ind-badge ind-badge-success">
            <span class="status-dot"></span>
            {custom_status}
        </span>
        <span class="ind-badge ind-badge-primary">
            <span style="font-weight: 700; color: #0066FF;">TF</span>
            TensorFlow 2.x
        </span>
        <span class="ind-badge ind-badge-secondary">
            <span style="font-size: 11px;">⚙</span>
            Turbofan #TF-04A
        </span>
        <span class="ind-badge">
            <span style="color: #8B98A8;">Mode:</span>
            <span style="color: #F5F7FA; font-weight: 600;">Multivariate</span>
        </span>
    </div>
</div>
"""
    st.markdown(textwrap.dedent(raw_html).strip(), unsafe_allow_html=True)
