"""Industrial AI Theme - Tokens, Typography, and Streamlit Dark Industrial Styles.

Color Palette:
- Dark Charcoal Background: #0B0F14
- Surface Base: #111820
- Elevated Surface / Card: #161E27
- Border / Outline: #26313D
- Border Subtle: #1A232E
- Primary Accent (Siemens Blue): #0066FF
- Secondary Accent (AI Cyan / Petrol): #00A6A6
- Success: #22C55E
- Warning: #F59E0B
- Danger: #EF4444
- Text Primary: #F5F7FA
- Text Muted: #8B98A8
- Text Dim: #505F70
"""

import streamlit as st

# Color Tokens
COLOR_BG = "#0B0F14"
COLOR_SURFACE = "#111820"
COLOR_SURFACE_ELEVATED = "#161E27"
COLOR_SURFACE_HOVER = "#1C2633"
COLOR_BORDER = "#26313D"
COLOR_BORDER_SUBTLE = "#1A232E"
COLOR_PRIMARY = "#0066FF"
COLOR_SECONDARY = "#00A6A6"
COLOR_SUCCESS = "#22C55E"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#EF4444"
COLOR_TEXT_PRIMARY = "#F5F7FA"
COLOR_TEXT_MUTED = "#8B98A8"
COLOR_TEXT_DIM = "#505F70"

INDUSTRIAL_CSS = f"""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global reset and base typography */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: {COLOR_TEXT_PRIMARY};
        background-color: {COLOR_BG};
    }}

    /* Streamlit Main App Container */
    .stApp {{
        background-color: {COLOR_BG};
    }}

    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 98% !important;
    }}

    /* Hide Streamlit Header, Toolbar, and Footer */
    #MainMenu, header[data-testid="stHeader"], footer {{
        visibility: hidden;
        height: 0%;
        display: none !important;
    }}

    /* Compact Sidebar */
    section[data-testid="stSidebar"] {{
        width: 250px !important;
        min-width: 250px !important;
        max-width: 250px !important;
        background-color: {COLOR_SURFACE} !important;
        border-right: 1px solid {COLOR_BORDER} !important;
        padding-top: 1rem !important;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding: 1rem 0.85rem !important;
    }}

    /* Custom Navigation Radio in Sidebar */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > label {{
        display: none !important;
    }}

    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{
        display: flex;
        flex-direction: column;
        gap: 3px;
    }}

    /* Hide standard radio circle */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {{
        background: transparent;
        border: 1px solid transparent;
        border-left: 3px solid transparent;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 0;
        cursor: pointer;
        transition: all 0.15s ease-in-out;
        color: {COLOR_TEXT_MUTED};
        font-size: 12.5px;
        font-weight: 500;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        width: 100%;
    }}

    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label input[type="radio"] {{
        display: none !important;
    }}

    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {{
        width: 100%;
    }}

    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
        background-color: rgba(255, 255, 255, 0.035);
        color: {COLOR_TEXT_PRIMARY};
    }}

    /* Active navigation item */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
        background-color: rgba(0, 102, 255, 0.12) !important;
        border-left: 3px solid {COLOR_PRIMARY} !important;
        border-radius: 4px;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}

    /* Common Card Styles */
    .ind-card {{
        background-color: {COLOR_SURFACE_ELEVATED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }}

    .ind-card-sm {{
        background-color: {COLOR_SURFACE_ELEVATED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 12px 14px;
    }}

    .kpi-card {{
        background-color: {COLOR_SURFACE_ELEVATED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        transition: border-color 0.2s ease;
    }}
    .kpi-card:hover {{
        border-color: #384656;
    }}

    .kpi-label {{
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {COLOR_TEXT_MUTED};
        margin-bottom: 6px;
    }}

    .kpi-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: {COLOR_TEXT_PRIMARY};
        line-height: 1.15;
        margin-bottom: 6px;
    }}

    .kpi-subtext {{
        font-size: 11.5px;
        color: {COLOR_TEXT_MUTED};
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .status-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: {COLOR_SUCCESS};
        box-shadow: 0 0 6px {COLOR_SUCCESS};
    }}

    .status-dot-amber {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: {COLOR_WARNING};
        box-shadow: 0 0 6px {COLOR_WARNING};
    }}

    .status-dot-red {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: {COLOR_DANGER};
        box-shadow: 0 0 6px {COLOR_DANGER};
    }}

    /* Top Header Bar */
    .top-header-bar {{
        background-color: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }}

    .header-eyebrow {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {COLOR_SECONDARY};
        margin-bottom: 2px;
    }}

    .header-title {{
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: {COLOR_TEXT_PRIMARY};
        margin: 0;
        line-height: 1.2;
    }}

    .header-badges {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }}

    .ind-badge {{
        background-color: {COLOR_SURFACE_ELEVATED};
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 500;
        color: {COLOR_TEXT_MUTED};
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}

    .ind-badge-primary {{
        background-color: rgba(0, 102, 255, 0.12);
        border: 1px solid rgba(0, 102, 255, 0.3);
        color: #70A9FF;
    }}

    .ind-badge-secondary {{
        background-color: rgba(0, 166, 166, 0.12);
        border: 1px solid rgba(0, 166, 166, 0.3);
        color: #38D9D9;
    }}

    .ind-badge-success {{
        background-color: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ADE80;
    }}

    /* Form Controls & Streamlit Overrides */
    div[data-baseweb="select"] > div {{
        background-color: {COLOR_SURFACE_ELEVATED} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 6px !important;
        color: {COLOR_TEXT_PRIMARY} !important;
    }}

    div[data-baseweb="select"] * {{
        color: {COLOR_TEXT_PRIMARY} !important;
    }}

    div[data-baseweb="popover"] ul {{
        background-color: {COLOR_SURFACE_ELEVATED} !important;
        border: 1px solid {COLOR_BORDER} !important;
    }}

    div[data-baseweb="popover"] li {{
        background-color: {COLOR_SURFACE_ELEVATED} !important;
        color: {COLOR_TEXT_PRIMARY} !important;
    }}

    div[data-baseweb="popover"] li:hover {{
        background-color: {COLOR_SURFACE_HOVER} !important;
    }}

    /* Slider Styling */
    div[data-testid="stSlider"] {{
        padding: 0.2rem 0;
    }}

    /* Buttons */
    div[data-testid="stButton"] button {{
        background-color: {COLOR_SURFACE_ELEVATED};
        color: {COLOR_TEXT_PRIMARY};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        font-weight: 500;
        font-size: 13px;
        padding: 6px 14px;
        transition: all 0.15s ease;
    }}

    div[data-testid="stButton"] button:hover {{
        background-color: {COLOR_SURFACE_HOVER};
        border-color: {COLOR_PRIMARY};
        color: #FFFFFF;
    }}

    /* Tables & DataFrames */
    div[data-testid="stDataFrame"] {{
        background-color: {COLOR_SURFACE_ELEVATED} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 6px !important;
    }}

    /* Divider */
    hr {{
        border: none;
        height: 1px;
        background-color: {COLOR_BORDER};
        margin: 20px 0;
    }}

    /* Scrollbars */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLOR_BG};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLOR_BORDER};
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #384656;
    }}
</style>
"""


def apply_industrial_theme():
    """Inject custom CSS for dark industrial enterprise control center."""
    st.markdown(INDUSTRIAL_CSS, unsafe_allow_html=True)
