"""Application entry point: global configuration and routing.

This deployed variant has two pages only. They are registered explicitly with
:func:`st.navigation` using ``position="hidden"`` so the default Streamlit page
menu never appears. The Dashboard renders its own fixed sidebar for its five
internal views.

The four mandatory datasets are bundled in ``dataset_input`` and the analysis is
always run with K = 5; the Beranda button loads the cached result and switches
straight to the Dashboard.

Flow: Beranda -> Dashboard.
"""

from __future__ import annotations

import streamlit as st

from src.config import FIXED_K

st.set_page_config(
    page_title="Klasterisasi Pelabuhan Impor Tiongkok",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "n_clusters": FIXED_K,
    "analysis_result": None,
    "processing_error": None,
    "dashboard_view": "Ringkasan",
    "pipeline_summary": None,
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
BASE_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800&display=swap');

/* ─── Base ─────────────────────────────────────────────────────────── */
html, body, [class*='css'],
[data-testid='stAppViewContainer'],
[data-testid='stMain'],
[data-testid='stMainBlockContainer'] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

.stApp, [data-testid='stAppViewContainer'] {
    background: #F0F2F8;
    color: #1A1C2E;
}

header[data-testid='stHeader'] { display: none !important; height: 0 !important; min-height: 0 !important; }
div[data-testid='stDecoration'], [data-testid='stToolbar'] { display: none !important; height: 0 !important; }

.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 3rem !important;
    max-width: 1500px;
}

/* ─── Topbar ────────────────────────────────────────────────────────── */
.app-topbar {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(226,229,241,0.8);
    border-radius: 14px;
    padding: 14px 22px;
    margin: 0 0 24px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    box-shadow: 0 2px 10px rgba(79,107,237,0.07);
}
.app-topbar-title  { font-size: 17px; font-weight: 700; color: #1A1C2E; line-height: 1.4; }
.app-topbar-subtitle { font-size: 12px; color: #6B7280; white-space: nowrap; }

/* ─── Metric cards ──────────────────────────────────────────────────── */
[data-testid='stMetric'] {
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(226,229,241,0.8);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(79,107,237,0.06);
}
[data-testid='stMetricLabel'] p { font-size: 13px !important; color: #6B7280 !important; font-weight: 500 !important; }
[data-testid='stMetricValue']   { font-size: 26px !important; color: #1A1C2E !important; font-weight: 700 !important; }
div[data-testid='stVerticalBlockBorderWrapper'] { border-radius: 14px; }

/* ─── Hero panel (Beranda) ──────────────────────────────────────────── */
.hero-panel {
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(226,229,241,0.7);
    border-radius: 20px;
    padding: clamp(30px, 5vw, 64px) clamp(24px, 6vw, 72px);
    text-align: center;
    margin: 0 0 24px 0;
    box-shadow: 0 4px 20px rgba(79,107,237,0.08);
}
.hero-title {
    color: #1A1C2E;
    font-size: clamp(22px, 2.6vw, 38px);
    line-height: 1.25;
    font-weight: 800;
    margin: 0 auto 14px auto;
    max-width: 900px;
    letter-spacing: -0.3px;
}
.hero-subtitle {
    display: inline-block;
    background: rgba(79,107,237,0.10);
    color: #4F6BED;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 20px;
}
.hero-copy {
    color: #4B5563;
    font-size: 15.5px;
    line-height: 1.75;
    max-width: 830px;
    margin: 0 auto;
}

/* ─── Section cards (Beranda step cards) ────────────────────────────── */
.section-card {
    background: rgba(255,255,255,0.90);
    border: 1px solid rgba(226,229,241,0.8);
    border-radius: 16px;
    padding: 22px 24px;
    height: 100%;
    box-shadow: 0 2px 10px rgba(79,107,237,0.06);
}
.section-card h4 {
    margin: 0 0 10px 0;
    font-size: 15px;
    font-weight: 700;
    color: #1A1C2E;
    letter-spacing: -0.1px;
}
.section-card p {
    margin: 0;
    color: #4B5563;
    font-size: 14px;
    line-height: 1.65;
}

/* ─── Status badges ─────────────────────────────────────────────────── */
.file-ok      { color: #16A34A; font-weight: 600; }
.file-missing { color: #DC2626; font-weight: 600; }

/* ─── Caption / small text ──────────────────────────────────────────── */
[data-testid='stCaptionContainer'] p,
.stCaption p {
    font-size: 13px !important;
    color: #6B7280 !important;
    line-height: 1.6 !important;
}

/* ─── Buttons ───────────────────────────────────────────────────────── */
[data-testid='stBaseButton-primary'] > button,
button[kind='primary'] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 22px !important;
    box-shadow: 0 2px 8px rgba(79,107,237,0.18) !important;
}
[data-testid='stBaseButton-secondary'] > button,
button[kind='secondary'] {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* ─── Dataframes & tables ───────────────────────────────────────────── */
[data-testid='stDataFrame'] > div {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(226,229,241,0.9) !important;
    box-shadow: 0 1px 6px rgba(79,107,237,0.05);
}

/* ─── Expander ──────────────────────────────────────────────────────── */
[data-testid='stExpander'] {
    border-radius: 12px !important;
    border: 1px solid rgba(226,229,241,0.8) !important;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(79,107,237,0.04);
}

/* ─── Alerts / info / warning ───────────────────────────────────────── */
[data-testid='stAlert'] {
    border-radius: 12px !important;
}

/* ─── File uploader ─────────────────────────────────────────────────── */
[data-testid='stFileUploader'] {
    border-radius: 12px !important;
}
[data-testid='stFileUploader'] > section {
    border-radius: 12px !important;
    border: 1.5px dashed rgba(79,107,237,0.30) !important;
    background: rgba(79,107,237,0.03) !important;
}

/* ─── Slider ────────────────────────────────────────────────────────── */
[data-testid='stSlider'] {
    padding: 4px 0;
}

/* ─── Selectbox / radio ─────────────────────────────────────────────── */
[data-baseweb='select'] > div {
    border-radius: 10px !important;
}
[data-testid='stSelectbox'] {
    font-size: 14px !important;
}

/* ─── Progress bar ──────────────────────────────────────────────────── */
[data-testid='stProgress'] > div > div {
    border-radius: 6px !important;
}

/* ─── Status widget ─────────────────────────────────────────────────── */
[data-testid='stStatus'] {
    border-radius: 12px !important;
    border: 1px solid rgba(226,229,241,0.8) !important;
}

/* ─── Divider ───────────────────────────────────────────────────────── */
hr {
    border-color: rgba(226,229,241,0.8) !important;
    margin: 1.5rem 0 !important;
}

/* ─── Sidebar hidden by default ─────────────────────────────────────── */
[data-testid='stSidebar'] { display: none !important; }
[data-testid='stSidebarCollapsedControl'] { display: none !important; }

#MainMenu, footer { visibility: hidden; }
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page registration (hidden navigation)
# ---------------------------------------------------------------------------
pages = [
    st.Page("pages/1_Beranda.py", title="Beranda", default=True),
    st.Page("pages/2_Dashboard.py", title="Dashboard"),
]

navigation = st.navigation(pages, position="hidden")
navigation.run()
