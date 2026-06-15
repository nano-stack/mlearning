"""
MLearning — La app definitiva para aprender Machine Learning.
Punto de entrada principal (Streamlit).
"""
import streamlit as st
from pathlib import Path

# ── Page config (debe ir primero) ────────────────────────────────────────────
st.set_page_config(
    page_title="MLearning",
    page_icon="assets/icons/logo.png" if Path("assets/icons/logo.png").exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS global ────────────────────────────────────────────────────────────────
with open("assets/styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Hover override injected after Streamlit's own styles to win specificity race
st.markdown("""
<style>
.stButton > button:hover,
.stButton > button:hover:focus,
div[data-testid="stBaseButton-secondary"]:hover,
div[data-testid="stBaseButton-primary"]:hover,
section[data-testid] .stButton > button:hover {
    background-color: #f0a500 !important;
    background: #f0a500 !important;
    color: #0d1b2a !important;
    border-color: #f0a500 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(240,165,0,0.35) !important;
}

/* Tab spacing */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px !important;
    padding-bottom: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 24px !important;
    font-size: 14px !important;
    letter-spacing: 0.01em !important;
}
</style>
""", unsafe_allow_html=True)

# ── Estado de sesión ──────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page":        "landing",      # landing | select_type | select_model | data | params | results
        "algo_type":   None,           # "supervisado" | "no_supervisado"
        "model_key":   None,           # "linear_regression" | "logistic_regression" | "svm" | "kmeans" | "pca"
        "df":          None,
        "X":           None,
        "y":           None,
        "feature_cols": [],
        "target_col":  None,
        "params":      {},
        "result":      None,
        "plots":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Router ────────────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "landing":
    from pages.landing     import render; render()
elif page == "select_type":
    from pages.select_type import render; render()
elif page == "select_model":
    from pages.select_model import render; render()
elif page == "data":
    from pages.data_loader import render; render()
elif page == "params":
    from pages.hyperparams import render; render()
elif page == "results":
    from pages.results     import render; render()
