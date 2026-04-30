"""
Streamlit app — interactive demo for the loan-approval system.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_generation import (  # noqa: E402
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERICAL_FEATURES,
)
from src.preprocessing import get_feature_names  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — sidebar collapsed since we put all inputs on the main page
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Loan Approval ML",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════
PALETTE = {
    "primary": "#6366F1",
    "primary_dark": "#4F46E5",
    "primary_soft": "#EEF2FF",
    "success": "#10B981",
    "success_soft": "#D1FAE5",
    "warning": "#F59E0B",
    "warning_soft": "#FEF3C7",
    "danger": "#EF4444",
    "danger_soft": "#FEE2E2",
    "ink": "#0F172A",
    "ink_soft": "#1E293B",
    "muted": "#64748B",
    "border": "#E2E8F0",
    "surface": "#FFFFFF",
    "bg": "#F8FAFC",
    "bg_alt": "#F1F5F9",
}

PLOTLY_LAYOUT = {
    "font": {"family": "Inter, -apple-system, system-ui, sans-serif",
             "color": PALETTE["ink"], "size": 13},
    "title": {"font": {"size": 16, "color": PALETTE["ink"], "family": "Inter"},
              "x": 0, "xanchor": "left"},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "colorway": [PALETTE["primary"], PALETTE["success"], PALETTE["warning"],
                 PALETTE["danger"], "#8B5CF6", "#EC4899"],
    "xaxis": {"gridcolor": PALETTE["border"], "zerolinecolor": PALETTE["border"],
              "linecolor": PALETTE["border"], "tickfont": {"color": PALETTE["muted"]}},
    "yaxis": {"gridcolor": PALETTE["border"], "zerolinecolor": PALETTE["border"],
              "linecolor": PALETTE["border"], "tickfont": {"color": PALETTE["muted"]}},
    "hoverlabel": {"bgcolor": PALETTE["ink"], "bordercolor": PALETTE["ink"],
                   "font": {"color": "white", "family": "Inter", "size": 12}},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
}


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">

    <style>
    :root {{
        --primary: {PALETTE["primary"]};
        --primary-dark: {PALETTE["primary_dark"]};
        --primary-soft: {PALETTE["primary_soft"]};
        --success: {PALETTE["success"]};
        --success-soft: {PALETTE["success_soft"]};
        --warning: {PALETTE["warning"]};
        --warning-soft: {PALETTE["warning_soft"]};
        --danger: {PALETTE["danger"]};
        --danger-soft: {PALETTE["danger_soft"]};
        --ink: {PALETTE["ink"]};
        --ink-soft: {PALETTE["ink_soft"]};
        --muted: {PALETTE["muted"]};
        --border: {PALETTE["border"]};
        --surface: {PALETTE["surface"]};
        --bg: {PALETTE["bg"]};
        --bg-alt: {PALETTE["bg_alt"]};

        --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.04);
        --shadow-md: 0 4px 12px rgba(15, 23, 42, 0.06);
        --shadow-lg: 0 12px 32px rgba(15, 23, 42, 0.08);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;

        --ease: cubic-bezier(0.4, 0, 0.2, 1);
    }}

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
        color: var(--ink);
    }}
    .stApp {{
        background:
            radial-gradient(ellipse 1200px 800px at top right, rgba(99, 102, 241, 0.08), transparent 50%),
            radial-gradient(ellipse 1000px 700px at bottom left, rgba(16, 185, 129, 0.05), transparent 50%),
            var(--bg);
    }}
    .main .block-container {{
        max-width: 1320px;
        padding-top: 1rem;
        padding-bottom: 4rem;
    }}

    /* ──── FIX #1: Hide the black bar at the top properly ──── */
    [data-testid="stHeader"] {{
        background: transparent !important;
        height: 0 !important;
        display: none !important;
    }}
    [data-testid="stToolbar"] {{ display: none !important; }}
    [data-testid="stDecoration"] {{ display: none !important; }}
    .stDeployButton {{ display: none !important; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    h1 {{
        font-weight: 800 !important;
        letter-spacing: -0.035em;
        font-size: 2.4rem !important;
        line-height: 1.1 !important;
        color: var(--ink);
        margin-bottom: 0.25rem !important;
    }}
    h2, h3 {{
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: var(--ink);
    }}
    h2 {{ font-size: 1.5rem !important; }}
    h3 {{ font-size: 1.15rem !important; }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: var(--muted) !important;
        font-size: 0.9rem !important;
    }}

    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ──── Hero ──── */
    .hero {{
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        color: white;
        border-radius: var(--radius-xl);
        padding: 1.75rem 2.25rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
        animation: fadeUp 0.5s var(--ease);
    }}
    .hero::before {{
        content: "";
        position: absolute; inset: 0;
        background:
            radial-gradient(circle at 90% 10%, rgba(255,255,255,0.2), transparent 40%),
            radial-gradient(circle at 10% 90%, rgba(255,255,255,0.1), transparent 40%);
        pointer-events: none;
    }}
    .hero h1 {{
        color: white !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.4rem !important;
    }}
    .hero p {{
        color: rgba(255,255,255,0.92);
        font-size: 0.98rem;
        margin: 0;
        max-width: 720px;
        line-height: 1.5;
    }}

    /* ──── Inputs panel on main page ──── */
    .input-panel {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-sm);
    }}
    .input-section-label {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* Decision card */
    .decision-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 1.75rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        position: relative; overflow: hidden;
        transition: all 0.35s var(--ease);
    }}
    .decision-card.approved {{
        border-color: rgba(16, 185, 129, 0.4);
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.04), var(--surface) 60%);
    }}
    .decision-card.denied {{
        border-color: rgba(239, 68, 68, 0.4);
        background: linear-gradient(180deg, rgba(239, 68, 68, 0.04), var(--surface) 60%);
    }}
    .decision-card.borderline {{
        border-color: rgba(245, 158, 11, 0.4);
        background: linear-gradient(180deg, rgba(245, 158, 11, 0.04), var(--surface) 60%);
    }}
    .decision-label {{
        font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.12em; color: var(--muted);
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }}
    .decision-pill {{
        display: inline-flex; align-items: center; gap: 8px;
        padding: 10px 22px; border-radius: 999px;
        font-weight: 700; font-size: 1rem;
        letter-spacing: 0.02em;
        margin-top: 0.5rem;
    }}
    .pill-good {{ background: var(--success-soft); color: #065F46; }}
    .pill-warn {{ background: var(--warning-soft); color: #92400E; }}
    .pill-bad  {{ background: var(--danger-soft); color: #991B1B; }}

    /* Flag rows */
    .flag {{
        display: flex; align-items: flex-start; gap: 12px;
        padding: 12px 16px; margin-bottom: 8px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--muted);
        border-radius: var(--radius-md);
        font-size: 0.92rem;
        transition: all 0.2s var(--ease);
    }}
    .flag.good {{ border-left-color: var(--success); }}
    .flag.warn {{ border-left-color: var(--warning); }}
    .flag.bad  {{ border-left-color: var(--danger); }}
    .flag-icon {{ font-size: 1.1rem; line-height: 1.4; }}

    /* KPI strip */
    .kpi-row {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }}
    .kpi {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        transition: all 0.25s var(--ease);
    }}
    .kpi-label {{
        font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.08em; color: var(--muted);
        text-transform: uppercase;
    }}
    .kpi-value {{
        font-size: 1.5rem; font-weight: 800;
        color: var(--ink);
        font-feature-settings: "tnum";
        letter-spacing: -0.02em;
        margin-top: 4px;
    }}
    .kpi-sub {{ font-size: 0.78rem; color: var(--muted); margin-top: 2px; }}

    /* ──── FIX #4: Slider — make track indigo, value text BLACK (not red) ──── */
    .stSlider [data-baseweb="slider"] [role="slider"] {{
        background-color: var(--primary) !important;
        border: 2px solid white !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.35) !important;
    }}
    /* The filled track */
    .stSlider [data-baseweb="slider"] > div > div > div:first-child {{
        background: var(--primary) !important;
    }}
    /* The current value tooltip / number above thumb */
    .stSlider [data-testid="stThumbValue"] {{
        color: var(--ink) !important;
        background: transparent !important;
        font-weight: 600 !important;
    }}
    /* Min and max labels at slider ends */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {{
        color: var(--muted) !important;
        font-weight: 500 !important;
    }}
    /* Catch-all for any colored slider text */
    .stSlider span,
    .stSlider div[class*="StyledThumbValue"],
    .stSlider div[class*="StyledTickBar"] {{
        color: var(--ink) !important;
    }}

    /* Selectbox + toggle */
    .stSelectbox [data-baseweb="select"] > div {{
        border-radius: var(--radius-sm) !important;
        border-color: var(--border) !important;
    }}
    .stSelectbox [data-baseweb="select"] > div:focus-within {{
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: var(--bg-alt);
        padding: 4px;
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: var(--radius-sm) !important;
        padding: 8px 16px !important;
        height: auto !important;
        font-weight: 600 !important;
        color: var(--muted) !important;
        transition: all 0.2s var(--ease) !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(99, 102, 241, 0.08) !important;
        color: var(--primary) !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: var(--surface) !important;
        color: var(--ink) !important;
        box-shadow: var(--shadow-sm) !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
    .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}

    /* Plotly chart container */
    [data-testid="stPlotlyChart"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 0.75rem;
        box-shadow: var(--shadow-sm);
    }}

    /* ──── FIX #6: Smaller metric value font ──── */
    [data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 12px 16px;
        transition: all 0.25s var(--ease);
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em;
        color: var(--muted) !important;
        text-transform: uppercase;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: 0.7rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        font-feature-settings: "tnum";
        line-height: 1.3 !important;
    }}
    [data-testid="stMetricValue"] > div {{
        font-size: 1.25rem !important;
        font-weight: 700 !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--border);
    }}

    hr {{
        margin: 1.5rem 0 !important;
        border: none !important;
        border-top: 1px solid var(--border) !important;
    }}

    .section-h {{
        display: flex; align-items: center; gap: 10px;
        margin: 1.5rem 0 1rem;
    }}
    .section-h::before {{
        content: "";
        display: block;
        width: 4px; height: 20px;
        background: linear-gradient(180deg, var(--primary), var(--primary-dark));
        border-radius: 2px;
    }}
    .section-h h3 {{
        margin: 0 !important;
        font-size: 1.05rem !important;
    }}

    code {{
        font-family: 'JetBrains Mono', monospace !important;
        background: var(--bg-alt) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-size: 0.85em !important;
        color: var(--primary-dark) !important;
    }}

    /* Toggle */
    .stToggle [data-baseweb="toggle"] {{
        background: var(--bg-alt);
        padding: 6px 10px;
        border-radius: var(--radius-sm);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════
# ASSET LOADING
# ══════════════════════════════════════════════════════════════════════════
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "loan_data.csv"


@st.cache_resource
def load_artifacts():
    if not (MODELS_DIR / "best_model.pkl").exists():
        return None
    best_pipe = joblib.load(MODELS_DIR / "best_model.pkl")
    all_pipes = joblib.load(MODELS_DIR / "all_pipelines.pkl")
    with open(MODELS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    return {"best_pipe": best_pipe, "all_pipes": all_pipes, "metrics": metrics, "df": df}


@st.cache_resource
def get_shap_explainer(_pipe):
    clf = _pipe.named_steps["clf"]
    try:
        return shap.TreeExplainer(clf), True
    except Exception:
        return None, False


artifacts = load_artifacts()
if artifacts is None:
    from src.quick_train import quick_train

    st.markdown(
        """
        <div class='hero'>
            <h1>👋 First launch</h1>
            <p>Training models now (≈30–60 seconds). This happens once. Subsequent launches load instantly from disk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    progress_bar = st.progress(0.0, text="Starting…")
    status = st.empty()

    def _ui_progress(frac: float, msg: str):
        progress_bar.progress(min(max(frac, 0.0), 1.0), text=msg)
        status.caption(msg)

    try:
        artifacts = quick_train(ROOT, progress=_ui_progress)
    except Exception as e:
        st.error(f"❌ Training failed: `{type(e).__name__}: {e}`")
        st.stop()

    load_artifacts.clear()
    progress_bar.empty()
    status.empty()
    st.success("✅ Training complete.")
    st.balloons()

best_pipe = artifacts["best_pipe"]
all_pipes = artifacts["all_pipes"]
metrics = artifacts["metrics"]
df = artifacts["df"]
best_name = metrics["best_model"]
optimal_thresh = metrics["optimal_threshold"]


# ══════════════════════════════════════════════════════════════════════════
# HERO HEADER — FIX #5: badges removed
# ══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class='hero'>
        <h1>🏦 Loan Approval ML</h1>
        <p>Build a binary classification model from start to finish, compare different models, check how well their predicted probabilities match reality (calibration), 
        and use SHAP to explain the predictions. You can tweak the applicant inputs and see the prediction update instantly. Adjust the inputs and the prediction updates live.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════
# INPUT PANEL — FIX #2: inputs moved from sidebar to main page
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<div class='input-panel'>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; "
    "color: var(--muted); text-transform: uppercase; margin-bottom: 1rem;'>"
    "Applicant profile · adjust to update prediction</div>",
    unsafe_allow_html=True,
)

col_fin, col_credit, col_demo = st.columns(3, gap="large")

with col_fin:
    st.markdown("<div class='input-section-label'>💰 Financial</div>", unsafe_allow_html=True)
    annual_income = st.slider("Annual income ($)", 15_000, 200_000, 65_000, step=1_000, format="$%d")
    loan_amount = st.slider("Loan amount ($)", 5_000, 500_000, 80_000, step=1_000, format="$%d")
    savings_balance = st.slider("Savings balance ($)", 0, 200_000, 12_000, step=500, format="$%d")
    dti_ratio = st.slider("Debt-to-income ratio (%)", 1.0, 65.0, 28.0, step=0.5, format="%.1f%%")

with col_credit:
    st.markdown("<div class='input-section-label'>📊 Credit & Employment</div>", unsafe_allow_html=True)
    credit_score = st.slider("Credit score", 300, 850, 685, step=5)
    employment_years = st.slider("Employment (years)", 0, 40, 5)
    num_credit_lines = st.slider("Number of credit lines", 0, 25, 4)
    use_optimal_thresh = st.toggle(
        "Use cost-optimal threshold",
        value=False,
        help=f"Default 0.50 vs. cost-optimal {optimal_thresh:.2f} (FN penalized 3× FP).",
    )

with col_demo:
    st.markdown("<div class='input-section-label'>👤 Demographics</div>", unsafe_allow_html=True)
    age = st.slider("Age", 18, 75, 35)
    education = st.selectbox("Education", ["high_school", "some_college", "bachelors", "graduate"], index=2)
    home_ownership = st.selectbox("Home ownership", ["rent", "mortgage", "own", "other"], index=1)
    loan_purpose = st.selectbox(
        "Loan purpose",
        ["debt_consolidation", "home_improvement", "major_purchase", "medical", "education", "other"],
    )

st.markdown("</div>", unsafe_allow_html=True)


def build_input_row():
    return pd.DataFrame([{
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "dti_ratio": dti_ratio,
        "age": age,
        "num_credit_lines": num_credit_lines,
        "savings_balance": savings_balance,
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership,
        "education": education,
        "loan_to_income": round(loan_amount / annual_income, 3),
        "savings_to_loan": round(savings_balance / loan_amount, 3) if loan_amount > 0 else 0,
    }])


x_input = build_input_row()
threshold = optimal_thresh if use_optimal_thresh else 0.50
proba = best_pipe.predict_proba(x_input)[0, 1]
prediction = int(proba >= threshold)
all_probas = {name: pipe.predict_proba(x_input)[0, 1] for name, pipe in all_pipes.items()}


# ══════════════════════════════════════════════════════════════════════════
# TABS — FIX #3: only Predict, Models, Data (Explain & About removed)
# ══════════════════════════════════════════════════════════════════════════
tab_predict, tab_compare, tab_data = st.tabs(["🎯 Predict", "📊 Models", "📈 Data"])

# ───────────────────────────────  PREDICT  ────────────────────────────────
with tab_predict:
    left, right = st.columns([1, 1], gap="large")

    with left:
        gauge_color = (
            PALETTE["success"] if proba >= 0.70
            else PALETTE["warning"] if proba >= 0.50
            else PALETTE["danger"]
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 56, "color": PALETTE["ink"], "family": "Inter"}, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": PALETTE["muted"],
                         "tickfont": {"size": 11, "color": PALETTE["muted"]}, "ticklen": 4},
                "bar": {"color": gauge_color, "thickness": 0.32},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.10)"},
                    {"range": [50, 70], "color": "rgba(245, 158, 11, 0.10)"},
                    {"range": [70, 100], "color": "rgba(16, 185, 129, 0.10)"},
                ],
                "threshold": {"line": {"color": PALETTE["ink"], "width": 3},
                              "thickness": 0.85, "value": threshold * 100},
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig.update_layout(
            height=300, margin=dict(l=20, r=20, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if prediction == 1:
            if proba >= 0.7:
                card_class, pill_class, icon, text = "approved", "pill-good", "✓", "APPROVED"
            else:
                card_class, pill_class, icon, text = "borderline", "pill-warn", "✓", "APPROVED · BORDERLINE"
        else:
            card_class, pill_class, icon, text = "denied", "pill-bad", "✕", "DENIED"

        st.markdown(
            f"""
            <div class='decision-card {card_class}'>
                <div class='decision-label'>Decision · threshold {threshold:.2f}</div>
                <div class='decision-pill {pill_class}'>{icon} {text}</div>
                <div style='margin-top: 12px; font-size: 0.85rem; color: var(--muted);'>
                    Probability: <strong style='color: var(--ink); font-feature-settings: "tnum";'>{proba:.1%}</strong>
                    &nbsp;·&nbsp; Margin: <strong style='color: var(--ink); font-feature-settings: "tnum";'>{(proba - threshold)*100:+.1f}pp</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("<div class='section-h'><h3>Quick read on this applicant</h3></div>", unsafe_allow_html=True)

        loan_to_income = loan_amount / annual_income
        monthly_loan_payment = loan_amount * 0.006

        flags = []
        if credit_score < 620:
            flags.append(("🚨", "bad", "Credit score below 620 — subprime range"))
        elif credit_score >= 740:
            flags.append(("✅", "good", "Excellent credit score (740+)"))
        if dti_ratio > 43:
            flags.append(("🚨", "bad", "DTI above 43% — Qualified Mortgage rule of thumb exceeded"))
        elif dti_ratio < 28:
            flags.append(("✅", "good", "Healthy DTI under 28%"))
        if loan_to_income > 4:
            flags.append(("🚨", "bad", f"Loan is {loan_to_income:.1f}× annual income — high"))
        if employment_years < 2:
            flags.append(("⚠️", "warn", "Less than 2 years employment — limited stability signal"))
        if savings_balance >= loan_amount * 0.2:
            flags.append(("✅", "good", "Savings ≥ 20% of loan — strong cushion"))
        if not flags:
            flags.append(("ℹ️", "warn", "No strong signals either way — model decides on the margins."))

        for ic, kind, msg in flags:
            st.markdown(
                f"<div class='flag {kind}'><div class='flag-icon'>{ic}</div><div>{msg}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Loan-to-income", f"{loan_to_income:.2f}×")
        c2.metric("Monthly payment", f"${monthly_loan_payment:,.0f}")
        c3.metric("Savings cushion",
                  f"{(savings_balance / loan_amount * 100):.0f}%" if loan_amount else "—")


# ────────────────────────────────  MODELS  ────────────────────────────────
with tab_compare:
    st.markdown("<div class='section-h'><h3>Live prediction comparison</h3></div>", unsafe_allow_html=True)
    st.caption("Each model's probability of approval for the current applicant. Drag the inputs above to watch them shift.")

    proba_df = pd.DataFrame({
        "Model": list(all_probas.keys()),
        "P(approve)": list(all_probas.values()),
    }).sort_values("P(approve)", ascending=True)

    bar_colors_live = [
        PALETTE["danger"] if v < 0.5 else PALETTE["warning"] if v < 0.7 else PALETTE["success"]
        for v in proba_df["P(approve)"]
    ]
    # FIX #7: explicit name on every trace, hover template fully labeled
    fig = go.Figure(go.Bar(
        x=proba_df["P(approve)"],
        y=proba_df["Model"],
        orientation="h",
        name="Probability of approval",
        marker=dict(color=bar_colors_live, line=dict(color="rgba(0,0,0,0.05)", width=1)),
        text=[f"{v:.1%}" for v in proba_df["P(approve)"]],
        textposition="outside",
        textfont=dict(size=12, family="Inter", color=PALETTE["ink"]),
        hovertemplate="<b>%{y}</b><br>Probability of approval: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title="Predicted probability of approval — by model",
        height=340, margin=dict(l=20, r=60, t=50, b=40),
        xaxis=dict(range=[0, 1.05], tickformat=".0%", title="Probability of approval"),
        yaxis=dict(title="Model"),
        bargap=0.4,
        showlegend=False,
    )
    fig = apply_theme(fig)
    fig.add_vline(x=threshold, line_dash="dash", line_color=PALETTE["ink"], line_width=2,
                  annotation_text=f"  threshold {threshold:.2f}", annotation_position="top",
                  annotation_font=dict(color=PALETTE["ink"], size=11))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-h'><h3>Test-set metrics</h3></div>", unsafe_allow_html=True)
    st.caption(f"Held-out test set, n = {metrics['n_test']:,}. Best per metric is highlighted.")

    rows = []
    for name, info in metrics["models"].items():
        m = info["metrics"]
        rows.append({
            "Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"], "Brier": m["brier"],
        })
    metrics_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    def highlight_best(s):
        is_best = s == s.max() if s.name != "Brier" else s == s.min()
        return [f"background-color: {PALETTE['success_soft']}; font-weight: 700; color: #065F46" if v else "" for v in is_best]

    styled = (
        metrics_df.style
        .format({c: "{:.4f}" for c in metrics_df.columns if c != "Model"})
        .apply(highlight_best, subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-h'><h3>Diagnostic plots</h3></div>", unsafe_allow_html=True)
    st.caption("Interactive — hover for details. Each model has consistent color across all plots.")

    model_names_sorted = list(all_probas.keys())
    palette_seq = [PALETTE["primary"], PALETTE["success"], PALETTE["warning"],
                   PALETTE["danger"], "#8B5CF6"]
    model_color_map = {name: palette_seq[i % len(palette_seq)] for i, name in enumerate(model_names_sorted)}

    if df is not None:
        from sklearn.metrics import roc_curve, precision_recall_curve
        from sklearn.calibration import calibration_curve
        from sklearn.model_selection import train_test_split

        feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
        X_full = df[feature_cols]
        y_full = df["approved"].values
        _, X_test_p, _, y_test_p = train_test_split(X_full, y_full, test_size=0.20, random_state=42, stratify=y_full)

        plot_col1, plot_col2 = st.columns(2)

        # ROC
        with plot_col1:
            fig = go.Figure()
            for name, pipe in all_pipes.items():
                proba_t = pipe.predict_proba(X_test_p)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_p, proba_t)
                auc = metrics["models"][name]["metrics"]["roc_auc"]
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
                    line=dict(color=model_color_map[name], width=2.5),
                    hovertemplate=f"<b>{name}</b><br>False Positive Rate: %{{x:.3f}}<br>True Positive Rate: %{{y:.3f}}<extra></extra>",
                ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random chance",
                line=dict(color=PALETTE["muted"], dash="dash", width=1.5),
                hovertemplate="Random chance baseline<extra></extra>",
            ))
            fig.update_layout(
                title="ROC Curves",
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                height=380, margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                            bgcolor="rgba(255,255,255,0.85)", bordercolor=PALETTE["border"], borderwidth=1,
                            title=None),
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # PR
        with plot_col2:
            fig = go.Figure()
            for name, pipe in all_pipes.items():
                proba_t = pipe.predict_proba(X_test_p)[:, 1]
                prec, rec, _ = precision_recall_curve(y_test_p, proba_t)
                pr_auc = metrics["models"][name]["metrics"]["pr_auc"]
                fig.add_trace(go.Scatter(
                    x=rec, y=prec, mode="lines", name=f"{name} (AUC {pr_auc:.3f})",
                    line=dict(color=model_color_map[name], width=2.5),
                    hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
                ))
            base = float(y_test_p.mean())
            fig.add_hline(y=base, line_dash="dash", line_color=PALETTE["muted"], line_width=1.5,
                          annotation_text=f"baseline {base:.2f}", annotation_position="bottom right",
                          annotation_font=dict(color=PALETTE["muted"], size=10))
            fig.update_layout(
                title="Precision-Recall Curves",
                xaxis_title="Recall", yaxis_title="Precision",
                height=380, margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(yanchor="bottom", y=0.02, xanchor="left", x=0.02,
                            bgcolor="rgba(255,255,255,0.85)", bordercolor=PALETTE["border"], borderwidth=1,
                            title=None),
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Calibration
        with plot_col1:
            fig = go.Figure()
            for name, pipe in all_pipes.items():
                proba_t = pipe.predict_proba(X_test_p)[:, 1]
                frac_pos, mean_pred = calibration_curve(y_test_p, proba_t, n_bins=10, strategy="quantile")
                fig.add_trace(go.Scatter(
                    x=mean_pred, y=frac_pos, mode="lines+markers", name=name,
                    line=dict(color=model_color_map[name], width=2),
                    marker=dict(size=7, line=dict(color="white", width=1.5)),
                    hovertemplate=f"<b>{name}</b><br>Mean predicted: %{{x:.3f}}<br>Fraction positive: %{{y:.3f}}<extra></extra>",
                ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                line=dict(color=PALETTE["muted"], dash="dash", width=1.5),
                hovertemplate="Perfect calibration<extra></extra>",
            ))
            fig.update_layout(
                title="Calibration / Reliability",
                xaxis_title="Mean predicted probability", yaxis_title="Fraction of positives",
                height=380, margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98,
                            bgcolor="rgba(255,255,255,0.85)", bordercolor=PALETTE["border"], borderwidth=1,
                            title=None),
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Score distribution
        with plot_col2:
            best_proba_test = best_pipe.predict_proba(X_test_p)[:, 1]
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=best_proba_test[y_test_p == 1], name="Approved (actual)",
                marker_color=PALETTE["success"], opacity=0.7, nbinsx=40,
                hovertemplate="Predicted probability: %{x:.2f}<br>Count: %{y}<extra>Approved (actual)</extra>",
            ))
            fig.add_trace(go.Histogram(
                x=best_proba_test[y_test_p == 0], name="Denied (actual)",
                marker_color=PALETTE["danger"], opacity=0.7, nbinsx=40,
                hovertemplate="Predicted probability: %{x:.2f}<br>Count: %{y}<extra>Denied (actual)</extra>",
            ))
            fig.add_vline(x=threshold, line_dash="dash", line_color=PALETTE["ink"], line_width=2,
                          annotation_text=f"  t={threshold:.2f}", annotation_position="top right",
                          annotation_font=dict(color=PALETTE["ink"], size=11))
            fig.update_layout(
                title=f"Score distribution — {best_name}",
                xaxis_title="Predicted probability of approval", yaxis_title="Count",
                barmode="overlay", height=380, margin=dict(l=50, r=20, t=50, b=50),
                legend=dict(yanchor="top", y=0.98, xanchor="center", x=0.5,
                            bgcolor="rgba(255,255,255,0.85)", bordercolor=PALETTE["border"], borderwidth=1,
                            orientation="h", title=None),
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-h'><h3>McNemar's test — pairwise comparison</h3></div>", unsafe_allow_html=True)
    st.caption("Are differences between models statistically significant on the test set? *** p<0.001 · ** p<0.01 · * p<0.05 · ns = not significant.")
    mc_rows = []
    for r in metrics["mcnemar_pairwise"]:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
        mc_rows.append({
            "Model A": r["model_a"], "Model B": r["model_b"],
            "χ²": f"{r['chi2']:.3f}", "p-value": f"{r['p_value']:.4f}",
            "Significant": sig,
        })
    st.dataframe(pd.DataFrame(mc_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────  DATA  ─────────────────────────────────
with tab_data:
    st.markdown("<div class='section-h'><h3>Dataset overview</h3></div>", unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset not found.")
    else:
        approval_rate = df["approved"].mean()
        n_features = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES) + len(ENGINEERED_FEATURES)
        imbalance = df["approved"].value_counts().min() / df["approved"].value_counts().max()
        st.markdown(
            f"""
            <div class='kpi-row'>
                <div class='kpi'>
                    <div class='kpi-label'>Total rows</div>
                    <div class='kpi-value'>{len(df):,}</div>
                    <div class='kpi-sub'>synthetic, seed=42</div>
                </div>
                <div class='kpi'>
                    <div class='kpi-label'>Features</div>
                    <div class='kpi-value'>{n_features}</div>
                    <div class='kpi-sub'>{len(NUMERICAL_FEATURES)} num · {len(CATEGORICAL_FEATURES)} cat · {len(ENGINEERED_FEATURES)} engineered</div>
                </div>
                <div class='kpi'>
                    <div class='kpi-label'>Approval rate</div>
                    <div class='kpi-value'>{approval_rate:.1%}</div>
                    <div class='kpi-sub'>positive class</div>
                </div>
                <div class='kpi'>
                    <div class='kpi-label'>Class balance</div>
                    <div class='kpi-value'>{imbalance:.2f}</div>
                    <div class='kpi-sub'>minor / major</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-h'><h3>Distribution by approval outcome</h3></div>", unsafe_allow_html=True)
        feat = st.selectbox("Feature", NUMERICAL_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES)

        # FIX #7: explicit hovertemplates so nothing reads "undefined"
        if feat in NUMERICAL_FEATURES + ENGINEERED_FEATURES:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df["approved"] == 1][feat], name="Approved",
                marker_color=PALETTE["success"], opacity=0.7, nbinsx=40,
                hovertemplate=f"{feat}: %{{x}}<br>Count: %{{y}}<extra>Approved</extra>",
            ))
            fig.add_trace(go.Histogram(
                x=df[df["approved"] == 0][feat], name="Denied",
                marker_color=PALETTE["danger"], opacity=0.7, nbinsx=40,
                hovertemplate=f"{feat}: %{{x}}<br>Count: %{{y}}<extra>Denied</extra>",
            ))
            fig.update_layout(
                title=f"Distribution of {feat} — approved vs. denied applicants",
                barmode="overlay", height=440,
                xaxis_title=feat, yaxis_title="Count",
                margin=dict(l=50, r=20, t=20, b=50),
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                            bgcolor="rgba(255,255,255,0.85)", bordercolor=PALETTE["border"], borderwidth=1,
                            title="Outcome"),
            )
        else:
            counts = df.groupby([feat, "approved"]).size().reset_index(name="count")
            counts["Outcome"] = counts["approved"].map({0: "Denied", 1: "Approved"})
            fig = go.Figure()
            for outcome_label, color in [("Approved", PALETTE["success"]), ("Denied", PALETTE["danger"])]:
                sub = counts[counts["Outcome"] == outcome_label]
                fig.add_trace(go.Bar(
                    x=sub[feat], y=sub["count"], name=outcome_label,
                    marker_color=color,
                    hovertemplate=f"{feat}: %{{x}}<br>Count: %{{y}}<extra>{outcome_label}</extra>",
                ))
            fig.update_layout(
                title=f"Counts of {feat} — approved vs. denied applicants",
                barmode="group", height=440,
                xaxis_title=feat, yaxis_title="Count",
                margin=dict(l=50, r=20, t=20, b=50),
                legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98,
                            bgcolor="rgba(255,255,255,0.85)", bordercolor=PALETTE["border"], borderwidth=1,
                            title="Outcome"),
            )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div class='section-h'><h3>Correlation heatmap</h3></div>", unsafe_allow_html=True)
        corr = df[NUMERICAL_FEATURES + ENGINEERED_FEATURES + ["approved"]].corr().round(2)
        # FIX #7: explicit colorbar title + hover labels
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            name="Correlation",
            colorscale=[[0.0, "#1E3A8A"], [0.25, "#60A5FA"], [0.5, "#FFFFFF"],
                        [0.75, "#FCA5A5"], [1.0, "#7F1D1D"]],
            zmid=0, zmin=-1, zmax=1,
            text=corr.values, texttemplate="%{text:.2f}",
            textfont={"size": 11, "family": "Inter"},
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Pearson r: %{z:.3f}<extra></extra>",
            colorbar=dict(
                title=dict(text="Pearson r", font=dict(size=11, color=PALETTE["muted"])),
                thickness=14, len=0.7,
                tickfont=dict(size=10, color=PALETTE["muted"]),
            ),
        ))
        fig.update_layout(
            title="Pearson correlation between numeric features",
            height=560, margin=dict(l=120, r=20, t=50, b=120),
            xaxis=dict(side="bottom", tickangle=-30),
            yaxis=dict(autorange="reversed"),
        )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with st.expander("📋 Browse raw data (first 200 rows)"):
            st.dataframe(df.head(200), use_container_width=True)
