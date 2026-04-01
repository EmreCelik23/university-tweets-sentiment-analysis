import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import textwrap

from predict import get_multi_model_prediction  # GERÇEK TAHMİN MOTORU

# =====================================================================
# CONFIG & SETUP
# =====================================================================
st.set_page_config(
    page_title="Üniversite Duygu Analiz Platformu",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# SABİT METRİKLER (TEST SONUÇLARINDAN)
# =====================================================================
MODEL_METRICS = {
    "BERTurk": {"Accuracy": 0.9018, "Macro F1": 0.8796, "Precision": 0.8642, "Recall": 0.9013, "Support": 1110},
    "Electra": {"Accuracy": 0.9198, "Macro F1": 0.8948, "Precision": 0.9010, "Recall": 0.8892, "Support": 1110},
    "CNN-BiLSTM": {"Accuracy": 0.8811, "Macro F1": 0.8446, "Precision": 0.8486, "Recall": 0.8408, "Support": 1110},
    "BiLSTM": {"Accuracy": 0.8532, "Macro F1": 0.8216, "Precision": 0.8073, "Recall": 0.8440, "Support": 1110},
    "CNN": {"Accuracy": 0.8523, "Macro F1": 0.8082, "Precision": 0.8096, "Recall": 0.8068, "Support": 1110},
}

MODEL_CLASS_METRICS = {
    "BERTurk": {
        "0_olumsuz": {"precision": 0.9622, "recall": 0.9023, "f1": 0.9313, "support": 819},
        "1_olumlu": {"precision": 0.7661, "recall": 0.9003, "f1": 0.8278, "support": 291},
    },
    "Electra": {
        "0_olumsuz": {"precision": 0.9387, "recall": 0.9536, "f1": 0.9461, "support": 819},
        "1_olumlu": {"precision": 0.8633, "recall": 0.8247, "f1": 0.8436, "support": 291},
    },
    "CNN-BiLSTM": {
        "0_olumsuz": {"precision": 0.9144, "recall": 0.9255, "f1": 0.9199, "support": 819},
        "1_olumlu": {"precision": 0.7829, "recall": 0.7560, "f1": 0.7692, "support": 291},
    },
    "BiLSTM": {
        "0_olumsuz": {"precision": 0.9327, "recall": 0.8632, "f1": 0.8966, "support": 819},
        "1_olumlu": {"precision": 0.6818, "recall": 0.8247, "f1": 0.7465, "support": 291},
    },
    "CNN": {
        "0_olumsuz": {"precision": 0.8979, "recall": 0.9023, "f1": 0.9001, "support": 819},
        "1_olumlu": {"precision": 0.7213, "recall": 0.7113, "f1": 0.7163, "support": 291},
    },
}


@st.cache_data
def load_main_dataset(path: str = "data/tweetVeriseti.xlsx"):
    try:
        return pd.read_excel(path)
    except Exception:
        return None


DATA_DF = load_main_dataset()

# =====================================================================
# SESSION STATE INIT
# =====================================================================
if "active_pool" not in st.session_state:
    st.session_state["active_pool"] = []
if "last_input" not in st.session_state:
    st.session_state["last_input"] = ""
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Karanlık"

IS_LIGHT_THEME = st.session_state["theme_mode"] == "Aydınlık"
POS_COLOR = "#16a34a" if IS_LIGHT_THEME else "#3fb950"
NEG_COLOR = "#dc2626" if IS_LIGHT_THEME else "#f85149"
PLOT_FONT_COLOR = "#1f2937" if IS_LIGHT_THEME else "#c9d1d9"
PLOT_GRID_COLOR = "rgba(15, 23, 42, 0.14)" if IS_LIGHT_THEME else "rgba(255,255,255,0.05)"
PLOT_LEGEND_BG = "rgba(255,255,255,0.92)" if IS_LIGHT_THEME else "rgba(22, 27, 34, 0.8)"
PLOT_LEGEND_BORDER = "rgba(148, 163, 184, 0.55)" if IS_LIGHT_THEME else "rgba(240, 246, 252, 0.15)"
PLOT_TITLE_COLOR = "#0f172a" if IS_LIGHT_THEME else "#c9d1d9"
PLOT_AXIS_TICK_SIZE = 15
HYPE_LEGEND_FONT_SIZE = 18
PIE_BORDER_COLOR = "#e2e8f0" if IS_LIGHT_THEME else "#0d1117"
TURKEY_AVG_LINE_COLOR = "#64748b" if IS_LIGHT_THEME else "#8b949e"
ANALYSIS_HEADING_COLOR = "#2563eb" if IS_LIGHT_THEME else "#79c0ff"
METRIC_LABEL_COLOR = "#334155" if IS_LIGHT_THEME else "#8b949e"
METRIC_VALUE_COLOR = "#0f172a" if IS_LIGHT_THEME else "#ffffff"
METRIC_VALUE_SHADOW = "none" if IS_LIGHT_THEME else "0 0 10px rgba(88,166,255,0.2)"

TRANSFORMER_MODELS = ["BERTurk", "Electra", "TabiBERT"]
CLASSICAL_MODELS = ["CNN-BiLSTM", "BiLSTM", "CNN"]

# =====================================================================
# HELPERS
# =====================================================================
def _escape_html(s: str) -> str:
    s = "" if s is None else str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_lab_card(text: str, res: dict):
    txt_safe = _escape_html(text)

    chips = []
    for model_name in TRANSFORMER_MODELS + CLASSICAL_MODELS:
        pred = res.get(model_name, (0, 0))[0] if res.get(model_name) else 0
        dot_color = POS_COLOR if pred == 1 else NEG_COLOR

        chip_html = f"""
<div class="pred-chip">
  <div class="pred-left">
    <span class="dot" style="background:{dot_color};"></span>
    <span>{model_name}</span>
  </div>
</div>
""".strip()
        chips.append(chip_html)

    chips_html = "\n".join(chips)

    html = f"""
<div class="glass-card lab-card">
  <div class="lab-meta">ÖRNEK METİN</div>
  <div class="lab-textbox">{txt_safe}</div>

  <div class="lab-meta">MODEL TAHMİNLERİ</div>
  <div class="lab-preds">
    {chips_html}
  </div>
</div>
""".strip()

    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


# =====================================================================
# DATA PROCESSING HELPERS FOR VISUALIZATIONS
# =====================================================================
def extract_year(date_str):
    """Extract year from Twitter createdAt format."""
    try:
        # Format: "Tue Nov 04 12:45:47 +0000 2025"
        return int(str(date_str).split()[-1])
    except Exception:
        return None


def min_max_normalize(series):
    """Normalize a pandas Series using max-normalization."""
    if len(series) == 0:
        return pd.Series(dtype=float)
    max_val = series.max()
    if pd.isna(max_val) or max_val == 0:
        return pd.Series([0] * len(series), index=series.index)
    return series / max_val


@st.cache_data
def prepare_hype_data(df):
    """Prepare normalized tweet count data by university and year."""
    if df is None or 'createdAt' not in df.columns or 'university' not in df.columns:
        return None
    
    # Extract years
    df_copy = df.copy()
    df_copy['year'] = df_copy['createdAt'].apply(extract_year)
    df_copy = df_copy.dropna(subset=['year'])
    df_copy['year'] = df_copy['year'].astype(int)
    
    # Count tweets by university and year
    hype_df = df_copy.groupby(['university', 'year']).size().reset_index(name='tweet_count')
    
    # Apply rolling mean for smoother trends (especially for universities with sparse data)
    # Increased to 3-period moving average for even smoother curves
    hype_df['smoothed_count'] = hype_df.groupby('university')['tweet_count'].transform(
        lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
    )
    
    # Normalize smoothed counts per university (Min-Max)
    hype_df['normalized_count'] = hype_df.groupby('university')['smoothed_count'].transform(min_max_normalize)
    
    # Add total tweet count for each university (for highlighting top universities)
    total_tweets = df_copy.groupby('university').size().reset_index(name='total_tweets')
    hype_df = hype_df.merge(total_tweets, on='university', how='left')
    
    return hype_df


@st.cache_data
def prepare_sentiment_trend_data(df):
    """Prepare sentiment trend data by university and year."""
    if df is None or 'createdAt' not in df.columns or 'university' not in df.columns or 'tags' not in df.columns:
        return None, None
    
    # Extract years
    df_copy = df.copy()
    df_copy['year'] = df_copy['createdAt'].apply(extract_year)
    df_copy = df_copy.dropna(subset=['year'])
    df_copy['year'] = df_copy['year'].astype(int)
    
    # Calculate average sentiment by university and year
    sentiment_df = df_copy.groupby(['university', 'year'])['tags'].mean().reset_index(name='avg_sentiment')
    sentiment_df['avg_sentiment_pct'] = sentiment_df['avg_sentiment'] * 100
    
    # Calculate Turkey average (overall)
    turkey_avg = df_copy.groupby('year')['tags'].mean().reset_index(name='avg_sentiment')
    turkey_avg['avg_sentiment_pct'] = turkey_avg['avg_sentiment'] * 100
    
    return sentiment_df, turkey_avg


@st.cache_data
def prepare_heatmap_data(df):
    """Prepare heatmap data for university sentiment by year."""
    if df is None or 'createdAt' not in df.columns or 'university' not in df.columns or 'tags' not in df.columns:
        return None
    
    # Extract years
    df_copy = df.copy()
    df_copy['year'] = df_copy['createdAt'].apply(extract_year)
    df_copy = df_copy.dropna(subset=['year'])
    df_copy['year'] = df_copy['year'].astype(int)
    
    # Calculate average sentiment by university and year
    heatmap_df = df_copy.groupby(['university', 'year'])['tags'].mean().reset_index(name='avg_sentiment')
    heatmap_df['avg_sentiment_pct'] = heatmap_df['avg_sentiment'] * 100
    
    # Pivot to create heatmap structure
    pivot_df = heatmap_df.pivot(index='university', columns='year', values='avg_sentiment_pct')
    
    return pivot_df


# =====================================================================
# PREMIUM CSS STYLING
# =====================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    /* Remove top black bar / header + tighten padding */
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Remove extra top padding that creates a "black bar" feeling */
    .block-container { padding-top: 1.0rem !important; }
    div[data-testid="stAppViewContainer"] > .main { padding-top: 0rem !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

    /* App Background */
    .stApp {
        background-color: #0d1117;
        background-image:
            radial-gradient(circle at 10% 20%, rgba(88, 166, 255, 0.08) 0%, transparent 25%),
            radial-gradient(circle at 90% 80%, rgba(63, 185, 80, 0.06) 0%, transparent 25%);
        font-family: 'Inter', sans-serif;
        color: #c9d1d9;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    p { line-height: 1.6; font-weight: 300; color: #c9d1d9; }

    .main-title {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #79c0ff 0%, #2f81f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(47, 129, 247, 0.2);
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 2.0rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(240, 246, 252, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        height: 100%;
    }
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(63, 185, 80, 0.25);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        background: rgba(22, 27, 34, 0.82);
    }

    /* Metric card */
    .metric-card {
        background: rgba(22, 27, 34, 0.6);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(240, 246, 252, 0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        background: rgba(22, 27, 34, 0.8);
        border-color: #58a6ff;
        transform: scale(1.02);
    }

    /* Model cards */
    .model-card-positive {
        border-top: 4px solid #3fb950;
        background: linear-gradient(180deg, rgba(63, 185, 80, 0.24) 0%, rgba(22, 27, 34, 0.18) 100%);
        box-shadow: 0 10px 28px rgba(63, 185, 80, 0.16), inset 0 0 0 1px rgba(63, 185, 80, 0.22);
    }
    .model-card-negative {
        border-top: 4px solid #f85149;
        background: linear-gradient(180deg, rgba(248, 81, 73, 0.24) 0%, rgba(22, 27, 34, 0.18) 100%);
        box-shadow: 0 10px 28px rgba(248, 81, 73, 0.16), inset 0 0 0 1px rgba(248, 81, 73, 0.24);
    }
    .result-card {
        text-align: center;
        margin-bottom: 20px;
        padding: 18px;
        min-height: 165px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .result-model-name {
        margin: 0;
        min-height: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #8b949e;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        font-size: 0.78rem;
    }
    .result-emoji {
        font-size: 2.7rem;
        margin: 9px 0;
        line-height: 1;
    }
    .result-sentiment {
        margin: 0;
        letter-spacing: 0.5px;
        font-size: 1.72rem;
        line-height: 1.08;
        font-weight: 800;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: rgba(1, 4, 9, 0.6) !important;
        border: 1px solid #30363d !important;
        color: #ffffff !important;
        border-radius: 12px;
        transition: border-color 0.2s, box-shadow 0.2s;
        font-size: 1rem;
    }
    .stTextArea textarea:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.2) !important;
    }

    /* Buttons (force solid) */
    div.stButton > button,
    div.stButton > button[kind],
    button[kind="primary"],
    button[kind="secondary"],
    button[kind="tertiary"] {
        opacity: 1 !important;
        filter: none !important;
        -webkit-filter: none !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.2rem !important;
        width: 100% !important;
        text-transform: none !important;
    }
    div.stButton > button[kind="primary"], button[kind="primary"] {
        background: linear-gradient(180deg, rgba(46, 160, 67, 0.82) 0%, rgba(35, 134, 54, 0.76) 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(240, 246, 252, 0.24) !important;
        box-shadow: 0 8px 20px rgba(46, 160, 67, 0.20) !important;
    }
    div.stButton > button[kind="secondary"], button[kind="secondary"] {
        background: rgba(56, 139, 253, 0.12) !important;
        color: #c9d1d9 !important;
        border: 1px solid rgba(56, 139, 253, 0.35) !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover,
    div.stButton > button[kind]:hover,
    button[kind="primary"]:hover,
    button[kind="secondary"]:hover {
        opacity: 1 !important;
        filter: none !important;
        transform: translateY(-1px) !important;
    }
    div.stButton > button:disabled, button:disabled {
        opacity: 0.55 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
        padding-bottom: 5px;
        border-bottom: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #8b949e;
        padding: 10px 0px;
        font-weight: 600;
        font-size: 1rem;
        transition: color 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #c9d1d9; }
    .stTabs [aria-selected="true"] { background-color: transparent !important; color: #58a6ff !important; }

    /* DataFrame */
    div[data-testid="stDataFrame"] {
        border: 1px solid #30363d;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Unified section header for dashboard */
    .section-header{
        font-size: 1.15rem;
        font-weight: 800;
        color: #c9d1d9;
        letter-spacing: -0.2px;
        margin: 0 0 12px 0;
    }

    .theme-switch-title {
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.9px;
        text-transform: uppercase;
        color: #8b949e;
        margin: 4px 0 6px 2px;
    }

    /* Theme segmented control readability */
    div[data-testid="stSegmentedControl"] {
        padding: 4px;
        border-radius: 12px;
        border: 1px solid #30363d;
        background: rgba(13, 17, 23, 0.42);
    }
    div[data-testid="stSegmentedControl"] button,
    div[data-testid="stSegmentedControl"] [role="radio"] {
        color: #c9d1d9 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
    div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
        color: #ffffff !important;
        background: rgba(56, 139, 253, 0.40) !important;
        border: 1px solid rgba(121, 192, 255, 0.65) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-pressed="false"],
    div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="false"] {
        color: #9fb0c1 !important;
        background: transparent !important;
    }
    div[data-baseweb="button-group"] {
        padding: 4px;
        border-radius: 12px;
        border: 1px solid #30363d;
        background: rgba(13, 17, 23, 0.42);
    }
    div[data-baseweb="button-group"] button {
        color: #c9d1d9 !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="button-group"] button[aria-pressed="true"] {
        color: #ffffff !important;
        background: rgba(56, 139, 253, 0.38) !important;
        border: 1px solid rgba(121, 192, 255, 0.6) !important;
    }
    div[data-baseweb="button-group"] button[aria-pressed="false"] {
        color: #9fb0c1 !important;
        background: transparent !important;
    }
    div[data-testid="stSegmentedControl"] button:nth-child(1),
    div[data-testid="stSegmentedControl"] [role="radio"]:nth-child(1),
    div[data-baseweb="button-group"] button:nth-child(1) {
        background: #e2e8f0 !important;
        border: 1px solid #93c5fd !important;
    }
    div[data-testid="stSegmentedControl"] button:nth-child(1) *,
    div[data-testid="stSegmentedControl"] [role="radio"]:nth-child(1) *,
    div[data-baseweb="button-group"] button:nth-child(1) * {
        color: #1e293b !important;
    }

    /* =========================
       DATA LAB SAMPLE CARDS
       ========================= */
    .lab-card {
        height: 460px;
        padding: 18px !important;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .lab-textbox {
        flex: 1;
        background: rgba(1, 4, 9, 0.45);
        border: 1px solid rgba(240, 246, 252, 0.08);
        border-radius: 12px;
        padding: 12px 12px;
        overflow-y: auto;
        line-height: 1.65;
        color: #c9d1d9;
        font-size: 0.95rem;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .lab-meta {
        font-size: 0.75rem;
        color: #8b949e;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 800;
    }
    .lab-preds {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
    }
    .pred-chip {
        background: rgba(22, 27, 34, 0.65);
        border: 1px solid rgba(240, 246, 252, 0.08);
        border-radius: 12px;
        padding: 10px 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 800;
        font-size: 0.85rem;
    }
    .pred-left {
        display: flex;
        gap: 8px;
        align-items: center;
        color: #c9d1d9;
    }
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        flex: 0 0 auto;
    }
    .pred-right {
        font-size: 0.8rem;
        letter-spacing: 0.6px;
        opacity: 0.95;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if IS_LIGHT_THEME:
    st.markdown(
        """
        <style>
        ::-webkit-scrollbar { background: #e2e8f0 !important; }
        ::-webkit-scrollbar-thumb { background: #94a3b8 !important; }
        ::-webkit-scrollbar-thumb:hover { background: #2563eb !important; }

        .stApp {
            background-color: #f5f7fb !important;
            background-image:
                radial-gradient(circle at 12% 16%, rgba(37, 99, 235, 0.11) 0%, transparent 28%),
                radial-gradient(circle at 84% 82%, rgba(22, 163, 74, 0.10) 0%, transparent 26%) !important;
            color: #0f172a !important;
        }

        h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
        p, span, label { color: #1f2937; }
        .main-title {
            background: linear-gradient(135deg, #0f4ec4 0%, #2563eb 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            text-shadow: none !important;
        }
        .subtitle { color: #334155 !important; }

        .glass-card {
            background: rgba(255, 255, 255, 0.94) !important;
            border: 1px solid rgba(15, 23, 42, 0.12) !important;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10) !important;
        }
        .glass-card:hover {
            border-color: rgba(37, 99, 235, 0.35) !important;
            background: rgba(255, 255, 255, 0.98) !important;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.14) !important;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.96) !important;
            border: 1px solid rgba(15, 23, 42, 0.10) !important;
        }
        .metric-card:hover {
            background: #ffffff !important;
            border-color: #2563eb !important;
        }

        .model-card-positive {
            border-top-color: #16a34a !important;
            background: linear-gradient(180deg, rgba(22, 163, 74, 0.27) 0%, rgba(255, 255, 255, 0.98) 100%) !important;
            box-shadow: 0 10px 24px rgba(22, 163, 74, 0.20), inset 0 0 0 1px rgba(22, 163, 74, 0.28) !important;
        }
        .model-card-negative {
            border-top-color: #dc2626 !important;
            background: linear-gradient(180deg, rgba(220, 38, 38, 0.23) 0%, rgba(255, 255, 255, 0.98) 100%) !important;
            box-shadow: 0 10px 24px rgba(220, 38, 38, 0.20), inset 0 0 0 1px rgba(220, 38, 38, 0.26) !important;
        }
        .result-model-name { color: #475569 !important; }

        .stTextArea textarea {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            color: #0f172a !important;
        }
        .stTextArea textarea::placeholder { color: #64748b !important; }
        .stTextArea textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18) !important;
        }

        div.stButton > button[kind="primary"], button[kind="primary"] {
            background: linear-gradient(180deg, rgba(34, 197, 94, 0.74) 0%, rgba(22, 163, 74, 0.72) 100%) !important;
            border: 1px solid rgba(15, 23, 42, 0.15) !important;
            color: #ffffff !important;
            box-shadow: 0 8px 18px rgba(22, 163, 74, 0.20) !important;
        }
        div.stButton > button[kind="secondary"], button[kind="secondary"] {
            background: rgba(37, 99, 235, 0.10) !important;
            color: #1e293b !important;
            border: 1px solid rgba(37, 99, 235, 0.35) !important;
        }

        .stTabs [data-baseweb="tab-list"] { border-bottom-color: #d1d5db !important; }
        .stTabs [data-baseweb="tab"] { color: #475569 !important; }
        .stTabs [data-baseweb="tab"]:hover { color: #1e293b !important; }
        .stTabs [aria-selected="true"] { color: #2563eb !important; }

        div[data-testid="stDataFrame"] { border-color: #cbd5e1 !important; }
        .section-header { color: #0f172a !important; }
        .theme-switch-title { color: #475569 !important; }

        div[data-testid="stSegmentedControl"] {
            border-color: #cbd5e1 !important;
            background: #e2e8f0 !important;
        }
        div[data-testid="stSegmentedControl"] button,
        div[data-testid="stSegmentedControl"] [role="radio"] {
            color: #334155 !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
        div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
            color: #0f172a !important;
            background: #ffffff !important;
            border: 1px solid #2563eb !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-pressed="false"],
        div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="false"] {
            color: #334155 !important;
            background: transparent !important;
        }
        div[data-baseweb="button-group"] {
            border-color: #cbd5e1 !important;
            background: #e2e8f0 !important;
        }
        div[data-baseweb="button-group"] button {
            color: #334155 !important;
        }
        div[data-baseweb="button-group"] button[aria-pressed="true"] {
            color: #0f172a !important;
            background: #ffffff !important;
            border: 1px solid #2563eb !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
        }
        div[data-baseweb="button-group"] button[aria-pressed="false"] {
            color: #334155 !important;
            background: transparent !important;
        }
        div[data-testid="stSegmentedControl"] button:nth-child(1),
        div[data-testid="stSegmentedControl"] [role="radio"]:nth-child(1),
        div[data-baseweb="button-group"] button:nth-child(1) {
            background: #e2e8f0 !important;
            border: 1px solid #93c5fd !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.14) !important;
        }
        div[data-testid="stSegmentedControl"] button:nth-child(1) *,
        div[data-testid="stSegmentedControl"] [role="radio"]:nth-child(1) *,
        div[data-baseweb="button-group"] button:nth-child(1) * {
            color: #1e293b !important;
        }

        .lab-textbox {
            background: #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            color: #1e293b !important;
        }
        .lab-meta { color: #475569 !important; }
        .pred-chip {
            background: #ffffff !important;
            border: 1px solid #d1d5db !important;
        }
        .pred-left { color: #0f172a !important; }

        [style*="color:#ffffff"], [style*="color: #ffffff"] { color: #0f172a !important; }
        [style*="color:#c9d1d9"], [style*="color: #c9d1d9"] { color: #1f2937 !important; }
        [style*="color:#8b949e"], [style*="color: #8b949e"] { color: #475569 !important; }
        [style*="border: 2px dashed #30363d"] { border: 2px dashed #94a3b8 !important; }
        [style*="border:1px dashed rgba(48,54,61,0.55)"] { border: 1px dashed rgba(71, 85, 105, 0.55) !important; }
        [style*="text-shadow:0 0 10px rgba(88,166,255,0.2)"] { text-shadow: none !important; }

        div[data-testid="stCodeBlock"] pre {
            background: #f8fafc !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =====================================================================
# HEADER SECTION
# =====================================================================
col_brand, col_title, col_theme = st.columns([1, 4.4, 1.6])
with col_brand:
    try:
        st.image("assets/ytu_logo.png", width=120)
    except Exception:
        st.markdown(f"<h1 style='font-size:4rem; color:{ANALYSIS_HEADING_COLOR};'>S</h1>", unsafe_allow_html=True)

with col_title:
    st.markdown('<h1 class="main-title">Türk Üniversiteleri Duygu Analiz Modeli</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Üniversite yorumları için duygu analizi platformu</p>', unsafe_allow_html=True)

with col_theme:
    st.markdown('<div class="theme-switch-title">Tema Modu</div>', unsafe_allow_html=True)
    st.segmented_control(
        "Tema",
        ["Karanlık", "Aydınlık"],
        format_func=lambda x: "🌙 Karanlık" if x == "Karanlık" else "☀️ Aydınlık",
        key="theme_mode",
        label_visibility="collapsed",
    )

# =====================================================================
# MAIN TABS
# =====================================================================
tab_live, tab_dashboard, tab_lab = st.tabs(["CANLI ANALİZ", "DASHBOARD", "DATA LAB"])

# =====================================================================
# TAB 1: CANLI ANALİZ
# =====================================================================
with tab_live:
    st.write("")
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="margin-top:0; color:#c9d1d9;">Yorum Analizi</h3>
                <p style="color:#8b949e; font-size:0.95rem; margin-bottom:18px;">
                    Aşağıya bir metin girin ve 6 farklı modelin (BERTurk, Electra, TabiBERT, CNN-BiLSTM, BiLSTM, CNN) anlık duygu analizini izleyin.
                </p>
            """,
            unsafe_allow_html=True,
        )

        txt_input = st.text_area(
            "Metin Girişi",
            height=140,
            placeholder="Örn: Kampüs hayatı harika ama yemekhane sırası çok uzun...",
            label_visibility="collapsed",
        )

        # Buttons side by side
        c_btn1, c_btn2 = st.columns([2.5, 1], gap="small")
        with c_btn1:
            analyze_btn = st.button("ANALİZİ BAŞLAT", type="primary", use_container_width=True)
        with c_btn2:
            clean_btn = st.button("Temizle", type="secondary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if clean_btn:
            st.session_state["last_input"] = ""
            st.session_state["last_results"] = None
            st.rerun()

        if analyze_btn:
            if txt_input.strip():
                with st.spinner("Modeller çalışıyor..."):
                    time.sleep(0.35)
                    try:
                        results = get_multi_model_prediction(txt_input)
                        st.session_state["last_input"] = txt_input
                        st.session_state["last_results"] = results
                    except Exception as e:
                        st.error(f"Tahmin Hatası: {e}")
            else:
                st.warning("Lütfen bir metin giriniz.")

        # Active Learning
        if st.session_state["last_results"]:
            st.write("")
            st.markdown(
                """
                <div class="glass-card" style="border:1px dashed rgba(48,54,61,0.55);">
                    <h4 style="margin-top:0;">Veri Havuzu</h4>
                    <p style="color:#8b949e; font-size:0.85rem; margin-bottom:10px;">
                        Modelin yanıldığı durumları düzeltip havuza ekleyerek gelecekteki eğitimlere katkıda bulunun.
                    </p>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([2, 1], gap="medium")
            with c1:
                st.caption("Son Analiz Edilen Metin:")
                preview = st.session_state["last_input"] or ""
                st.code(f"{preview[:220]}{'...' if len(preview) > 220 else ''}", language="text")
            with c2:
                st.caption("Doğru Etiket:")
                tag_choice = st.selectbox("Etiket Seç", ["Pozitif (1)", "Negatif (0)"], label_visibility="collapsed")

            if st.button("Veri Setine Ekle (+)", type="secondary"):
                tag_value = 1 if "Pozitif" in tag_choice else 0
                st.session_state["active_pool"].append({"text": st.session_state["last_input"], "tags": tag_value})

                try:
                    try:
                        existing = pd.read_excel("active_learning_pool.xlsx")
                    except Exception:
                        existing = pd.DataFrame(columns=["text", "tags"])

                    new_row = pd.DataFrame([{"text": st.session_state["last_input"], "tags": tag_value}])
                    pd.concat([existing, new_row], ignore_index=True).to_excel("active_learning_pool.xlsx", index=False)
                    st.toast("Veri başarıyla kaydedildi.", icon="✅")
                except Exception as e:
                    st.error(f"Kayıt Hatası: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

    with col_results:
        st.markdown("### Sonuçlar")

        if st.session_state["last_results"]:
            results = st.session_state["last_results"]
            
            # Üst satır: Transformer modeller (BERTurk, Electra, TabiBERT)
            st.markdown('<div style="margin-bottom:10px;"><span style="color:#8b949e; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; font-weight:700;">Transformer Models</span></div>', unsafe_allow_html=True)
            transformer_cols = st.columns(len(TRANSFORMER_MODELS), gap="medium")
            
            for i, model_name in enumerate(TRANSFORMER_MODELS):
                data = results.get(model_name)
                target_col = transformer_cols[i]
                
                with target_col:
                    if data is None:
                        st.info(f"{model_name} henüz yüklenmedi")
                    else:
                        pred, _ = data
                        sentiment = "POZİTİF" if pred == 1 else "NEGATİF"
                        card_class = "model-card-positive" if pred == 1 else "model-card-negative"
                        emoji = "😊" if pred == 1 else "😡"
                        text_color = POS_COLOR if pred == 1 else NEG_COLOR

                        st.markdown(
                            f"""
                            <div class="glass-card {card_class} result-card">
                                <div class="result-model-name">{model_name}</div>
                                <div class="result-emoji">{emoji}</div>
                                <div class="result-sentiment" style="color:{text_color};">{sentiment}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            
            # Alt satır: Klasik modeller (CNN-BiLSTM, BiLSTM, CNN)
            st.markdown('<div style="margin-bottom:10px; margin-top:20px;"><span style="color:#8b949e; font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; font-weight:700;">Classical Models</span></div>', unsafe_allow_html=True)
            classical_cols = st.columns(len(CLASSICAL_MODELS), gap="small")
            
            for i, model_name in enumerate(CLASSICAL_MODELS):
                data = results.get(model_name)
                target_col = classical_cols[i]
                
                with target_col:
                    if data is None:
                        st.info(f"{model_name} N/A")
                    else:
                        pred, _ = data
                        sentiment = "POZİTİF" if pred == 1 else "NEGATİF"
                        card_class = "model-card-positive" if pred == 1 else "model-card-negative"
                        emoji = "😊" if pred == 1 else "😡"
                        text_color = POS_COLOR if pred == 1 else NEG_COLOR

                        st.markdown(
                            f"""
                            <div class="glass-card {card_class} result-card">
                                <div class="result-model-name">{model_name}</div>
                                <div class="result-emoji">{emoji}</div>
                                <div class="result-sentiment" style="color:{text_color};">{sentiment}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center; padding:60px 40px; border: 2px dashed #30363d; opacity:0.7; background:transparent;">
                    <div style="font-size:4rem; margin-bottom:20px; opacity:0.5; filter: grayscale(100%);">📡</div>
                    <h3 style="color:#8b949e;">Bekleniyor...</h3>
                    <p>Analiz sonuçları burada görüntülenecek.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =====================================================================
# TAB 2: DASHBOARD
# =====================================================================
with tab_dashboard:
    st.write("")

    total_tweets = len(DATA_DF) if DATA_DF is not None else 0
    if DATA_DF is not None and "tags" in DATA_DF.columns:
        pos_count = int((DATA_DF["tags"] == 1).sum())
        neg_count = int((DATA_DF["tags"] == 0).sum())
    else:
        pos_count, neg_count = 1374, 3669

    total_labeled = pos_count + neg_count if (pos_count + neg_count) > 0 else 1
    pos_percent = (pos_count / total_labeled) * 100
    neg_percent = (neg_count / total_labeled) * 100

    m1, m2, m3, m4 = st.columns(4, gap="medium")

    def render_metric_card(label, value, col):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="color:{METRIC_LABEL_COLOR}; font-size:0.9rem; margin-bottom:5px;">{label}</div>
                    <div style="color:{METRIC_VALUE_COLOR}; font-size:2rem; font-weight:800; text-shadow:{METRIC_VALUE_SHADOW};">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_metric_card("Toplam Yorum", f"{total_tweets:,}", m1)
    render_metric_card("Pozitif", f"{pos_count:,}", m2)
    render_metric_card("Negatif", f"{neg_count:,}", m3)
    render_metric_card("Pozitif / Negatif", f"%{pos_percent:.1f} / %{neg_percent:.1f}", m4)

    st.markdown("---")

    metrics_df = pd.DataFrame(MODEL_METRICS).T.reset_index().rename(columns={"index": "Model"})

    col_main_chart, col_pie = st.columns([2, 1], gap="large")

    with col_main_chart:
        st.markdown(
            '''
            <div class="glass-card">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Model Başarı Sıralaması (Macro F1)</h4>
                <p style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Beş farklı modelin genel performans karşılaştırması - yüksek F1 skoru daha dengeli tahmin anlamına gelir.</p>
            ''',
            unsafe_allow_html=True
        )

        fig_bar = px.bar(
            metrics_df,
            x="Model",
            y="Macro F1",
            text=metrics_df["Macro F1"].apply(lambda x: f"%{x*100:.1f}"),
            color="Macro F1",
            color_continuous_scale=["#1a7f37", POS_COLOR],
            height=320,
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=PLOT_FONT_COLOR),
            yaxis=dict(
                showgrid=True,
                gridcolor=PLOT_GRID_COLOR,
                range=[0, 1.05],
                tickfont=dict(color=PLOT_FONT_COLOR, size=PLOT_AXIS_TICK_SIZE),
                title_font=dict(color=PLOT_TITLE_COLOR, size=13),
                zerolinecolor=PLOT_GRID_COLOR,
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(color=PLOT_FONT_COLOR, size=PLOT_AXIS_TICK_SIZE),
                title_font=dict(color=PLOT_TITLE_COLOR, size=13),
                linecolor=PLOT_GRID_COLOR,
            ),
            coloraxis_showscale=False,
            margin=dict(t=10, l=0, r=0, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_pie:
        st.markdown(
            '''
            <div class="glass-card">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Model Performans Özeti</h4>
                <p style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Accuracy, F1, Precision ve Recall metriklerinin detaylı karşılaştırması.</p>
            ''',
            unsafe_allow_html=True
        )

        perf_table = metrics_df[["Model", "Accuracy", "Macro F1", "Precision", "Recall"]].copy()
        st.dataframe(
            perf_table.style.format("{:.3f}", subset=["Accuracy", "Macro F1", "Precision", "Recall"])
            .background_gradient(cmap="Greens", subset=["Accuracy"], vmin=0.75, vmax=0.95)
            .background_gradient(cmap="Greens", subset=["Macro F1"], vmin=0.70, vmax=0.92)
            .background_gradient(cmap="Greens", subset=["Precision"], vmin=0.75, vmax=0.92)
            .background_gradient(cmap="Greens", subset=["Recall"], vmin=0.70, vmax=0.92),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    col_tbl1, col_tbl2 = st.columns(2, gap="large")

    with col_tbl1:
        st.markdown(
            '''
            <div class="glass-card">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Üniversite Bazlı Dağılım</h4>
                <p style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Seçilen üniversitenin pozitif ve negatif yorum oranlarını görselleştirin.</p>
            ''',
            unsafe_allow_html=True
        )

        if DATA_DF is not None and {"tags", "university"}.issubset(DATA_DF.columns):
            uni_list = sorted(DATA_DF["university"].dropna().unique().tolist())
            selected_uni = st.selectbox("Üniversite Filtrele", ["Tümü"] + uni_list)

            subset = DATA_DF if selected_uni == "Tümü" else DATA_DF[DATA_DF["university"] == selected_uni]

            if len(subset) > 0:
                p = int((subset["tags"] == 1).sum())
                n = int((subset["tags"] == 0).sum())
                fig_pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Pozitif", "Negatif"],
                            values=[p, n],
                            hole=0.6,
                            marker=dict(colors=[POS_COLOR, NEG_COLOR], line=dict(color=PIE_BORDER_COLOR, width=2)),
                        )
                    ]
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=PLOT_FONT_COLOR),
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=280,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        x=0.15,
                        y=-0.1,
                        font=dict(color=PLOT_FONT_COLOR, size=12),
                        bgcolor=PLOT_LEGEND_BG,
                        bordercolor=PLOT_LEGEND_BORDER,
                        borderwidth=1,
                    ),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Veri yok.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_tbl2:
        st.markdown(
            '''
            <div class="glass-card">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Detaylı Sınıf Analizi</h4>
                <p style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Her modelin pozitif ve negatif sınıflardaki performansı ayrı ayrı incelenir.</p>
            ''',
            unsafe_allow_html=True
        )

        rows = []
        for model_name, class_dict in MODEL_CLASS_METRICS.items():
            for cls_name, vals in class_dict.items():
                rows.append(
                    {"Model": model_name, "Sınıf": cls_name, "Precision": vals["precision"], "Recall": vals["recall"], "F1": vals["f1"]}
                )
        class_df = pd.DataFrame(rows)

        st.dataframe(
            class_df.style.format("{:.3f}", subset=["Precision", "Recall", "F1"]),
            use_container_width=True,
            hide_index=True,
            height=388,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================================
    # NEW VISUALIZATIONS: TEMPORAL ANALYSIS
    # =====================================================================
    st.write("")
    st.write("")
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; margin: 30px 0 20px 0;">
            <h2 style="color:{ANALYSIS_HEADING_COLOR}; font-weight:800; font-size:2rem; letter-spacing:-0.5px;">Yıllara Göre Analiz</h2>
            <p style="color:#8b949e; font-size:1rem;">Üniversitelerin yıllar içindeki popülerlik ve duygu trendleri...</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    # Visualization 1: "Hype" Graph (Normalized Tweet Counts)
    st.markdown(
        '''
        <div class="glass-card">
            <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Hype Grafiği</h4>
            <p style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Hangi üniversite hangi yıl daha çok konuşuldu? Min-Max normalizasyonu ile tüm üniversiteler adil şekilde karşılaştırılır.</p>
        ''',
        unsafe_allow_html=True
    )

    hype_data = prepare_hype_data(DATA_DF)
    if hype_data is not None and len(hype_data) > 0:
        # Manually selected default universities (excluding BILKENT and HACETTEPE)
        default_unis = ['ODTU', 'BOUN', 'ITU', 'YTU', 'ISTANBUL_UNI']
        all_universities = sorted(hype_data['university'].unique())
        
        # Get top universities for highlighting (for star icons)
        top_unis = hype_data.groupby('university')['total_tweets'].first().nlargest(6).index.tolist()
        
        # Multi-select for universities (default: manually selected 5)
        selected_universities = st.multiselect(
            "Üniversiteleri Seç (Karşılaştırma)",
            options=all_universities,
            default=default_unis,
            help="En fazla 10 üniversite seçebilirsiniz. Karşılaştırma için 5 önemli üniversite varsayılan olarak seçilmiştir."
        )
        
        # Limit to 10 universities for readability
        if len(selected_universities) > 10:
            st.warning("⚠️ En fazla 10 üniversite seçebilirsiniz. İlk 10'u gösteriyorum.")
            selected_universities = selected_universities[:10]
        
        if len(selected_universities) > 0:
            fig_hype = go.Figure()
            
            # Use Plotly's distinct color scales to ensure unique colors
            # Combine multiple color scales for maximum variety
            color_scale_1 = px.colors.qualitative.Plotly  # 10 colors
            color_scale_2 = px.colors.qualitative.D3  # 10 colors
            color_scale_3 = px.colors.qualitative.G10  # 10 colors
            all_colors = color_scale_1 + color_scale_2 + color_scale_3  # 30 unique colors
            
            # Get all years for x-axis tick values
            all_years = sorted(hype_data['year'].unique())
            
            for idx, uni in enumerate(selected_universities):
                uni_data = hype_data[hype_data['university'] == uni].sort_values('year')
                is_top = uni in top_unis
                
                # Highlight top universities with thicker lines
                line_width = 3.0 if is_top else 2.0
                
                # Use unique color from combined palette
                color_idx = idx % len(all_colors)
                
                fig_hype.add_trace(go.Scatter(
                    x=uni_data['year'],
                    y=uni_data['normalized_count'],
                    mode='lines',
                    name=uni,
                    line=dict(
                        width=line_width, 
                        color=all_colors[color_idx],
                        shape='spline',
                        smoothing=1.3
                    ),
                    hovertemplate=f'<b>{uni}</b><br>Yıl: %{{x}}<br>Yoğunluk: %{{y:.2f}}<extra></extra>'
                ))
            
            fig_hype.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PLOT_FONT_COLOR),
                yaxis=dict(
                    title="Etkileşim Yoğunluğu (Normalize)",
                    showgrid=True,
                    gridcolor=PLOT_GRID_COLOR,
                    range=[0, 1.05],
                    tickfont=dict(color=PLOT_FONT_COLOR, size=PLOT_AXIS_TICK_SIZE),
                    title_font=dict(color=PLOT_TITLE_COLOR, size=14),
                    zerolinecolor=PLOT_GRID_COLOR,
                ),
                xaxis=dict(
                    title=None,
                    showgrid=False,
                    tickmode='array',
                    tickvals=all_years,
                    ticktext=[str(year) for year in all_years],
                    range=[min(all_years) - 0.3, max(all_years) + 0.3],  # Add padding to show 2025 fully
                    tickfont=dict(color=PLOT_FONT_COLOR, size=PLOT_AXIS_TICK_SIZE),
                    title_font=dict(color=PLOT_TITLE_COLOR, size=14),
                    linecolor=PLOT_GRID_COLOR,
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.46,
                    xanchor="center",
                    x=0.5,
                    bgcolor=PLOT_LEGEND_BG,
                    bordercolor=PLOT_LEGEND_BORDER,
                    borderwidth=1,
                    font=dict(size=HYPE_LEGEND_FONT_SIZE, color=PLOT_FONT_COLOR),
                ),
                height=550,  # Increased height for better visibility
                margin=dict(t=10, l=0, r=0, b=140),
                hovermode='x unified'
            )
            st.plotly_chart(fig_hype, use_container_width=True)
        else:
            st.info("👆 Lütfen en az bir üniversite seçin.")
    else:
        st.info("Zaman serisi verisi bulunamadı.")
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    # Row 2: Happiness Curve + Heatmap
    col_happiness, col_heatmap = st.columns([3, 2], gap="large")
    
    # Visualization 2: Happiness Change Curve
    with col_happiness:
        st.markdown(
            '''
            <div class="glass-card">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Mutluluk Değişim Eğrisi</h4>
                <p style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Üniversiteler yıllar geçtikçe daha mı mutlu?</p>
            ''',
            unsafe_allow_html=True
        )
        
        sentiment_data, turkey_avg = prepare_sentiment_trend_data(DATA_DF)
        
        if sentiment_data is not None and len(sentiment_data) > 0:
            uni_list_sentiment = sorted(sentiment_data['university'].unique())
            selected_uni_sentiment = st.selectbox(
                "Üniversite Seç",
                ["Tümü (Türkiye Ortalaması)"] + uni_list_sentiment,
                key="sentiment_uni_select"
            )
            
            # Get all years for x-axis
            all_years_sentiment = sorted(sentiment_data['year'].unique())
            
            fig_sentiment = go.Figure()
            
            # Add Turkey average (dashed line)
            if turkey_avg is not None:
                fig_sentiment.add_trace(go.Scatter(
                    x=turkey_avg['year'],
                    y=turkey_avg['avg_sentiment_pct'],
                    mode='lines',
                    name='Türkiye Ortalaması',
                    line=dict(width=2, color=TURKEY_AVG_LINE_COLOR, dash='dash'),
                    hovertemplate='<b>Türkiye Ort.</b><br>Yıl: %{x}<br>Pozitiflik: %{y:.1f}%<extra></extra>'
                ))
            
            # Add selected university line (solid)
            if selected_uni_sentiment != "Tümü (Türkiye Ortalaması)":
                uni_sentiment_data = sentiment_data[sentiment_data['university'] == selected_uni_sentiment].sort_values('year')
                fig_sentiment.add_trace(go.Scatter(
                    x=uni_sentiment_data['year'],
                    y=uni_sentiment_data['avg_sentiment_pct'],
                    mode='lines+markers',
                    name=selected_uni_sentiment,
                    line=dict(width=4, color=POS_COLOR),
                    marker=dict(size=10, symbol='circle'),
                    hovertemplate=f'<b>{selected_uni_sentiment}</b><br>Yıl: %{{x}}<br>Pozitiflik: %{{y:.1f}}%<extra></extra>'
                ))
            
            fig_sentiment.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PLOT_FONT_COLOR),
                yaxis=dict(
                    title="Pozitiflik Oranı (%)",
                    showgrid=True,
                    gridcolor=PLOT_GRID_COLOR,
                    range=[0, 100],
                    tickfont=dict(color=PLOT_FONT_COLOR, size=PLOT_AXIS_TICK_SIZE),
                    title_font=dict(color=PLOT_TITLE_COLOR, size=14),
                    zerolinecolor=PLOT_GRID_COLOR,
                ),
                xaxis=dict(
                    title=None,
                    showgrid=False,
                    tickmode='array',
                    tickvals=all_years_sentiment,
                    ticktext=[str(year) for year in all_years_sentiment],
                    range=[min(all_years_sentiment) - 0.3, max(all_years_sentiment) + 0.3],  # Add padding to show 2025 fully
                    tickfont=dict(color=PLOT_FONT_COLOR, size=PLOT_AXIS_TICK_SIZE),
                    title_font=dict(color=PLOT_TITLE_COLOR, size=14),
                    linecolor=PLOT_GRID_COLOR,
                ),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=0.02,
                    bgcolor=PLOT_LEGEND_BG,
                    bordercolor=PLOT_LEGEND_BORDER,
                    borderwidth=1,
                    font=dict(color=PLOT_FONT_COLOR, size=12),
                ),
                height=400,
                margin=dict(t=10, l=0, r=0, b=40),
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("Duygu trend verisi bulunamadı.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualization 3: Yearly Report Cards (Heatmap)
    with col_heatmap:
        st.markdown(
            '''
            <div class="glass-card">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Yıllık Karneler</h4>
                <p style="color:#8b949e; font-size:0.85rem; margin-bottom:12px;">Her üniversitenin yıllara göre karnesi</p>
            ''',
            unsafe_allow_html=True
        )
        
        heatmap_data = prepare_heatmap_data(DATA_DF)
        
        if heatmap_data is not None and not heatmap_data.empty:
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=[
                    [0, NEG_COLOR],      # Red for negative
                    [0.5, '#e3b341'],    # Yellow for neutral
                    [1, POS_COLOR]       # Green for positive
                ],
                text=heatmap_data.values.round(1),
                texttemplate='%{text:.1f}%',
                textfont={"size": 10, "color": PLOT_FONT_COLOR},
                colorbar=dict(
                    title=dict(text="Pozitif %", font=dict(color=PLOT_TITLE_COLOR)),
                    tickmode="linear",
                    tick0=0,
                    dtick=25,
                    thickness=15,
                    len=0.7,
                    tickfont=dict(color=PLOT_FONT_COLOR, size=PLOT_AXIS_TICK_SIZE),
                ),
                hovertemplate='<b>%{y}</b><br>Yıl: %{x}<br>Pozitiflik: %{z:.1f}%<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=PLOT_FONT_COLOR),
                xaxis=dict(
                    title=None,
                    side="bottom",
                    dtick=1,
                    tickfont=dict(color=PLOT_FONT_COLOR, size=PLOT_AXIS_TICK_SIZE),
                    title_font=dict(color=PLOT_TITLE_COLOR, size=14),
                    linecolor=PLOT_GRID_COLOR,
                ),
                yaxis=dict(
                    title=None,
                    tickfont=dict(color=PLOT_FONT_COLOR, size=PLOT_AXIS_TICK_SIZE),
                    title_font=dict(color=PLOT_TITLE_COLOR, size=14),
                ),
                height=400,
                margin=dict(t=10, l=120, r=0, b=40),
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Heatmap verisi bulunamadı.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# TAB 3: DATA LAB
# =====================================================================
with tab_lab:
    st.write("")
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0;">Toplu Test & Veri Laboratuvarı</h3>
            <p>Excel (.xlsx) veya CSV yükleyip toplu analiz yapabilir, rastgele örneklerle modelleri hızlıca test edebilirsiniz.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    uploaded_file = st.file_uploader("Dosya Yükle (Sürükle-Bırak)", type=["xlsx", "csv"])

    if uploaded_file:
        df_up = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)

        with st.expander("📂 Yüklenen Dosya İçeriği", expanded=True):
            st.dataframe(df_up.head(10), use_container_width=True)

        st.write("")
        if st.button("Rastgele 3 Örnek Analiz Et", type="primary"):
            text_col = next(
                (c for c in df_up.columns if "text" in c.lower() or "tweet" in c.lower() or "yorum" in c.lower()),
                None,
            )

            if not text_col:
                st.error("Hata: Dosyada 'text', 'tweet' veya 'yorum' sütunu bulunamadı.")
            else:
                samples = df_up.sample(3)
                cols_lab = st.columns(3, gap="large")

                for idx, (_, row) in enumerate(samples.iterrows()):
                    raw_txt = "" if pd.isna(row[text_col]) else str(row[text_col])
                    try:
                        res = get_multi_model_prediction(raw_txt)
                        with cols_lab[idx]:
                            render_lab_card(raw_txt, res)
                    except Exception as e:
                        with cols_lab[idx]:
                            st.error(f"Hata: {str(e)[:160]}")
