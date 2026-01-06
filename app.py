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
    "BERTurk": {"Accuracy": 0.9247, "Macro F1": 0.9045, "Precision": 0.9064, "Recall": 0.9027, "Support": 757},
    "CNN-BiLSTM": {"Accuracy": 0.8534, "Macro F1": 0.7899, "Precision": 0.8528, "Recall": 0.7610, "Support": 757},
    "BiLSTM": {"Accuracy": 0.8296, "Macro F1": 0.7697, "Precision": 0.7949, "Recall": 0.7538, "Support": 757},
    "CNN": {"Accuracy": 0.8151, "Macro F1": 0.7558, "Precision": 0.7702, "Recall": 0.7453, "Support": 757},
}

MODEL_CLASS_METRICS = {
    "BERTurk": {
        "0_olumsuz": {"precision": 0.9458, "recall": 0.9510, "f1": 0.9484, "support": 551},
        "1_olumlu": {"precision": 0.8670, "recall": 0.8544, "f1": 0.8606, "support": 206},
    },
    "CNN-BiLSTM": {
        "0_olumsuz": {"precision": 0.8564, "recall": 0.9201, "f1": 0.8871, "support": 551},
        "1_olumlu": {"precision": 0.7333, "recall": 0.5874, "f1": 0.6523, "support": 206},
    },
    "BiLSTM": {
        "0_olumsuz": {"precision": 0.8549, "recall": 0.8984, "f1": 0.8761, "support": 551},
        "1_olumlu": {"precision": 0.6854, "recall": 0.5922, "f1": 0.6354, "support": 206},
    },
    "CNN": {
        "0_olumsuz": {"precision": 0.8537, "recall": 0.9637, "f1": 0.9054, "support": 551},
        "1_olumlu": {"precision": 0.8519, "recall": 0.5583, "f1": 0.6745, "support": 206},
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

# =====================================================================
# HELPERS
# =====================================================================
def _escape_html(s: str) -> str:
    s = "" if s is None else str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_lab_card(text: str, res: dict):
    txt_safe = _escape_html(text)

    chips = []
    for model_name in ["BERTurk", "CNN-BiLSTM", "BiLSTM", "CNN"]:
        pred = res.get(model_name, (0, 0))[0] if res.get(model_name) else 0
        dot_color = "#3fb950" if pred == 1 else "#f85149"
        label = "POZİTİF" if pred == 1 else "NEGATİF"

        chip_html = f"""
<div class="pred-chip">
  <div class="pred-left">
    <span class="dot" style="background:{dot_color};"></span>
    <span>{model_name}</span>
  </div>
  <div class="pred-right" style="color:{dot_color};">{label}</div>
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
        background: linear-gradient(180deg, rgba(63, 185, 80, 0.08) 0%, rgba(22, 27, 34, 0.1) 100%);
    }
    .model-card-negative {
        border-top: 4px solid #f85149;
        background: linear-gradient(180deg, rgba(248, 81, 73, 0.08) 0%, rgba(22, 27, 34, 0.1) 100%);
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
        background: linear-gradient(180deg, #2ea043 0%, #238636 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(240, 246, 252, 0.18) !important;
        box-shadow: 0 8px 20px rgba(46, 160, 67, 0.28) !important;
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
        grid-template-columns: 1fr 1fr;
        gap: 10px;
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

# =====================================================================
# HEADER SECTION
# =====================================================================
col_brand, col_title = st.columns([1, 6])
with col_brand:
    try:
        st.image("assets/ytu_logo.png", width=120)
    except Exception:
        st.markdown("<h1 style='font-size:4rem; color:#58a6ff;'>S</h1>", unsafe_allow_html=True)

with col_title:
    st.markdown('<h1 class="main-title">Turkish Universities Sentiment Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Üniversite yorumları için duygu analizi platformu</p>', unsafe_allow_html=True)

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
                    Aşağıya bir metin girin ve 4 farklı modelin (BERTurk, CNN-BiLSTM, BiLSTM, CNN) anlık duygu analizini izleyin.
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
            c_res1, c_res2 = st.columns(2, gap="large")
            models_list = ["BERTurk", "CNN-BiLSTM", "BiLSTM", "CNN"]

            for i, model_name in enumerate(models_list):
                data = results.get(model_name)
                target_col = c_res1 if i % 2 == 0 else c_res2

                with target_col:
                    if data is None:
                        st.error(f"{model_name} N/A")
                    else:
                        pred, _ = data
                        sentiment = "POZİTİF" if pred == 1 else "NEGATİF"
                        card_class = "model-card-positive" if pred == 1 else "model-card-negative"
                        emoji = "😊" if pred == 1 else "😡"
                        text_color = "#3fb950" if pred == 1 else "#f85149"

                        st.markdown(
                            f"""
                            <div class="glass-card {card_class}" style="text-align:center; margin-bottom:20px; padding:20px; min-height:160px; display:flex; flex-direction:column; justify-content:center;">
                                <h5 style="color:#8b949e; margin:0; font-weight:700; text-transform:uppercase; letter-spacing:1px; font-size:0.8rem;">{model_name}</h5>
                                <div style="font-size:3rem; margin:10px 0;">{emoji}</div>
                                <h3 style="color:{text_color}; margin:0; letter-spacing:0.5px;">{sentiment}</h3>
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
                    <div style="color:#8b949e; font-size:0.9rem; margin-bottom:5px;">{label}</div>
                    <div style="color:#ffffff; font-size:2rem; font-weight:800; text-shadow:0 0 10px rgba(88,166,255,0.2);">{value}</div>
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
            ''',
            unsafe_allow_html=True
        )

        fig_bar = px.bar(
            metrics_df,
            x="Model",
            y="Macro F1",
            text=metrics_df["Macro F1"].apply(lambda x: f"%{x*100:.1f}"),
            color="Macro F1",
            color_continuous_scale=["#1a7f37", "#3fb950"],
            height=320,
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9d1d9"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[0, 1.05]),
            xaxis=dict(showgrid=False),
            coloraxis_showscale=False,
            margin=dict(t=10, l=0, r=0, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_pie:
        st.markdown(
            '''
            <div class="glass-card">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Üniversite Bazlı Dağılım</h4>
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
                            marker=dict(colors=["#3fb950", "#f85149"], line=dict(color="#0d1117", width=2)),
                        )
                    ]
                )
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9d1d9"),
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=220,
                    showlegend=True,
                    legend=dict(orientation="h", x=0.15, y=-0.1),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Veri yok.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    col_tbl1, col_tbl2 = st.columns(2, gap="large")

    with col_tbl1:
        st.markdown(
            '''
            <div class="glass-card">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Model Performans Özeti</h4>
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

    with col_tbl2:
        st.markdown(
            '''
            <div class="glass-card">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c9d1d9; font-weight:800; font-size:1.15rem; letter-spacing:-0.2px;">Detaylı Sınıf Analizi</h4>
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
            height=300,
        )
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