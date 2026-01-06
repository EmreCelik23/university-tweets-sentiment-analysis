import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from predict import get_multi_model_prediction  # GERÇEK TAHMİN MOTORU

# =====================================================================
# SABİT METRİKLER (TEST SONUÇLARINDAN)
# =====================================================================

MODEL_METRICS = {
    "BERTurk": {
        "Accuracy": 0.9247,
        "Macro F1": 0.9045,
        "Precision": 0.9064,
        "Recall": 0.9027,
        "Support": 757,
    },
    "CNN-BiLSTM": {
        "Accuracy": 0.8534,
        "Macro F1": 0.7899,
        "Precision": 0.8528,
        "Recall": 0.7610,
        "Support": 757,
    },
    "BiLSTM": {
        "Accuracy": 0.8296,
        "Macro F1": 0.7697,
        "Precision": 0.7949,
        "Recall": 0.7538,
        "Support": 757,
    },
    "CNN": {
        "Accuracy": 0.8151,
        "Macro F1": 0.7558,
        "Precision": 0.7702,
        "Recall": 0.7453,
        "Support": 757,
    },
}

# Her model için SINIF (olumsuz / olumlu) bazlı metrikler
# BERTurk değerleri senin classification_report çıktından alınmış.
# Diğer modelleri elindeki çıktılara göre doldurabilirsin.
MODEL_CLASS_METRICS = {
    "BERTurk": {
        "0_olumsuz": {
            "precision": 0.9458,
            "recall": 0.9510,
            "f1": 0.9484,
            "support": 551,
        },
        "1_olumlu": {
            "precision": 0.8670,
            "recall": 0.8544,
            "f1": 0.8606,
            "support": 206,
        },
    },
    "CNN-BiLSTM": {
        # TODO: burayı kendi CNN-BiLSTM classification_report çıktına göre doldur
        "0_olumsuz": {"precision": 0.8564, "recall": 0.9201, "f1": 0.8871, "support": 551},
        "1_olumlu": {"precision": 0.7333, "recall": 0.5874, "f1": 0.6523, "support": 206},
    },
    "BiLSTM": {
        # TODO: burayı kendi BiLSTM classification_report çıktına göre doldur
        "0_olumsuz": {"precision": 0.8549, "recall": 0.8984, "f1": 0.8761, "support": 551},
        "1_olumlu": {"precision": 0.6854, "recall": 0.5922, "f1": 0.6354, "support": 206},
    },
    "CNN": {
        # TODO: burayı kendi CNN classification_report çıktına göre doldur
        "0_olumsuz": {"precision": 0.8537, "recall": 0.9637, "f1": 0.9054, "support": 551},
        "1_olumlu": {"precision": 0.8519, "recall": 0.5583, "f1": 0.6745, "support": 206},
    },
}


@st.cache_data
def load_main_dataset(path: str = "data/tweetVeriseti.xlsx"):
    """Dashboard için ana veri setini yükler."""
    try:
        df = pd.read_excel(path)
        return df
    except Exception:
        return None


DATA_DF = load_main_dataset()

# === SESSION STATE: Aktif öğrenme havuzu & son analiz ===
if "active_pool" not in st.session_state:
    st.session_state["active_pool"] = []

if "last_input" not in st.session_state:
    st.session_state["last_input"] = ""

if "last_results" not in st.session_state:
    st.session_state["last_results"] = None

# =====================================================================
# SAYFA AYARLARI & CSS
# =====================================================================

st.set_page_config(
    page_title="Üniversite Duygu Analiz Platformu",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Genel Arka Plan */
    .stApp {
        background-color: #000000;
        background-image: radial-gradient(circle at 50% 50%, #1e1e2f 0%, #000000 100%);
    }

    /* Üst Başlık Stili */
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800;
        text-shadow: 0 0 10px #00d2ff;
    }

    /* Glassmorphism Kartlar */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: #00d2ff;
    }

    /* Model Kartları Renklendirme */
    .model-card-positive {
        border-left: 5px solid #00ff88;
    }
    .model-card-negative {
        border-left: 5px solid #ff0055;
    }

    /* Butonlar */
    .stButton>button {
        background: linear-gradient(45deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #00d2ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# HEADER
# =====================================================================

col_h1, col_h2 = st.columns([1, 6])
with col_h1:
    st.image("assets/ytu_logo.png", width=90)
with col_h2:
    st.title("Türkiye Üniversiteleri Duygu Analiz Platformu")
    st.markdown(
        "BERTurk ve derin öğrenme tabanlı çoklu model ile tweet duygu analizi"
    )

st.markdown("---")

# =====================================================================
# TABS
# =====================================================================

tab1, tab2, tab3 = st.tabs(
    ["🚀 CANLI ARENA (Multi-Model)", "📊 DASHBOARD & İSTATİSTİK", "📂 VERİ LABORATUVARI"]
)

# =====================================================================
# TAB 1: CANLI ARENA
# =====================================================================
with tab1:
    st.markdown(
        "<div class='glass-card'><h3>🧠 Anlık Analiz Modülü</h3>"
        "<p>Metni girin, 4 farklı yapay zeka modeli aynı anda analiz etsin.</p></div>",
        unsafe_allow_html=True,
    )

    txt_input = st.text_area(
        "Analiz edilecek yorumu giriniz:",
        height=100,
        placeholder="Örn: İTÜ çok güzel, kampüsü harika ama ulaşım biraz zor...",
    )

    if st.button("ANALİZİ BAŞLAT", type="primary", use_container_width=True):
        if txt_input.strip():
            try:
                with st.spinner("Modeller çalışıyor..."):
                    results = get_multi_model_prediction(txt_input)
                st.session_state["last_input"] = txt_input
                st.session_state["last_results"] = results
            except Exception as e:
                st.error(f"Model tahmini sırasında hata oluştu: {e}")
        else:
            st.warning("Lütfen analiz edilecek bir metin giriniz.")

    # Eğer daha önce analiz yapılmışsa sonuçları göster
    if st.session_state["last_results"] is not None:
        results = st.session_state["last_results"]

        st.markdown("### 🧬 Model Sonuçları")
        cols = st.columns(4)

        for i, model_name in enumerate(["BERTurk", "CNN-BiLSTM", "BiLSTM", "CNN"]):
            data = results.get(model_name, None)

            with cols[i]:
                if data is None:
                    st.markdown(
                        f"""
                        <div class='glass-card model-card-negative' style='text-align:center; opacity:0.7;'>
                            <h4 style='color:#ccc'>{model_name}</h4>
                            <h2 style='color:#ffcc00'>⚠ Kullanılamıyor</h2>
                            <p>Bu model için tahmin üretilemedi.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    continue

                pred, conf = data  # conf'u hesaplıyoruz ama UI'da göstermiyoruz
                sentiment = "POZİTİF" if pred == 1 else "NEGATİF"
                color = "#00ff88" if pred == 1 else "#ff0055"
                icon = "😊" if pred == 1 else "😡"

                st.markdown(
                    f"""
                    <div class='glass-card model-card-{'positive' if pred == 1 else 'negative'}' style='text-align:center;'>
                        <h4 style='color:#ccc'>{model_name}</h4>
                        <h2 style='color:{color}'>{icon} {sentiment}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # === AKTİF ÖĞRENME MODÜLÜ ===
        st.markdown("---")
        st.markdown(
            "<div class='glass-card'><h3>📚 Aktif Öğrenme Modülü</h3>"
            "<p>Son analiz edilen cümleyi <b>text / tags</b> formatında Excel havuzuna kaydedebilirsin. "
            "<code>text</code> kolonunda metin, <code>tags</code> kolonunda ise "
            "pozitif için 1, negatif için 0 tutulur.</p></div>",
            unsafe_allow_html=True,
        )

        col_al1, col_al2 = st.columns([3, 1])

        with col_al1:
            st.markdown("**text**")
            text_for_label = st.text_area(
                "",
                value=st.session_state["last_input"],
                height=100,
                placeholder="Son analiz edilen metin burada görünecek...",
            )

        with col_al2:
            st.markdown("**tags**")
            tag_choice = st.radio(
                "",
                ["Pozitif (1)", "Negatif (0)"],
                index=0,
                key="active_tag_radio",
            )

            if st.button("Excel'e kaydet"):
                if not text_for_label.strip():
                    st.warning("Kaydetmeden önce bir metin olmalı.")
                else:
                    tag_value = 1 if "Pozitif" in tag_choice else 0
                    # Session içi havuza ekle
                    st.session_state["active_pool"].append(
                        {"text": text_for_label, "tags": tag_value}
                    )
                    # Excel'e yaz / append et
                    try:
                        try:
                            existing = pd.read_excel("active_learning_pool.xlsx")
                        except FileNotFoundError:
                            existing = pd.DataFrame(columns=["text", "tags"])

                        new_row = pd.DataFrame(
                            [{"text": text_for_label, "tags": tag_value}]
                        )
                        out_df = pd.concat([existing, new_row], ignore_index=True)
                        out_df.to_excel("active_learning_pool.xlsx", index=False)
                        st.success("Örnek active_learning_pool.xlsx dosyasına kaydedildi ✅")
                    except Exception as e:
                        st.error(f"Excel kaydı sırasında hata oluştu: {e}")

        if st.session_state["active_pool"]:
            al_df = pd.DataFrame(st.session_state["active_pool"])
            st.markdown("#### 🎯 Etiket Havuzu (Session içindeki son kayıtlar)")
            st.dataframe(al_df.tail(10), use_container_width=True)

            csv_bytes = al_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Session etiket havuzunu CSV olarak indir",
                data=csv_bytes,
                file_name="active_learning_pool_session.csv",
                mime="text/csv",
            )

# =====================================================================
# TAB 2: DASHBOARD & İSTATİSTİK
# =====================================================================
with tab2:
    total_tweets = len(DATA_DF) if DATA_DF is not None else 0

    if DATA_DF is not None and "tags" in DATA_DF.columns:
        pos_count = int((DATA_DF["tags"] == 1).sum())
        neg_count = int((DATA_DF["tags"] == 0).sum())
        total_labeled = pos_count + neg_count if (pos_count + neg_count) > 0 else 1
        pos_ratio = pos_count / total_labeled
        neg_ratio = neg_count / total_labeled
    else:
        pos_count, neg_count = 1374, 3669
        total_labeled = pos_count + neg_count
        pos_ratio = pos_count / total_labeled
        neg_ratio = neg_count / total_labeled

    avg_macro_f1 = (
        sum(m["Macro F1"] for m in MODEL_METRICS.values()) / len(MODEL_METRICS)
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Toplam Veri", f"{total_tweets:,}".replace(",", "."))
    m2.metric(
        "Toplam Memnuniyet (Pozitif / Negatif)",
        f"%{pos_ratio * 100:.1f} / %{neg_ratio * 100:.1f}",
    )
    m3.metric("Ortalama Macro F1", f"%{avg_macro_f1 * 100:.1f}")

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    # --- Model Liderlik Tablosu ---
    with col_g1:
        st.markdown(
            "<div class='glass-card'><h4>🏆 Model Liderlik Tablosu</h4></div>",
            unsafe_allow_html=True,
        )

        metrics_df = pd.DataFrame(MODEL_METRICS).T.reset_index()
        metrics_df = metrics_df.rename(columns={"index": "Model"})

        fig_bar = px.bar(
            metrics_df,
            x="Model",
            y="Macro F1",
            text=metrics_df["Macro F1"].apply(lambda x: f"%{x*100:.1f}"),
        )
        fig_bar.update_layout(
            yaxis_range=[0, 1],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption("Not: Değerler test seti sonuçlarından (Macro F1) alınmıştır.")

        # Özet tablo
        st.markdown("#### 📊 Model Bazlı Özet Performans Tablosu")
        perf_table = metrics_df[
            ["Model", "Accuracy", "Macro F1", "Precision", "Recall", "Support"]
        ].copy()
        st.dataframe(
            perf_table.style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Macro F1": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "Support": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

        # Detaylı sınıf bazlı tablo
        st.markdown("#### 🔬 Detaylı (Olumlu / Olumsuz) Sınıf Metrikleri")
        rows = []
        for model_name, class_dict in MODEL_CLASS_METRICS.items():
            for cls_name, vals in class_dict.items():
                rows.append(
                    {
                        "Model": model_name,
                        "Sınıf": cls_name,
                        "Precision": vals["precision"],
                        "Recall": vals["recall"],
                        "F1": vals["f1"],
                        "Support": vals["support"],
                    }
                )
        class_df = pd.DataFrame(rows)
        st.dataframe(
            class_df.style.format(
                {
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1": "{:.3f}",
                    "Support": "{:.0f}",
                }
            ),
            use_container_width=True,
        )

    # --- Üniversite bazlı memnuniyet dağılımı ---
    with col_g2:
        st.markdown(
            "<div class='glass-card'><h4>🎓 Üniversite Bazlı Memnuniyet Dağılımı</h4></div>",
            unsafe_allow_html=True,
        )

        if DATA_DF is not None and {"tags", "university"}.issubset(DATA_DF.columns):
            uni_list = (
                sorted(DATA_DF["university"].dropna().unique().tolist())
                if len(DATA_DF) > 0
                else []
            )
            selected_uni = st.selectbox(
                "Üniversite seçiniz:",
                ["Tüm Üniversiteler"] + uni_list,
            )

            if selected_uni == "Tüm Üniversiteler":
                subset = DATA_DF
            else:
                subset = DATA_DF[DATA_DF["university"] == selected_uni]

            if subset is not None and len(subset) > 0:
                pos_u = int((subset["tags"] == 1).sum())
                neg_u = int((subset["tags"] == 0).sum())
                values = [pos_u, neg_u]
            else:
                values = [0, 0]

            labels = ["Pozitif", "Negatif"]
            fig_pie_uni = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.6,
                        marker_colors=["#00ff88", "#ff0055"],
                    )
                ]
            )
            fig_pie_uni.update_layout(
                title=f"{selected_uni} Memnuniyet Dağılımı",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_pie_uni, use_container_width=True)
        else:
            st.info("Üniversite ve tags kolonları bulunamadı, grafik gösterilemiyor.")

# =====================================================================
# TAB 3: VERİ LABORATUVARI
# =====================================================================
with tab3:
    st.markdown("### 📂 Toplu Veri Yükle & Test Et")
    uploaded_file = st.file_uploader(
        "Excel/CSV Dosyasını Sürükle", type=["xlsx", "csv"]
    )

    if uploaded_file:
        df = (
            pd.read_excel(uploaded_file)
            if uploaded_file.name.endswith("xlsx")
            else pd.read_csv(uploaded_file)
        )
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Rastgele 3 örnek çek ve analiz et"):
            samples = df.sample(3)

            text_col = next(
                (col for col in df.columns if "text" in col.lower() or "tweet" in col.lower()),
                None,
            )
            uni_col = next(
                (col for col in df.columns if "uni" in col.lower()), None
            )

            if not text_col:
                st.error(
                    "Metin sütunu bulunamadı. Lütfen 'text' veya 'tweet' içeren bir sütun adı kullanın."
                )
            else:
                for _, row in samples.iterrows():
                    txt = str(row[text_col])
                    uni = (
                        str(row[uni_col])
                        if (uni_col and pd.notna(row[uni_col]))
                        else "Genel"
                    )

                    try:
                        res = get_multi_model_prediction(txt, university=uni)
                    except Exception as e:
                        st.error(f"Bu örnek için model tahmini yapılamadı: {e}")
                        continue
                    bert_pred = res.get("BERTurk", (0, 0.0))[0]

                    color = "#00ff88" if bert_pred == 1 else "#ff0055"
                    border = f"4px solid {color}"

                    def label_for(model_name: str) -> str:
                        data = res.get(model_name, None)
                        if data is None:
                            return "ÇALIŞMIYOR"
                        pred, _ = data
                        return "POZİTİF" if pred == 1 else "NEGATİF"

                    st.markdown(
                        f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-top: 10px; border-left: {border};">
                            <small style="color: #888;">{uni}</small>
                            <p style="font-size: 1.1em; color: white;">"{txt}"</p>
                            <div style="display:flex; gap:8px; flex-wrap:wrap;">
                                <span style="background:{color}; color:black; padding:2px 8px; border-radius:4px; font-weight:bold;">
                                    BERTurk: {label_for('BERTurk')}
                                </span>
                                <span style="background:#222; color:#ccc; padding:2px 8px; border-radius:4px;">
                                    CNN-BiLSTM: {label_for('CNN-BiLSTM')}
                                </span>
                                <span style="background:#222; color:#ccc; padding:2px 8px; border-radius:4px;">
                                    BiLSTM: {label_for('BiLSTM')}
                                </span>
                                <span style="background:#222; color:#ccc; padding:2px 8px; border-radius:4px;">
                                    CNN: {label_for('CNN')}
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Analiz için bir Excel/CSV dosyası yükleyebilirsin.")