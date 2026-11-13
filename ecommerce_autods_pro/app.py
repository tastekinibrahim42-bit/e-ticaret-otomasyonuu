"""
app.py
AutoDS Tarzı E-Ticaret Otomasyonu & İleri Veri Madenciliği SaaS Platformu
Dark / Executive UI — Streamlit + Plotly
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.autods_engine import AutoDSEngine, PricingRuleConfig
from src.deep_anomaly import DeepAnomalyDetector
from src.predictive_ml import PredictiveChurnEngine
from src.preprocessing import DataPreprocessor
from src.rfm_clustering import RFMClusterEngine
from src.scraper_engine import MarketplaceScraper, generate_synthetic_ecommerce_transactions

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="AutoDS Pro • E-Ticaret Otomasyonu & Veri Madenciliği",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark / Executive SaaS Özel CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0F19;
        color: #E5E7EB;
    }
    section[data-testid="stSidebar"] {
        background-color: #0F1624;
        border-right: 1px solid #1F2937;
    }
    .saas-header {
        background: linear-gradient(135deg, #111827 0%, #1E293B 100%);
        border: 1px solid #263248;
        border-radius: 14px;
        padding: 20px 26px;
        margin-bottom: 18px;
    }
    .saas-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F9FAFB;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .saas-sub {
        font-size: 0.92rem;
        color: #9CA3AF;
        margin-top: 4px;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px 18px;
        transition: border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #3B82F6;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #F9FAFB;
        margin-top: 4px;
    }
    .kpi-delta {
        font-size: 0.80rem;
        color: #10B981;
        margin-top: 2px;
    }
    div[data-testid="stTabs"] button {
        background-color: #111827;
        border-radius: 8px;
        color: #9CA3AF;
        padding: 8px 16px;
        border: 1px solid #1F2937;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #1D4ED8;
        color: #FFFFFF;
        border-color: #3B82F6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_kpi_card(label: str, value: str, delta: str = "", delta_color: str = "#10B981") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta" style="color: {delta_color};">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_or_bootstrap_data(n_products: int, n_customers: int, n_tx: int, seed: int = 42):
    scraper = MarketplaceScraper()
    prod_path = RAW_DIR / "scraped_products.csv"
    tx_path = RAW_DIR / "ecommerce_transactions.csv"

    if not prod_path.exists():
        products_df = scraper.generate_realistic_marketplace_catalog(n_products=n_products, seed=seed, output_path=prod_path)
    else:
        products_df = pd.read_csv(prod_path)

    if not tx_path.exists():
        tx_raw_df = generate_synthetic_ecommerce_transactions(
            n_customers=n_customers, n_transactions=n_tx, seed=seed, output_path=tx_path
        )
    else:
        tx_raw_df = pd.read_csv(tx_path)

    return products_df, tx_raw_df


def main() -> None:
    st.markdown(
        """
        <div class="saas-header">
            <div class="saas-title">⚡ AutoDS Pro • E-Ticaret Otomasyonu & Veri Madenciliği Platformu</div>
            <div class="saas-sub">
                Çoklu Pazaryeri Fiyat/Stok Otomasyonu (Trendyol, Hepsiburada, N11, Amazon TR) • RFM & K-Means • Data Leakage Korumalı Churn • Autoencoder & Isolation Forest
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Kontrol Paneli")
        st.subheader("1. AutoDS Fiyatlandırma Kuralları")
        target_margin = st.slider("Hedef Net Kâr Marjı (%)", 10.0, 45.0, 22.0, 1.0)
        min_margin = st.slider("Minimum Kâr Tabanı (%)", 5.0, 25.0, 10.0, 0.5)
        undercut_step = st.slider("BuyBox Altına İnme (%)", 0.5, 5.0, 1.0, 0.5)
        low_stock_thr = st.number_input("Kritik Stok Eşiği", min_value=1, max_value=30, value=5)

        st.divider()
        st.subheader("2. Veri Madenciliği & ML")
        n_clusters = st.slider("K-Means Küme Sayısı (k)", 2, 7, 4)
        pred_window = st.select_slider("Churn Tahmin Penceresi (Gün)", options=[60, 90, 120], value=90)
        contamination = st.slider("Anomali Hassasiyeti (Contamination)", 0.02, 0.12, 0.05, 0.01)

        st.divider()
        st.subheader("3. Veri Yönetimi")
        if st.button("🔄 Sentetik Pazaryeri Verisini Yenile", use_container_width=True):
            scraper = MarketplaceScraper()
            scraper.generate_realistic_marketplace_catalog(n_products=96, seed=99, output_path=RAW_DIR / "scraped_products.csv")
            generate_synthetic_ecommerce_transactions(
                n_customers=900, n_transactions=6500, seed=99, output_path=RAW_DIR / "ecommerce_transactions.csv"
            )
            st.cache_data.clear()
            st.rerun()

    products_raw, tx_raw = load_or_bootstrap_data(n_products=96, n_customers=900, n_tx=6500)

    # Pipeline çalıştırma
    autods = AutoDSEngine(
        PricingRuleConfig(
            target_margin_pct=target_margin,
            min_margin_pct=min_margin,
            undercut_step_pct=undercut_step,
            low_stock_threshold=int(low_stock_thr),
        )
    )
    evaluated_products = autods.evaluate_catalog(products_raw)
    arbitrage_df = autods.find_arbitrage_opportunities(evaluated_products, min_roi_pct=18.0)
    kpis = autods.summarize_kpis(evaluated_products)

    preprocessor = DataPreprocessor(iqr_multiplier=1.5)
    tx_clean, prep_report = preprocessor.clean_transactions(tx_raw, output_path=PROCESSED_DIR / "cleaned_transactions.csv")

    rfm_engine = RFMClusterEngine()
    rfm_table = rfm_engine.compute_rfm_table(tx_clean)
    k_eval_df = rfm_engine.evaluate_k_range(rfm_table, k_min=2, k_max=7)
    clustered_rfm, cluster_summary = rfm_engine.fit_predict_clusters(
        rfm_table, n_clusters=n_clusters, model_save_path=MODELS_DIR / "kmeans_rfm.pkl"
    )

    churn_engine = PredictiveChurnEngine(prediction_window_days=pred_window)
    ml_metrics_df, churn_scored_df, feat_imp_df, cutoff_date = churn_engine.train_and_evaluate(
        tx_clean, model_save_path=MODELS_DIR / "churn_model.pkl"
    )

    anomaly_detector = DeepAnomalyDetector(contamination=contamination)
    anomaly_df, anomaly_summary = anomaly_detector.detect_anomalies(
        tx_clean, model_save_path=MODELS_DIR / "anomaly_models.pkl"
    )

    tabs = st.tabs([
        "📊 Yönetici Özeti",
        "🤖 AutoDS Fiyat & Arbitraj",
        "🔎 Veri Ön İşleme (IQR)",
        "🎯 RFM & K-Means (PCA)",
        "📈 Churn Tahmini (ML)",
        "🛡️ Derin Anomali",
    ])

    # 1. YÖNETİCİ ÖZETİ
    with tabs[0]:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            render_kpi_card("Takip Edilen Ürün", f"{kpis['total_products']}", "4 Pazaryeri Aktif")
        with c2:
            render_kpi_card("Ort. Net Marj", f"%{kpis['avg_margin_pct']}", f"Ort. ROI: %{kpis['avg_roi_pct']}")
        with c3:
            render_kpi_card("BuyBox Kazanım", f"%{kpis['buybox_win_rate_pct']}", "Dinamik Fiyatlama")
        with c4:
            render_kpi_card("Kritik Alarmlar", f"{kpis['critical_alerts']}", "Stok / Marj Uyarısı", "#F59E0B")
        with c5:
            render_kpi_card(
                "Tespit Edilen Anomali",
                f"{anomaly_summary['anomaly_count']} Müşteri",
                f"Oran: %{anomaly_summary['anomaly_rate_pct']}",
                "#EF4444",
            )

        st.markdown("### ")
        col_left, col_right = st.columns(2)
        with col_left:
            mp_rev = tx_clean.groupby("marketplace", as_index=False)["total_amount"].sum()
            fig_mp = px.bar(
                mp_rev,
                x="marketplace",
                y="total_amount",
                color="marketplace",
                title="Pazaryeri Bazlı Toplam Ciro (₺)",
                template="plotly_dark",
            )
            fig_mp.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827", showlegend=False)
            st.plotly_chart(fig_mp, use_container_width=True)

        with col_right:
            alert_counts = evaluated_products["alert_status"].value_counts().reset_index()
            alert_counts.columns = ["Durum", "Adet"]
            fig_al = px.pie(
                alert_counts,
                names="Durum",
                values="Adet",
                hole=0.52,
                title="AutoDS Katalog Sağlık & Alarm Dağılımı",
                template="plotly_dark",
            )
            fig_al.update_layout(paper_bgcolor="#111827")
            st.plotly_chart(fig_al, use_container_width=True)

    # 2. AUTODS FİYAT & ARBİTRAJ
    with tabs[1]:
        st.subheader("Canlı Ürün URL Tarama & Dinamik Fiyatlandırma")
        col_url1, col_url2, col_url3 = st.columns([4, 1.5, 1.2])
        with col_url1:
            target_url = st.text_input("Pazaryeri Ürün URL'si", placeholder="https://www.trendyol.com/...")
        with col_url2:
            target_mp = st.selectbox("Pazaryeri", MarketplaceScraper.SUPPORTED_MARKETPLACES)
        with col_url3:
            st.markdown("<br>", unsafe_allow_html=True)
            scrape_btn = st.button("URL Tara", use_container_width=True)

        if scrape_btn and target_url.strip():
            live_scraper = MarketplaceScraper(use_playwright=False)
            res = live_scraper.scrape_product_url(target_url.strip(), marketplace=target_mp)
            if res is not None:
                st.success(f"Ürün çekildi: {res.title} — {res.current_price} ₺")
            else:
                st.warning("Canlı URL bot korumasına takıldı veya fiyat etiketi okunamadı; mevcut katalog verisiyle devam ediliyor.")

        mp_filter = st.multiselect(
            "Pazaryeri Filtresi",
            options=sorted(evaluated_products["marketplace"].unique()),
            default=sorted(evaluated_products["marketplace"].unique()),
        )
        filtered_eval = evaluated_products[evaluated_products["marketplace"].isin(mp_filter)]

        st.dataframe(
            filtered_eval[
                [
                    "product_id", "title", "marketplace", "supplier_cost", "current_price",
                    "competitor_min_price", "recommended_price", "net_profit", "net_margin_pct",
                    "roi_pct", "stock_qty", "buybox_status", "alert_status",
                ]
            ],
            use_container_width=True,
            height=340,
        )

        st.subheader("💎 Yüksek ROI Arbitraj Fırsatları")
        st.dataframe(
            arbitrage_df[
                ["product_id", "title", "marketplace", "supplier_cost", "recommended_price", "net_profit", "roi_pct", "arbitrage_score", "stock_qty"]
            ].head(15),
            use_container_width=True,
        )

    # 3. VERİ ÖN İŞLEME (IQR)
    with tabs[2]:
        st.subheader("Veri Temizleme, Eksik Değer & IQR Aykırı Değer Raporu")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            render_kpi_card("Ham Kayıt", f"{prep_report.raw_rows:,}", "Sipariş Satırı")
        with p2:
            render_kpi_card("Temiz Kayıt", f"{prep_report.cleaned_rows:,}", f"Silinen Tekrar: {prep_report.duplicates_removed}")
        with p3:
            render_kpi_card("Doldurulan Eksik", f"{prep_report.missing_values_imputed}", "Medyan / Mod İle")
        with p4:
            total_iqr = sum(prep_report.iqr_capped_counts.values())
            render_kpi_card("IQR Baskılanan", f"{total_iqr}", "Winsorization (1.5x IQR)")

        fig_iqr = go.Figure()
        fig_iqr.add_trace(go.Box(y=tx_clean["total_amount_raw"], name="Ham Sepet Tutarı (₺)", marker_color="#EF4444"))
        fig_iqr.add_trace(go.Box(y=tx_clean["total_amount_iqr_capped"], name="IQR Baskılanmış Tutar (₺)", marker_color="#10B981"))
        fig_iqr.update_layout(
            title="IQR Aykırı Değer Baskılama Karşılaştırması",
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
        )
        st.plotly_chart(fig_iqr, use_container_width=True)

    # 4. RFM & K-MEANS (PCA)
    with tabs[3]:
        st.subheader("RFM Segmentasyonu, Elbow/Silhouette & 2D PCA Kümeleme")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            fig_elbow = px.line(
                k_eval_df, x="k", y="inertia", markers=True,
                title="Elbow Analizi (WCSS / Inertia)", template="plotly_dark",
            )
            fig_elbow.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827")
            st.plotly_chart(fig_elbow, use_container_width=True)
        with col_e2:
            fig_sil = px.bar(
                k_eval_df, x="k", y="silhouette_score",
                title="Silhouette Skoru (Küme Ayrışma Kalitesi)", template="plotly_dark",
            )
            fig_sil.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827")
            st.plotly_chart(fig_sil, use_container_width=True)

        fig_pca = px.scatter(
            clustered_rfm,
            x="pca_1",
            y="pca_2",
            color="cluster_name",
            size="monetary",
            hover_data=["customer_id", "recency", "frequency", "monetary", "estimated_cltv", "rfm_segment"],
            title=f"K-Means (k={n_clusters}) • 2D PCA Müşteri Uzayı",
            template="plotly_dark",
        )
        fig_pca.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827", height=460)
        st.plotly_chart(fig_pca, use_container_width=True)

        st.dataframe(cluster_summary, use_container_width=True)

    # 5. CHURN TAHMİNİ (ML)
    with tabs[4]:
        st.subheader("Data Leakage Korumalı Churn & Tekrar Satın Alma Modelleri")
        st.info(
            f"Zamansal Kesim Tarihi (Cutoff): **{cutoff_date.strftime('%Y-%m-%d')}** — "
            f"Özellikler yalnızca bu tarihten önceki gözlem penceresinden çıkarıldı; hedef değişken sonraki {pred_window} günlük pencereden üretildi."
        )

        col_m1, col_m2 = st.columns([1.2, 1])
        with col_m1:
            st.markdown("#### Model Performans Karşılaştırması")
            st.dataframe(ml_metrics_df, use_container_width=True)
        with col_m2:
            fig_imp = px.bar(
                feat_imp_df.sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                title=f"En İyi Model ({churn_engine.best_model_name}) Öznitelik Önemleri",
                template="plotly_dark",
            )
            fig_imp.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827", height=320)
            st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### Yüksek Churn Riskli Müşteriler (Aksiyon Listesi)")
        st.dataframe(
            churn_scored_df.sort_values("churn_probability", ascending=False).head(25),
            use_container_width=True,
        )

    # 6. DERİN ANOMALİ (AUTOENCODER + ISOLATION FOREST)
    with tabs[5]:
        st.subheader("Autoencoder (MLPRegressor) + Isolation Forest Anomali Tespiti")
        fig_anom = px.scatter(
            anomaly_df,
            x="ae_recon_error",
            y="iso_score",
            color=anomaly_df["is_anomaly"].map({0: "Normal", 1: "Anomali / Şüpheli"}),
            size="total_spend",
            hover_data=["customer_id", "return_rate", "avg_basket", "avg_discount", "anomaly_reason"],
            title="Autoencoder Yeniden Yapılandırma Hatası vs. Isolation Forest Skoru",
            template="plotly_dark",
            color_discrete_map={"Normal": "#3B82F6", "Anomali / Şüpheli": "#EF4444"},
        )
        fig_anom.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827", height=450)
        st.plotly_chart(fig_anom, use_container_width=True)

        st.markdown("#### Tespit Edilen Anormal / Şüpheli Müşteri Profilleri")
        st.dataframe(
            anomaly_df[anomaly_df["is_anomaly"] == 1][
                ["customer_id", "tx_count", "total_spend", "avg_basket", "return_rate", "avg_discount", "ensemble_anomaly_score", "anomaly_reason"]
            ],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
