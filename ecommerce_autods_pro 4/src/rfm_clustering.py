"""
rfm_clustering.py
RFM skorlama, CLTV (Customer Lifetime Value) hesabı,
Elbow & Silhouette analizli K-Means kümeleme ve PCA görselleştirme modülü.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class RFMClusterEngine:
    """RFM analizi, CLTV hesabı, K-Means kümeleme ve PCA boyut indirgeme."""

    def __init__(self, profit_margin: float = 0.22) -> None:
        self.profit_margin = profit_margin
        self.scaler = StandardScaler()
        self.kmeans: Optional[KMeans] = None
        self.pca = PCA(n_components=2, random_state=42)

    @staticmethod
    def compute_rfm_table(df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Temizlenmiş işlem tablosundan müşteri bazlı RFM ve CLTV metriklerini çıkarır."""
        work = df.copy()
        work["order_date"] = pd.to_datetime(work["order_date"])
        ref_date = reference_date or (work["order_date"].max() + pd.Timedelta(days=1))

        rfm = (
            work.groupby("customer_id")
            .agg(
                recency=("order_date", lambda s: int((ref_date - s.max()).days)),
                frequency=("order_id", "nunique"),
                monetary=("total_amount", "sum"),
                avg_order_value=("total_amount", "mean"),
                return_rate=("is_returned", "mean"),
                avg_discount=("discount_rate", "mean"),
                primary_marketplace=("marketplace", lambda s: s.mode().iloc[0] if not s.mode().empty else "Trendyol"),
            )
            .reset_index()
        )

        # 1-5 RFM skorları (Recency ters orantılı: küçük gün = 5 puan)
        rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
        rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

        # Basit ampirik CLTV (12 aylık projeksiyon * net kâr marjı)
        rfm["estimated_cltv"] = np.round(rfm["monetary"] * 0.22 * (1.0 - rfm["return_rate"] * 0.5) * (1 + rfm["f_score"] * 0.15), 2)

        def assign_segment(row: pd.Series) -> str:
            r, f, m = row["r_score"], row["f_score"], row["m_score"]
            if r >= 4 and f >= 4 and m >= 4:
                return "Şampiyonlar (VIP)"
            if r >= 3 and f >= 3:
                return "Sadık Müşteriler"
            if r >= 4 and f <= 2:
                return "Yeni / Potansiyel"
            if r <= 2 and f >= 3:
                return "Riskli (Kazanılmalı)"
            return "Kayıp / Uyuyan"

        rfm["rfm_segment"] = rfm.apply(assign_segment, axis=1)
        return rfm

    def evaluate_k_range(self, rfm_df: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
        """Elbow (WCSS/Inertia) ve Silhouette skorlarını k=2..8 aralığında hesaplar."""
        features = np.log1p(rfm_df[["recency", "frequency", "monetary"]].clip(lower=0))
        scaled = self.scaler.fit_transform(features)

        results: List[Dict] = []
        max_k = min(k_max, len(rfm_df) - 1)
        for k in range(k_min, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(scaled)
            sil = silhouette_score(scaled, labels) if len(set(labels)) > 1 else 0.0
            results.append({"k": k, "inertia": round(float(km.inertia_), 2), "silhouette_score": round(float(sil), 4)})
        return pd.DataFrame(results)

    def fit_predict_clusters(
        self,
        rfm_df: pd.DataFrame,
        n_clusters: int = 4,
        model_save_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """K-Means kümeleme ve 2D PCA koordinatlarını hesaplar; modeli .pkl olarak kaydeder."""
        work = rfm_df.copy()
        features = np.log1p(work[["recency", "frequency", "monetary"]].clip(lower=0))
        scaled = self.scaler.fit_transform(features)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        work["cluster_id"] = self.kmeans.fit_predict(scaled)

        pca_coords = self.pca.fit_transform(scaled)
        work["pca_1"] = np.round(pca_coords[:, 0], 4)
        work["pca_2"] = np.round(pca_coords[:, 1], 4)

        cluster_summary = (
            work.groupby("cluster_id")
            .agg(
                customer_count=("customer_id", "count"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                avg_cltv=("estimated_cltv", "mean"),
            )
            .round(2)
            .reset_index()
        )

        # Küme adlarını harcamaya göre anlamlı etiketle
        ordered = cluster_summary.sort_values("avg_monetary", ascending=False)["cluster_id"].tolist()
        tier_names = ["Elmas Küme", "Altın Küme", "Gümüş Küme", "Bronz / Risk Kümesi", "İzleme Kümesi", "Mikro Küme"]
        name_map = {cid: tier_names[idx] if idx < len(tier_names) else f"Küme {cid}" for idx, cid in enumerate(ordered)}
        work["cluster_name"] = work["cluster_id"].map(name_map)
        cluster_summary["cluster_name"] = cluster_summary["cluster_id"].map(name_map)

        if model_save_path is not None:
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "scaler": self.scaler,
                    "kmeans": self.kmeans,
                    "pca": self.pca,
                    "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
                    "cluster_name_map": name_map,
                },
                model_save_path,
            )

        return work, cluster_summary
