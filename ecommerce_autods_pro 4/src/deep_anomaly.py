"""
deep_anomaly.py
Autoencoder (MLPRegressor bottleneck mimarisi) davranış temsili ve
Isolation Forest çok boyutlu anomali tespit modülü.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler


class DeepAnomalyDetector:
    """Autoencoder + Isolation Forest hibrit e-ticaret anomali tespit motoru."""

    BEHAVIOR_COLS: List[str] = [
        "tx_count",
        "total_spend",
        "avg_basket",
        "max_basket",
        "avg_qty",
        "return_rate",
        "avg_discount",
        "avg_shipping_days",
    ]

    def __init__(self, contamination: float = 0.05) -> None:
        self.contamination = contamination
        self.scaler = RobustScaler()
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(16, 4, 16),
            activation="relu",
            solver="adam",
            max_iter=350,
            random_state=42,
        )
        self.iso_forest = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42,
        )

    @staticmethod
    def build_behavior_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """İşlem verisinden müşteri davranış matrisini oluşturur."""
        beh = (
            df.groupby("customer_id")
            .agg(
                tx_count=("order_id", "nunique"),
                total_spend=("total_amount", "sum"),
                avg_basket=("total_amount", "mean"),
                max_basket=("total_amount", "max"),
                avg_qty=("quantity", "mean"),
                return_rate=("is_returned", "mean"),
                avg_discount=("discount_rate", "mean"),
                avg_shipping_days=("shipping_days", "mean"),
            )
            .reset_index()
        )
        return beh

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        model_save_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        """
        1. Davranış matrisini RobustScaler ile ölçekler.
        2. Autoencoder (MLPRegressor) ile X -> X yeniden yapılandırma hatasını (MSE) hesaplar.
        3. Isolation Forest ile izolasyon anomali skorunu hesaplar.
        4. İki skoru normalize edip birleştirerek hibrit anomali tespiti yapar.
        """
        beh = self.build_behavior_matrix(df)
        X = beh[self.BEHAVIOR_COLS].values
        X_scaled = self.scaler.fit_transform(X)

        # Autoencoder: Girdiyi bottleneck üzerinden yeniden üret
        self.autoencoder.fit(X_scaled, X_scaled)
        X_recon = self.autoencoder.predict(X_scaled)
        recon_mse = np.mean((X_scaled - X_recon) ** 2, axis=1)

        # Isolation Forest
        iso_pred = self.iso_forest.fit_predict(X_scaled)  # -1 anomali, 1 normal
        iso_raw_score = -self.iso_forest.decision_function(X_scaled)

        # 0-1 aralığına min-max normalizasyon
        ae_norm = (recon_mse - recon_mse.min()) / np.maximum(recon_mse.max() - recon_mse.min(), 1e-9)
        iso_norm = (iso_raw_score - iso_raw_score.min()) / np.maximum(iso_raw_score.max() - iso_raw_score.min(), 1e-9)

        beh["ae_recon_error"] = np.round(recon_mse, 4)
        beh["iso_score"] = np.round(iso_norm, 4)
        beh["ensemble_anomaly_score"] = np.round(0.5 * ae_norm + 0.5 * iso_norm, 4)

        threshold = float(np.quantile(beh["ensemble_anomaly_score"], 1.0 - self.contamination))
        beh["is_anomaly"] = ((beh["ensemble_anomaly_score"] >= threshold) | (iso_pred == -1)).astype(int)

        # Anomali nedenini yorumla
        ret_q95 = beh["return_rate"].quantile(0.95)
        basket_q95 = beh["avg_basket"].quantile(0.95)
        disc_q95 = beh["avg_discount"].quantile(0.95)
        qty_q95 = beh["avg_qty"].quantile(0.95)

        reasons: List[str] = []
        for _, row in beh.iterrows():
            if row["is_anomaly"] == 0:
                reasons.append("Normal Davranış")
                continue
            flags = []
            if row["return_rate"] >= ret_q95:
                flags.append("Yüksek İade İstismarı")
            if row["avg_basket"] >= basket_q95:
                flags.append("Olağandışı Sepet Tutarı")
            if row["avg_discount"] >= disc_q95:
                flags.append("Aşırı İndirim/Kupon Kullanımı")
            if row["avg_qty"] >= qty_q95:
                flags.append("Toplu/Bot Adet Siparişi")
            reasons.append(", ".join(flags) if flags else "Çok Boyutlu Davranış Sapması")

        beh["anomaly_reason"] = reasons

        summary = {
            "total_customers": int(len(beh)),
            "anomaly_count": int(beh["is_anomaly"].sum()),
            "anomaly_rate_pct": round(float(beh["is_anomaly"].mean() * 100.0), 2),
            "ae_threshold_95": round(float(np.quantile(recon_mse, 0.95)), 4),
            "ensemble_threshold": round(threshold, 4),
        }

        if model_save_path is not None:
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "scaler": self.scaler,
                    "autoencoder": self.autoencoder,
                    "iso_forest": self.iso_forest,
                    "behavior_cols": self.BEHAVIOR_COLS,
                    "summary": summary,
                },
                model_save_path,
            )

        return beh.sort_values("ensemble_anomaly_score", ascending=False).reset_index(drop=True), summary
