"""
predictive_ml.py
Data Leakage (Veri Sızıntısı) Korumalı Churn & Tekrar Satın Alma Tahmin Modülü.
Zamansal kesim (Temporal Split / Observation vs Prediction Window) uygulayarak
Logistic Regression, Random Forest ve Gradient Boosting modellerini eğitir ve karşılaştırır.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PredictiveChurnEngine:
    """Zamansal veri sızıntısı korumalı Churn ve Tekrar Satın Alma tahmin motoru."""

    FEATURE_COLS: List[str] = [
        "obs_recency_days",
        "obs_frequency",
        "obs_monetary",
        "obs_avg_basket",
        "obs_return_rate",
        "obs_avg_discount",
        "obs_avg_shipping_days",
        "obs_category_diversity",
        "obs_marketplace_diversity",
    ]

    def __init__(self, prediction_window_days: int = 90) -> None:
        self.prediction_window_days = prediction_window_days
        self.models: Dict[str, Pipeline] = {}
        self.best_model_name: Optional[str] = None

    def build_temporal_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """
        Veri sızıntısını önlemek için veriyi iki zamansal pencereye böler:
        - Gözlem Penceresi (order_date <= cutoff_date): Yalnızca buradan özellik (X) üretilir.
        - Tahmin Penceresi (order_date > cutoff_date): Buradaki satın alma durumu hedef (y) olur.
          churned = 1 -> tahmin penceresinde hiç sipariş vermedi (tekrar satın almadı)
          repurchased = 1 - churned
        """
        work = df.copy()
        work["order_date"] = pd.to_datetime(work["order_date"])
        max_date = work["order_date"].max()
        cutoff_date = max_date - pd.Timedelta(days=self.prediction_window_days)

        obs_df = work[work["order_date"] <= cutoff_date].copy()
        pred_df = work[work["order_date"] > cutoff_date].copy()

        if obs_df.empty:
            raise ValueError("Gözlem penceresi boş. Veri tarih aralığı yetersiz.")

        features = (
            obs_df.groupby("customer_id")
            .agg(
                obs_recency_days=("order_date", lambda s: int((cutoff_date - s.max()).days)),
                obs_frequency=("order_id", "nunique"),
                obs_monetary=("total_amount", "sum"),
                obs_avg_basket=("total_amount", "mean"),
                obs_return_rate=("is_returned", "mean"),
                obs_avg_discount=("discount_rate", "mean"),
                obs_avg_shipping_days=("shipping_days", "mean"),
                obs_category_diversity=("category", "nunique"),
                obs_marketplace_diversity=("marketplace", "nunique"),
            )
            .reset_index()
        )

        active_in_future = set(pred_df["customer_id"].unique())
        features["repurchased"] = features["customer_id"].isin(active_in_future).astype(int)
        features["churned"] = 1 - features["repurchased"]
        return features, cutoff_date

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        model_save_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        """
        LR, RF ve GB modellerini eğitir, performans tablosunu,
        özellik önem skorlarını ve müşteri bazlı churn olasılıklarını döndürür.
        """
        dataset, cutoff_date = self.build_temporal_dataset(df)
        X = dataset[self.FEATURE_COLS]
        y = dataset["churned"]

        # Eğer tek sınıf kalırsa stratify hatasını önle
        stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify_target
        )

        candidates = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]),
            "Random Forest": Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=180, max_depth=7, class_weight="balanced", random_state=42
                )),
            ]),
            "Gradient Boosting": Pipeline([
                ("clf", GradientBoostingClassifier(
                    n_estimators=140, learning_rate=0.06, max_depth=3, random_state=42
                )),
            ]),
        }

        metrics_rows: List[Dict] = []
        best_auc = -1.0

        for name, pipe in candidates.items():
            pipe.fit(X_train, y_train)
            self.models[name] = pipe
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred.astype(float)

            auc_val = float(roc_auc_score(y_test, y_prob)) if y_test.nunique() > 1 else 0.5
            metrics_rows.append({
                "Model": name,
                "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "F1 Score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "ROC-AUC": round(auc_val, 4),
            })

            if auc_val > best_auc:
                best_auc = auc_val
                self.best_model_name = name

        metrics_df = pd.DataFrame(metrics_rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

        # En iyi model ile tüm müşteriler için churn ve tekrar satın alma olasılığı hesapla
        best_pipe = self.models[self.best_model_name]
        churn_probs = best_pipe.predict_proba(X)[:, 1]
        scored_customers = dataset[["customer_id", "obs_recency_days", "obs_frequency", "obs_monetary", "churned"]].copy()
        scored_customers["churn_probability"] = np.round(churn_probs, 4)
        scored_customers["repurchase_probability"] = np.round(1.0 - churn_probs, 4)
        scored_customers["risk_tier"] = pd.cut(
            scored_customers["churn_probability"],
            bins=[-0.01, 0.35, 0.65, 1.01],
            labels=["Düşük Risk", "Orta Risk", "Yüksek Churn Riski"],
        ).astype(str)

        # Feature Importance (ağaç bazlı modellerden veya LR katsayılarından)
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            importances = np.zeros(len(self.FEATURE_COLS))

        importance_df = pd.DataFrame({
            "feature": self.FEATURE_COLS,
            "importance": np.round(importances / np.maximum(importances.sum(), 1e-9), 4),
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        if model_save_path is not None:
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "best_model_name": self.best_model_name,
                    "best_pipeline": best_pipe,
                    "feature_cols": self.FEATURE_COLS,
                    "cutoff_date": str(cutoff_date),
                    "metrics": metrics_rows,
                },
                model_save_path,
            )

        return metrics_df, scored_customers, importance_df, cutoff_date
