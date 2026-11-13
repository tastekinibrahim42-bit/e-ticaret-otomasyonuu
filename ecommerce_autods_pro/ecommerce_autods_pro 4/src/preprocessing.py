"""
preprocessing.py
Ham e-ticaret işlem verilerini temizleme, eksik değer doldurma,
IQR tabanlı aykırı değer yönetimi (winsorization) ve sipariş kategorizasyon modülü.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreprocessingReport:
    raw_rows: int
    cleaned_rows: int
    duplicates_removed: int
    missing_values_imputed: int
    iqr_capped_counts: Dict[str, int]


class DataPreprocessor:
    """E-ticaret veri temizleme, eksik değer doldurma, IQR ve kategorizasyon."""

    NUMERIC_COLS = ["quantity", "unit_price", "discount_rate", "total_amount", "shipping_days"]
    CATEGORICAL_COLS = ["marketplace", "category", "city", "payment_method"]

    def __init__(self, iqr_multiplier: float = 1.5) -> None:
        self.iqr_multiplier = iqr_multiplier

    def clean_transactions(
        self,
        df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, PreprocessingReport]:
        """
        İşlem verisini temizler ve kategorize eder:
        1. Tekrarlayan order_id kayıtlarını temizler.
        2. Tarih sütununu datetime formatına çevirir.
        3. Eksik sayısal değerleri medyan, kategorik değerleri mod ile doldurur.
        4. Negatif/sıfır fiyat ve miktarları filtreler.
        5. IQR yöntemiyle aykırı değerleri baskılar (capping).
        6. Fiyat segmenti, indirim grubu ve sipariş tipi kategorilerini üretir.
        """
        work = df.copy()
        raw_rows = len(work)

        work = work.drop_duplicates(subset=["order_id"]).reset_index(drop=True)
        duplicates_removed = raw_rows - len(work)

        work["order_date"] = pd.to_datetime(work["order_date"], errors="coerce")
        work = work.dropna(subset=["order_date", "customer_id"]).reset_index(drop=True)

        missing_before = int(work[self.NUMERIC_COLS + self.CATEGORICAL_COLS].isna().sum().sum())

        for col in self.NUMERIC_COLS:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
                median_val = work[col].median()
                work[col] = work[col].fillna(0.0 if pd.isna(median_val) else median_val)

        for col in self.CATEGORICAL_COLS:
            if col in work.columns:
                mode_series = work[col].mode()
                fill_val = mode_series.iloc[0] if not mode_series.empty else "Bilinmiyor"
                work[col] = work[col].fillna(fill_val).astype(str)

        missing_after = int(work[self.NUMERIC_COLS + self.CATEGORICAL_COLS].isna().sum().sum())
        missing_imputed = max(0, missing_before - missing_after)

        work = work[(work["quantity"] > 0) & (work["unit_price"] > 0) & (work["total_amount"] > 0)].reset_index(drop=True)

        work["total_amount_raw"] = work["total_amount"]
        iqr_capped: Dict[str, int] = {}
        for col in ["unit_price", "total_amount", "quantity"]:
            q1 = work[col].quantile(0.25)
            q3 = work[col].quantile(0.75)
            iqr = q3 - q1
            if iqr <= 0:
                iqr_capped[col] = 0
                continue
            lower = max(0.0, q1 - self.iqr_multiplier * iqr)
            upper = q3 + self.iqr_multiplier * iqr
            outliers_mask = (work[col] < lower) | (work[col] > upper)
            iqr_capped[col] = int(outliers_mask.sum())
            work[f"{col}_iqr_capped"] = np.clip(work[col], lower, upper)

        # Sipariş ve Fiyat Kategorizasyonu
        work["fiyat_segmenti"] = pd.cut(
            work["unit_price"],
            bins=[0, 500, 1200, 2500, float("inf")],
            labels=["Ekonomik (0-500 ₺)", "Orta (500-1200 ₺)", "Premium (1200-2500 ₺)", "Lüks (2500+ ₺)"],
        ).astype(str)

        work["indirim_grubu"] = pd.cut(
            work["discount_rate"],
            bins=[-0.01, 0.05, 0.15, 0.25, 1.0],
            labels=["Düşük (%0-5)", "Standart (%5-15)", "Kampanya (%15-25)", "Agresif (%25+)"],
        ).astype(str)

        work["siparis_tipi"] = np.where(
            work["quantity"] >= 4,
            "Toplu / Toptan Alım",
            np.where(work["quantity"] >= 2, "Çoklu Sepet", "Tekli Sipariş"),
        )

        work["order_month"] = work["order_date"].dt.to_period("M").astype(str)
        work["order_dayofweek"] = work["order_date"].dt.dayofweek
        work["is_weekend"] = (work["order_dayofweek"] >= 5).astype(int)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            work.to_csv(output_path, index=False)

        report = PreprocessingReport(
            raw_rows=raw_rows,
            cleaned_rows=len(work),
            duplicates_removed=duplicates_removed,
            missing_values_imputed=missing_imputed,
            iqr_capped_counts=iqr_capped,
        )
        return work, report

    @staticmethod
    def report_to_dict(report: PreprocessingReport) -> Dict:
        return asdict(report)
