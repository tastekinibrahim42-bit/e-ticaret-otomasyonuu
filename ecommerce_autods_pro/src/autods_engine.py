"""
autods_engine.py
AutoDS tarzı otomatik fiyatlandırma, kâr marjı hesaplama, çapraz pazaryeri arbitraj tarama
ve kritik stok/marj alarm motoru.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class PricingRuleConfig:
    target_margin_pct: float = 20.0
    min_margin_pct: float = 10.0
    vat_rate_pct: float = 20.0
    undercut_step_pct: float = 1.0
    low_stock_threshold: int = 5
    return_buffer_pct: float = 2.5


class AutoDSEngine:
    """Otomatik fiyatlandırma, net kâr marjı, arbitraj ve stok alarm motoru."""

    MARKETPLACE_COMMISSIONS: Dict[str, float] = {
        "Trendyol": 0.19,
        "Hepsiburada": 0.17,
        "N11": 0.15,
        "Amazon TR": 0.13,
    }

    def __init__(self, config: PricingRuleConfig | None = None) -> None:
        self.config = config or PricingRuleConfig()

    def evaluate_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ürün kataloğu üzerinde komisyon, kargo, fire/iade payı ve hedef kâr marjını
        hesaplayarak önerilen AutoDS fiyatını, net kârı ve alarmları üretir.
        """
        work = df.copy()
        commissions = work["marketplace"].map(self.MARKETPLACE_COMMISSIONS).fillna(0.18)
        work["commission_rate"] = commissions

        # Minimum kâr tabanını koruyan taban fiyat formülü:
        # Fiyat * (1 - komisyon - iade_payı - min_marj) >= Tedarik + Kargo
        min_margin = self.config.min_margin_pct / 100.0
        target_margin = self.config.target_margin_pct / 100.0
        ret_buf = self.config.return_buffer_pct / 100.0

        denom_min = np.maximum(0.15, 1.0 - commissions - ret_buf - min_margin)
        denom_target = np.maximum(0.15, 1.0 - commissions - ret_buf - target_margin)

        base_cost = work["supplier_cost"] + work["shipping_cost"]
        work["min_viable_price"] = np.round(base_cost / denom_min, 2)
        work["target_price"] = np.round(base_cost / denom_target, 2)

        # Rakip fiyatının undercut_step_pct kadar altına in, ancak min_viable_price altına düşme
        undercut_price = work["competitor_min_price"] * (1.0 - self.config.undercut_step_pct / 100.0)
        work["recommended_price"] = np.round(np.maximum(work["min_viable_price"], undercut_price), 2)

        # Mevcut ve önerilen net kâr
        work["commission_fee"] = np.round(work["current_price"] * commissions, 2)
        work["net_profit"] = np.round(
            work["current_price"] - work["supplier_cost"] - work["shipping_cost"] - work["commission_fee"] - (work["current_price"] * ret_buf),
            2,
        )
        work["net_margin_pct"] = np.round((work["net_profit"] / np.maximum(work["current_price"], 1.0)) * 100.0, 2)
        work["roi_pct"] = np.round((work["net_profit"] / np.maximum(work["supplier_cost"], 1.0)) * 100.0, 2)

        work["recommended_profit"] = np.round(
            work["recommended_price"] * (1.0 - commissions - ret_buf) - work["supplier_cost"] - work["shipping_cost"],
            2,
        )
        work["buybox_status"] = np.where(
            work["current_price"] <= work["competitor_min_price"],
            "BuyBox Kazanıldı",
            "Rakip Daha Ucuz",
        )

        # Stok ve marj alarmları
        alerts: List[str] = []
        for _, row in work.iterrows():
            if row["stock_qty"] == 0:
                alerts.append("OUT_OF_STOCK")
            elif row["stock_qty"] <= self.config.low_stock_threshold:
                alerts.append("LOW_STOCK")
            elif row["net_margin_pct"] < self.config.min_margin_pct:
                alerts.append("MARGIN_EROSION")
            elif row["current_price"] > row["competitor_min_price"] * 1.05:
                alerts.append("BUYBOX_RISK")
            else:
                alerts.append("HEALTHY")
        work["alert_status"] = alerts
        return work

    def find_arbitrage_opportunities(self, evaluated_df: pd.DataFrame, min_roi_pct: float = 18.0) -> pd.DataFrame:
        """
        Tedarik maliyetine göre yüksek ROI sunan ve stoğu müsait arbitraj adaylarını sıralar.
        """
        candidates = evaluated_df[
            (evaluated_df["stock_qty"] > self.config.low_stock_threshold)
            & (evaluated_df["roi_pct"] >= min_roi_pct)
            & (evaluated_df["rating"] >= 4.0)
        ].copy()

        candidates["arbitrage_score"] = np.round(
            (candidates["roi_pct"] * 0.55)
            + (candidates["net_margin_pct"] * 0.25)
            + (candidates["rating"] * 4.0),
            2,
        )
        return candidates.sort_values("arbitrage_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def summarize_kpis(evaluated_df: pd.DataFrame) -> Dict[str, float]:
        """Dashboard üst KPI kartları için özet metrikleri döndürür."""
        return {
            "total_products": int(len(evaluated_df)),
            "avg_margin_pct": round(float(evaluated_df["net_margin_pct"].mean()), 2),
            "avg_roi_pct": round(float(evaluated_df["roi_pct"].mean()), 2),
            "total_potential_profit": round(float((evaluated_df["net_profit"] * np.minimum(evaluated_df["stock_qty"], 10)).sum()), 2),
            "buybox_win_rate_pct": round(float((evaluated_df["buybox_status"] == "BuyBox Kazanıldı").mean() * 100.0), 2),
            "critical_alerts": int(evaluated_df["alert_status"].isin(["OUT_OF_STOCK", "LOW_STOCK", "MARGIN_EROSION"]).sum()),
        }
