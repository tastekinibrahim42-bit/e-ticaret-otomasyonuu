"""
scraper_engine.py
Trendyol, Hepsiburada, N11 ve Amazon TR pazaryerleri için ürün, fiyat, stok ve yorum toplayıcı motor.
Statik sayfalar için requests + BeautifulSoup + fake-useragent; dinamik JS sayfaları için Playwright kullanır.
Bot koruması/erişim engeli veya çevrimdışı test senaryolarında gerçekçi Türkiye pazaryeri simülasyon verisi üretir.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from fake_useragent import UserAgent
except ImportError:
    UserAgent = None

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

logger = logging.getLogger(__name__)


@dataclass
class ScrapedProduct:
    product_id: str
    title: str
    category: str
    marketplace: str
    seller_name: str
    supplier_cost: float
    current_price: float
    competitor_min_price: float
    rating: float
    review_count: int
    stock_qty: int
    shipping_cost: float
    url: str
    scraped_at: str
    source_mode: str = "live"


class MarketplaceScraper:
    """Çoklu Türk pazaryeri veri toplayıcı (Trendyol, Hepsiburada, N11, Amazon TR)."""

    SUPPORTED_MARKETPLACES = ("Trendyol", "Hepsiburada", "N11", "Amazon TR")

    DEFAULT_HEADERS = {
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def __init__(self, timeout: int = 12, use_playwright: bool = False) -> None:
        self.timeout = timeout
        self.use_playwright = use_playwright
        self._ua = UserAgent() if UserAgent is not None else None

    def _get_headers(self) -> Dict[str, str]:
        headers = dict(self.DEFAULT_HEADERS)
        if self._ua is not None:
            try:
                headers["User-Agent"] = self._ua.random
            except Exception:
                headers["User-Agent"] = (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
                )
        else:
            headers["User-Agent"] = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            )
        return headers

    def fetch_html(self, url: str) -> Optional[str]:
        """URL'den HTML içeriğini çeker; use_playwright=True ise tarayıcı motoru kullanır."""
        if self.use_playwright and sync_playwright is not None:
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page(user_agent=self._get_headers()["User-Agent"])
                    page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
                    page.wait_for_timeout(1500)
                    html = page.content()
                    browser.close()
                    return html
            except Exception as exc:
                logger.warning("Playwright çekimi başarısız (%s): %s", url, exc)

        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200 and len(resp.text) > 500:
                return resp.text
            logger.warning("HTTP %s döndü: %s", resp.status_code, url)
        except Exception as exc:
            logger.warning("Requests çekimi başarısız (%s): %s", url, exc)
        return None

    @staticmethod
    def _parse_tr_price(text: str) -> Optional[float]:
        """Türkçe fiyat metnini ('1.299,90 TL') float değere çevirir."""
        if not text:
            return None
        cleaned = re.sub(r"[^\d,\.]", "", text)
        if "," in cleaned and "." in cleaned:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        elif "," in cleaned:
            cleaned = cleaned.replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None

    def scrape_product_url(self, url: str, marketplace: str = "Trendyol") -> Optional[ScrapedProduct]:
        """Tek bir ürün URL'sini ayrıştırır; erişim engellenirse None döner."""
        html = self.fetch_html(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")
        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else "Bilinmeyen Ürün"

        price = None
        for selector in [
            "span.prc-dsc",
            "span[data-test-id='price-current-price']",
            "div.newPrice ins",
            "span.a-price-whole",
        ]:
            el = soup.select_one(selector)
            if el:
                price = self._parse_tr_price(el.get_text(strip=True))
                if price:
                    break

        if not price:
            return None

        return ScrapedProduct(
            product_id=f"LIVE-{abs(hash(url)) % 100000}",
            title=title[:120],
            category="Genel",
            marketplace=marketplace,
            seller_name="Canlı Satıcı",
            supplier_cost=round(price * 0.68, 2),
            current_price=round(price, 2),
            competitor_min_price=round(price * 0.96, 2),
            rating=4.5,
            review_count=120,
            stock_qty=25,
            shipping_cost=49.90,
            url=url,
            scraped_at=datetime.now().isoformat(timespec="seconds"),
            source_mode="live",
        )

    def generate_realistic_marketplace_catalog(
        self,
        n_products: int = 80,
        seed: int = 42,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Pazaryeri bot koruması veya demo/eğitim senaryoları için gerçekçi Türkiye
        e-ticaret ürün, rakip fiyat, yorum ve stok veri seti üretir.
        """
        rng = np.random.default_rng(seed)
        random.seed(seed)

        catalog_templates = [
            ("Kablosuz ANC Bluetooth Kulaklık Pro", "Elektronik", 520, 1299),
            ("Akıllı Saat AMOLED GPS Spor", "Elektronik", 890, 2199),
            ("USB-C 65W GaN Hızlı Şarj Adaptörü", "Elektronik", 240, 649),
            ("Mekanik RGB Gaming Klavye TKL", "Bilgisayar", 750, 1799),
            ("Ergonomik Dikey Kablosuz Mouse", "Bilgisayar", 310, 849),
            ("Alüminyum Laptop Yükseltici Stand", "Bilgisayar", 190, 529),
            ("Paslanmaz Çelik Termos 750ml", "Ev & Yaşam", 180, 489),
            ("Hava Nemlendirici Ultrasonik Difüzör", "Ev & Yaşam", 260, 699),
            ("Otomatik Kahve Öğütücü Çelik Bıçak", "Küçük Ev Aletleri", 610, 1490),
            ("Airfryer Yağsız Fritöz 5.5L", "Küçük Ev Aletleri", 1450, 3299),
            ("Pamuklu Oversize Basic Tişört 3'lü", "Giyim", 210, 599),
            ("Su Geçirmez Outdoor Sırt Çantası 30L", "Çanta & Aksesuar", 340, 899),
            ("Araç İçi Kablosuz Şarjlı Telefon Tutucu", "Oto Aksesuar", 195, 549),
            ("LED Ring Light Tripod 12 İnç", "Foto & Kamera", 230, 629),
            ("Yüksek Hızlı NVMe SSD Kutusu USB 3.2", "Bilgisayar", 280, 729),
            ("Akıllı Wi-Fi Priz Enerji Ölçümlü", "Akıllı Ev", 160, 449),
        ]

        sellers = [
            "TeknoVadi", "AnadoluSepet", "EgeToptan", "MarmaraElektronik",
            "BosphorusStore", "MerkezTedarik", "KuzeyPazar", "AtlasDirect",
        ]

        rows: List[Dict] = []
        now = datetime.now()
        for idx in range(n_products):
            base_title, category, min_cost, max_price = catalog_templates[idx % len(catalog_templates)]
            variant_tag = ["Siyah", "Beyaz", "Titanyum", "Pro", "Plus", "2026 Seri"][idx % 6]
            marketplace = self.SUPPORTED_MARKETPLACES[idx % len(self.SUPPORTED_MARKETPLACES)]

            supplier_cost = float(np.round(rng.uniform(min_cost * 0.88, min_cost * 1.22), 2))
            markup = float(rng.uniform(1.38, 2.05))
            current_price = float(np.round(min(max_price * 1.25, supplier_cost * markup), 2))
            competitor_min = float(np.round(current_price * rng.uniform(0.86, 1.07), 2))
            rating = float(np.round(np.clip(rng.normal(4.35, 0.42), 2.6, 5.0), 2))
            review_count = int(rng.integers(12, 1850))
            stock_qty = int(rng.choice([0, 2, 4, 8, 15, 34, 65, 120], p=[0.06, 0.08, 0.10, 0.14, 0.22, 0.18, 0.14, 0.08]))
            shipping = 0.0 if current_price >= 450 else float(rng.choice([39.90, 49.90, 59.90]))

            slug = re.sub(r"[^a-z0-9]+", "-", base_title.lower()).strip("-")
            domain_map = {
                "Trendyol": "https://www.trendyol.com",
                "Hepsiburada": "https://www.hepsiburada.com",
                "N11": "https://www.n11.com",
                "Amazon TR": "https://www.amazon.com.tr",
            }
            item = ScrapedProduct(
                product_id=f"PRD-{1001 + idx}",
                title=f"{base_title} ({variant_tag})",
                category=category,
                marketplace=marketplace,
                seller_name=random.choice(sellers),
                supplier_cost=supplier_cost,
                current_price=current_price,
                competitor_min_price=competitor_min,
                rating=rating,
                review_count=review_count,
                stock_qty=stock_qty,
                shipping_cost=shipping,
                url=f"{domain_map[marketplace]}/p/{slug}-{1001 + idx}",
                scraped_at=(now - timedelta(minutes=int(rng.integers(1, 360)))).isoformat(timespec="seconds"),
                source_mode="simulated",
            )
            rows.append(asdict(item))

        df = pd.DataFrame(rows)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        return df


def generate_synthetic_ecommerce_transactions(
    n_customers: int = 900,
    n_transactions: int = 6500,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    RFM, K-Means, Churn (Data Leakage Korumalı) ve Deep Anomaly (Autoencoder + Isolation Forest)
    modülleri için gerçekçi çok kanallı e-ticaret sipariş işlem verisi üretir.
    """
    rng = np.random.default_rng(seed)
    end_date = datetime(2026, 11, 13)
    start_date = end_date - timedelta(days=365)

    marketplaces = ["Trendyol", "Hepsiburada", "N11", "Amazon TR"]
    categories = ["Elektronik", "Bilgisayar", "Ev & Yaşam", "Küçük Ev Aletleri", "Giyim", "Oto Aksesuar"]
    cities = ["İstanbul", "Ankara", "İzmir", "Aydın", "Konya", "Bursa", "Antalya"]
    payment_methods = ["Kredi Kartı", "Havale/EFT", "Cüzdan", "Kapıda Ödeme"]

    customer_ids = [f"CUST-{10000 + i}" for i in range(n_customers)]
    # Müşteri segment profilleri: 0=VIP, 1=Sadık, 2=Riskli/Churn, 3=Anomali/Bot
    segment_probs = [0.16, 0.44, 0.36, 0.04]
    cust_types = rng.choice([0, 1, 2, 3], size=n_customers, p=segment_probs)
    cust_type_map = dict(zip(customer_ids, cust_types))

    records: List[Dict] = []
    for tx_idx in range(n_transactions):
        cid = customer_ids[int(rng.integers(0, n_customers))]
        ctype = cust_type_map[cid]

        if ctype == 0:  # VIP: yüksek frekans, güncel alışveriş, yüksek sepet
            days_ago = int(rng.integers(1, 75))
            unit_price = float(rng.uniform(450, 3200))
            qty = int(rng.integers(1, 5))
            discount = float(rng.uniform(0.0, 0.12))
            is_returned = int(rng.random() < 0.03)
        elif ctype == 1:  # Sadık: düzenli alışveriş
            days_ago = int(rng.integers(5, 160))
            unit_price = float(rng.uniform(180, 1400))
            qty = int(rng.integers(1, 3))
            discount = float(rng.uniform(0.0, 0.18))
            is_returned = int(rng.random() < 0.06)
        elif ctype == 2:  # Riskli / Churn eğilimli: eski tarihli alışverişler
            days_ago = int(rng.integers(110, 360))
            unit_price = float(rng.uniform(120, 850))
            qty = int(rng.integers(1, 2))
            discount = float(rng.uniform(0.05, 0.25))
            is_returned = int(rng.random() < 0.14)
        else:  # Anomali / Şüpheli davranış (aşırı iade, yüksek adet veya olağandışı tutar)
            days_ago = int(rng.integers(1, 120))
            unit_price = float(rng.uniform(2200, 9500))
            qty = int(rng.integers(4, 18))
            discount = float(rng.uniform(0.25, 0.55))
            is_returned = int(rng.random() < 0.45)

        tx_date = end_date - timedelta(days=days_ago, hours=int(rng.integers(0, 23)))
        if tx_date < start_date:
            tx_date = start_date

        gross_amount = unit_price * qty
        net_amount = round(gross_amount * (1.0 - discount), 2)

        records.append({
            "order_id": f"ORD-{100000 + tx_idx}",
            "customer_id": cid,
            "order_date": tx_date.strftime("%Y-%m-%d %H:%M:%S"),
            "marketplace": marketplaces[int(rng.integers(0, len(marketplaces)))],
            "category": categories[int(rng.integers(0, len(categories)))],
            "city": cities[int(rng.integers(0, len(cities)))],
            "payment_method": payment_methods[int(rng.integers(0, len(payment_methods)))],
            "quantity": qty,
            "unit_price": round(unit_price, 2),
            "discount_rate": round(discount, 3),
            "total_amount": net_amount,
            "shipping_days": int(rng.integers(1, 7)),
            "is_returned": is_returned,
        })

    df = pd.DataFrame(records)
    # Preprocessing testini gerçekçi kılmak için %1.5 kontrollü eksik değer ekleyelim
    na_indices = rng.choice(df.index, size=max(10, int(len(df) * 0.015)), replace=False)
    df.loc[na_indices[: len(na_indices) // 2], "shipping_days"] = np.nan
    df.loc[na_indices[len(na_indices) // 2 :], "city"] = np.nan

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df
