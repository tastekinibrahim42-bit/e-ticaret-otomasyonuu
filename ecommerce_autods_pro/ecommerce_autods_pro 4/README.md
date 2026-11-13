# AutoDS Tarzı E-Ticaret Otomasyonu & Veri Madenciliği Platformu (`ecommerce_autods_pro`)

Çoklu pazaryeri (Trendyol, Hepsiburada, N11, Amazon TR) ürün/fiyat/stok otomasyonu ile ileri seviye veri madenciliği ve makine öğrenmesi modüllerini birleştiren Dark/Executive SaaS Streamlit uygulaması.

## Kurulum ve Çalıştırma

```bash
cd ecommerce_autods_pro
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium  # Opsiyonel: JS ağırlıklı sayfalar için
streamlit run app.py
```

## Modül Yapısı
- `src/scraper_engine.py`: Pazaryeri HTML/Playwright tarayıcısı ve gerçekçi simülasyon veri üreteci.
- `src/autods_engine.py`: Komisyon/kargo/iade payı bazlı dinamik fiyatlandırma, BuyBox ve arbitraj motoru.
- `src/preprocessing.py`: Tekrar temizleme, medyan/mod eksik değer doldurma ve IQR aykırı değer baskılama.
- `src/rfm_clustering.py`: RFM skorlama, CLTV hesabı, Elbow/Silhouette analizli K-Means ve 2D PCA.
- `src/predictive_ml.py`: Zamansal kesim (Data Leakage korumalı) Churn & Tekrar Satın Alma modelleri (LR, RF, GB).
- `src/deep_anomaly.py`: Autoencoder (`MLPRegressor`) + `IsolationForest` hibrit anomali tespiti.
