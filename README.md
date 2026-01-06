# Bankacı Plus / Bankaci Plus

## English

**What is it?**  
Bankacı Plus is a Streamlit-based decision support app that brings together three banking verticals: Credit Risk, Customer Churn Prevention, and Smart Sales (Next Best Action – K-Means).

### Features
- Credit Risk Module: Lite & Pro XGBoost models, quick risk scoring.
- Churn Module: LightGBM churn classifier with action strategies.
- NBA Module: K-Means segmentation + rule-based product recommendations.
- Interactive UI with graphs, strategy tables, and manual input forms.

### Datasets
- Lending Club cleaned credit dataset (40k+ records) for credit risk.
- Bank Customer Churn dataset (10k records) for churn and NBA.
*(Large CSVs are not included in the repo; provide your own or follow download instructions.)*

### Models
- Credit Risk Lite: XGBoost (8 variables, fast screening).
- Credit Risk Pro: XGBoost (16 variables, deeper analysis).
- Churn: LightGBM classifier (50-fold CV, ROC-AUC ~0.87).
- NBA: K-Means (6 clusters) + rule-based product mapping.
*(Model .pkl files are not in the repo; store externally or load at runtime.)*

### Requirements & Run
1. Python 3.9+ and Streamlit.
2. Install deps: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Provide data/model files via local paths or add download steps before launch.

### Structure (key files)
- `app.py` – Streamlit app.
- `requirements.txt` – Python dependencies.
- `datasets/` – (optional) place small sample data here.
- `models/` – (optional) place external models here (ignored by git).

### Notes
- Do not commit secrets (`.streamlit/secrets.toml`, `.env`, API keys).
- Add download links or instructions for datasets and models in the repo or wiki.

---

## Türkçe

**Nedir?**  
Bankacı Plus, üç bankacılık dikeyini bir araya getiren Streamlit tabanlı bir karar destek uygulamasıdır: Kredi Risk, Müşteri Kayıp (Churn) Önleme ve Akıllı Satış (Next Best Action – K-Means).

### Özellikler
- Kredi Risk Modülü: Lite & Pro XGBoost modelleri, hızlı risk skoru.
- Churn Modülü: LightGBM churn sınıflandırıcı ve aksiyon stratejileri.
- NBA Modülü: K-Means segmentasyon + kural tabanlı ürün önerileri.
- Grafikler, strateji tabloları ve manuel giriş formları ile etkileşimli arayüz.

### Veri Setleri
- Lending Club temizlenmiş kredi verisi (40k+ kayıt) – kredi riski.
- Bank Customer Churn verisi (10k kayıt) – churn ve NBA.
*(Büyük CSV dosyaları repoda yok; kendi verinizi ekleyin veya indirme adımlarını izleyin.)*

### Modeller
- Kredi Risk Lite: XGBoost (8 değişken, hızlı ön tarama).
- Kredi Risk Pro: XGBoost (16 değişken, detaylı analiz).
- Churn: LightGBM (50-fold CV, ROC-AUC ~0.87).
- NBA: K-Means (6 küme) + kural tabanlı ürün eşleme.
*(Model .pkl dosyaları repoda yok; harici saklayın veya çalışırken yükleyin.)*

### Gereksinimler ve Çalıştırma
1. Python 3.9+ ve Streamlit.
2. Bağımlılık kurulumu: `pip install -r requirements.txt`
3. Çalıştır: `streamlit run app.py`
4. Veri/model dosyalarını yerel dizinlerden sağlayın veya çalıştırmadan önce indirin.

### Yapı (ana dosyalar)
- `app.py` – Streamlit uygulaması.
- `requirements.txt` – Bağımlılıklar.
- `datasets/` – (opsiyonel) küçük örnek veriler için.
- `models/` – (opsiyonel) harici modeller (git tarafından yok sayılır).

### Notlar
- Gizli bilgileri (`.streamlit/secrets.toml`, `.env`, API anahtarları) commit etmeyin.
- Veri seti ve model indirme linklerini veya adımlarını README’de ya da wiki’de belirtin.


