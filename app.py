import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os


# --- 0. OTOMATİK DARK MODE AYARLAYICI (NATIVE STREAMLIT CONFIG) ---
def setup_config():
    # .streamlit klasörü yoksa oluştur
    if not os.path.exists(".streamlit"):
        os.makedirs(".streamlit")

    # config.toml dosyası yoksa veya içeriği hatalıysa oluştur/güncelle
    config_path = ".streamlit/config.toml"
    config_content = """
[theme]
base="dark"
primaryColor="#00f0ff"
backgroundColor="#0a0a12"
secondaryBackgroundColor="#12121f"
textColor="#ffffff"
font="sans serif"
    """

    # Dosya yoksa yaz
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(config_content)


# Konfigürasyonu uygula (İlk çalıştırmada bir kere rerun gerekebilir)
setup_config()

# --- 1. GENEL AYARLAR ---
# Proje kök dizinini belirle - Streamlit çalışma dizinini kullanır
PROJECT_ROOT = os.getcwd()

# Streamlit Cloud için alternatif path kontrolü
# Streamlit Cloud'da dosyalar genellikle proje kök dizininde olmalı
if not os.path.exists(os.path.join(PROJECT_ROOT, 'credit_risk_model_20fold.pkl')):
    # Alternatif: Windows local path (sadece local development için)
    alt_path = r"G:\My Drive\BankaciPlus"
    if os.path.exists(os.path.join(alt_path, 'credit_risk_model_20fold.pkl')):
        PROJECT_ROOT = alt_path

st.set_page_config(page_title="Bankacı Plus", page_icon="🏦", layout="wide")


def set_design():
    # 1. Google Fonts
    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@100..900&family=Syne:wght@400..800&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    # 2. NEXAVERSE CSS
    st.markdown("""
    <style>
        /* =========================================
           1. TEMEL DEĞİŞKENLER
           ========================================= */
        :root {
            --primary: #00f0ff;
            --secondary: #ff00d4;
            --accent: #9d4edd;
            --dark-1: #0a0a12;
            --dark-2: #12121f;
            --dark-3: #1a1a2e;
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glow-cyan: rgba(0, 240, 255, 0.4);
            --glow-magenta: rgba(255, 0, 212, 0.4);
        }

        .stApp {
            background-color: var(--dark-1) !important;
            font-family: 'Outfit', sans-serif !important;
            color: #ffffff !important;
        }

        /* Başlıklar */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Syne', sans-serif !important;
            background: linear-gradient(135deg, var(--primary) 0%, #ffffff 50%, var(--secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700 !important;
        }

        /* Metinler */
        p, label, span, div, li, small {
            color: rgba(255, 255, 255, 0.9) !important;
            font-family: 'Outfit', sans-serif !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--dark-2) !important;
            border-right: 1px solid var(--glass-border) !important;
            backdrop-filter: blur(20px);
        }

        /* =========================================
           2. NAVİGASYON (RADIO) DÜZELTMESİ
           ========================================= */
        /* Dikey radio butonlar (sidebar için) */
        div[role="radiogroup"]:not(.horizontal-radio) {
            display: flex; flex-direction: column; gap: 15px;
        }
        div[role="radiogroup"]:not(.horizontal-radio) label {
            background-color: var(--dark-3) !important;
            border: 1px solid var(--glass-border) !important;
            padding: 15px 20px !important;
            border-radius: 15px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            width: 100%;
            display: flex; align-items: center;
        }
        div[role="radiogroup"]:not(.horizontal-radio) label:hover {
            border-color: var(--primary) !important;
            background-color: rgba(0, 240, 255, 0.1) !important;
            transform: translateX(5px);
        }
        div[role="radiogroup"]:not(.horizontal-radio) label[aria-checked="true"] {
            background: linear-gradient(90deg, rgba(0, 240, 255, 0.15), transparent) !important;
            border-color: var(--primary) !important;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.2) !important;
        }
        div[role="radiogroup"]:not(.horizontal-radio) label[aria-checked="true"] p {
            color: #fff !important; font-weight: bold !important;
            text-shadow: 0 0 10px var(--glow-cyan);
        }
        
        /* Yatay radio butonlar (kompakt, sola yaslı) - Ana sayfa için */
        .stRadio > div[role="radiogroup"] {
            display: flex !important;
            flex-direction: row !important;
            gap: 10px !important;
            justify-content: flex-start !important;
            width: fit-content !important;
            max-width: fit-content !important;
            align-items: flex-start !important;
        }
        .stRadio > div[role="radiogroup"] > label {
            background-color: var(--dark-3) !important;
            border: 1px solid var(--glass-border) !important;
            padding: 8px 16px !important;
            border-radius: 20px !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            width: auto !important;
            min-width: fit-content !important;
            max-width: fit-content !important;
            display: inline-flex !important;
            align-items: center !important;
            white-space: nowrap !important;
            flex-shrink: 0 !important;
        }
        .stRadio > div[role="radiogroup"] > label:hover {
            border-color: var(--primary) !important;
            background-color: rgba(0, 240, 255, 0.1) !important;
            transform: translateY(-2px);
        }
        .stRadio > div[role="radiogroup"] > label[aria-checked="true"] {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(255, 0, 212, 0.2)) !important;
            border-color: var(--primary) !important;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.3) !important;
        }
        /* Sidebar radio butonlarını etkileme */
        [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
            flex-direction: column !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
            width: 100% !important;
        }

        /* =========================================
           3. SEKMELER (TABS) DÜZELTMESİ
           ========================================= */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px !important; background-color: transparent !important; padding-bottom: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: var(--dark-3) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 50px !important;
            padding: 12px 30px !important;
            color: rgba(255, 255, 255, 0.6) !important;
            font-family: 'Syne', sans-serif !important;
            font-size: 14px !important;
            height: auto !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(0, 240, 255, 0.1) !important;
            color: white !important; border-color: var(--primary) !important;
            transform: translateY(-2px);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(255, 0, 212, 0.2)) !important;
            border-color: var(--primary) !important;
            color: #fff !important;
            font-weight: bold !important;
            box-shadow: 0 0 20px rgba(0, 240, 255, 0.3) !important;
        }

        /* =========================================
           4. DİĞER BİLEŞENLER
           ========================================= */

        /* Dropdown (Fixed) */
        div[data-baseweb="select"] > div {
            background-color: var(--dark-3) !important;
            border-color: var(--glass-border) !important;
            color: white !important;
        }
        div[data-baseweb="popover"], ul[data-baseweb="menu"] {
            background-color: var(--dark-2) !important;
            border: 1px solid var(--primary) !important;
        }
        li[data-baseweb="option"] { color: white !important; }
        li[data-baseweb="option"] * { color: white !important; }
        li[data-baseweb="option"]:hover, li[aria-selected="true"] {
            background: linear-gradient(90deg, var(--primary), transparent) !important;
            color: black !important;
        }
        li[data-baseweb="option"]:hover * { color: black !important; }

        /* Inputlar */
        div[data-baseweb="input"] > div {
            background-color: var(--dark-3) !important;
            border-color: var(--glass-border) !important;
        }
        input { color: #ffffff !important; caret-color: var(--primary) !important; }

        /* Butonlar */
        div.stButton > button {
            background: var(--glass-bg) !important;
            border: 1px solid var(--primary) !important;
            color: var(--primary) !important;
            border-radius: 50px !important;
            font-family: 'Syne', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 1px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.1) !important;
        }
        div.stButton > button:hover {
            background: var(--primary) !important;
            color: #000 !important;
            box-shadow: 0 0 30px var(--glow-cyan) !important;
            transform: translateY(-2px);
        }
        button[kind="secondary"] {
            border-color: var(--secondary) !important; color: var(--secondary) !important;
        }
        button[kind="secondary"]:hover {
            background: var(--secondary) !important; color: #fff !important;
            box-shadow: 0 0 30px var(--glow-magenta) !important;
        }

        /* Metrik Kartları */
        div[data-testid="stMetric"] {
            background: var(--glass-bg) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 15px !important;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1) !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: var(--primary) !important;
            text-shadow: 0 0 10px var(--glow-cyan);
        }

        .streamlit-expanderHeader, div[data-testid="stDataFrame"] {
            background-color: var(--dark-2) !important;
            border: 1px solid var(--glass-border) !important;
        }

        /* Expander klavye kısayolu metinlerini ve tooltip'leri gizle */
        [data-testid="stExpander"] [title],
        [data-testid="stExpander"] .streamlit-expanderHeader [title],
        .streamlit-expanderHeader [title],
        [data-testid="stExpander"] p[title],
        [data-testid="stExpander"] span[title],
        [data-testid="stExpander"] div[title] {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
        }
        
        /* Expander header içindeki ikinci ve sonraki child elementleri gizle (sadece başlık kalsın) */
        .streamlit-expanderHeader > *:not(:first-child) {
            display: none !important;
        }
        
        /* Expander içindeki tüm tooltip ve hint elementlerini gizle */
        [data-testid="stExpander"] [title*="Press"],
        [data-testid="stExpander"] [title*="keyboard"],
        [data-testid="stExpander"] [title*="Enter"],
        [data-testid="stExpander"] [title*="↓"],
        [data-testid="stExpander"] [title*="▲"],
        [data-testid="stExpander"] [title*="▼"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Tablo ve DataFrame Şeffaf Arka Plan ve Beyaz Metin Fix */
        [data-testid="stDataFrame"], 
        [data-testid="stDataFrame"] *, 
        [role="grid"],
        [role="gridcell"],
        [role="columnheader"],
        .stTable, 
        .stTable *,
        table, 
        table *,
        td, 
        th {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        /* Satır ve hücre bazlı zorlama */
        div[role="gridcell"] > div, 
        div[role="columnheader"] > div {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        /* Tablo kenarlıkları */
        table, th, td {
            border: 1px solid var(--glass-border) !important;
        }
        
        header[data-testid="stHeader"] { background-color: transparent !important; }

        /* =========================================
           5. SIDEBAR TOGGLE BUTON DÜZELTMESİ
           ========================================= */
        /* SADECE sidebar toggle butonunu hedefle - header içindeki ilk buton */
        header[data-testid="stHeader"] > div:first-child button,
        header[data-testid="stHeader"] button:first-of-type,
        button[kind="header"]:first-of-type {
            font-size: 0 !important;
            min-width: 40px !important;
            width: 40px !important;
            height: 40px !important;
            padding: 0 !important;
            position: relative !important;
        }
        
        /* SADECE sidebar toggle butonunun içindeki elementleri gizle - diğer butonları etkileme */
        header[data-testid="stHeader"] > div:first-child button *,
        header[data-testid="stHeader"] button:first-of-type *,
        button[kind="header"]:first-of-type * {
            display: none !important;
            visibility: hidden !important;
            font-size: 0 !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
        }
        
        /* SADECE sidebar toggle butonunun metnini gizle */
        header[data-testid="stHeader"] > div:first-child button:not(::before):not(::after),
        header[data-testid="stHeader"] button:first-of-type:not(::before):not(::after),
        button[kind="header"]:first-of-type:not(::before):not(::after) {
            text-indent: -9999px !important;
            overflow: hidden !important;
        }
        
        /* Ok simgesini ekle - SADECE sidebar toggle butonuna */
        header[data-testid="stHeader"] > div:first-child button::before,
        header[data-testid="stHeader"] button:first-of-type::before,
        button[kind="header"]:first-of-type::before {
            content: "←" !important;
            font-size: 24px !important;
            color: var(--primary) !important;
            display: inline-block !important;
            visibility: visible !important;
            opacity: 1 !important;
            font-weight: bold !important;
            line-height: 1 !important;
            position: absolute !important;
            left: 50% !important;
            top: 50% !important;
            transform: translate(-50%, -50%) !important;
            text-indent: 0 !important;
            width: auto !important;
            height: auto !important;
            z-index: 999 !important;
        }
        
        /* Sidebar AÇIKKEN ok yönünü değiştir (sağa ok) */
        [data-testid="stSidebar"][aria-expanded="true"] ~ * header[data-testid="stHeader"] > div:first-child button::before,
        [data-testid="stSidebar"][aria-expanded="true"] ~ * header[data-testid="stHeader"] button:first-of-type::before,
        body:has([data-testid="stSidebar"][aria-expanded="true"]) header[data-testid="stHeader"] > div:first-child button::before,
        body:has([data-testid="stSidebar"][aria-expanded="true"]) header[data-testid="stHeader"] button:first-of-type::before {
            content: "→" !important;
        }
        
        /* Sidebar KAPALIYKEN ok yönü (sola ok) */
        [data-testid="stSidebar"][aria-expanded="false"] ~ * header[data-testid="stHeader"] > div:first-child button::before,
        [data-testid="stSidebar"][aria-expanded="false"] ~ * header[data-testid="stHeader"] button:first-of-type::before,
        body:has([data-testid="stSidebar"][aria-expanded="false"]) header[data-testid="stHeader"] > div:first-child button::before,
        body:has([data-testid="stSidebar"][aria-expanded="false"]) header[data-testid="stHeader"] button:first-of-type::before {
            content: "←" !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # 3. BACKGROUND
    st.markdown("""
    <div class="ambient-bg" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: -1; overflow: hidden;">
        <div style="position: absolute; border-radius: 50%; filter: blur(80px); width: 600px; height: 600px; background: radial-gradient(circle, rgba(0, 240, 255, 0.15) 0%, transparent 70%); top: -200px; left: -200px;"></div>
        <div style="position: absolute; border-radius: 50%; filter: blur(80px); width: 500px; height: 500px; background: radial-gradient(circle, rgba(255, 0, 212, 0.15) 0%, transparent 70%); bottom: -150px; right: -150px;"></div>
    </div>
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-image: linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px); background-size: 80px 80px; pointer-events: none; z-index: -1;"></div>
    """, unsafe_allow_html=True)


set_design()


# --- 2. SESSION STATE BAŞLATMA ---
def init_session_state():
    defaults = {
        'c_id': None, 'c_score': 650, 'c_geo': 'France', 'c_gen': 'Male',
        'c_age': 30, 'c_tenure': 5, 'c_bal': 0.0, 'c_prod': 1,
        'c_card': 'Evet', 'c_active': 'Aktif', 'c_sal': 50000.0,
        'c_spending': 50, 'has_bes': 0, 'has_kredi': 0, 'has_yatirim': 0,
        'has_vadesiz': 1, 'c_segment': 'Bilinmiyor', 'analysis_mode': None,
        'l_inc': 50000.0, 'l_loan': 10000.0
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


from sklearn.metrics import silhouette_score

# --- 3. VERİ ZENGİNLEŞTİRME VE KÜMELEME ---
def enhance_data_with_products(df):
    np.random.seed(42)
    # Maaş skoru: Sabit maksimum (200,000$) ile normalize edilmiş, max 50 puan
    MAX_SALARY = 200000.0
    salary_score = (df['EstimatedSalary'] / MAX_SALARY * 50).clip(0, 50)
    
    # Yaş skoru: Yaş gruplarına göre daha mantıklı bir dağılım
    def calculate_age_score(age):
        if age <= 35:
            return 30  # Gençler: Yüksek harcama potansiyeli
        elif age <= 45:
            return 25  # Genç-orta yaş: Yüksek harcama
        elif age <= 55:
            return 20  # Orta yaş: Orta harcama
        elif age <= 65:
            return 15  # Orta-ileri yaş: Düşük-orta harcama
        else:
            return 10  # İleri yaş: Düşük ama sıfır olmayan harcama
    
    age_score = df['Age'].apply(calculate_age_score)
    
    # Kredi kartı skoru: Kredi kartı varsa +20 puan
    cc_score = df['HasCrCard'] * 20
    
    # Rastgele gürültü kaldırıldı - daha deterministik skor
    df['Spending_Score'] = (salary_score + age_score + cc_score).clip(1, 100).astype(int)
    df['Has_Vadesiz'] = 1

    def assign_extra_products(row):
        current_count = 1 + row['HasCrCard']
        target_count = row['NumOfProducts']
        has_bes, has_kredi, has_yatirim = 0, 0, 0
        while current_count < target_count:
            if row['Balance'] > 50000 and has_yatirim == 0: has_yatirim = 1; current_count += 1; continue
            if row['EstimatedSalary'] > 60000 and row[
                'Age'] > 28 and has_bes == 0: has_bes = 1; current_count += 1; continue
            if row['CreditScore'] < 650 and has_kredi == 0: has_kredi = 1; current_count += 1; continue
            options = [];
            if has_bes == 0: options.append('BES')
            if has_kredi == 0: options.append('Kredi')
            if has_yatirim == 0: options.append('Yatırım')
            if not options: break
            choice = np.random.choice(options)
            if choice == 'BES':
                has_bes = 1
            elif choice == 'Kredi':
                has_kredi = 1
            elif choice == 'Yatırım':
                has_yatirim = 1
            current_count += 1
        return pd.Series([has_bes, has_kredi, has_yatirim])

    df[['Has_BES', 'Has_Kredi', 'Has_Yatirim']] = df.apply(assign_extra_products, axis=1)

    scaler = MinMaxScaler()
    # YENİ DEĞİŞKENLER: Balance, EstimatedSalary, NumOfProducts, Tenure, IsActiveMember
    features = ['Balance', 'EstimatedSalary', 'NumOfProducts', 'Tenure', 'IsActiveMember']
    X_scaled = scaler.fit_transform(df[features])
    
    # --- KÜMELEME ---
    # n_init=10000 ile en az 10000 farklı başlangıç noktası denenir, en iyi varyasyon seçilir
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10000, max_iter=300)
    df['Cluster_Label'] = kmeans.fit_predict(X_scaled)

    # Model Doğrulama: Silüet Skoru
    sample_size = min(2000, len(X_scaled))
    sil_score = silhouette_score(X_scaled[:sample_size], df['Cluster_Label'][:sample_size])
    
    # Kümeleri finansal özelliklerine göre detaylı isimlendirme mantığı (Centroid analizi)
    # center[0] = Balance (normalized), center[1] = EstimatedSalary (normalized)
    # center[2] = NumOfProducts (normalized), center[3] = Tenure (normalized)
    # center[4] = IsActiveMember (normalized)
    centroids = kmeans.cluster_centers_
    
    # Her segment için ortalama değerleri hesapla (gerçek ölçekte)
    segment_stats = {}
    for i in range(6):
        cluster_data = df[df['Cluster_Label'] == i]
        segment_stats[i] = {
            'avg_balance': cluster_data['Balance'].mean(),
            'avg_salary': cluster_data['EstimatedSalary'].mean(),
            'avg_products': cluster_data['NumOfProducts'].mean(),
            'avg_tenure': cluster_data['Tenure'].mean(),
            'avg_active': cluster_data['IsActiveMember'].mean(),
            'size': len(cluster_data)
        }
    
    # YENİ 6 SEGMENT İSİMLENDİRME MANTIĞI (5 değişkene göre)
    # Her cluster için benzersiz isim garantisi: Centroid değerlerine göre sıralama
    cluster_names = {}
    
    # Her cluster için skor hesapla (öncelik sırası belirlemek için)
    cluster_scores = []
    for i, center in enumerate(centroids):
        stats = segment_stats[i]
        balance_norm = center[0]
        salary_norm = center[1]
        products_norm = center[2]
        tenure_norm = center[3]
        active_norm = center[4]
        
        # Toplam skor: Yüksek değerli müşteriler önce
        total_score = (balance_norm * 0.3 + salary_norm * 0.3 + products_norm * 0.2 + 
                      tenure_norm * 0.1 + active_norm * 0.1)
        cluster_scores.append((i, total_score, balance_norm, salary_norm, products_norm, 
                              tenure_norm, active_norm, stats))
    
    # Skora göre sırala (yüksekten düşüğe)
    cluster_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Her cluster için benzersiz isim ata
    segment_templates = [
        "💎 Elit / Servet Yönetimi",
        "🚀 Dinamik / Aktif Müşteri", 
        "💰 Güvenli / Birikimci",
        "⚠️ Riskli / Pasif Müşteri",
        "🌱 Temel Mevduat / Giriş",
        "📊 Standart Bankacılık"
    ]
    
    # Her rank için direkt olarak farklı segment ismi ata (6 farklı isim garantisi)
    for rank, (cluster_id, total_score, bal, sal, prod, ten, act, stats) in enumerate(cluster_scores):
        # Rank'a göre direkt isim ata - her rank farklı segment
        # Bu şekilde 6 farklı isim garanti edilir
        name = segment_templates[rank]
        cluster_names[cluster_id] = name
    
    # Kontrol: Her cluster için benzersiz isim olduğundan emin ol
    assert len(set(cluster_names.values())) == 6, f"Benzersiz segment sayısı 6 değil: {len(set(cluster_names.values()))}"

    df['Segment_Name'] = df['Cluster_Label'].map(cluster_names)
    return df, kmeans, scaler, cluster_names, sil_score


# --- 4. KAYNAKLARI YÜKLEME ---
# Cache ile optimize edilmiş yükleme (versiyon 2 - yeni segmentler için)
@st.cache_resource(ttl=3600, show_spinner=False)
def load_all_resources():
    try:
        # Dosya yollarını kontrol et
        required_files = {
            'credit_risk_model_20fold.pkl': 'Credit Risk Pro Model',
            'credit_risk_lite_model.pkl': 'Credit Risk Lite Model',
            'lending_club_cleaned.csv': 'Lending Club Dataset',
            'churn_model_v1.pkl': 'Churn Prediction Model'
        }
        
        missing_files = []
        for filename, description in required_files.items():
            filepath = os.path.join(PROJECT_ROOT, filename)
            if not os.path.exists(filepath):
                missing_files.append(f"{description} ({filename})")
        
        if missing_files:
            error_msg = f"⚠️ Eksik dosyalar bulundu:\n\n"
            error_msg += "\n".join([f"• {f}" for f in missing_files])
            error_msg += f"\n\n📁 Arama yapılan dizin: `{PROJECT_ROOT}`"
            error_msg += f"\n\n💡 Lütfen bu dosyaları proje kök dizinine ekleyin."
            st.error(error_msg)
            return None, None, None, None, None, None, None, None, None
        
        # PROJECT_ROOT global değişkenini kullan
        pro_m = joblib.load(os.path.join(PROJECT_ROOT, 'credit_risk_model_20fold.pkl'))
        lite_m = joblib.load(os.path.join(PROJECT_ROOT, 'credit_risk_lite_model.pkl'))
        df_risk = pd.read_csv(os.path.join(PROJECT_ROOT, 'lending_club_cleaned.csv'))
        churn_m = joblib.load(os.path.join(PROJECT_ROOT, 'churn_model_v1.pkl'))
        
        # İşlenmiş veri seti dosyası (cluster bilgisiyle birlikte)
        processed_file = os.path.join(PROJECT_ROOT, 'churn_processed_with_clusters.csv')
        # Yeni rasyonel EstimatedSalary'li veri setini kullan
        raw_file = os.path.join(PROJECT_ROOT, 'churn_processed_data_with_rational_salary.csv')
        
        # Eğer cluster bilgisiyle işlenmiş dosya varsa direkt yükle
        kmeans_file = os.path.join(PROJECT_ROOT, 'kmeans_model.pkl')
        scaler_file = os.path.join(PROJECT_ROOT, 'scaler_model.pkl')
        
        # YENİ ÖZELLİKLER: Balance, EstimatedSalary, NumOfProducts, Tenure, IsActiveMember
        expected_features = ['Balance', 'EstimatedSalary', 'NumOfProducts', 'Tenure', 'IsActiveMember']
        
        if os.path.exists(processed_file) and os.path.exists(kmeans_file) and os.path.exists(scaler_file):
            try:
                # Hızlı yükleme - cluster bilgisi ve modeller kayıtlı
                df_churn_proc = pd.read_csv(processed_file)
                
                # Özellik uyumluluğunu kontrol et
                missing_features = [f for f in expected_features if f not in df_churn_proc.columns]
                if missing_features:
                    raise ValueError(f"Eksik özellikler: {missing_features}")
                
                # Cluster names map'i oluştur
                cluster_map = {}
                for cluster_id in sorted(df_churn_proc['Cluster_Label'].unique()):
                    segment_name = df_churn_proc[df_churn_proc['Cluster_Label'] == cluster_id]['Segment_Name'].iloc[0]
                    cluster_map[cluster_id] = segment_name
                
                # Kayıtlı modelleri yükle (manuel segment tahmini için)
                kmeans_m = joblib.load(kmeans_file)
                scaler_m = joblib.load(scaler_file)
                
                # Özellik uyumluluğunu test et
                test_data = df_churn_proc[expected_features].iloc[:1]
                scaler_m.transform(test_data)  # Eğer hata verirse exception fırlatır
                
                # Silhouette score'u hesapla (hızlı)
                X_scaled = scaler_m.transform(df_churn_proc[expected_features])
                sample_size = min(2000, len(X_scaled))
                sil_val = silhouette_score(X_scaled[:sample_size], df_churn_proc['Cluster_Label'][:sample_size])
            except (ValueError, KeyError, AttributeError) as e:
                # Eski model/veri uyumsuz, yeniden oluştur
                st.warning(f"⚠️ Eski model uyumsuz, yeniden oluşturuluyor: {e}")
                if os.path.exists(processed_file):
                    os.remove(processed_file)
                if os.path.exists(kmeans_file):
                    os.remove(kmeans_file)
                if os.path.exists(scaler_file):
                    os.remove(scaler_file)
                # Yeniden oluştur
                if not os.path.exists(raw_file):
                    st.error(f"❌ Churn veri dosyası bulunamadı: {raw_file}")
                    df_churn_proc = pd.DataFrame()
                    kmeans_m = None
                    scaler_m = None
                    cluster_map = {}
                    sil_val = 0.0
                else:
                    df_churn_raw = pd.read_csv(raw_file)
                    df_churn_proc, kmeans_m, scaler_m, cluster_map, sil_val = enhance_data_with_products(df_churn_raw)
                    df_churn_proc.to_csv(processed_file, index=False)
                    joblib.dump(kmeans_m, kmeans_file)
                    joblib.dump(scaler_m, scaler_file)
                    st.info(f"✅ Yeni cluster bilgileri ve modeller hesaplandı ve kaydedildi.")
            else:
                # Başarıyla yüklendi, devam et
                pass
        else:
            # İlk kez çalışıyor - cluster hesapla ve kaydet
            if not os.path.exists(raw_file):
                st.warning(f"⚠️ Churn veri dosyası bulunamadı: {raw_file}\n\nNBA modülü çalışmayacak.")
                df_churn_proc = pd.DataFrame()
                kmeans_m = None
                scaler_m = None
                cluster_map = {}
                sil_val = 0.0
            else:
                try:
                    df_churn_raw = pd.read_csv(raw_file)
                    df_churn_proc, kmeans_m, scaler_m, cluster_map, sil_val = enhance_data_with_products(df_churn_raw)
                    
                    # İşlenmiş veriyi kaydet (cluster bilgisiyle birlikte)
                    df_churn_proc.to_csv(processed_file, index=False)
                    
                    # Modelleri kaydet (manuel segment tahmini için)
                    joblib.dump(kmeans_m, kmeans_file)
                    joblib.dump(scaler_m, scaler_file)
                    
                    st.info(f"✅ Cluster bilgileri ve modeller hesaplandı ve kaydedildi.")
                except Exception as e:
                    st.error(f"❌ Churn veri işleme hatası: {e}")
                    df_churn_proc = pd.DataFrame()
                    kmeans_m = None
                    scaler_m = None
                    cluster_map = {}
                    sil_val = 0.0
        
        if 'User_ID' not in df_churn_proc.columns:
            np.random.seed(42)
            ids = np.random.choice(range(1000000, 9999999), size=len(df_churn_proc), replace=False)
            df_churn_proc.insert(0, 'User_ID', ids)
            
        return pro_m, lite_m, df_risk, churn_m, df_churn_proc, kmeans_m, scaler_m, cluster_map, sil_val
    except FileNotFoundError as e:
        st.error(f"❌ Dosya bulunamadı: {e}\n\n📁 Arama yapılan dizin: `{PROJECT_ROOT}`\n\n💡 Lütfen gerekli model ve veri dosyalarını proje kök dizinine ekleyin.")
        return None, None, None, None, None, None, None, None, None
    except AttributeError as e:
        if '_RemainderColsList' in str(e) or 'ColumnTransformer' in str(e):
            st.error(f"❌ Scikit-learn versiyon uyumsuzluğu hatası!\n\n"
                    f"**Hata:** {e}\n\n"
                    f"**Çözüm:** Model dosyaları farklı bir scikit-learn versiyonu ile kaydedilmiş.\n\n"
                    f"**Yapılacaklar:**\n"
                    f"1. `requirements.txt` dosyasında `scikit-learn==1.3.2` olduğundan emin olun\n"
                    f"2. Streamlit Cloud'da paketleri yeniden yükleyin\n"
                    f"3. Gerekirse model dosyalarını mevcut scikit-learn versiyonu ile yeniden kaydedin")
        else:
            st.error(f"❌ Dosya yükleme hatası: {e}\n\n📁 Arama yapılan dizin: `{PROJECT_ROOT}`")
        return None, None, None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"❌ Dosya yükleme hatası: {e}\n\n📁 Arama yapılan dizin: `{PROJECT_ROOT}`")
        return None, None, None, None, None, None, None, None, None


pro_model, lite_model, df_original, churn_model, df_churn, kmeans_model, scaler_model, cluster_names_map, silhouette_val = load_all_resources()

# --- 5. YARDIMCI FONKSİYONLAR ---
PURPOSE_MAP = {"Borç Birleştirme": "debt_consolidation", "Kredi Kartı": "credit_card",
               "Ev Tadilatı": "home_improvement", "Büyük Harcama": "major_purchase", "Küçük İşletme": "small_business",
               "Araba": "car", "Düğün": "wedding", "Diğer": "other"}
HOME_MAP = {"Kiracı": "RENT", "İpotekli": "MORTGAGE", "Ev Sahibi": "OWN", "Diğer": "ANY"}
VERIF_MAP = {"Doğrulanmış": "Verified", "Kaynak Doğrulanmış": "Source Verified", "Doğrulanmamış": "Not Verified"}
EMP_MAP = {"1 yıldan az": "< 1 year", "1 yıl": "1 year", "2 yıl": "2 years", "10 yıl ve üzeri": "10+ years"}
REVERSE_HOME = {v: k for k, v in HOME_MAP.items()}
REVERSE_PURPOSE = {v: k for k, v in PURPOSE_MAP.items()}
REVERSE_EMP = {v: k for k, v in EMP_MAP.items()}


def clean_emp_length_input(k):
    if pd.isna(k) or k not in EMP_MAP: return 0
    val = EMP_MAP[k]
    return 10 if '+' in val else (0 if '<' in val else int(re.findall(r'\d+', val)[0]))


def map_term(t): return " 36 months" if t <= 36 else " 60 months"


def calculate_manual_spending_score(salary, age, has_card):
    """
    İyileştirilmiş harcama skoru hesaplama:
    - Maaş: 200,000$'a normalize edilmiş, max 50 puan
    - Yaş: Yaş gruplarına göre 10-30 puan arası
    - Kredi Kartı: Varsa +20 puan
    """
    MAX_SALARY = 200000.0
    
    # Maaş skoru (max 50 puan)
    salary_score = min(50, (salary / MAX_SALARY) * 50)
    
    # Yaş skoru (yaş grubuna göre)
    if age <= 35:
        age_score = 30  # Gençler: Yüksek harcama potansiyeli
    elif age <= 45:
        age_score = 25  # Genç-orta yaş: Yüksek harcama
    elif age <= 55:
        age_score = 20  # Orta yaş: Orta harcama
    elif age <= 65:
        age_score = 15  # Orta-ileri yaş: Düşük-orta harcama
    else:
        age_score = 10  # İleri yaş: Düşük ama sıfır olmayan harcama
    
    # Kredi kartı skoru
    cc_score = 20 if has_card else 0
    
    # Toplam skor (1-100 arası)
    total_score = int(salary_score + age_score + cc_score)
    return min(100, max(1, total_score))


def get_strategy_details(strategy_name):
    """
    Strateji adına göre detaylı açıklama döndürür
    """
    strategy_details = {
        "🚨 VIP MÜDAHALE": {
            "title": "🚨 VIP MÜDAHALE",
            "description": "Değerli müşteri için acil müdahale stratejisi",
            "details": """
            **Ne Zaman Uygulanır?**
            - Risk skoru > %60 ve hesap bakiyesi > $50,000
            
            **Aksiyon Planı:**
            1. **Özel Müşteri Temsilcisi Atama:** Hemen VIP müşteri temsilcisi atanır
            2. **Kişisel Görüşme:** 24 saat içinde telefon veya yüz yüze görüşme planlanır
            3. **Sorun Dinleme:** Müşterinin şikayetleri ve beklentileri detaylı dinlenir
            4. **Özel Çözümler:** Özel faiz indirimleri, ücretsiz hizmetler veya özel kampanyalar sunulur
            5. **Takip:** Düzenli takip görüşmeleri planlanır
            
            **Hedef:** Değerli müşteriyi kaybetmemek, ilişkiyi güçlendirmek
            """,
            "timeline": "Acil - 24 saat içinde müdahale"
        },
        "🔄 SADELEŞTİRME": {
            "title": "🔄 SADELEŞTİRME",
            "description": "Ürün paradoksu çözüm stratejisi",
            "details": """
            **Ne Zaman Uygulanır?**
            - Risk skoru > %60 ve ürün sayısı >= 3
            
            **Aksiyon Planı:**
            1. **Ürün Analizi:** Müşterinin tüm ürünleri gözden geçirilir
            2. **Gereksiz Ürünleri Kapatma:** Kullanılmayan veya gereksiz ürünler kapatılır
            3. **Konsolidasyon:** Benzer ürünler birleştirilir (örn: birden fazla kredi kartı)
            4. **Basitleştirme:** Ürün yönetimini kolaylaştıracak çözümler sunulur
            5. **Eğitim:** Müşteriye ürün kullanımı hakkında bilgi verilir
            
            **Hedef:** Müşteriyi boğmamak, sadakati artırmak, karmaşayı azaltmak
            """,
            "timeline": "1 hafta içinde uygulanır"
        },
        "📞 ARAMA": {
            "title": "📞 ARAMA",
            "description": "Proaktif iletişim stratejisi",
            "details": """
            **Ne Zaman Uygulanır?**
            - Risk skoru > %60 ancak özel koşullar sağlanmıyor
            
            **Aksiyon Planı:**
            1. **Proaktif Arama:** Müşteri temsilcisi tarafından doğrudan telefon araması yapılır
            2. **Sorun Tespiti:** Müşterinin memnuniyetsizlik nedenleri araştırılır
            3. **Çözüm Önerileri:** Müşteriye özel çözümler ve alternatifler sunulur
            4. **Kampanya Bilgilendirme:** Mevcut kampanyalar ve fırsatlar paylaşılır
            5. **İlişki Güçlendirme:** Müşteri ile duygusal bağ kurulmaya çalışılır
            
            **Hedef:** Müşteriyi geri kazanmak, ilişkiyi canlandırmak
            """,
            "timeline": "3 gün içinde arama yapılır"
        },
        "🔔 UYANDIRMA": {
            "title": "🔔 UYANDIRMA",
            "description": "Pasif müşteri canlandırma stratejisi",
            "details": """
            **Ne Zaman Uygulanır?**
            - Risk skoru %40-60 ve müşteri pasif (aktif değil)
            
            **Aksiyon Planı:**
            1. **Özel Bonus Kampanyaları:** Hesap kullanımı için özel bonuslar sunulur
            2. **Faiz İndirimleri:** Kredi ürünleri için özel faiz indirimleri teklif edilir
            3. **Hediye Puanlar:** Aktivite için hediye puanlar veya cashback önerilir
            4. **E-posta/SMS Kampanyaları:** Düzenli iletişim ile müşteri hatırlatılır
            5. **Yeni Ürün Tanıtımları:** İlgi çekici yeni ürünler tanıtılır
            
            **Hedef:** İlişkiyi canlandırmak, unutulmuş müşteriyi geri kazanmak
            """,
            "timeline": "2 hafta içinde kampanya başlatılır"
        },
        "🎁 LIFESTYLE HEDİYE": {
            "title": "🎁 LIFESTYLE HEDİYE",
            "description": "Genç müşteri için yaşam tarzı odaklı strateji",
            "details": """
            **Ne Zaman Uygulanır?**
            - Risk skoru %40-60 ve müşteri yaşı < 35
            
            **Aksiyon Planı:**
            1. **Yaşam Tarzı Hediyeleri:** Konser bileti, spor salonu üyeliği, teknoloji ürünleri
            2. **Sosyal Medya Kampanyaları:** Instagram, TikTok gibi platformlarda özel içerikler
            3. **Genç Odaklı Etkinlikler:** Networking etkinlikleri, workshop'lar, konserler
            4. **Teknoloji Ürünleri:** Akıllı saat, kulaklık gibi teknoloji hediyeleri
            5. **Deneyim Paketleri:** Seyahat, yemek, eğlence deneyimleri
            
            **Hedef:** Genç müşterilerle duygusal bağ kurmak, marka sadakati oluşturmak
            """,
            "timeline": "1 ay içinde hediye programı başlatılır"
        },
        "💳 TEŞVİK": {
            "title": "💳 TEŞVİK",
            "description": "Aktif müşteri için genel teşvik stratejisi",
            "details": """
            **Ne Zaman Uygulanır?**
            - Risk skoru %40-60 ve müşteri aktif ancak özel koşullar yok
            
            **Aksiyon Planı:**
            1. **Genel Teşvik Kampanyaları:** Cashback, puan kazanma, özel indirimler
            2. **Ürün Kullanım Teşvikleri:** Kredi kartı kullanımı için bonuslar
            3. **Sadakat Programları:** Uzun vadeli sadakat programlarına dahil etme
            4. **Özel Fırsatlar:** Sınırlı süreli özel fırsatlar ve kampanyalar
            5. **Referans Programları:** Arkadaş getirme kampanyaları
            
            **Hedef:** Müşteriyi aktif tutmak, ilişkiyi güçlendirmek, kullanımı artırmak
            """,
            "timeline": "2 hafta içinde kampanya başlatılır"
        },
        "💰 YATIRIM ÇAPRAZ SATIŞ": {
            "title": "💰 YATIRIM ÇAPRAZ SATIŞ",
            "description": "Yüksek bakiyeli müşteri için yatırım stratejisi",
            "details": """
            **Ne Zaman Uygulanır?**
            - Risk skoru <= %40 ve hesap bakiyesi > $100,000
            
            **Aksiyon Planı:**
            1. **Yatırım Ürünleri Önerisi:** Likit fon, altın, yatırım hesabı gibi ürünler
            2. **Finansal Danışmanlık:** Kişisel finansal danışman atama
            3. **Portföy Yönetimi:** Yatırım portföyü oluşturma ve yönetim hizmetleri
            4. **Eğitim Seminerleri:** Yatırım ve finansal planlama eğitimleri
            5. **Özel Yatırım Fırsatları:** Özel yatırım fırsatları ve alternatifler
            
            **Hedef:** Müşterinin parasını değerlendirmesine yardımcı olmak, banka ile ilişkiyi derinleştirmek
            """,
            "timeline": "1 ay içinde yatırım danışmanlığı başlatılır"
        },
        "🤝 İLİŞKİ YÖNETİMİ": {
            "title": "🤝 İLİŞKİ YÖNETİMİ",
            "description": "Standart müşteri ilişkisi yönetimi",
            "details": """
            **Ne Zaman Uygulanır?**
            - Risk skoru <= %40 ve özel koşul yok
            
            **Aksiyon Planı:**
            1. **Düzenli İletişim:** Aylık bültenler, e-posta kampanyaları
            2. **Genel Kampanyalar:** Tüm müşterilere açık genel kampanyalar
            3. **Müşteri Memnuniyeti Takibi:** Düzenli anketler ve geri bildirim toplama
            4. **Ürün Güncellemeleri:** Yeni ürün ve hizmet bilgilendirmeleri
            5. **Doğum Günü/Özel Günler:** Özel günlerde tebrik mesajları ve küçük hediyeler
            
            **Hedef:** Mevcut durumu korumak, müşteriyi mutlu tutmak, ilişkiyi sürdürmek
            """,
            "timeline": "Sürekli devam eden süreç"
        }
    }
    
    return strategy_details.get(strategy_name, {
        "title": strategy_name,
        "description": "Strateji açıklaması",
        "details": "Detaylı bilgi bulunamadı.",
        "timeline": "Belirtilmemiş"
    })


def advanced_strategy(row):
    """
    Strateji Mantığı sekmesindeki mantıkla uyumlu strateji belirleme fonksiyonu.
    Risk seviyesine ve müşteri özelliklerine göre kişiselleştirilmiş strateji önerir.
    """
    prob = row['Risk_Probability']
    bal = row['Balance']
    prod = row['NumOfProducts']
    act = row['IsActiveMember']
    age = row.get('Age', 40)  # Yaş bilgisi varsa kullan, yoksa varsayılan 40
    
    # YÜKSEK RİSK (prob > 0.60)
    if prob > 0.60:
        if bal > 50000:
            return "🚨 VIP MÜDAHALE"
        elif prod >= 3:
            return "🔄 SADELEŞTİRME"
        else:
            return "📞 ARAMA"
    
    # ORTA RİSK (0.40 < prob <= 0.60)
    elif prob > 0.40:
        if act == 0:  # Pasif üye
            return "🔔 UYANDIRMA"
        elif age < 35:  # Genç müşteri
            return "🎁 LIFESTYLE HEDİYE"
        else:
            return "💳 TEŞVİK"
    
    # DÜŞÜK RİSK (prob <= 0.40)
    else:
        if bal > 100000:
            return "💰 YATIRIM ÇAPRAZ SATIŞ"
        else:
            return "🤝 İLİŞKİ YÖNETİMİ"


def get_next_best_action(row, segment_name=None):
    """
    Segment bazlı Next Best Action önerileri.
    6 segment için özelleştirilmiş ürün önerileri.
    """
    # Pandas Series için güvenli erişim fonksiyonu
    def safe_get(row, key, default=0):
        try:
            if isinstance(row, dict):
                return row.get(key, default)
            else:  # pandas Series
                return row[key] if key in row.index else default
        except:
            return default
    
    # Segment bilgisi varsa öncelikle segment bazlı öner
    if segment_name and segment_name in cluster_names_map.values():
        # Segment bazlı öneriler
        
        if "💎 Elit / Servet Yönetimi" in segment_name:
            if safe_get(row, 'Has_Yatirim', 0) == 0:
                return {"Product": "Özel Yatırım Danışmanlığı", "Prob": 92,
                       "Reason": "Elit segment - Yüksek değerli müşteri için özel hizmet.",
                       "Script": "Kişisel yatırım danışmanınızla tanışmak ister misiniz?"}
            elif safe_get(row, 'Has_BES', 0) == 0:
                return {"Product": "Premium BES Paketi", "Prob": 85,
                       "Reason": "Elit müşteriler için özel emeklilik planı.",
                       "Script": "Geleceğinizi premium seviyede planlayalım."}
            else:
                return {"Product": "VIP Müşteri Hizmetleri", "Prob": 80,
                       "Reason": "Elit segment için özel avantajlar.",
                       "Script": "Size özel avantajlardan haberdar mısınız?"}
        
        elif "🚀 Dinamik / Aktif Müşteri" in segment_name:
            # Aktif müşteri + Yüksek ürün sayısı + Yüksek maaş
            if safe_get(row, 'HasCrCard', 0) == 0:
                return {"Product": "Premium Kredi Kartı (Mil Puan)", "Prob": 88,
                       "Reason": "Aktif müşteri - Yüksek harcama potansiyeli, mil puan kazanma fırsatı.",
                       "Script": "Her harcamanızda mil puan kazanın, seyahatlerinizi ücretsiz yapın!"}
            elif safe_get(row, 'NumOfProducts', 1) < 3:
                return {"Product": "BES + Yatırım Paketi", "Prob": 85,
                       "Reason": "Aktif müşteri - Ürün portföyünü genişletme fırsatı.",
                       "Script": "Geleceğinizi planlayın, birikimlerinizi değerlendirin."}
            else:
                return {"Product": "Lifestyle Ödül Programı", "Prob": 80,
                       "Reason": "Aktif müşteri - Yaşam tarzına uygun ödüller.",
                       "Script": "Konser, spor, teknoloji ürünlerinde özel indirimler."}
        
        elif "💰 Güvenli / Birikimci" in segment_name:
            if safe_get(row, 'Balance', 0) > 50000 and safe_get(row, 'Has_Yatirim', 0) == 0:
                return {"Product": "Likit Fon / Altın Yatırımı", "Prob": 90,
                       "Reason": "Yüksek bakiye + Birikimci profil - Enflasyona karşı koruma.",
                       "Script": "Paranızı enflasyona karşı koruyalım, değer kazandıralım."}
            elif safe_get(row, 'EstimatedSalary', 0) > 60000:
                return {"Product": "Vadeli Mevduat (Yüksek Faiz)", "Prob": 85,
                       "Reason": "Birikimci segment - Güvenli ve yüksek getiri.",
                       "Script": "Birikimlerinize yüksek faiz kazandıralım."}
            else:
                return {"Product": "Otomatik Birikim Planı", "Prob": 78,
                       "Reason": "Birikimci segment - Düzenli tasarruf alışkanlığı.",
                       "Script": "Her ay otomatik birikim yaparak hedeflerinize ulaşın."}
        
        elif "⚠️ Riskli / Pasif Müşteri" in segment_name:
            # Pasif müşteri + Düşük maaş + Düşük bakiye
            if safe_get(row, 'IsActiveMember', 1) == 0:
                return {"Product": "Müşteri Aktivasyon Programı", "Prob": 85,
                       "Reason": "Pasif müşteri - Aktivasyon ve ilişki güçlendirme.",
                       "Script": "Size özel avantajlarla bankacılık deneyiminizi canlandıralım."}
            elif safe_get(row, 'Balance', 0) < 10000:
                return {"Product": "Dijital Bankacılık Eğitimi + Teşvik", "Prob": 75,
                       "Reason": "Pasif müşteri - Dijital kanalları kullanma teşviki.",
                       "Script": "Dijital bankacılık avantajlarını keşfedin, özel teşviklerden faydalanın."}
            else:
                return {"Product": "Finansal Danışmanlık", "Prob": 65,
                       "Reason": "Pasif müşteri - Finansal planlama ve ilişki yönetimi.",
                       "Script": "Ücretsiz finansal danışmanlık hizmetimizden faydalanın."}
        
        elif "🌱 Temel Mevduat / Giriş" in segment_name:
            if safe_get(row, 'HasCrCard', 0) == 0:
                return {"Product": "Temel Kredi Kartı", "Prob": 75,
                       "Reason": "Giriş seviyesi - İlk kredi kartı fırsatı.",
                       "Script": "İlk kredi kartınızı alın, güvenli alışveriş yapın."}
            elif safe_get(row, 'EstimatedSalary', 0) > 30000:
                return {"Product": "Dijital Bankacılık Eğitimi", "Prob": 70,
                       "Reason": "Giriş seviyesi - Dijital bankacılık öğrenimi.",
                       "Script": "Dijital bankacılık avantajlarını keşfedin."}
            else:
                return {"Product": "Genç Müşteri Paketi", "Prob": 65,
                       "Reason": "Giriş segmenti - Özel genç müşteri avantajları.",
                       "Script": "Size özel avantajlı paketlerimizi inceleyin."}
        
        elif "📊 Standart Bankacılık" in segment_name:
            if safe_get(row, 'EstimatedSalary', 0) > 50000 and safe_get(row, 'Age', 30) > 25 and safe_get(row, 'Age', 30) < 55 and safe_get(row, 'Has_BES', 0) == 0:
                return {"Product": "Bireysel Emeklilik (BES)", "Prob": 78,
                       "Reason": "Standart segment - Gelecek planlaması.",
                       "Script": "Devlet katkısından faydalanarak emekliliğinizi planlayın."}
            elif safe_get(row, 'Spending_Score', 0) > 50 and safe_get(row, 'HasCrCard', 0) == 0:
                return {"Product": "Standart Kredi Kartı", "Prob": 72,
                       "Reason": "Orta harcama potansiyeli - Kredi kartı ihtiyacı.",
                       "Script": "Günlük alışverişlerinizde kolaylık sağlayın."}
            else:
                return {"Product": "Otomatik Ödeme Sistemi", "Prob": 68,
                       "Reason": "Standart segment - Kolaylık odaklı.",
                       "Script": "Faturalarınızı otomatik ödeyin, zaman kazanın."}
    
    # Segment bilgisi yoksa genel kurallar (geriye dönük uyumluluk)
    if safe_get(row, 'Balance', 0) > 40000 and safe_get(row, 'Has_Yatirim', 0) == 0: 
        return {"Product": "Likit Fon / Altın", "Prob": 88,
                                                                   "Reason": "Vadesiz hesapta yüksek atıl bakiye.",
                                                                   "Script": "Paranızı enflasyona karşı koruyalım."}
    if safe_get(row, 'EstimatedSalary', 0) > 50000 and safe_get(row, 'Age', 30) > 25 and safe_get(row, 'Age', 30) < 55 and safe_get(row, 'Has_BES', 0) == 0: 
        return {"Product": "Bireysel Emeklilik (BES)", "Prob": 78, 
               "Reason": "Gelir yüksek, gelecek güvencesi yok.",
        "Script": "Devlet katkısından faydalanın."}
    if safe_get(row, 'Spending_Score', 50) > 60 and safe_get(row, 'HasCrCard', 0) == 0: 
        return {"Product": "Platinum Kredi Kartı", "Prob": 72,
                                                                     "Reason": "Harcama potansiyeli yüksek.",
                                                                     "Script": "Mil puan kazanmak ister misiniz?"}
    if safe_get(row, 'CreditScore', 650) < 650 and safe_get(row, 'Balance', 0) < 5000 and safe_get(row, 'Has_Kredi', 0) == 0: 
        return {"Product": "İhtiyaç Kredisi", "Prob": 65, 
               "Reason": "Nakit sıkışıklığı sinyali.",
        "Script": "3 ay ertelemeli kredi ister misiniz?"}
    return {"Product": "Otomatik Ödeme", "Prob": 45, 
           "Reason": "Mevcut ürünler yeterli.",
            "Script": "Faturalarınızı otomatik ödeyelim."}


# --- 6. VERİ GETİRME ---
def get_random_risk_customer():
    if df_original is None: return
    row = df_original.sample(1).iloc[0]
    st.session_state.update({'l_inc': float(row['annual_inc']), 'l_loan': float(row['loan_amnt']),
                             'l_term': int(str(row['term']).split()[0]), 'l_grade': row['grade'],
                             'l_home': REVERSE_HOME.get(row['home_ownership'], "Kiracı"),
                             'l_purp': REVERSE_PURPOSE.get(row['purpose'], "Diğer"),
                             'l_emp': REVERSE_EMP.get(row['emp_length'], "10 yıl ve üzeri")})
    st.toast("🎲 Kredi Verisi Yüklendi", icon="✅")


def get_random_churn_customer():
    """Risk skorlarına göre ardışık aralıklarda müşteri seçimi"""
    if df_churn is None or churn_model is None: 
        st.warning("Veri yükleniyor, lütfen bekleyin...")
        return
    
    # Risk skoru aralıkları (sırayla)
    risk_ranges = [
        (0, 20),      # 1. basış: [0-20]
        (81, 100),    # 2. basış: [81-100]
        (21, 40),     # 3. basış: [21-40]
        (61, 80),     # 4. basış: [61-80]
        (41, 60),     # 5. basış: [41-60]
    ]
    
    # Session state'te sayaç başlat (döngüsel)
    if 'churn_range_index' not in st.session_state:
        st.session_state['churn_range_index'] = 0
    
    # Mevcut aralığı al
    current_index = st.session_state['churn_range_index']
    min_risk, max_risk = risk_ranges[current_index]
    
    # Sayaç artır (bir sonraki basış için)
    st.session_state['churn_range_index'] = (current_index + 1) % len(risk_ranges)
    
    try:
        # Tüm müşterilerin risk skorlarını hesapla (cache için)
        if 'df_churn_with_risk' not in st.session_state:
            cols_to_drop = ['User_ID', 'Has_Vadesiz', 'Has_BES', 'Has_Kredi', 'Has_Yatirim', 'Spending_Score',
                           'Cluster_Label', 'Segment_Name']
            X_all = df_churn.drop(columns=cols_to_drop, errors='ignore')
            risk_probs = churn_model.predict_proba(X_all)[:, 1]
            df_with_risk = df_churn.copy()
            df_with_risk['Risk_Probability'] = risk_probs
            st.session_state['df_churn_with_risk'] = df_with_risk
        
        df_with_risk = st.session_state['df_churn_with_risk']
        
        # Risk skorunu yüzde olarak hesapla
        df_with_risk['Risk_Percent'] = df_with_risk['Risk_Probability'] * 100
        
        # Belirtilen aralıktaki müşterileri filtrele
        filtered_df = df_with_risk[
            (df_with_risk['Risk_Percent'] >= min_risk) & 
            (df_with_risk['Risk_Percent'] <= max_risk)
        ]
        
        if len(filtered_df) == 0:
            # Eğer bu aralıkta müşteri yoksa, en yakın aralıktan seç
            st.warning(f"⚠️ [{min_risk}-{max_risk}] aralığında müşteri bulunamadı. En yakın aralıktan seçiliyor...")
            filtered_df = df_with_risk
        
        # Rastgele bir müşteri seç
        row = filtered_df.sample(n=1, random_state=None).iloc[0]
        actual_risk = row['Risk_Percent']
        
        # Session state'e hızlı güncelleme
        st.session_state.update({
            'c_id': str(row['User_ID']), 
            'c_score': int(row['CreditScore']), 
            'c_geo': row['Geography'],
            'c_gen': row['Gender'], 
            'c_age': int(row['Age']), 
            'c_tenure': int(row['Tenure']),
            'c_bal': float(row['Balance']), 
            'c_prod': int(row['NumOfProducts']),
            'c_card': "Evet" if row['HasCrCard'] == 1 else "Hayır",
            'c_active': "Aktif" if row['IsActiveMember'] == 1 else "Pasif",
            'c_sal': float(row['EstimatedSalary']), 
            'c_spending': int(row['Spending_Score']),
            'has_bes': int(row['Has_BES']), 
            'has_kredi': int(row['Has_Kredi']), 
            'has_yatirim': int(row['Has_Yatirim']),
            'has_vadesiz': int(row['Has_Vadesiz']), 
            'c_segment': str(row['Segment_Name']), 
            'analysis_mode': 'random'
        })
        
        # Toast mesajı (hangi aralıktan geldiğini göster)
        st.toast(f"🎲 Müşteri Yüklendi (Risk: %{actual_risk:.1f} - Aralık: [{min_risk}-{max_risk}])", icon="👤")
    except Exception as e:
        st.error(f"Müşteri yükleme hatası: {e}")


# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="width: 80px; height: 80px; margin: 0 auto; background: radial-gradient(circle, rgba(0, 240, 255, 0.2), transparent); border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(0, 240, 255, 0.3); box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);">
                <span style="font-size: 40px;">🏦</span>
            </div>
            <h2 style="font-family: 'Syne', sans-serif; background: linear-gradient(135deg, #00f0ff, #ff00d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-top: 10px; font-weight: 800;">BANKACI<br>PLUS</h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    # Menü başlığı
    st.markdown("### 📋 Menü")
    page = st.radio("", ["🛡️ Kredi Risk Tahmini", "📉 Müşteri Kayıp (Churn)", "🎯 Fırsatlar & Satış (NBA - K-Means)",
                             "ℹ️ Proje Hakkında"], label_visibility="collapsed")
    st.markdown("---")
    
    # Geliştirici ile İletişim
    st.markdown("### 📧 Geliştirici ile İletişim")
    st.markdown("""
    <div style="padding: 15px; background-color: rgba(0, 240, 255, 0.1); border: 1px solid rgba(0, 240, 255, 0.3); border-radius: 10px;">
        <h3 style="color: #00f0ff; margin-bottom: 10px; font-size: 16px; font-weight: bold;">EMRE AÇAR</h3>
        <p style="color: rgba(255, 255, 255, 0.9); margin: 5px 0;">
            <a href="https://www.linkedin.com/in/emreacarc/" target="_blank" style="color: #00f0ff; text-decoration: none;">LinkedIn Profilim</a>
        </p>
        <p style="color: rgba(255, 255, 255, 0.9); margin: 5px 0;">
            <strong>E-mail:</strong> <a href="mailto:ar.emreacar@gmail.com" style="color: #00f0ff; text-decoration: none;">ar.emreacar@gmail.com</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# SAYFA 1: KREDİ RİSK TAHMİNİ
# =========================================================
if page == "🛡️ Kredi Risk Tahmini":
    st.title("🛡️ Kredi Risk Tahmin Modülü")
    col_r1, _ = st.columns([1, 6])
    with col_r1:
        st.button("🎲 Rastgele Getir", on_click=get_random_risk_customer, use_container_width=True)

    t1, t2 = st.tabs(["🚀 Hızlı Analiz (Lite)", "📈 Detaylı Analiz (Pro)"])

    with t1:
        c1, c2 = st.columns(2)
        l_inc = c1.number_input("Yıllık Gelir ($)", 10000.0, 1000000.0, key="l_inc")
        l_loan = c1.number_input("Kredi Tutarı ($)", 1000.0, 50000.0, key="l_loan")
        l_term = c1.selectbox("Vade (Ay)", [12, 24, 36, 48, 60], key="l_term")
        l_grade = c1.selectbox("Not (Grade)", ["A", "B", "C", "D", "E", "F", "G"], key="l_grade")
        l_home = c2.selectbox("Ev Durumu", list(HOME_MAP.keys()), key="l_home")
        l_purp = c2.selectbox("Amaç", list(PURPOSE_MAP.keys()), key="l_purp")
        l_emp = c2.selectbox("Çalışma", list(EMP_MAP.keys()), key="l_emp")

        if st.button("🚀 ANALİZ ET (LITE)", type="primary", use_container_width=True):
            if lite_model:
                df = pd.DataFrame(
                    {'annual_inc': [l_inc], 'loan_amnt': [l_loan], 'term': [map_term(l_term)], 'grade': [l_grade],
                     'home_ownership': [HOME_MAP[l_home]], 'purpose': [PURPOSE_MAP[l_purp]],
                     'emp_length': [clean_emp_length_input(l_emp)]})
                df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
                prob = float(lite_model.predict_proba(df)[0][1])
                pred = lite_model.predict(df)[0]
                st.divider();
                k1, k2, k3 = st.columns(3)
                k1.metric("Risk Skoru", f"%{prob * 100:.1f}")
                k2.metric("Karar", "RED" if pred == 1 else "ONAY")
                k3.metric("Güven", "0.70")
                st.progress(prob, text="Risk Seviyesi")
        
        # Model Bilgileri
        st.divider()
        with st.expander("📊 Lite Model Hakkında Detaylı Bilgi", expanded=False):
            st.markdown("""
            ### 🚀 Lite Model (XGBoost Classifier)
            
            **Model Tipi:** XGBoost (Extreme Gradient Boosting) Classifier
            
            **Kullanılan Değişkenler (7 Değişken):**
            - `annual_inc` - Yıllık Gelir
            - `loan_amnt` - Kredi Tutarı
            - `term` - Vade (12, 24, 36, 48, 60 ay)
            - `grade` - Kredi Notu (A, B, C, D, E, F, G)
            - `home_ownership` - Ev Durumu
            - `purpose` - Kredi Amacı
            - `emp_length` - Çalışma Süresi
            - `loan_to_income` - Türetilmiş: Kredi/Gelir Oranı
            
            **Model Parametreleri (Optimize Edilmiş):**
            - `n_estimators`: 100 (Ağaç sayısı)
            - `learning_rate`: 0.1 (Öğrenme hızı)
            - `max_depth`: 5 (Ağaç derinliği)
            - `subsample`: 0.8 (Alt örnekleme oranı)
            - `colsample_bytree`: 0.7 (Sütun alt örnekleme)
            - `min_child_weight`: 1 (Minimum çocuk ağırlığı)
            
            **Optimizasyon:**
            - **RandomizedSearchCV** ile hiperparametre optimizasyonu yapılmıştır
            - 150 rastgele kombinasyon test edilmiştir
            - Accuracy ve ROC-AUC skorları optimize edilmiştir
            
            **Doğrulama:**
            - **3-Fold Cross Validation** ile optimize edilmiştir
            - Model kararlılığı ve güvenilirliği test edilmiştir
            
            **Model Performans Metrikleri (Optimize Edilmiş):**
            - **Test Set Accuracy:** %65.29
            - **Test Set ROC-AUC:** %70.31
            - **Optimizasyon Öncesi ROC-AUC:** %70.50
            - **Optimizasyon Sonrası İyileştirme:** Accuracy +0.99%, ROC-AUC -0.19% (yakın performans)
            
            **📊 Performans Değerlendirmesi:**
            - **Accuracy:** Orta seviye (%65.29) - İyileştirilebilir
            - **ROC-AUC:** Kabul edilebilir (%70.31) - Model riskli/risksiz ayırt etme konusunda rastgele tahminden daha iyi
            - **Kararlılık:** Yüksek (Optimize edilmiş parametreler ile tutarlı sonuçlar)
            
            **Avantajlar:**
            - ⚡ Hızlı tahmin süresi (az değişken)
            - 💡 Basit ve anlaşılır girdi gereksinimleri
            - 🔄 Gerçek zamanlı analiz için optimize edilmiştir
            - 📈 Model kararlılığı yüksek (tutarlı sonuçlar)
            
            **Kullanım Senaryosu:**
            Hızlı karar verme gerektiren durumlarda, minimum bilgi ile risk değerlendirmesi yapmak için idealdir.
            """)

    with t2:
        with st.form("pro_form"):
            c1, c2, c3 = st.columns(3)
            p_loan = c1.number_input("Tutar", 1000.0, 50000.0, value=st.session_state.get('l_loan', 10000.0))
            p_term = c1.selectbox("Vade", [12, 24, 36, 48, 60], index=2)
            p_int = c1.number_input("Faiz %", 5.0, 30.0, 12.5)
            p_inst = c1.number_input("Taksit", 50.0, 2000.0, 350.0)
            p_inc = c2.number_input("Yıllık Gelir", 10000.0, 1000000.0, value=st.session_state.get('l_inc', 60000.0))
            p_emp = c2.selectbox("Çalışma", list(EMP_MAP.keys()))
            p_home = c2.selectbox("Ev", list(HOME_MAP.keys()))
            p_grade = c2.selectbox("Not", ["A", "B", "C", "D", "E", "F", "G"])
            p_dti = c3.number_input("DTI", 0.0, 100.0, 15.0)
            p_rev = c3.number_input("Kart Borcu", 0, 100000, 5000)
            p_acc = c3.number_input("Hesap Sayısı", 1, 100, 15)
            p_ver = c3.selectbox("Teyit", list(VERIF_MAP.keys()))
            analyze_pro = st.form_submit_button("📊 ANALİZ ET (PRO)", type="primary", use_container_width=True)

        if analyze_pro:
            if pro_model:
                df = pd.DataFrame(
                    {'loan_amnt': [p_loan], 'term': [map_term(p_term)], 'int_rate': [p_int], 'installment': [p_inst],
                     'grade': [p_grade], 'sub_grade': ['B1'], 'emp_length': [clean_emp_length_input(p_emp)],
                     'home_ownership': [HOME_MAP[p_home]], 'annual_inc': [p_inc],
                     'verification_status': [VERIF_MAP[p_ver]], 'purpose': ['debt_consolidation'], 'dti': [p_dti],
                     'revol_bal': [p_rev], 'revol_util': [40.0], 'total_acc': [p_acc]})
                df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
                df['installment_to_income'] = df['installment'] / ((df['annual_inc'] / 12) + 1)
                df['balance_income_ratio'] = df['revol_bal'] / (df['annual_inc'] + 1)
                prob = float(pro_model.predict_proba(df)[0][1])
                st.divider();
                k1, k2, k3 = st.columns(3)
                k1.metric("Risk Skoru", f"%{prob * 100:.1f}");
                k2.metric("Karar", "RED" if prob > 0.5 else "ONAY");
                k3.metric("Güven", "0.70")
                st.progress(prob, text="Kredi Risk Seviyesi")
        
        # Model Bilgileri
        st.divider()
        with st.expander("📊 Pro Model Hakkında Detaylı Bilgi", expanded=False):
            st.markdown("""
            ### 📈 Pro Model (XGBoost Classifier)
            
            **Model Tipi:** XGBoost (Extreme Gradient Boosting) Classifier
            
            **Kullanılan Değişkenler (13+ Değişken):**
            
            **Temel Değişkenler:**
            - `loan_amnt` - Kredi Tutarı
            - `term` - Vade
            - `int_rate` - Faiz Oranı
            - `installment` - Aylık Taksit
            - `grade` - Kredi Notu
            - `sub_grade` - Alt Not
            - `emp_length` - Çalışma Süresi
            - `home_ownership` - Ev Durumu
            - `annual_inc` - Yıllık Gelir
            - `verification_status` - Gelir Teyit Durumu
            - `purpose` - Kredi Amacı
            - `dti` - Debt-to-Income (Borç/Gelir Oranı)
            - `revol_bal` - Döner Kredi Bakiyesi
            - `revol_util` - Döner Kredi Kullanım Oranı
            - `total_acc` - Toplam Hesap Sayısı
            
            **Türetilmiş Özellikler (Feature Engineering):**
            - `loan_to_income` - Kredi/Gelir Oranı
            - `installment_to_income` - Taksit/Aylık Gelir Oranı (PTI)
            - `balance_income_ratio` - Bakiye/Gelir Oranı
            
            **Model Parametreleri (Optimize Edilmiş):**
            - `n_estimators`: 350 (Ağaç sayısı)
            - `learning_rate`: 0.03 (Öğrenme hızı)
            - `max_depth`: 4 (Ağaç derinliği)
            - `subsample`: 0.75 (Alt örnekleme oranı)
            - `colsample_bytree`: 0.75 (Sütun alt örnekleme)
            - `min_child_weight`: 2 (Minimum çocuk ağırlığı)
            - `gamma`: 0 (Minimum kayıp azaltma)
            
            **Optimizasyon:**
            - **RandomizedSearchCV** ile hiperparametre optimizasyonu yapılmıştır
            - 100 rastgele kombinasyon test edilmiştir
            - Accuracy ve ROC-AUC skorları optimize edilmiştir
            
            **Doğrulama:**
            - **3-Fold Cross Validation** ile optimize edilmiştir
            - Model kararlılığı ve güvenilirliği test edilmiştir
            
            **Model Performans Metrikleri (Optimize Edilmiş):**
            - **Test Set Accuracy:** %65.71
            - **Test Set ROC-AUC:** %71.24
            - **Optimizasyon Öncesi ROC-AUC:** %71.01
            - **Optimizasyon Sonrası İyileştirme:** Accuracy +0.73%, ROC-AUC +0.22%
            
            **📊 Performans Değerlendirmesi:**
            - **Accuracy:** Orta seviye (%65.71) - İyileştirilebilir
            - **ROC-AUC:** Kabul edilebilir (%71.24) - Model riskli/risksiz ayırt etme konusunda rastgele tahminden daha iyi
            - **Kararlılık:** Yüksek (Optimize edilmiş parametreler ile tutarlı sonuçlar)
            
            **Avantajlar:**
            - 📊 Kapsamlı risk analizi (çok değişkenli)
            - 🔍 Detaylı finansal profil değerlendirmesi
            - 💼 Kurumsal seviye karar desteği
            - 📈 Model kararlılığı yüksek (tutarlı sonuçlar)
            
            **Kullanım Senaryosu:**
            Büyük tutarlı krediler, kurumsal müşteriler veya detaylı risk analizi gerektiren durumlarda kullanılır.
            """)

# =========================================================
# SAYFA 2: MÜŞTERİ KAYIP (CHURN)
# =========================================================
elif page == "📉 Müşteri Kayıp (Churn)":
    st.title("📉 Müşteri Kayıp (Churn) Önleme Paneli")
    tab_single, tab_batch, tab_analytics, tab_logic, tab_models = st.tabs(
        ["🔍 Tekil Müşteri Analizi", "📋 Toplu Risk Listesi", "📊 Segment Bazlı Analiz", "🧠 Strateji Mantığı", "🔬 Model Denemeleri"])

    with tab_single:
        col_c1, _ = st.columns([1, 6])
        with col_c1:
            st.button("🎲 Rastgele Müşteri Getir", on_click=get_random_churn_customer, use_container_width=True,
                      key="c_rand_tab")
        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            c_geo = st.selectbox("Ülke", ["France", "Germany", "Spain"], key="c_geo")
            c_gen = st.selectbox("Cinsiyet", ["Female", "Male"], key="c_gen")
            c_age = st.number_input("Yaş", 18, 100, key="c_age")
        with c_col2:
            c_score = st.number_input("Kredi Skoru", 300, 850, key="c_score")
            c_bal = st.number_input("Hesap Bakiyesi ($)", 0.0, 300000.0, key="c_bal")
            c_sal = st.number_input("Tahmini Maaş ($)", 0.0, 200000.0, key="c_sal")
        with c_col3:
            c_prod = st.selectbox("Ürün Sayısı", [1, 2, 3, 4], key="c_prod")
            c_card = st.selectbox("Kredi Kartı Var mı?", ["Evet", "Hayır"], key="c_card")
            c_active = st.selectbox("Üyelik Durumu", ["Aktif", "Pasif"], key="c_active")

        if st.button("🔍 KAYIP RİSKİNİ HESAPLA", type="primary", use_container_width=True):
            if churn_model:
                age_grp = 'Young' if c_age <= 30 else 'Adult' if c_age <= 45 else 'Middle' if c_age <= 60 else 'Senior'
                input_df = pd.DataFrame(
                    {'CreditScore': [c_score], 'Geography': [c_geo], 'Gender': [c_gen], 'Age': [c_age], 'Tenure': [5],
                     'Balance': [c_bal], 'NumOfProducts': [c_prod], 'HasCrCard': [1 if c_card == "Evet" else 0],
                     'IsActiveMember': [1 if c_active == "Aktif" else 0], 'EstimatedSalary': [c_sal],
                     'Balance_per_Product': [c_bal / (c_prod + 0.1)], 'Age_Group': [age_grp],
                     'Credit_Score_Age_Ratio': [c_score / (c_age + 1)],
                     'Is_High_Value_Active': [1 if (c_active == "Aktif" and c_bal > 70000) else 0]})
                prob = float(churn_model.predict_proba(input_df)[0][1])
                
                # advanced_strategy fonksiyonu ile aynı mantık
                if prob > 0.60:
                    if c_bal > 50000:
                        strategy_text = "🚨 VIP MÜDAHALE"
                    elif c_prod >= 3:
                        strategy_text = "🔄 SADELEŞTİRME"
                    else:
                        strategy_text = "📞 ARAMA"
                elif prob > 0.40:
                    if c_active == "Pasif":
                        strategy_text = "🔔 UYANDIRMA"
                    elif c_age < 35:
                        strategy_text = "🎁 LIFESTYLE HEDİYE"
                    else:
                        strategy_text = "💳 TEŞVİK"
                else:  # prob <= 0.40
                    if c_bal > 100000:
                        strategy_text = "💰 YATIRIM ÇAPRAZ SATIŞ"
                    else:
                        strategy_text = "🤝 İLİŞKİ YÖNETİMİ"
                st.divider();
                res1, res2 = st.columns([1, 2])
                with res1:
                    st.metric("Risk Skoru", f"%{prob * 100:.1f}")
                    if prob > 0.6:
                        st.error("⚠️ YÜKSEK RİSK")
                    elif prob > 0.4:
                        st.warning("⚡ ORTA RİSK")
                    else:
                        st.success("✅ DÜŞÜK RİSK")
                    
                    # Risk skalası (sayı doğrusu)
                    st.markdown("---")
                    st.markdown("**📊 Risk Skalası:**")
                    
                    # Sayı doğrusu görselleştirmesi
                    risk_percent = prob * 100
                    scale_html = f"""
                    <div style="position: relative; width: 100%; margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; font-size: 10px; color: #888; margin-bottom: 5px;">
                            <span>0%</span>
                            <span>20%</span>
                            <span>40%</span>
                            <span>60%</span>
                            <span>80%</span>
                            <span>100%</span>
                        </div>
                        <div style="position: relative; width: 100%; height: 8px; background: linear-gradient(to right, #00ff00 0%, #ffff00 40%, #ff8000 60%, #ff0000 100%); border-radius: 4px; margin-bottom: 5px;"></div>
                        <div style="position: relative; width: 100%; height: 20px;">
                            <div style="position: absolute; left: {risk_percent}%; transform: translateX(-50%); width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 10px solid #fff;"></div>
                        </div>
                    </div>
                    """
                    st.markdown(scale_html, unsafe_allow_html=True)
                    
                    # Risk seviyesi açıklaması
                    if prob <= 0.2:
                        st.caption("🟢 Çok Düşük Risk")
                    elif prob <= 0.4:
                        st.caption("🟡 Düşük Risk")
                    elif prob <= 0.6:
                        st.caption("🟠 Orta Risk")
                    elif prob <= 0.8:
                        st.caption("🔴 Yüksek Risk")
                    else:
                        st.caption("🔴🔴 Çok Yüksek Risk")
                        
                with res2:
                    st.markdown("##### 💡 Kişiselleştirilmiş Strateji")
                    strategy_info = get_strategy_details(strategy_text)
                    
                    # Strateji başlığı ve açıklama
                    st.markdown(f"### {strategy_info['title']}")
                    st.caption(f"*{strategy_info['description']}*")
                    
                    # Detaylı bilgileri expander içinde göster
                    with st.expander("📋 Detaylı Aksiyon Planı", expanded=True):
                        st.markdown(strategy_info['details'])
                        st.info(f"⏱️ **Zaman Çizelgesi:** {strategy_info['timeline']}")

    with tab_batch:
        if churn_model and df_churn is not None:
            # CSS ile selectbox ve buton genişliklerini eşitle
            st.markdown("""
                <style>
                div[data-testid="column"]:first-child [data-baseweb="select"] > div {
                    width: 100% !important;
                }
                div[data-testid="column"]:first-child button {
                    width: 100% !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Tek bir column kullan, buton selectbox'ın altında
            col_list1, _ = st.columns([1, 3])
            with col_list1:
                top_n = st.selectbox("Görüntülenecek Müşteri Sayısı", [10, 100, 500, 1000], index=1)
            all_probs = churn_model.predict_proba(df_churn.drop(
                columns=['User_ID', 'Has_Vadesiz', 'Has_BES', 'Has_Kredi', 'Has_Yatirim', 'Spending_Score',
                         'Cluster_Label', 'Segment_Name'], errors='ignore'))[:, 1]
            df_res = df_churn.copy();
            df_res['Risk_Probability'] = all_probs;
            df_res['Strategy'] = df_res.apply(advanced_strategy, axis=1)
            df_top = df_res.sort_values('Risk_Probability', ascending=False).head(top_n)
            
            # CSV verisini önceden hazırla (buton için)
            csv = df_top[['User_ID', 'Risk_Probability', 'Strategy']].to_csv(index=False).encode('utf-8-sig')
            
            # Butonu selectbox'ın altına yerleştir (aynı genişlikte)
            with col_list1:
                st.download_button(label=f"📥 Aksiyon Planını İndir", data=csv, file_name=f"AdvancedActionPlan.csv",
                                   mime="text/csv", use_container_width=True)
            
            st.subheader(f"📋 En Yüksek Riskli {top_n} Müşteri ve Aksiyon Planı")
            display_cols = ['User_ID', 'Risk_Probability', 'Strategy', 'Balance', 'NumOfProducts']
            st.table(df_top[display_cols].style.format({
                'Risk_Probability': '{:.1%}',
                'Balance': '${:,.0f}'
            }))

    with tab_analytics:
        if churn_model and df_churn is not None:
            all_probs_all = churn_model.predict_proba(df_churn.drop(
                columns=['User_ID', 'Has_Vadesiz', 'Has_BES', 'Has_Kredi', 'Has_Yatirim', 'Spending_Score',
                         'Cluster_Label', 'Segment_Name'], errors='ignore'))[:, 1]
            df_analysis = df_churn.copy()
            df_analysis['Risk_Probability'] = all_probs_all
            age_order = ['Young', 'Adult', 'Middle', 'Senior']
            df_analysis['Age_Group'] = pd.Categorical(df_analysis['Age_Group'], categories=age_order, ordered=True)

            # --- FİLTRELEME PANELİ ---
            st.markdown("### 🔍 Filtreleme Seçenekleri")
            st.caption("💡 **İpucu:** Dropdown menülerden istediğiniz kategorileri seçin. Hiçbir şey seçmezseniz o kategoride tüm veri gösterilir.")
            
            # Session state başlatma (filtreler için)
            filter_key_prefix = "analytics_filter_"
            if f"{filter_key_prefix}initialized" not in st.session_state:
                st.session_state[f"{filter_key_prefix}initialized"] = True
                # Varsayılan değerler: hiçbiri seçili değil (boş liste)
                st.session_state[f"{filter_key_prefix}countries"] = []
                st.session_state[f"{filter_key_prefix}age_groups"] = []
                st.session_state[f"{filter_key_prefix}genders"] = []
                st.session_state[f"{filter_key_prefix}show_graphs"] = False
            
            with st.expander("📊 Filtreleri Aç/Kapat", expanded=True):
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    # Ülke filtresi
                    available_countries = sorted(df_analysis['Geography'].unique().tolist())
                    all_countries_option = "✅ Tümünü Seç"
                    countries_options = [all_countries_option] + available_countries
                    
                    # Session state'ten mevcut seçimleri al
                    current_countries = st.session_state.get(f"{filter_key_prefix}countries", [])
                    # Eğer tüm seçenekler seçiliyse, "Tümünü Seç"i de göster
                    if set(current_countries) == set(available_countries) and len(current_countries) == len(available_countries):
                        default_countries = [all_countries_option]
                    else:
                        default_countries = current_countries
                    
                    selected_countries_raw = st.multiselect(
                        "🌍 Ülke",
                        options=countries_options,
                        default=default_countries,
                        help="Analiz edilecek ülkeleri seçin (boş bırakırsanız tümü gösterilir)"
                    )
                    # "Tümünü Seç" kontrolü
                    if all_countries_option in selected_countries_raw:
                        if len(selected_countries_raw) == 1:
                            # Sadece "Tümünü Seç" seçiliyse, tümünü seç
                            selected_countries = available_countries.copy()
                        else:
                            # "Tümünü Seç" + başka seçenekler varsa, "Tümünü Seç"i kaldır
                            selected_countries = [c for c in selected_countries_raw if c != all_countries_option]
                    else:
                        selected_countries = selected_countries_raw
                        # Eğer tüm seçenekler manuel olarak seçildiyse, otomatik olarak tümünü seç
                        if set(selected_countries) == set(available_countries) and len(selected_countries) == len(available_countries):
                            selected_countries = available_countries.copy()
                    st.session_state[f"{filter_key_prefix}countries"] = selected_countries
                    
                    # Yaş grubu filtresi
                    available_age_groups = ['Young', 'Adult', 'Middle', 'Senior']
                    all_age_option = "✅ Tümünü Seç"
                    age_options = [all_age_option] + available_age_groups
                    
                    # Session state'ten mevcut seçimleri al
                    current_age_groups = st.session_state.get(f"{filter_key_prefix}age_groups", [])
                    # Eğer tüm seçenekler seçiliyse, "Tümünü Seç"i de göster
                    if set(current_age_groups) == set(available_age_groups) and len(current_age_groups) == len(available_age_groups):
                        default_age_groups = [all_age_option]
                    else:
                        default_age_groups = current_age_groups
                    
                    selected_age_groups_raw = st.multiselect(
                        "👥 Yaş Grubu",
                        options=age_options,
                        default=default_age_groups,
                        help="Analiz edilecek yaş gruplarını seçin (boş bırakırsanız tümü gösterilir)"
                    )
                    # "Tümünü Seç" kontrolü
                    if all_age_option in selected_age_groups_raw:
                        if len(selected_age_groups_raw) == 1:
                            # Sadece "Tümünü Seç" seçiliyse, tümünü seç
                            selected_age_groups = available_age_groups.copy()
                        else:
                            # "Tümünü Seç" + başka seçenekler varsa, "Tümünü Seç"i kaldır
                            selected_age_groups = [a for a in selected_age_groups_raw if a != all_age_option]
                    else:
                        selected_age_groups = selected_age_groups_raw
                        # Eğer tüm seçenekler manuel olarak seçildiyse, otomatik olarak tümünü seç
                        if set(selected_age_groups) == set(available_age_groups) and len(selected_age_groups) == len(available_age_groups):
                            selected_age_groups = available_age_groups.copy()
                    st.session_state[f"{filter_key_prefix}age_groups"] = selected_age_groups
                
                with filter_col2:
                    # Cinsiyet filtresi
                    available_genders = sorted(df_analysis['Gender'].unique().tolist())
                    all_genders_option = "✅ Tümünü Seç"
                    genders_options = [all_genders_option] + available_genders
                    
                    # Session state'ten mevcut seçimleri al
                    current_genders = st.session_state.get(f"{filter_key_prefix}genders", [])
                    # Eğer tüm seçenekler seçiliyse, "Tümünü Seç"i de göster
                    if set(current_genders) == set(available_genders) and len(current_genders) == len(available_genders):
                        default_genders = [all_genders_option]
                    else:
                        default_genders = current_genders
                    
                    selected_genders_raw = st.multiselect(
                        "⚧️ Cinsiyet",
                        options=genders_options,
                        default=default_genders,
                        help="Analiz edilecek cinsiyetleri seçin (boş bırakırsanız tümü gösterilir)"
                    )
                    # "Tümünü Seç" kontrolü
                    if all_genders_option in selected_genders_raw:
                        if len(selected_genders_raw) == 1:
                            # Sadece "Tümünü Seç" seçiliyse, tümünü seç
                            selected_genders = available_genders.copy()
                        else:
                            # "Tümünü Seç" + başka seçenekler varsa, "Tümünü Seç"i kaldır
                            selected_genders = [g for g in selected_genders_raw if g != all_genders_option]
                    else:
                        selected_genders = selected_genders_raw
                        # Eğer tüm seçenekler manuel olarak seçildiyse, otomatik olarak tümünü seç
                        if set(selected_genders) == set(available_genders) and len(selected_genders) == len(available_genders):
                            selected_genders = available_genders.copy()
                    st.session_state[f"{filter_key_prefix}genders"] = selected_genders
                    
                    # Bakiye aralığı (10000'in katları)
                    min_balance_raw = float(df_analysis['Balance'].min())
                    max_balance_raw = float(df_analysis['Balance'].max())
                    
                    # Min ve max değerleri 10000'in katına yuvarla (aşağı ve yukarı)
                    min_balance = int((int(min_balance_raw // 10000)) * 10000)
                    max_balance = int((int(max_balance_raw // 10000) + 1) * 10000)
                    
                    # Session state'ten mevcut değerleri al veya varsayılan değerleri kullan
                    current_balance_range = st.session_state.get(f"{filter_key_prefix}balance_range", (min_balance, max_balance))
                    # Mevcut değerleri 10000'in katına yuvarla
                    current_min = int((int(current_balance_range[0] // 10000)) * 10000)
                    current_max = int((int(current_balance_range[1] // 10000) + 1) * 10000) if current_balance_range[1] % 10000 != 0 else int(current_balance_range[1])
                    
                    balance_range = st.slider(
                        "💰 Bakiye Aralığı ($)",
                        min_value=min_balance,
                        max_value=max_balance,
                        value=(current_min, current_max),
                        step=10000,
                        format="$%.0f",
                        help="Bakiye aralığını belirleyin (sadece 10000'in katları seçilebilir)"
                    )
                    st.session_state[f"{filter_key_prefix}balance_range"] = balance_range
                
                # Grafikleri Göster butonu
                st.markdown("---")
                show_graphs_col1, show_graphs_col2, show_graphs_col3 = st.columns([1, 2, 1])
                with show_graphs_col2:
                    if st.button("📊 Grafikleri Göster", use_container_width=True, type="primary"):
                        st.session_state[f"{filter_key_prefix}show_graphs"] = True
                        st.rerun()
            
            # --- FİLTRELEME UYGULAMA ---
            df_filtered = df_analysis.copy()
            
            # Ülke filtresi (boşsa tümü göster)
            if selected_countries:
                df_filtered = df_filtered[df_filtered['Geography'].isin(selected_countries)]
            
            # Yaş grubu filtresi (boşsa tümü göster)
            if selected_age_groups:
                df_filtered = df_filtered[df_filtered['Age_Group'].isin(selected_age_groups)]
            
            # Cinsiyet filtresi (boşsa tümü göster)
            if selected_genders:
                df_filtered = df_filtered[df_filtered['Gender'].isin(selected_genders)]
            
            # Bakiye filtresi (slider - her zaman uygulanır)
            df_filtered = df_filtered[
                (df_filtered['Balance'] >= balance_range[0]) & 
                (df_filtered['Balance'] <= balance_range[1])
            ]
            
            # Filtre sonuç bilgisi
            total_count = len(df_analysis)
            filtered_count = len(df_filtered)
            filter_info = f"📊 **Filtrelenmiş Veri:** {filtered_count:,} müşteri (Toplam: {total_count:,})"
            
            if filtered_count < total_count:
                st.success(filter_info)
            else:
                st.info(filter_info)
            
            if len(df_filtered) == 0:
                st.warning("⚠️ Seçilen filtrelerle eşleşen müşteri bulunamadı. Lütfen filtreleri gevşetin.")
                st.stop()
            
            # Grafikleri göstermek için butona basılmış mı kontrol et
            show_graphs = st.session_state.get(f"{filter_key_prefix}show_graphs", False)
            
            if not show_graphs:
                st.info("👆 Yukarıdaki **'Grafikleri Göster'** butonuna basarak filtrelenmiş verilerin grafiklerini görüntüleyebilirsiniz.")
            else:
                # Grafikler sadece butona basıldığında gösterilir
                st.markdown("### 📊 Filtrelenmiş Veri Grafikleri")
                st.divider()

                col_row1_1, col_row1_2 = st.columns(2)
                with col_row1_1:
                    # Filtrelenmiş veri ile grafik
                    if len(df_filtered) > 0:
                        geo_age_data = df_filtered.groupby(['Geography', 'Age_Group'], observed=True)['Risk_Probability'].mean().reset_index()
                        if len(geo_age_data) > 0:
                            fig1 = px.bar(geo_age_data, x="Age_Group", y="Risk_Probability",
                                  color="Geography", barmode="group", title="1. Ülke & Yaş Grubu Bazlı Risk",
                                  color_discrete_sequence=['#00f0ff', '#ff00d4', '#9d4edd'])
                            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                               font=dict(color="white"), xaxis=dict(showgrid=False),
                                               yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                            st.plotly_chart(fig1, use_container_width=True)
                        else:
                            st.info("Bu filtrelerle grafik oluşturulamadı.")
                with col_row1_2:
                    # Filtrelenmiş veri ile aktiflik grafiği
                    if len(df_filtered) > 0:
                        active_data = df_filtered.groupby('IsActiveMember')['Risk_Probability'].mean().reset_index()
                        active_data['IsActiveMember'] = active_data['IsActiveMember'].map({1: 'Aktif', 0: 'Pasif'})
                        if len(active_data) > 0:
                            fig2 = px.pie(active_data, values='Risk_Probability', names='IsActiveMember', hole=.5,
                                  title="2. Aktiflik Durumuna Göre Risk", color_discrete_sequence=['#00f0ff', '#ff00d4'])
                            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info("Bu filtrelerle grafik oluşturulamadı.")

                st.divider()
                col_row2_1, col_row2_2 = st.columns(2)
                with col_row2_1:
                    # Filtrelenmiş veri ile scatter plot
                    if len(df_filtered) > 0:
                        sample_size = min(2000, len(df_filtered))
                        scatter_data = df_filtered.sample(n=sample_size, random_state=42) if len(df_filtered) > sample_size else df_filtered
                        if len(scatter_data) > 0:
                            fig3 = px.scatter(scatter_data, x="CreditScore", y="Risk_Probability", 
                                              color="Age_Group", opacity=0.5,
                                  title="3. Kredi Skoru & Risk İlişkisi",
                                  color_discrete_sequence=px.colors.qualitative.Bold)
                            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                               font=dict(color="white"), xaxis=dict(showgrid=False),
                                               yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'))
                            st.plotly_chart(fig3, use_container_width=True)
                        else:
                            st.info("Bu filtrelerle grafik oluşturulamadı.")
                with col_row2_2:
                    # Filtrelenmiş veri ile ürün sayısı grafiği
                    if len(df_filtered) > 0:
                        product_data = df_filtered.groupby('NumOfProducts')['Risk_Probability'].mean().reset_index()
                        if len(product_data) > 0:
                            fig4 = px.bar(product_data, x="NumOfProducts", y="Risk_Probability", 
                                          title="4. Ürün Sayısına Göre Churn",
                              color="Risk_Probability", color_continuous_scale="RdBu_r")
                            fig4.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1, showgrid=False),
                                               yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                               font=dict(color="white"))
                            st.plotly_chart(fig4, use_container_width=True)
                        else:
                            st.info("Bu filtrelerle grafik oluşturulamadı.")
                st.divider();
                st.error("⚠️ **STRATEJİK ANALİZ:** 2 Ürün 'Güvenli Bölge' iken, 3+ Ürün 'Yüksek Risk' bölgesidir.")

    with tab_logic:
        st.subheader("🧠 Algoritmanın Karar Matrisi")
        logic_data = {"Risk Seviyesi": ["🚨 Yüksek", "🚨 Yüksek", "🚨 Yüksek", "⚡ Orta", "⚡ Orta", "⚡ Orta", "✅ Düşük", "✅ Düşük"],
                      "Ek Koşul": ["Bakiye > 50K", "Ürün >= 3", "Diğer", "Pasif Üye", "Yaş < 35", "Diğer", "Bakiye > 100K", "Standart"],
                      "Strateji": ["VIP Müdahale", "Sadeleştirme", "Arama", "Uyandırma", "Lifestyle Hediye", "Teşvik",
                                   "Yatırım Çapraz Satış", "İlişki Yönetimi"]}
        st.table(pd.DataFrame(logic_data))
        
        st.markdown("---")
        st.markdown("##### 📖 Strateji Mantığı Açıklamaları")
        st.markdown("""
        **🚨 YÜKSEK RİSK SEVİYESİ (Risk Olasılığı > %60):**
        
        - **VIP MÜDAHALE (Bakiye > 50K):** Müşterinin hesabında 50.000$'dan fazla bakiye varsa ve churn riski yüksekse, 
          bu müşteri değerli bir varlıktır. Hemen özel müşteri temsilcisi atanır, kişisel görüşme yapılır ve 
          sorunlar dinlenir. Amaç: Değerli müşteriyi kaybetmemek.
        
        - **SADELEŞTİRME (Ürün >= 3):** Müşterinin 3 veya daha fazla ürünü varsa, bu "ürün paradoksu" işaretidir. 
          Çok fazla ürün, müşterinin aşırı borçlanma veya karmaşa yaşadığını gösterir. Strateji: Ürünleri sadeleştir, 
          gereksiz ürünleri kapat, müşteriyi rahatlat. Amaç: Müşteriyi boğmamak, sadakati artırmak.
        
        - **ARAMA (Diğer Yüksek Risk Durumları):** Yüksek riskli ancak yukarıdaki özel koşulları sağlamayan müşteriler için 
          proaktif iletişim stratejisi. Müşteri temsilcisi tarafından doğrudan arama yapılır, sorunlar dinlenir ve çözüm önerilir.
        
        **⚡ ORTA RİSK SEVİYESİ (Risk Olasılığı %40-60):**
        
        - **UYANDIRMA (Pasif Üye):** Müşteri uzun süredir hesabını kullanmıyorsa (pasif), bankayı unutmuş olabilir. 
          Strateji: Özel bonus kampanyaları, faiz indirimleri veya hediye puanlar sunarak müşteriyi tekrar aktif hale getir. 
          Amaç: İlişkiyi canlandırmak, unutulmuş müşteriyi geri kazanmak.
        
        - **LIFESTYLE HEDİYE (Yaş < 35):** Genç müşteriler (35 yaş altı) genelde teknolojiye meraklıdır ve sosyal medyada aktiftir. 
          Strateji: Onlara yaşam tarzına uygun hediyeler (konser bileti, spor salonu üyeliği, teknoloji ürünleri) sun. 
          Amaç: Genç müşterilerle duygusal bağ kurmak, marka sadakati oluşturmak.
        
        - **TEŞVİK (Diğer Orta Risk Durumları):** Aktif ancak orta risk seviyesindeki müşteriler için genel teşvik kampanyaları 
          ve özel fırsatlar sunulur. Amaç: Müşteriyi aktif tutmak ve ilişkiyi güçlendirmek.
        
        **✅ DÜŞÜK RİSK SEVİYESİ (Risk Olasılığı <= %40):**
        
        - **YATIRIM ÇAPRAZ SATIŞ (Bakiye > 100K):** Müşterinin hesabında 100.000$'dan fazla bakiye varsa ve risk düşükse, 
          bu müşteri yatırım yapmaya hazırdır. Strateji: Likit fon, altın, yatırım hesabı gibi ürünler öner. 
          Amaç: Müşterinin parasını değerlendirmesine yardımcı olmak, banka ile ilişkiyi derinleştirmek.
        
        - **İLİŞKİ YÖNETİMİ (Standart):** Risk düşük ve özel bir koşul yoksa, standart müşteri ilişkisi yönetimi uygulanır. 
          Strateji: Düzenli iletişim, genel kampanyalar, müşteri memnuniyeti takibi. 
          Amaç: Mevcut durumu korumak, müşteriyi mutlu tutmak.
        """)

    with tab_models:
        st.subheader("🔬 Model Karşılaştırma ve Denemeler")
        st.markdown("""
        Bu sekmede, churn tahmin modeli için yapılan **XGBoost, LightGBM ve CatBoost** modellerinin 
        kapsamlı karşılaştırma sonuçları gösterilmektedir.
        """)
        
        # Log dosyasını oku
        try:
            with open('model_comparison_log.txt', 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Başlangıç ve bitiş zamanlarını bul
            import re
            start_match = re.search(r'Baslangic Zamani: (.+)', log_content)
            end_match = re.search(r'Bitis Zamani: (.+)', log_content)
            
            if start_match and end_match:
                st.info(f"**Test Tarihi:** {start_match.group(1)} - {end_match.group(1)}")
            
            # Test Metodolojisi
            with st.expander("📋 Test Metodolojisi", expanded=True):
                st.markdown("""
                **Cross-Validation Yöntemi:**
                - **50-Katlı Stratified Cross-Validation** kullanılmıştır
                - Her model için aynı veri seti ve aynı fold'lar kullanılmıştır (adil karşılaştırma)
                - Her fold'da hem **ROC-AUC** hem **Accuracy** metrikleri hesaplanmıştır
                - Model eğitimi tek bir CV döngüsünde yapılmıştır (gereksiz tekrar önlendi)
                
                **Test Edilen Modeller:**
                1. **XGBoost (Mevcut Model)** - Extreme Gradient Boosting
                2. **LightGBM** - Light Gradient Boosting Machine
                3. **CatBoost** - Categorical Boosting
                
                **Veri Seti:**
                - Toplam kayıt: 10,000
                - Özellik sayısı: 18 (temel + türetilmiş özellikler)
                - Hedef değişken dağılımı: {0: 7962, 1: 2038}
                """)
            
            # Performans Karşılaştırması
            st.markdown("---")
            st.markdown("### 📊 Performans Karşılaştırması")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏆 En İyi ROC-AUC: CatBoost", "87.17%", "+0.48% (XGBoost'a göre)")
            
            with col2:
                st.metric("🏆 En İyi Accuracy: CatBoost", "86.50%", "+0.16% (XGBoost'a göre)")
            
            with col3:
                st.metric("⚡ En Hızlı Model: LightGBM", "4.74 saniye", "XGBoost'tan 1.47s daha hızlı")
            
            # Detaylı Sonuçlar Tablosu
            st.markdown("---")
            st.markdown("### 📈 Detaylı Performans Metrikleri")
            
            # ROC-AUC Sonuçları
            st.markdown("#### ROC-AUC Skorları (50-Fold CV)")
            roc_auc_data = {
                'Model': ['XGBoost (Mevcut)', 'LightGBM', 'CatBoost'],
                'Ortalama': ['86.69%', '86.72%', '87.17%'],
                'Std Sapma': ['3.67%', '3.64%', '3.38%'],
                'Min': ['78.69%', '78.59%', '79.67%'],
                'Max': ['93.11%', '92.81%', '92.81%']
            }
            st.table(pd.DataFrame(roc_auc_data))
            
            # Accuracy Sonuçları
            st.markdown("#### Accuracy Skorları (50-Fold CV)")
            acc_data = {
                'Model': ['XGBoost (Mevcut)', 'LightGBM', 'CatBoost'],
                'Ortalama': ['86.34%', '86.45%', '86.50%'],
                'Std Sapma': ['1.98%', '1.96%', '1.86%'],
                'Min': ['81.50%', '82.00%', '83.00%'],
                'Max': ['89.50%', '89.50%', '91.00%']
            }
            st.table(pd.DataFrame(acc_data))
            
            # Hız Karşılaştırması
            st.markdown("---")
            st.markdown("### ⚡ Eğitim Süresi Karşılaştırması")
            speed_data = {
                'Model': ['XGBoost (Mevcut)', 'LightGBM', 'CatBoost'],
                'Süre (saniye)': ['6.21', '4.74', '21.93'],
                'Göreceli Hız': ['1.0x (Referans)', '1.31x (Daha Hızlı)', '0.28x (Daha Yavaş)']
            }
            st.table(pd.DataFrame(speed_data))
            
            # Model Seçimi
            st.markdown("---")
            st.markdown("### 🎯 Model Seçimi ve Sonuç")
            
            st.success("""
            **Seçilen Model: LightGBM**
            
            **Seçim Gerekçesi:**
            - ✅ **Hız:** XGBoost'tan %24 daha hızlı (4.74s vs 6.21s)
            - ✅ **Performans:** XGBoost'a çok yakın performans (ROC-AUC: %86.72 vs %86.69)
            - ✅ **Dengeli:** Performans ve hız arasında en iyi denge
            - ✅ **Pratiklik:** Üretim ortamında daha hızlı tahmin süreleri
            
            **Not:** CatBoost en yüksek performansı gösterdi ancak eğitim süresi 4.6x daha uzun olduğu için 
            pratik kullanım için LightGBM tercih edilmiştir.
            """)
            
            # Detaylı İstatistikler
            st.markdown("---")
            with st.expander("📊 Detaylı İstatistiksel Analiz", expanded=False):
                st.markdown("""
                **XGBoost (Mevcut):**
                - ROC-AUC Ortalama: 86.69% (Medyan: 87.22%, Q1: 84.74%, Q3: 89.60%)
                - Accuracy Ortalama: 86.34% (Medyan: 86.50%)
                
                **LightGBM:**
                - ROC-AUC Ortalama: 86.72% (Medyan: 87.21%, Q1: 84.35%, Q3: 89.70%)
                - Accuracy Ortalama: 86.45% (Medyan: 86.50%)
                
                **CatBoost:**
                - ROC-AUC Ortalama: 87.17% (Medyan: 87.57%, Q1: 84.89%, Q3: 89.61%)
                - Accuracy Ortalama: 86.50% (Medyan: 86.50%)
                """)
            
            # Log Dosyası Görüntüleme
            st.markdown("---")
            with st.expander("📄 Tam Log Dosyası", expanded=False):
                st.code(log_content, language='text')
        
        except FileNotFoundError:
            st.warning("⚠️ Model karşılaştırma log dosyası bulunamadı. Lütfen `compare_churn_models_100cv.py` scriptini çalıştırın.")
        except Exception as e:
            st.error(f"Log dosyası okunurken bir hata oluştu: {e}")

# =========================================================
# SAYFA 3: FIRSATLAR VE SATIŞ (NBA)
# =========================================================
elif page == "🎯 Fırsatlar & Satış (NBA - K-Means)":
    st.title("🎯 Fırsatlar & Akıllı Satış Ekranı (K-Means Clustering)")
    st.markdown("Müşteri odaklı **'Sıradaki En İyi Aksiyon' (Next Best Action)** önerileri.")
    tab_ind, tab_camp = st.tabs(["🎯 Bireysel Analiz", "📢 Toplu Kampanya Yönetimi"])

    with tab_ind:
        # Radio butonlarını sola yasla ve kompakt yap
        st.markdown("**Analiz Modu Seçiniz:**")
        mode = st.radio("", ["🎲 Simülasyon (Rastgele)", "✏️ Manuel Giriş"], horizontal=True, label_visibility="collapsed")
        st.divider()
        selected_row, segment_name, cust_id = None, "Manuel Analiz", "Manuel-001"

        if mode == "🎲 Simülasyon (Rastgele)":
            if st.button("🎲 Rastgele Müşteri Analiz Et", on_click=get_random_churn_customer,
                         use_container_width=True): pass
            if 'c_id' in st.session_state:
                s = st.session_state
                selected_row = {'Balance': s.get('c_bal', 0.0), 'EstimatedSalary': s.get('c_sal', 50000.0),
                                'Age': s.get('c_age', 30), 'Spending_Score': s.get('c_spending', 50),
                                'Has_Yatirim': s.get('has_yatirim', 0), 'Has_BES': s.get('has_bes', 0),
                                'HasCrCard': 1 if s.get('c_card') == "Evet" else 0, 'Has_Kredi': s.get('has_kredi', 0),
                                'CreditScore': s.get('c_score', 650), 'NumOfProducts': s.get('c_prod', 1),
                                'Tenure': s.get('c_tenure', 5), 'IsActiveMember': 1 if s.get('c_active') == "Aktif" else 0}
                segment_name, cust_id = s.get('c_segment', "Bilinmiyor"), s.get('c_id', "ID_YOK")
        else:
            # Session state'ten değerleri al veya varsayılan kullan
            m_age = st.session_state.get('manual_age', 30)
            m_sal = st.session_state.get('manual_sal', 50000.0)
            m_card = st.session_state.get('manual_card', True)
            m_bal = st.session_state.get('manual_bal', 10000.0)
            m_score = st.session_state.get('manual_score', 650)
            m_bes = st.session_state.get('manual_bes', False)
            m_prod = st.session_state.get('manual_prod', 1)
            m_yatirim = st.session_state.get('manual_yatirim', False)
            m_kredi = st.session_state.get('manual_kredi', False)
            m_tenure = st.session_state.get('manual_tenure', 5)
            m_active = st.session_state.get('manual_active', True)
            
            with st.form("manual_input_form"):
                c1, c2, c3 = st.columns(3)
                with c1: 
                    m_age = st.number_input("Yaş", 18, 90, m_age, key="manual_age_input")
                    m_sal = st.number_input("Maaş ($)", 0.0, 200000.0, m_sal, key="manual_sal_input")
                    m_card = st.checkbox("Kredi Kartı Var mı?", value=m_card, key="manual_card_input")
                with c2: 
                    m_bal = st.number_input("Bakiye ($)", 0.0, 500000.0, m_bal, key="manual_bal_input")
                    m_score = st.number_input("Kredi Skoru", 300, 850, m_score, key="manual_score_input")
                    m_bes = st.checkbox("BES Var mı?", value=m_bes, key="manual_bes_input")
                with c3: 
                    m_prod = st.number_input("Ürün Sayısı", 1, 4, m_prod, key="manual_prod_input")
                    m_tenure = st.number_input("Müşteri Süresi (Yıl)", 0, 10, m_tenure, key="manual_tenure_input")
                    m_active = st.checkbox("Aktif Üye mi?", value=m_active, key="manual_active_input")
                    m_yatirim = st.checkbox("Yatırım Hesabı Var mı?", value=m_yatirim, key="manual_yatirim_input")
                    m_kredi = st.checkbox("Aktif Kredisi Var mı?", value=m_kredi, key="manual_kredi_input")
                
                if st.form_submit_button("🔍 Müşteriyi Analiz Et", type="primary", use_container_width=True):
                    # Session state'e kaydet
                    st.session_state['manual_age'] = m_age
                    st.session_state['manual_sal'] = m_sal
                    st.session_state['manual_card'] = m_card
                    st.session_state['manual_bal'] = m_bal
                    st.session_state['manual_score'] = m_score
                    st.session_state['manual_bes'] = m_bes
                    st.session_state['manual_prod'] = m_prod
                    st.session_state['manual_yatirim'] = m_yatirim
                    st.session_state['manual_kredi'] = m_kredi
                    st.session_state['manual_tenure'] = m_tenure
                    st.session_state['manual_active'] = m_active
                    st.session_state['manual_analysis_done'] = True
                    st.rerun()
            
            # Eğer analiz yapıldıysa göster
            if st.session_state.get('manual_analysis_done', False):
                m_spending = calculate_manual_spending_score(m_sal, m_age, m_card)
                selected_row = {
                    'Balance': m_bal,
                    'EstimatedSalary': m_sal,
                    'Age': m_age,
                    'Spending_Score': m_spending,
                    'Has_Yatirim': 1 if m_yatirim else 0,
                    'Has_BES': 1 if m_bes else 0,
                    'HasCrCard': 1 if m_card else 0,
                    'Has_Kredi': 1 if m_kredi else 0,
                    'CreditScore': m_score,
                    'NumOfProducts': m_prod,
                    'Tenure': st.session_state.get('manual_tenure', 5),
                    'IsActiveMember': 1 if st.session_state.get('manual_active', True) else 0
                }
                
                # --- MANUEL SEGMENT TAHMİNİ ---
                if kmeans_model and scaler_model:
                    # K-Means için gereken özellikleri hazırla (YENİ: 5 değişken)
                    # Balance, EstimatedSalary, NumOfProducts, Tenure, IsActiveMember
                    m_tenure_val = st.session_state.get('manual_tenure', 5)
                    m_active_val = 1 if st.session_state.get('manual_active', True) else 0
                    input_features = ['Balance', 'EstimatedSalary', 'NumOfProducts', 'Tenure', 'IsActiveMember']
                    input_data = pd.DataFrame([[m_bal, m_sal, m_prod, m_tenure_val, m_active_val]], columns=input_features)
                    
                    # Ölçeklendir ve Tahmin Et
                    input_scaled = scaler_model.transform(input_data)
                    cluster_id = kmeans_model.predict(input_scaled)[0]
                    segment_name = cluster_names_map.get(cluster_id, "Bilinmiyor")
                else:
                    segment_name = "Analiz Edilemedi"
                
                cust_id = "Manuel-001"
            else:
                # Manuel giriş modunda henüz analiz yapılmadıysa
                selected_row = None

        # selected_row None ise bilgilendirme mesajı göster
        if not selected_row or selected_row is None:
            if mode == "🎲 Simülasyon (Rastgele)":
                st.info("👆 Yukarıdaki **'Rastgele Müşteri Analiz Et'** butonuna basarak bir müşteri seçebilirsiniz.")
            else:
                st.info("👆 Yukarıdaki formu doldurup **'Müşteriyi Analiz Et'** butonuna basarak analiz yapabilirsiniz.")
        
        if selected_row and selected_row is not None:
            with st.container():
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1: st.subheader(f"👤 {cust_id} | {segment_name}")
                with col_h2:
                    risk_color = "#ff00d4" if selected_row['CreditScore'] < 600 else "#00f0ff"
                    st.markdown(
                        f"<div style='border: 1px solid {risk_color}; color:{risk_color}; padding:5px; border-radius:15px; text-align:center'>Kredi Skoru: {selected_row['CreditScore']}</div>",
                        unsafe_allow_html=True)
            st.markdown("---")
            col_left, col_right = st.columns([1, 2], gap="large")
            with col_left:
                st.markdown("##### 🧬 Finansal DNA")
                
                # YENİ 5 DEĞİŞKENE GÖRE NORMALİZASYON (0-100 arası)
                # Veri setinden max değerleri al (dinamik)
                if df_churn is not None:
                    max_balance = df_churn['Balance'].max()
                    max_salary = df_churn['EstimatedSalary'].max()
                    max_products = 4  # Sabit max
                    max_tenure = 10   # Sabit max
                    
                    # Ortalama değerleri hesapla
                    avg_balance = df_churn['Balance'].mean()
                    avg_salary = df_churn['EstimatedSalary'].mean()
                    avg_products = df_churn['NumOfProducts'].mean()
                    avg_tenure = df_churn['Tenure'].mean()
                    avg_active = df_churn['IsActiveMember'].mean()
                    
                    # Ortalamaları normalize et (0-100)
                    avg_balance_norm = min((avg_balance / max_balance) * 100, 100) if max_balance > 0 else 0
                    avg_salary_norm = min((avg_salary / max_salary) * 100, 100) if max_salary > 0 else 0
                    avg_products_norm = min((avg_products / max_products) * 100, 100)
                    avg_tenure_norm = min((avg_tenure / max_tenure) * 100, 100)
                    avg_active_norm = avg_active * 100
                else:
                    # Varsayılan değerler (veri yüklenmemişse)
                    max_balance, max_salary = 250000, 80000
                    avg_balance_norm, avg_salary_norm = 31, 47
                    avg_products_norm, avg_tenure_norm, avg_active_norm = 38, 50, 52
                
                # Müşteri değerlerini normalize et
                val_balance = min((selected_row['Balance'] / max_balance) * 100, 100) if max_balance > 0 else 0
                val_salary = min((selected_row['EstimatedSalary'] / max_salary) * 100, 100) if max_salary > 0 else 0
                val_products = min((selected_row['NumOfProducts'] / max_products) * 100, 100)
                val_tenure = min((selected_row.get('Tenure', 5) / max_tenure) * 100, 100)
                val_active = selected_row.get('IsActiveMember', 1) * 100
                
                # Müşteri değerleri
                vals = [int(val_balance), int(val_salary), int(val_products), int(val_tenure), int(val_active)]
                
                # Ortalama değerler
                avgs = [int(avg_balance_norm), int(avg_salary_norm), int(avg_products_norm), 
                       int(avg_tenure_norm), int(avg_active_norm)]
                
                categories = ['Bakiye', 'Maaş', 'Ürün Sayısı', 'Müşteri Süresi', 'Aktif Üye']
                
                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(r=vals, theta=categories, fill='toself', name='Müşteri', line=dict(color='#00f0ff'),
                                    marker=dict(color='#00f0ff'), mode='lines+markers+text',
                                    text=[str(v) for v in vals], textposition='top center'))
                fig.add_trace(go.Scatterpolar(r=avgs, theta=categories, fill='toself', name='Ortalama', opacity=0.4,
                                              line=dict(color='#ff00d4'), marker=dict(color='#ff00d4'),
                                              mode='lines+markers'))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
                               bgcolor='rgba(0,0,0,0)'), showlegend=True, legend=dict(font=dict(color="white")),
                    height=350, margin=dict(t=30, b=30, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)
                
                # --- SKOR TABLOSU EKLEME ---
                st.markdown("###### 📊 DNA Skor Detayları")
                dna_data = {
                    "Metrik": categories,
                    "Müşteri Skoru": vals,
                    "Ortalama": avgs
                }
                st.table(pd.DataFrame(dna_data))
            with col_right:
                st.markdown("##### 🔥 Akıllı Öneri (NBA)")
                rec = get_next_best_action(selected_row, segment_name=segment_name)
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); padding: 25px; border-radius: 20px; border: 1px solid rgba(0, 240, 255, 0.2); box-shadow: 0 0 20px rgba(0, 240, 255, 0.1);">
                    <h3 style="margin:0; color:#00f0ff; font-family: 'Syne', sans-serif;">{rec['Product']}</h3>
                    <p style="margin-top:10px; font-size:18px; color: #fff;">Satış İhtimali: <b style="color: #ff00d4;">%{rec['Prob']}</b></p>
                    <hr style="border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 15px 0;">
                    <p style="color: rgba(255,255,255,0.8);"><i>💡 <b>Neden?</b> {rec['Reason']}</i></p>
                    <p style="color: rgba(255,255,255,0.8);">🗣️ <b>Script:</b> "{rec['Script']}"</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("###### 📦 Mevcut Portföy")
                products_html = "<span style='background-color:rgba(0,240,255,0.1); border:1px solid #00f0ff; padding:5px 10px; border-radius:50px; color:#00f0ff; margin-right:5px; font-size:12px;'>✅ Vadesiz Hesap</span>"
                if selected_row[
                    'HasCrCard']: products_html += "<span style='background-color:rgba(0,240,255,0.1); border:1px solid #00f0ff; padding:5px 10px; border-radius:50px; color:#00f0ff; margin-right:5px; font-size:12px;'>✅ Kredi Kartı</span>"
                if selected_row[
                    'Has_BES']: products_html += "<span style='background-color:rgba(157, 78, 221,0.1); border:1px solid #9d4edd; padding:5px 10px; border-radius:50px; color:#9d4edd; margin-right:5px; font-size:12px;'>✅ BES</span>"
                if selected_row[
                    'Has_Yatirim']: products_html += "<span style='background-color:rgba(255, 0, 212,0.1); border:1px solid #ff00d4; padding:5px 10px; border-radius:50px; color:#ff00d4; margin-right:5px; font-size:12px;'>✅ Yatırım</span>"
                if selected_row[
                    'Has_Kredi']: products_html += "<span style='background-color:rgba(255,255,255,0.1); border:1px solid #fff; padding:5px 10px; border-radius:50px; color:#fff; margin-right:5px; font-size:12px;'>✅ Kredi</span>"
                st.markdown(products_html, unsafe_allow_html=True)
            
            # Kredi Skoru Hakkında Bilgi
            st.markdown("---")
            with st.expander("ℹ️ Kredi Skoru Hakkında Bilgi", expanded=False):
                st.markdown("""
                ### 📊 Kredi Skoru Nedir?
                
                **Kredi Skoru**, bir müşterinin finansal geçmişine dayalı olarak hesaplanan, kredi verme kararlarında kullanılan sayısal bir göstergedir. 
                Bu skor, müşterinin borç ödeme geçmişini, mevcut borç durumunu, kredi kullanım süresini ve diğer finansal davranışlarını değerlendirir.
                
                ### 🎯 Skor Aralığı ve Anlamı
                
                Bu veri setinde kullanılan kredi skoru **300-850 aralığında** değerler alır:
                
                - **750-850:** Mükemmel (Excellent) - Çok düşük risk, en iyi kredi koşulları
                - **700-749:** İyi (Good) - Düşük risk, iyi kredi koşulları
                - **650-699:** Orta (Fair) - Orta risk, standart kredi koşulları
                - **600-649:** Zayıf (Poor) - Yüksek risk, sınırlı kredi seçenekleri
                - **300-599:** Çok Zayıf (Very Poor) - Çok yüksek risk, kredi almak zor
                
                ### 🌍 Derecelendirme Sistemi
                
                Bu veri setindeki kredi skoru, **FICO (Fair Isaac Corporation) kredi skoru** sistemine dayanmaktadır. 
                FICO skoru, dünya çapında en yaygın kullanılan kredi skorlama sistemidir ve özellikle **ABD, İngiltere ve Avrupa ülkelerinde** 
                finansal kurumlar tarafından kullanılmaktadır.
                
                **FICO Skoru Bileşenleri:**
                1. **Ödeme Geçmişi (Payment History)** - %35: Geçmişteki ödemelerin zamanında yapılıp yapılmadığı
                2. **Borç Miktarı (Amounts Owed)** - %30: Toplam borç ve kredi limiti kullanım oranı
                3. **Kredi Geçmişi Süresi (Length of Credit History)** - %15: Kredi hesaplarının ne kadar süredir açık olduğu
                4. **Yeni Kredi (New Credit)** - %10: Son zamanlarda açılan yeni kredi hesapları
                5. **Kredi Karışımı (Credit Mix)** - %10: Farklı kredi türlerinin (kredi kartı, kredi, ipotek vb.) kullanımı
                
                ### 📈 Veri Setindeki Kullanım
                
                Bu projede kredi skoru:
                - **Müşteri segmentasyonunda** kullanılmaktadır (K-Means clustering)
                - **Churn (müşteri kaybı) riski** tahmininde önemli bir faktördür
                - **Next Best Action (NBA)** önerilerinde dikkate alınmaktadır
                - Düşük kredi skorlu müşteriler için özel finansal danışmanlık önerileri sunulmaktadır
                
                ### ⚠️ Önemli Notlar
                
                - Kredi skoru, müşterinin **geçmiş finansal davranışlarını** yansıtır, geleceği garanti etmez
                - Skor, **dinamik bir göstergedir** ve zamanla değişebilir
                - Farklı ülkelerde farklı kredi skorlama sistemleri kullanılabilir (ör. Türkiye'de KKB, Findeks)
                - Bu veri seti **simüle edilmiş/örnek veriler** içermektedir ve gerçek müşteri bilgileri değildir
                """)
            

    with tab_camp:
        if churn_model and df_churn is not None:
            st.subheader("📂 Segment Bazlı Kampanya Yönetimi")
            
            # --- SEGMENT DASHBOARD PANEL ---
            cols_to_drop = ['User_ID', 'Has_Vadesiz', 'Has_BES', 'Has_Kredi', 'Has_Yatirim', 'Spending_Score',
                            'Cluster_Label', 'Segment_Name']
            X_all = df_churn.drop(columns=cols_to_drop, errors='ignore')
            all_risk_scores = churn_model.predict_proba(X_all)[:, 1]
            df_dash = df_churn.copy()
            df_dash['Risk_Probability'] = all_risk_scores

            with st.expander("📊 Tüm Segmentlerin Genel Görünümü (Dashboard)", expanded=True):
                # Doğrulama Metriği Gösterimi
                st.write(f"**Model Doğrulama (Silhouette Score):** `{silhouette_val:.3f}`")
                if silhouette_val > 0.3:
                    st.success("✅ Segmentler istatistiksel olarak iyi ayrışmış durumda.")
                else:
                    st.warning("⚠️ Segmentler birbirine çok yakın, daha fazla özellik mühendisliği gerekebilir.")
                
                col_d1, col_d2 = st.columns([1, 1])
                
                # 1. Segment Dağılımı (Pie Chart)
                segment_counts = df_dash['Segment_Name'].value_counts().reset_index()
                segment_counts.columns = ['Segment', 'Müşteri Sayısı']
                fig_pie = px.pie(segment_counts, values='Müşteri Sayısı', names='Segment', 
                                title="Segment Dağılımı", hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                col_d1.plotly_chart(fig_pie, use_container_width=True)
                
                # 2. Ortalama Metrikler (Bar Chart)
                metrics_avg = df_dash.groupby('Segment_Name')[['Balance', 'EstimatedSalary']].mean().reset_index()
                fig_bar = px.bar(metrics_avg, x='Segment_Name', y=['Balance', 'EstimatedSalary'], 
                                barmode='group', title="Ortalama Finansal Durum",
                                color_discrete_sequence=['#00f0ff', '#ff00d4'])
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="white"))
                col_d2.plotly_chart(fig_bar, use_container_width=True)
                
                # 3. Kapsamlı Segment Özellikleri Tablosu (Sadece clustering'de kullanılan değişkenler)
                st.markdown("##### 📋 Segment Karakteristikleri")
                summary_table = df_dash.groupby('Segment_Name').agg({
                    'User_ID': 'count',
                    'Balance': 'mean',
                    'EstimatedSalary': 'mean',
                    'NumOfProducts': 'mean',
                    'Tenure': 'mean',
                    'IsActiveMember': 'mean'
                }).reset_index()
                
                summary_table.columns = ['Segment', 'Üye Sayısı', 'Ort. Bakiye ($)', 'Ort. Maaş ($)', 'Ort. Ürün Sayısı', 'Ort. Müşteri Süresi (Yıl)', 'Aktif Üye Oranı']
                
                # Formatlama
                st.table(summary_table.style.format({
                    'Üye Sayısı': '{:,.0f}',
                    'Ort. Bakiye ($)': '${:,.0f}',
                    'Ort. Maaş ($)': '${:,.0f}',
                    'Ort. Ürün Sayısı': '{:.2f}',
                    'Ort. Müşteri Süresi (Yıl)': '{:.1f}',
                    'Aktif Üye Oranı': '{:.1%}'
                }))

            st.markdown("---")
            st.subheader("🎯 Hedefli Kampanya Listesi Oluştur")
            
            df_campaign = df_dash.copy()
            segments = sorted(df_campaign['Segment_Name'].unique().tolist())
            # "Tümünü Seç" seçeneğini ekle
            segment_options = ["📋 Tüm Segmentler"] + segments
            selected_segment = st.selectbox("Hedef Segmenti Seçiniz:", segment_options)
            
            # Tüm segmentler seçildiyse tüm veriyi al, değilse seçili segmenti filtrele
            if selected_segment == "📋 Tüm Segmentler":
                filtered_df = df_campaign.copy()
                display_segment_name = "Tüm Segmentler"
            else:
                filtered_df = df_campaign[df_campaign['Segment_Name'] == selected_segment].copy()
                display_segment_name = selected_segment


            def get_nba_product_only(row): 
                # Pandas Series için güvenli erişim
                try:
                    if isinstance(row, dict):
                        segment = row.get('Segment_Name', None)
                    else:  # pandas Series
                        segment = row['Segment_Name'] if 'Segment_Name' in row.index else None
                except:
                    segment = None
                return get_next_best_action(row, segment_name=segment)['Product']


            filtered_df['Onerilen_Urun'] = filtered_df.apply(get_nba_product_only, axis=1)

            st.write(f"**{display_segment_name}** için **{len(filtered_df)}** müşteri bulundu.")
            
            # Önizleme tablosu (ilk 10 kayıt)
            preview_df = filtered_df[['User_ID', 'Segment_Name', 'Balance', 'EstimatedSalary', 'Onerilen_Urun', 'Risk_Probability']].head(10)
            st.table(preview_df.style.format({
                    'Balance': '${:,.0f}',
                    'EstimatedSalary': '${:,.0f}',
                    'Risk_Probability': '{:.1%}'
                }))
            
            # CSV indirme - Segment isimlerinden emojileri kaldır
            csv_df = filtered_df[['User_ID', 'Segment_Name', 'Onerilen_Urun', 'Risk_Probability']].copy()
            # Segment_Name sütunundaki emojileri temizle
            csv_df['Segment_Name'] = csv_df['Segment_Name'].str.replace('💎 ', '', regex=False)
            csv_df['Segment_Name'] = csv_df['Segment_Name'].str.replace('🚀 ', '', regex=False)
            csv_df['Segment_Name'] = csv_df['Segment_Name'].str.replace('💰 ', '', regex=False)
            csv_df['Segment_Name'] = csv_df['Segment_Name'].str.replace('⚠️ ', '', regex=False)
            csv_df['Segment_Name'] = csv_df['Segment_Name'].str.replace('🌱 ', '', regex=False)
            csv_df['Segment_Name'] = csv_df['Segment_Name'].str.replace('📊 ', '', regex=False)
            csv_camp = csv_df.to_csv(index=False).encode('utf-8-sig')
            
            # Dosya adını belirle (özel karakterleri temizle)
            if selected_segment == "📋 Tüm Segmentler":
                file_name = "Campaign_All_Segments.csv"
                button_label = "📥 Tüm Segmentler Kampanya Listesini İndir"
            else:
                # Emoji ve özel karakterleri temizle
                clean_name = selected_segment.replace('💎 ', '').replace('🚀 ', '').replace('💰 ', '').replace('⚠️ ', '').replace('🌱 ', '').replace('📊 ', '')
                clean_name = clean_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                file_name = f"Campaign_{clean_name}.csv"
                button_label = f"📥 {selected_segment} Kampanya Listesini İndir"
            
            st.download_button(label=button_label, data=csv_camp,
                               file_name=file_name, mime="text/csv")
            
            # Silhouette Score Hakkında Bilgi
            st.markdown("---")
            with st.expander("ℹ️ Silhouette Score (Silüet Skoru) Hakkında Bilgi", expanded=False):
                st.markdown("""
                ### 📊 Silhouette Score Nedir?
                
                **Silhouette Score (Silüet Skoru)**, K-Means gibi kümeleme (clustering) algoritmalarının performansını değerlendirmek için kullanılan bir metrikdir. 
                Bu skor, her bir veri noktasının kendi kümesine ne kadar iyi uyduğunu ve diğer kümelerden ne kadar iyi ayrıldığını ölçer.
                
                ### 🧮 Nasıl Hesaplanır?
                
                Silhouette Score, her bir veri noktası için aşağıdaki formülle hesaplanır:
                
                ```
                s(i) = (b(i) - a(i)) / max(a(i), b(i))
                ```
                
                Burada:
                - **a(i)**: i. noktanın kendi kümesindeki diğer noktalara olan ortalama uzaklığı (iç küme uzaklığı)
                - **b(i)**: i. noktanın en yakın diğer kümedeki noktalara olan ortalama uzaklığı (dış küme uzaklığı)
                
                Tüm noktalar için hesaplanan skorların ortalaması alınarak genel Silhouette Score elde edilir.
                
                ### 📈 Skor Aralığı ve Yorumlama
                
                Silhouette Score **-1 ile +1 arasında** değerler alır:
                
                - **+1'e yakın (0.7-1.0):** Mükemmel kümeleme
                  - Noktalar kendi kümelerine çok yakın
                  - Kümeler birbirinden çok iyi ayrılmış
                  - Segmentasyon çok başarılı
                
                - **Orta değerler (0.3-0.7):** İyi kümeleme
                  - Kümeler makul şekilde ayrılmış
                  - Bazı noktalar sınırda olabilir
                  - Segmentasyon kullanılabilir
                
                - **Düşük değerler (0.0-0.3):** Zayıf kümeleme
                  - Kümeler birbirine çok yakın
                  - Noktalar hangi kümeye ait olduğundan emin değil
                  - Segmentasyon belirsiz
                
                - **Negatif değerler (-1.0-0.0):** Kötü kümeleme
                  - Noktalar yanlış kümeye atanmış olabilir
                  - Kümeler birbiriyle iç içe geçmiş
                  - Segmentasyon başarısız
                
                ### 🎯 Bu Projedeki Kullanım
                
                Bu projede Silhouette Score:
                - **K-Means clustering** modelinin kalitesini değerlendirmek için kullanılmaktadır
                - **6 segment** oluşturulurken segmentlerin birbirinden ne kadar iyi ayrıldığını gösterir
                - **5 değişken** (Balance, EstimatedSalary, NumOfProducts, Tenure, IsActiveMember) kullanılarak hesaplanmaktadır
                - Skor **0.340** ise, bu segmentlerin orta düzeyde iyi ayrıştığını gösterir
                
                ### 💡 Skorun Anlamı
                
                **0.340 Silhouette Score** değeri:
                - Segmentlerin **makul şekilde ayrıldığını** gösterir
                - Segmentasyon **kullanılabilir** seviyededir
                - Bazı müşteriler segment sınırlarında olabilir, ancak genel olarak segmentler **ayırt edilebilir**
                - İş uygulamaları için **yeterli** bir segmentasyon kalitesidir
                
                ### ⚙️ Hesaplama Detayları
                
                Bu projede Silhouette Score hesaplanırken:
                1. Tüm veri seti **MinMaxScaler** ile normalize edilir
                2. K-Means algoritması **n_init=10000** ile çalıştırılır (en iyi başlangıç noktası seçilir)
                3. **6 cluster** oluşturulur
                4. Hesaplama performansı için **2000 örnek** kullanılır (veri seti büyükse)
                5. Her noktanın kendi kümesine ve diğer kümelerine olan uzaklıkları hesaplanır
                6. Ortalama alınarak final skor elde edilir
                
                ### 📚 Referans
                
                Silhouette Score, **Peter J. Rousseeuw** tarafından 1987 yılında geliştirilmiştir ve kümeleme algoritmalarının 
                en yaygın kullanılan değerlendirme metriklerinden biridir.
                
                ### 🚀 Silhouette Score'u Nasıl Yükseltebiliriz?
                
                **1. n_init Artırmak:**
                - **Etkisi:** Sınırlı - n_init artırmak sadece farklı başlangıç noktalarını dener ve en iyi lokal minimum'u bulur
                - **Mevcut durum:** n_init=10000 ile çalışıyor (maksimum optimizasyon)
                - **Sonuç:** 10000 farklı başlangıç noktası denenerek en iyi sonuç seçilir
                - **Not:** Daha fazla artırmak çok az fark yaratır, hesaplama süresini önemli ölçüde artırır
                
                **2. Özellik Mühendisliği (En Etkili):**
                - **Yeni özellikler eklemek:** Örneğin Balance/EstimatedSalary oranı, Balance_per_Product gibi
                - **Etkileşim özellikleri:** Balance × EstimatedSalary gibi çarpım özellikleri
                - **Kategorik özellikler:** Geography, Gender gibi kategorik değişkenleri eklemek (One-Hot Encoding ile)
                - **Örnek:** `Balance_per_Salary = Balance / EstimatedSalary` gibi yeni özellikler segmentasyonu iyileştirebilir
                
                **3. Normalizasyon Yöntemi Değiştirmek:**
                - **MinMaxScaler (mevcut):** 0-1 arasına ölçekler
                - **StandardScaler:** Ortalama 0, standart sapma 1 yapar (z-score normalization)
                - **RobustScaler:** Outlier'lara daha dayanıklı
                - **Deneme:** Farklı scaler'ları deneyip en iyi sonucu seçmek
                
                **4. Outlier Temizleme:**
                - Aşırı değerli müşterileri (outlier) temizlemek
                - IQR (Interquartile Range) yöntemi ile outlier'ları tespit etmek
                - Segmentasyon kalitesini artırabilir
                
                **5. Özellik Seçimi:**
                - Daha ayırt edici özellikler seçmek
                - Korelasyon analizi yaparak gereksiz özellikleri çıkarmak
                - Feature importance analizi yapmak
                
                **6. Cluster Sayısını Optimize Etmek:**
                - 6 cluster sabit, ancak farklı sayılar deneyebilirsiniz (5, 7, 8)
                - Elbow method veya Silhouette Score grafiği ile optimal sayıyı bulmak
                
                **7. Farklı Algoritmalar:**
                - **DBSCAN:** Gürültüye dayanıklı, farklı şekilli kümeler bulabilir
                - **Hierarchical Clustering:** Hiyerarşik yapı oluşturur
                - **Gaussian Mixture Models (GMM):** Olasılıksal yaklaşım
                
                ### 💡 Pratik Öneriler (Bu Proje İçin)
                
                **Hızlı İyileştirmeler:**
                1. **Yeni özellik ekle:** `Balance_per_Salary = Balance / EstimatedSalary`
                2. **StandardScaler deneyin:** MinMaxScaler yerine
                3. **Outlier temizleme:** Balance > 200,000 veya EstimatedSalary > 80,000 gibi aşırı değerleri filtreleyin
                
                **Orta Vadeli İyileştirmeler:**
                1. **Kategorik özellikler ekle:** Geography, Gender (One-Hot Encoding ile)
                2. **Etkileşim özellikleri:** Balance × NumOfProducts gibi
                3. **PCA (Principal Component Analysis):** Boyut azaltma ve gürültü temizleme
                
                **Uzun Vadeli İyileştirmeler:**
                1. **Farklı algoritmalar:** DBSCAN veya GMM deneyin
                2. **Ensemble yöntemler:** Birden fazla algoritmanın sonuçlarını birleştirin
                3. **Domain knowledge:** İş mantığına göre özel özellikler oluşturun
                
                ### ⚠️ Önemli Notlar
                
                - **0.340 Silhouette Score** zaten kullanılabilir bir değerdir
                - Mükemmel skor (0.7+) genellikle gerçek dünya verilerinde nadirdir
                - İş uygulamaları için 0.3-0.5 arası skorlar genellikle yeterlidir
                - Skorun yükseltilmesi segmentasyon kalitesini artırır, ancak her zaman gerekli değildir
                """)

# =========================================================
# SAYFA 4: PROJE HAKKINDA (DETAYLI DOKÜMANTASYON)
# =========================================================
elif page == "ℹ️ Proje Hakkında":
    st.title("ℹ️ Bankacı Plus: Proje Teknik Raporu")

    st.markdown("""
    **Bankacı Plus**, finansal süreçlerde veriye dayalı karar almayı (Data-Driven Decision Making) sağlayan, yapay zeka tabanlı entegre bir **Karar Destek Sistemidir (DSS)**. 
    Bu platform; Kredi Risk, Müşteri Kayıp (Churn) ve Akıllı Satış (NBA) olmak üzere üç ana bankacılık dikeyini tek çatı altında toplar.
    """)

    st.divider()

    # 1. KREDİ RİSK MODÜLÜ
    st.header("1. 🛡️ Kredi Risk Modülü")
    c1, c2 = st.columns([1, 2])
    c1.info("**Amaç:** Kredi başvurusu yapan müşterinin temerrüde düşme (ödeyememe) riskini hesaplamak.")
    c2.markdown("""
    ### ⚙️ Teknik Detaylar
    
    **Veri Seti:** 
    Lending Club Dataset (2007-2015 arası gerçek P2P kredi verileri). 40,000+ kayıt içeren temizlenmiş ve işlenmiş veri seti.
    
    **Özellik Mühendisliği (Feature Engineering):**
    * `loan_to_income`: Kredi tutarı / Yıllık gelir oranı
    * `installment_to_income`: Aylık taksit / Aylık gelir oranı (PTI - Payment-to-Income)
    * `balance_income_ratio`: Döner kredi bakiyesi / Yıllık gelir oranı
    * Kategorik veriler (Ev Durumu, Amaç, Not) One-Hot Encoding ile sayısallaştırılmıştır.
    
    **Kullanılan Algoritmalar:**
    
    **🚀 Lite Model (Hızlı Analiz):**
    * **Algoritma:** XGBoost Classifier
    * **Değişken Sayısı:** 7 temel + 1 türetilmiş = 8 değişken
    * **Optimize Edilmiş Parametreler:** n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.7, min_child_weight=1
    * **Optimizasyon:** RandomizedSearchCV ile 150 kombinasyon test edildi
    * **Kullanım:** Hızlı karar verme, minimum bilgi gereksinimi
    * **Avantaj:** ⚡ Düşük hesaplama maliyeti, gerçek zamanlı analiz
    
    **📈 Pro Model (Detaylı Analiz):**
    * **Algoritma:** XGBoost Classifier
    * **Değişken Sayısı:** 13 temel + 3 türetilmiş = 16 değişken
    * **Optimize Edilmiş Parametreler:** n_estimators=350, learning_rate=0.03, max_depth=4, subsample=0.75, colsample_bytree=0.75, min_child_weight=2, gamma=0
    * **Optimizasyon:** RandomizedSearchCV ile 100 kombinasyon test edildi
    * **Kullanım:** Büyük tutarlı krediler, detaylı risk analizi
    * **Avantaj:** 🎯 En yüksek doğruluk, kapsamlı değerlendirme
    
    **Doğrulama (Validation):** 
    Her iki model de **RandomizedSearchCV ile 3-Fold Cross Validation** kullanılarak optimize edilmiştir. Bu yöntem, modelin farklı veri alt kümelerinde tutarlı performans göstermesini sağlar ve overfitting'i önler.
    
    **Model Performans Metrikleri (Optimize Edilmiş):**
    
    **🚀 Lite Model Performansı:**
    * **Test Set Accuracy:** %65.29
    * **Test Set ROC-AUC:** %70.31
    * **Optimizasyon Öncesi ROC-AUC:** %70.50
    * **Optimizasyon Sonrası:** Accuracy +0.99%, ROC-AUC -0.19% (yakın performans)
    
    **📈 Pro Model Performansı:**
    * **Test Set Accuracy:** %65.71
    * **Test Set ROC-AUC:** %71.24
    * **Optimizasyon Öncesi ROC-AUC:** %71.01
    * **Optimizasyon Sonrası:** Accuracy +0.73%, ROC-AUC +0.22%
    
    **📊 Genel Performans Değerlendirmesi:**
    * **Accuracy (%65-66):** Orta seviye - İyileştirilebilir (Hedef: %80+)
    * **ROC-AUC (%70-71):** Kabul edilebilir - Model rastgele tahminden daha iyi (Hedef: %80+)
    * **Kararlılık:** Yüksek - Optimize edilmiş parametreler ile tutarlı sonuçlar
    * **Sonuç:** Modeller optimize edilmiş durumda ve kullanılabilir seviyede
    
    **✅ Yapılan Optimizasyonlar:**
    * ✅ RandomizedSearchCV ile hiperparametre optimizasyonu tamamlandı
    * ✅ Accuracy ve ROC-AUC skorları optimize edildi
    * ✅ En iyi parametre kombinasyonları belirlendi ve modeller güncellendi
    
    **Metrik Açıklamaları:**
    * **Accuracy:** Doğru tahmin oranı (Rastgele: %50, İyi: %80+)
    * **ROC-AUC:** Modelin riskli/risksiz ayırt etme yeteneği (0.5 = rastgele, 0.7-0.8 = kabul edilebilir, 0.8+ = iyi, 0.9+ = mükemmel)
    * **Precision:** Riskli tahmin edilenlerin gerçekten riskli olma oranı
    * **Recall:** Gerçek riskli müşterilerin yakalanma oranı
    * **F1-Score:** Precision ve Recall'un harmonik ortalaması
    """)

    st.divider()

    # 2. CHURN ANALİZİ
    st.header("2. 📉 Müşteri Kayıp (Churn) Önleme Modülü")
    c1, c2 = st.columns([1, 2])
    c1.info("**Amaç:** Bankayı terk etme eğiliminde olan müşterileri tespit edip elde tutmak.")
    c2.markdown("""
    ### ⚙️ Teknik Detaylar
    
    **Veri Seti:** 
    Bank Customer Churn Modeling (10,000 gözlem). Müşteri demografik bilgileri, finansal durum ve bankacılık ürün kullanım verilerini içerir.
    
    **Keşfedilen İçgörü:** 
    "Ürün Paradoksu" - Ürün sayısı 3 ve 4 olan müşterilerin churn oranı %80 üzerindedir. Bu durum aşırı borçlanma belirtisi olarak yorumlanmıştır.
    
    **Kullanılan Model:**
    * **Algoritma:** LightGBM (Light Gradient Boosting Machine) Classifier
    * **Model Seçimi:** XGBoost, LightGBM ve CatBoost modelleri 50 katlı cross-validation ile karşılaştırılmıştır
    * **Seçim Gerekçesi:** LightGBM en hızlı eğitim süresi ve yüksek performans kombinasyonu ile seçilmiştir
    
    **Model Parametreleri:**
    * `n_estimators`: 100 (Ağaç sayısı)
    * `learning_rate`: 0.1 (Öğrenme hızı)
    * `max_depth`: 5 (Ağaç derinliği)
    * `subsample`: 0.8 (Alt örnekleme oranı)
    * `colsample_bytree`: 0.8 (Sütun alt örnekleme)
    * `boosting_type`: gbdt (Gradient Boosting Decision Tree)
    
    **Özellik Mühendisliği (Feature Engineering):**
    * `Balance_per_Product`: Ürün Başına Bakiye
    * `Age_Group`: Yaş Grubu (Young, Adult, Middle, Senior)
    * `Credit_Score_Age_Ratio`: Kredi Skoru/Yaş Oranı
    * `Is_High_Value_Active`: Yüksek Değerli Aktif Müşteri
    
    **Preprocessing:**
    * Sayısal değişkenler: **StandardScaler** ile ölçeklendirilmiştir
    * Kategorik değişkenler: **OneHotEncoder** ile kodlanmıştır
    * Pipeline yapısı ile ön işleme ve model eğitimi birleştirilmiştir
    
    **Eğitim ve Doğrulama:**
    * **50-Fold Stratified Cross Validation** ile eğitilmiştir
    * **ROC-AUC** ve **Accuracy** skorları ile değerlendirilmiştir
    * **Stratified Train-Test Split** (80-20) kullanılmıştır
    * Model kararlılığı ve güvenilirliği test edilmiştir
    
    **Model Performans Metrikleri (50-Fold CV):**
    * **ROC-AUC Ortalama:** %86.27 (Std: 3.51%)
    * **Accuracy Ortalama:** %86.39 (Std: 1.79%)
    * **Test Seti ROC-AUC:** %87.42
    * **Test Seti Accuracy:** %86.20
    
    **Avantajlar:**
    * ⚡ XGBoost'a göre daha hızlı eğitim süresi
    * 📊 Yüksek performans ve hız kombinasyonu
    * 💾 Daha az bellek kullanımı
    * 🎯 Müşteri churn riskini 0-1 arası olasılık olarak tahmin eder
    
    **Strateji Algoritması (Rule-Engine):**
    * `Risk > %60 + Bakiye > 50K` ➔ **VIP Müdahale**
    * `Risk > %60 + Ürün >= 3` ➔ **Sadeleştirme**
    * `Risk > %60` ➔ **Arama**
    * `Risk %40-%60 + Pasif Üye` ➔ **Uyandırma Kampanyası**
    * `Risk %40-%60 + Yaş < 35` ➔ **Lifestyle Hediye**
    * `Risk %40-%60` ➔ **Teşvik**
    * `Risk <= %40 + Bakiye > 100K` ➔ **Yatırım Çapraz Satış**
    * `Risk <= %40` ➔ **İlişki Yönetimi**
    
    **Çıktı:** 
    İndirilebilir CSV formatında aksiyon listesi ve kişiselleştirilmiş strateji önerileri.
    """)

    st.divider()

    # 3. AKILLI SATIŞ (NBA)
    st.header("3. 🎯 Fırsatlar & Akıllı Satış (Next Best Action - K-Means)")
    c1, c2 = st.columns([1, 2])
    c1.info("**Amaç:** Müşteriye doğru zamanda doğru ürünü satmak.")
    c2.markdown("""
    ### ⚙️ Teknik Detaylar (Hibrit Yapı)
    
    **Veri Seti:**
    Bank Customer Churn veri seti üzerinden segmentasyon yapılmaktadır. 5 temel özellik kullanılmaktadır:
    * Balance (Hesap Bakiyesi)
    * EstimatedSalary (Tahmini Maaş)
    * NumOfProducts (Ürün Sayısı)
    * Tenure (Müşteri Süresi)
    * IsActiveMember (Aktif Üyelik Durumu)
    
    **Adım 1: Kümeleme (Unsupervised Learning):**
    * **Algoritma:** K-Means Clustering
    * **Cluster Sayısı:** 6 farklı mikro-segment
    * **Normalizasyon:** MinMaxScaler ile 0-1 arasına ölçeklendirme
    * **Başlangıç Noktaları:** n_init=10000 (en iyi lokal minimum'u bulmak için)
    * **Segment İsimleri:** 
        * Genç Üniversiteli
        * Beyaz Yakalı
        * Orta Yaş Profesyonel
        * Emekli
        * Yüksek Gelirli
        * Standart Müşteri
    
    **Model Doğrulama:**
    * **Silhouette Score:** 0.340 (Kullanılabilir seviye)
    * Segmentler istatistiksel olarak iyi ayrışmış durumda
    
    **Adım 2: Kural Tabanlı Öneri (Rule-Based System):**
    * Yapay zeka segmenti bulur, iş kuralları ürünü önerir
    * **Örnek Kurallar:**
        * `Eğer (Maaş > 50K) VE (Yaş < 55) VE (BES Yok) ➔ ÖNER: BES`
        * `Eğer (Bakiye > 100K) VE (Yatırım Yok) ➔ ÖNER: Yatırım Hesabı`
        * `Eğer (Kredi Yok) VE (Gelir Yeterli) ➔ ÖNER: Kredi Ürünleri`
    
    **Görselleştirme:**
    * Plotly Scatterpolar (Radar Grafik) ile müşteri vs segment ortalaması karşılaştırması
    * Finansal DNA radar grafiği (5 değişkene göre)
    * Segment bazlı kampanya yönetimi dashboard'u
    
    **Kullanım Senaryoları:**
    * Bireysel müşteri analizi ve ürün önerileri
    * Segment bazlı toplu kampanya yönetimi
    * Kişiselleştirilmiş satış stratejileri geliştirme
    """)

    st.divider()

    # 4. TEKNOLOJİ ALTYAPISI
    st.header("4. 🛠️ Teknoloji Altyapısı")
    col1, col2, col3, col4 = st.columns(4)
    col1.success("**Frontend:**\nStreamlit")
    col2.success("**Veri İşleme:**\nPandas, NumPy")
    col3.success("**Machine Learning:**\nScikit-learn, XGBoost, LightGBM")
    col4.success("**Görselleştirme:**\nPlotly Express")
    
    st.markdown("""
    ### 📚 Kullanılan Kütüphaneler
    
    **Veri İşleme:**
    * `pandas` - Veri manipülasyonu ve analizi
    * `numpy` - Sayısal hesaplamalar
    
    **Machine Learning:**
    * `scikit-learn` - Preprocessing, model eğitimi ve değerlendirme
    * `xgboost` - Gradient boosting (Kredi Risk modelleri)
    * `lightgbm` - Light gradient boosting (Churn modeli)
    * `joblib` - Model serialization ve yükleme
    
    **Görselleştirme:**
    * `plotly.express` - İnteraktif grafikler ve görselleştirmeler
    * `plotly.graph_objects` - Gelişmiş grafik özellikleri
    
    **Clustering:**
    * `sklearn.cluster.KMeans` - K-Means kümeleme algoritması
    * `sklearn.metrics.silhouette_score` - Kümeleme kalite metriği
    
    **Frontend Framework:**
    * `streamlit` - Web uygulaması framework'ü
    """)

    st.divider()
    st.caption("© 2025 Bankacı Plus | Developed for FinTech Innovation")


