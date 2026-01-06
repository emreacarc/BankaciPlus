"""
NBA (Next Best Action) K-Means Clustering Model Eğitim Scripti
K-Means Clustering - 6 küme
Müşteri segmentasyonu için optimize edilmiş model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("NBA K-MEANS CLUSTERING MODEL EĞİTİMİ")
print("K-Means Clustering - 6 Küme")
print("=" * 80)
print()

# --- 1. VERİ YÜKLEME ---
print(">>> [1/5] Veri yükleniyor...")
INPUT_FILE = 'datasets/bank_customer_churn_data/Customer-Churn-Records.csv'
df = pd.read_csv(INPUT_FILE)
print(f"   Toplam kayıt: {len(df)}")
print(f"   Sütun sayısı: {len(df.columns)}")
print()

# --- 2. ÖZELLİK SEÇİMİ ---
print(">>> [2/5] Özellik seçimi yapılıyor...")

# K-Means için seçilen özellikler
features = ['Balance', 'EstimatedSalary', 'NumOfProducts', 'Tenure', 'IsActiveMember']

X = df[features].copy()

print(f"   Seçilen özellikler ({len(features)}):")
for i, feat in enumerate(features, 1):
    print(f"     {i}. {feat}")
print()

# Eksik değer kontrolü
print("   Eksik değer kontrolü:")
missing_count = X.isnull().sum().sum()
if missing_count > 0:
    print(f"     Toplam eksik değer: {missing_count}")
    for col in features:
        missing = X[col].isnull().sum()
        if missing > 0:
            print(f"       {col}: {missing} eksik değer")
            X[col].fillna(X[col].median(), inplace=True)
    print("     Eksik değerler medyan ile dolduruldu")
else:
    print("     Eksik değer yok")
print()

# --- 3. VERİ ÖN İŞLEME (NORMALİZASYON) ---
print(">>> [3/5] Veri normalizasyonu yapılıyor...")

# MinMaxScaler ile 0-1 arasına ölçeklendirme
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"   MinMaxScaler uygulandı")
print(f"   Özellikler 0-1 arasına ölçeklendirildi")
print(f"   İşlenmiş veri boyutu: {X_scaled.shape}")
print()

# Özellik istatistikleri (normalize edilmiş)
print("   Normalize edilmiş özellik istatistikleri:")
for i, feat in enumerate(features):
    print(f"     {feat}:")
    print(f"       Min: {X_scaled[:, i].min():.4f}")
    print(f"       Max: {X_scaled[:, i].max():.4f}")
    print(f"       Mean: {X_scaled[:, i].mean():.4f}")
    print(f"       Std: {X_scaled[:, i].std():.4f}")
print()

# --- 4. K-MEANS CLUSTERING ---
print(">>> [4/5] K-Means clustering yapılıyor...")

# K-Means parametreleri
N_CLUSTERS = 6
N_INIT = 10000  # 10,000 farklı başlangıç noktası
MAX_ITER = 300

print(f"   Parametreler:")
print(f"     n_clusters: {N_CLUSTERS}")
print(f"     n_init: {N_INIT} (10,000 farklı başlangıç noktası)")
print(f"     max_iter: {MAX_ITER}")
print()
print("   Model eğitiliyor (bu işlem biraz zaman alabilir)...")

# K-Means modeli
kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    n_init=N_INIT,
    max_iter=MAX_ITER,
    n_jobs=-1
)

# Model eğitimi
kmeans.fit(X_scaled)

# Küme etiketleri
cluster_labels = kmeans.predict(X_scaled)
df['Cluster_Label'] = cluster_labels

print("   Eğitim tamamlandı!")
print()

# --- 5. MODEL DOĞRULAMA (SILHOUETTE SCORE) ---
print(">>> [5/5] Model doğrulama (Silhouette Score)...")

# Silhouette Score hesaplama (örneklem boyutu sınırlı)
sample_size = min(2000, len(X_scaled))
sil_score = silhouette_score(X_scaled[:sample_size], cluster_labels[:sample_size])

print(f"   Silhouette Score: {sil_score:.4f}")
print(f"   Yorumlama:")
if sil_score >= 0.5:
    print("     Mükemmel: Kümeler çok iyi ayrılmış")
elif sil_score >= 0.3:
    print("     İyi: Kümeler kabul edilebilir şekilde ayrılmış")
elif sil_score >= 0.1:
    print("     Orta: Kümeler birbirine yakın")
else:
    print("     Zayıf: Kümeler çok yakın veya kötü ayrılmış")
print()

# --- 6. SEGMENT ANALİZİ ---
print("=" * 80)
print("SEGMENT ANALİZİ")
print("=" * 80)
print()

# Her küme için istatistikler
centroids = kmeans.cluster_centers_
segment_stats = {}

for i in range(N_CLUSTERS):
    cluster_data = df[df['Cluster_Label'] == i]
    segment_stats[i] = {
        'size': len(cluster_data),
        'avg_balance': cluster_data['Balance'].mean(),
        'avg_salary': cluster_data['EstimatedSalary'].mean(),
        'avg_products': cluster_data['NumOfProducts'].mean(),
        'avg_tenure': cluster_data['Tenure'].mean(),
        'avg_active': cluster_data['IsActiveMember'].mean(),
        'centroid': centroids[i]
    }

# Segment isimlendirme (centroid analizine göre)
cluster_scores = []
for i in range(N_CLUSTERS):
    stats = segment_stats[i]
    center = centroids[i]
    total_score = (center[0] * 0.3 + center[1] * 0.3 + center[2] * 0.2 + 
                  center[3] * 0.1 + center[4] * 0.1)
    cluster_scores.append((i, total_score, stats))

cluster_scores.sort(key=lambda x: x[1], reverse=True)

segment_templates = [
    "💎 Elit / Servet Yönetimi",
    "🚀 Dinamik / Aktif Müşteri", 
    "💰 Güvenli / Birikimci",
    "⚠️ Riskli / Pasif Müşteri",
    "🌱 Temel Mevduat / Giriş",
    "📊 Standart Bankacılık"
]

cluster_names = {}
for rank, (cluster_id, total_score, stats) in enumerate(cluster_scores):
    cluster_names[cluster_id] = segment_templates[rank]

df['Segment_Name'] = df['Cluster_Label'].map(cluster_names)

# Segment istatistiklerini yazdır
print("Segment İstatistikleri:")
print("-" * 80)
for i in range(N_CLUSTERS):
    stats = segment_stats[i]
    name = cluster_names[i]
    print(f"\n{i}. {name}")
    print(f"   Üye Sayısı: {stats['size']} ({(stats['size']/len(df)*100):.2f}%)")
    print(f"   Ortalama Bakiye: ${stats['avg_balance']:,.2f}")
    print(f"   Ortalama Maaş: ${stats['avg_salary']:,.2f}")
    print(f"   Ortalama Ürün Sayısı: {stats['avg_products']:.2f}")
    print(f"   Ortalama Tenure: {stats['avg_tenure']:.2f} yıl")
    print(f"   Aktif Üye Oranı: {stats['avg_active']:.2%}")
print()

# --- 7. MODEL KAYDETME ---
print("=" * 80)
print("MODEL KAYDEDİLİYOR...")
print("=" * 80)

# Model package
model_package = {
    'kmeans': kmeans,
    'scaler': scaler,
    'features': features,
    'cluster_names': cluster_names,
    'silhouette_score': sil_score,
    'n_clusters': N_CLUSTERS,
    'segment_stats': segment_stats
}

output_file = 'kmeans_model.pkl'
joblib.dump(model_package, output_file)
print(f"Model kaydedildi: {output_file}")
print()

# Scaler'ı ayrı kaydet (app.py uyumluluğu için)
scaler_file = 'scaler_model.pkl'
joblib.dump(scaler, scaler_file)
print(f"Scaler kaydedildi: {scaler_file}")
print()

# İşlenmiş veriyi kaydet (opsiyonel)
processed_file = 'churn_processed_with_clusters.csv'
df.to_csv(processed_file, index=False)
print(f"İşlenmiş veri kaydedildi: {processed_file}")
print()

print("=" * 80)
print("EĞİTİM TAMAMLANDI!")
print("=" * 80)

