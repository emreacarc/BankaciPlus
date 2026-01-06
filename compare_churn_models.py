"""
Churn Tahmin Modelleri Karşılaştırması
XGBoost, LightGBM ve CatBoost modellerini 20 katlı CV ile test eder
"""

import pandas as pd
import numpy as np
import time
import sys
from datetime import datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')
# LightGBM feature names uyarısını bastır
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

# Output'u hem konsola hem dosyaya yazdır
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Log dosyası aç
log_file = open('model_comparison_log.txt', 'w', encoding='utf-8')
sys.stdout = TeeOutput(sys.stdout, log_file)

# --- AYARLAR ---
INPUT_FILE = 'datasets/bank_customer_churn_data/Customer-Churn-Records.csv'
CV_FOLDS = 50  # 50 katlı CV
RANDOM_STATE = 42

print("=" * 80)
print("CHURN TAHMIN MODELLERI KARSILASTIRMASI")
print(f"{CV_FOLDS} Katli Cross-Validation Testi")
print("=" * 80)
print(f"Baslangic Zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("NOT: Ilerleme raporlari konsolda gorunecektir.")
print("     Eger gormuyorsaniz, scripti kendi terminalinizde calistirin.")
print()

# --- 1. VERI YUKLEME VE OZELLIK MUHENDISLIGI ---
print(">>> [1/5] Veri yukleniyor ve ozellik muhendisligi yapiliyor...")
df = pd.read_csv(INPUT_FILE)
print(f"   Toplam kayit: {len(df)}")

# Ozellik muhendisligi
df['Balance_per_Product'] = df['Balance'] / (df['NumOfProducts'] + 0.1)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
df['Credit_Score_Age_Ratio'] = df['CreditScore'] / (df['Age'] + 1)
df['Is_High_Value_Active'] = ((df['IsActiveMember'] == 1) & (df['Balance'] > df['Balance'].mean())).astype(int)

# Hedef ve Ozellikler
X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']

print(f"   Ozellik sayisi: {X.shape[1]}")
print(f"   Hedef degisken dagilimi: {y.value_counts().to_dict()}")

# --- 2. PREPROCESSING ---
print("\n>>> [2/5] Preprocessing yapiliyor...")
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Balance_per_Product', 'Credit_Score_Age_Ratio', 'Is_High_Value_Active']

cat_cols = ['Geography', 'Gender', 'Age_Group']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# --- 3. MODELLERI TANIMLA ---
print("\n>>> [3/5] Modeller tanimlaniyor...")

# XGBoost (Mevcut Model) - Optimize edilmiş parametreler
xgb_model = XGBClassifier(
    n_estimators=100,  # 250'den 100'e düşürüldü (hız için)
    learning_rate=0.1,  # 0.05'ten 0.1'e çıkarıldı (daha az iterasyon ile aynı sonuç)
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,  # Tüm CPU'ları kullan
    verbosity=0,
    tree_method='hist'  # Daha hızlı histogram yöntemi
)

# LightGBM - Optimize edilmiş
lgbm_model = LGBMClassifier(
    n_estimators=100,  # 250'den 100'e düşürüldü
    learning_rate=0.1,  # 0.05'ten 0.1'e çıkarıldı
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,  # Tüm CPU'ları kullan
    verbosity=-1,
    boosting_type='gbdt'  # Gradient Boosting Decision Tree (varsayılan, hızlı)
)

# CatBoost - Optimize edilmiş
catboost_model = CatBoostClassifier(
    iterations=100,  # 250'den 100'e düşürüldü
    learning_rate=0.1,  # 0.05'ten 0.1'e çıkarıldı
    depth=5,
    subsample=0.8,
    colsample_bylevel=0.8,
    random_state=RANDOM_STATE,
    verbose=False,
    thread_count=-1,  # Tüm CPU'ları kullan
    task_type='CPU'  # CPU kullanımını optimize et
)

# Pipeline'lar oluştur
# Not: CatBoost için kategorik değişkenleri ayrı işlemek daha iyi olabilir
# Ama karşılaştırma için aynı preprocessing kullanıyoruz

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

lgbm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgbm_model)
])

# CatBoost için aynı preprocessing (adil karşılaştırma için)
catboost_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', catboost_model)
])

models = {
    'XGBoost (Mevcut)': xgb_pipeline,
    'LightGBM': lgbm_pipeline,
    'CatBoost': catboost_pipeline
}

# --- 4. 20 KATLI CROSS-VALIDATION ---
print(f"\n>>> [4/5] {CV_FOLDS} katli Cross-Validation baslatiliyor...")
print("   Her fold'da ilerleme raporu gosterilecek...")
print()

# Stratified K-Fold (sınıf dağılımını korur)
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

results = {}

# İlerleme takibi için custom CV fonksiyonu - Hem ROC-AUC hem Accuracy
def cross_val_score_with_progress(estimator, X, y, cv, model_name):
    """İlerleme takibi ile cross validation - ROC-AUC ve Accuracy hesaplar"""
    auc_scores = []
    acc_scores = []
    fold_times = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        fold_start = time.time()
        
        # Train ve validation split
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Model eğitimi (sadece bir kez!)
        estimator.fit(X_train_fold, y_train_fold)
        
        # Hem ROC-AUC hem Accuracy tahminleri
        y_pred_proba = estimator.predict_proba(X_val_fold)[:, 1]
        y_pred = estimator.predict(X_val_fold)
        
        auc_score = roc_auc_score(y_val_fold, y_pred_proba)
        acc_score = accuracy_score(y_val_fold, y_pred)
        
        auc_scores.append(auc_score)
        acc_scores.append(acc_score)
        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        
        # İlerleme raporu
        progress = (fold_idx / CV_FOLDS) * 100
        avg_time = np.mean(fold_times)
        remaining_folds = CV_FOLDS - fold_idx
        estimated_time_left = avg_time * remaining_folds
        
        # İlerleme raporu - flush ile anında göster
        progress_msg = (f"      Fold {fold_idx}/{CV_FOLDS} tamamlandi | ROC-AUC: {auc_score:.4f} | "
                       f"Accuracy: {acc_score:.4f} | Ilerleme: %{progress:.1f} | Kalan: ~{estimated_time_left:.1f} saniye")
        print(progress_msg, flush=True)
    
    return np.array(auc_scores), np.array(acc_scores)

for idx, (model_name, model_pipeline) in enumerate(models.items(), 1):
    print(f"\n   [{idx}/{len(models)}] {model_name} test ediliyor...")
    print(f"   Toplam {CV_FOLDS} fold islenecek...")
    start_time = time.time()
    
    # Hem ROC-AUC hem Accuracy skorları (tek CV döngüsünde)
    print(f"\n      >>> ROC-AUC ve Accuracy skorlari hesaplaniyor...")
    cv_auc_scores, cv_acc_scores = cross_val_score_with_progress(
        model_pipeline, X, y, 
        cv=skf, 
        model_name=model_name
    )
    
    elapsed_time = time.time() - start_time
    
    results[model_name] = {
        'roc_auc_mean': cv_auc_scores.mean(),
        'roc_auc_std': cv_auc_scores.std(),
        'roc_auc_scores': cv_auc_scores,
        'accuracy_mean': cv_acc_scores.mean(),
        'accuracy_std': cv_acc_scores.std(),
        'accuracy_scores': cv_acc_scores,
        'time': elapsed_time
    }
    
    print(f"      Tamamlandi! (Sure: {elapsed_time:.2f} saniye)")

# --- 5. RAPOR OLUSTUR ---
print("\n>>> [5/5] Rapor olusturuluyor...")
print()
print("=" * 80)
print(f"SONUCLAR - {CV_FOLDS} KATLI CROSS-VALIDATION")
print("=" * 80)
print()

# ROC-AUC Sonuçları
print("ROC-AUC SKORLARI:")
print("-" * 80)
print(f"{'Model':<25} {'Ortalama':<15} {'Std Sapma':<15} {'Min':<15} {'Max':<15}")
print("-" * 80)

for model_name in models.keys():
    r = results[model_name]
    print(f"{model_name:<25} {r['roc_auc_mean']*100:>6.2f}%      {r['roc_auc_std']*100:>6.2f}%      "
          f"{r['roc_auc_scores'].min()*100:>6.2f}%      {r['roc_auc_scores'].max()*100:>6.2f}%")

print()

# Accuracy Sonuçları
print("ACCURACY SKORLARI:")
print("-" * 80)
print(f"{'Model':<25} {'Ortalama':<15} {'Std Sapma':<15} {'Min':<15} {'Max':<15}")
print("-" * 80)

for model_name in models.keys():
    r = results[model_name]
    print(f"{model_name:<25} {r['accuracy_mean']*100:>6.2f}%      {r['accuracy_std']*100:>6.2f}%      "
          f"{r['accuracy_scores'].min()*100:>6.2f}%      {r['accuracy_scores'].max()*100:>6.2f}%")

print()

# Performans Karşılaştırması
print("=" * 80)
print("PERFORMANS KARSILASTIRMASI")
print("=" * 80)
print()

# En iyi ROC-AUC
best_auc_model = max(results.keys(), key=lambda x: results[x]['roc_auc_mean'])
best_auc_score = results[best_auc_model]['roc_auc_mean']
xgb_auc = results['XGBoost (Mevcut)']['roc_auc_mean']

print(f"En Iyi ROC-AUC: {best_auc_model} ({best_auc_score*100:.2f}%)")
if best_auc_model != 'XGBoost (Mevcut)':
    improvement = (best_auc_score - xgb_auc) * 100
    print(f"   XGBoost'a gore iyilestirme: +{improvement:.2f}%")
else:
    print("   XGBoost en iyi performansi gosterdi!")

print()

# En iyi Accuracy
best_acc_model = max(results.keys(), key=lambda x: results[x]['accuracy_mean'])
best_acc_score = results[best_acc_model]['accuracy_mean']
xgb_acc = results['XGBoost (Mevcut)']['accuracy_mean']

print(f"En Iyi Accuracy: {best_acc_model} ({best_acc_score*100:.2f}%)")
if best_acc_model != 'XGBoost (Mevcut)':
    improvement = (best_acc_score - xgb_acc) * 100
    print(f"   XGBoost'a gore iyilestirme: +{improvement:.2f}%")
else:
    print("   XGBoost en iyi performansi gosterdi!")

print()

# Hız Karşılaştırması
print("=" * 80)
print("HIZ KARSILASTIRMASI")
print("=" * 80)
print(f"{'Model':<25} {'Sure (saniye)':<15}")
print("-" * 80)
for model_name in models.keys():
    print(f"{model_name:<25} {results[model_name]['time']:>6.2f}")
print()

# Detaylı İstatistikler
print("=" * 80)
print("DETAYLI ISTATISTIKLER")
print("=" * 80)
print()

for model_name in models.keys():
    r = results[model_name]
    print(f"\n{model_name}:")
    print(f"  ROC-AUC:")
    print(f"    Ortalama: {r['roc_auc_mean']*100:.4f}%")
    print(f"    Std Sapma: {r['roc_auc_std']*100:.4f}%")
    print(f"    Medyan: {np.median(r['roc_auc_scores'])*100:.4f}%")
    print(f"    Q1 (25%): {np.percentile(r['roc_auc_scores'], 25)*100:.4f}%")
    print(f"    Q3 (75%): {np.percentile(r['roc_auc_scores'], 75)*100:.4f}%")
    print(f"  Accuracy:")
    print(f"    Ortalama: {r['accuracy_mean']*100:.4f}%")
    print(f"    Std Sapma: {r['accuracy_std']*100:.4f}%")
    print(f"    Medyan: {np.median(r['accuracy_scores'])*100:.4f}%")

print()
print("=" * 80)
print(f"Bitis Zamani: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print(f"\nDetayli log dosyasi: model_comparison_log.txt")
log_file.close()
sys.stdout = sys.__stdout__

