"""
Credit Risk Pro Model Eğitim Scripti
XGBoost Classifier - 16 değişken (13 temel + 3 türetilmiş)
Detaylı risk analizi için optimize edilmiş model
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREDIT RISK PRO MODEL EĞİTİMİ")
print("=" * 80)
print()

# --- 1. VERİ YÜKLEME ---
print(">>> [1/6] Veri yükleniyor...")
INPUT_FILE = 'lending_club_cleaned.csv'
df = pd.read_csv(INPUT_FILE)
print(f"   Toplam kayıt: {len(df)}")
print(f"   Sütun sayısı: {len(df.columns)}")
print()

# --- 2. ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING) ---
print(">>> [2/6] Özellik mühendisliği yapılıyor...")

# Türetilmiş özellikler
df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
df['installment_to_income'] = df['installment'] / ((df['annual_inc'] / 12) + 1)  # PTI - Payment-to-Income
df['balance_income_ratio'] = df['revol_bal'] / (df['annual_inc'] + 1)

# Pro Model için 16 değişken seçimi (13 temel + 3 türetilmiş)
pro_features = [
    'loan_amnt',              # Kredi tutarı
    'annual_inc',             # Yıllık gelir
    'installment',            # Aylık taksit
    'int_rate',              # Faiz oranı
    'dti',                   # Debt-to-Income ratio
    'fico_range_low',        # FICO skoru (düşük)
    'fico_range_high',       # FICO skoru (yüksek)
    'revol_bal',             # Döner kredi bakiyesi
    'revol_util',            # Döner kredi kullanım oranı
    'total_acc',             # Toplam hesap sayısı
    'open_acc',              # Açık hesap sayısı
    'pub_rec',               # Kamu kayıtları
    'inq_last_6mths',       # Son 6 ayda sorgulama sayısı
    'loan_to_income',        # Türetilmiş: Kredi tutarı / Yıllık gelir
    'installment_to_income', # Türetilmiş: Aylık taksit / Aylık gelir (PTI)
    'balance_income_ratio'  # Türetilmiş: Döner kredi bakiyesi / Yıllık gelir
]

# Kategorik değişkenler (One-Hot Encoding için)
categorical_features = ['home_ownership', 'purpose', 'grade', 'emp_length', 'verification_status']

# Hedef değişken
target = 'loan_status_binary'  # 0: Ödendi, 1: Temerrüt

# Veri hazırlama
X = df[pro_features + categorical_features].copy()
y = df[target].copy()

print(f"   Seçilen özellik sayısı: {len(pro_features)} temel + {len(categorical_features)} kategorik")
print(f"   Türetilmiş özellikler:")
print(f"     - loan_to_income: Kredi tutarı / Yıllık gelir")
print(f"     - installment_to_income: Aylık taksit / Aylık gelir (PTI)")
print(f"     - balance_income_ratio: Döner kredi bakiyesi / Yıllık gelir")
print(f"   Hedef değişken dağılımı:")
print(f"     0 (Ödendi): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
print(f"     1 (Temerrüt): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
print()

# --- 3. VERİ ÖN İŞLEME (PREPROCESSING) ---
print(">>> [3/6] Veri ön işleme yapılıyor...")

# Eksik değer kontrolü
print(f"   Eksik değer kontrolü:")
for col in X.columns:
    missing = X[col].isnull().sum()
    if missing > 0:
        print(f"     {col}: {missing} eksik değer")

# Eksik değerleri doldur (sayısal için medyan, kategorik için mod)
for col in pro_features:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

for col in categorical_features:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].mode()[0], inplace=True)

# One-Hot Encoding için ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', pro_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop'
)

# Preprocessing uygula
X_processed = preprocessor.fit_transform(X)
feature_names = pro_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

print(f"   İşlenmiş özellik sayısı: {X_processed.shape[1]}")
print(f"   One-Hot Encoding uygulandı")
print()

# --- 4. VERİ BÖLME (TRAIN-TEST SPLIT) ---
print(">>> [4/6] Veri train-test split yapılıyor...")
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, 
    test_size=0.2, 
    stratify=y
)

print(f"   Train set: {X_train.shape[0]} kayıt")
print(f"   Test set: {X_test.shape[0]} kayıt")
print()

# --- 5. MODEL EĞİTİMİ VE OPTİMİZASYON ---
print(">>> [5/6] Model eğitimi ve hiperparametre optimizasyonu...")

# Base model
base_model = XGBClassifier(
    eval_metric='logloss',
    use_label_encoder=False,
    n_jobs=-1,
    verbosity=0
)

# Hiperparametre arama uzayı
param_distributions = {
    'n_estimators': [200, 300, 350, 400, 500],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.75, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.75, 0.8],
    'min_child_weight': [1, 2, 3],
    'gamma': [0, 0.1, 0.2]
}

# RandomizedSearchCV ile optimizasyon
print("   RandomizedSearchCV ile 100 kombinasyon test ediliyor (3-fold CV)...")
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=100,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"   En iyi parametreler:")
best_params = random_search.best_params_
for param, value in best_params.items():
    print(f"     {param}: {value}")
print(f"   En iyi CV skoru (ROC-AUC): {random_search.best_score_:.4f}")
print()

# Final model (en iyi parametrelerle)
final_model = XGBClassifier(**best_params, eval_metric='logloss', use_label_encoder=False, n_jobs=-1, verbosity=0)
final_model.fit(X_train, y_train)

# --- 6. MODEL DEĞERLENDİRME ---
print(">>> [6/6] Model değerlendirme...")

# Test set tahminleri
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

# Metrikler
test_accuracy = accuracy_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print()
print("=" * 80)
print("MODEL PERFORMANS METRİKLERİ (TEST SET)")
print("=" * 80)
print(f"Accuracy:        {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"ROC-AUC:         {test_roc_auc:.4f} ({test_roc_auc*100:.2f}%)")
print(f"Precision:       {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"Recall:          {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"F1-Score:        {test_f1:.4f} ({test_f1*100:.2f}%)")
print()

# Cross-validation skorları
print("Cross-Validation Skorları (5-fold):")
cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"   ROC-AUC Ortalama: {cv_scores.mean():.4f} (Std: {cv_scores.std():.4f})")
print()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ödendi', 'Temerrüt']))
print()

# --- 7. MODEL KAYDETME ---
print("=" * 80)
print("MODEL KAYDEDİLİYOR...")
print("=" * 80)

# Model ve preprocessor'ı birlikte kaydet
model_package = {
    'model': final_model,
    'preprocessor': preprocessor,
    'feature_names': feature_names,
    'best_params': best_params,
    'test_accuracy': test_accuracy,
    'test_roc_auc': test_roc_auc
}

output_file = 'credit_risk_model_20fold.pkl'
joblib.dump(model_package, output_file)
print(f"Model kaydedildi: {output_file}")
print()

print("=" * 80)
print("EĞİTİM TAMAMLANDI!")
print("=" * 80)

