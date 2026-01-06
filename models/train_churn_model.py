"""
Churn Prediction Model Eğitim Scripti
LightGBM Classifier - 50-Fold Stratified Cross Validation
Müşteri kayıp (churn) tahmini için optimize edilmiş model
"""

import pandas as pd
import numpy as np
import joblib
import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CHURN PREDICTION MODEL EĞİTİMİ")
print("LightGBM Classifier - 50-Fold Stratified Cross Validation")
print("=" * 80)
print()

# --- 1. VERİ YÜKLEME ---
print(">>> [1/6] Veri yükleniyor...")
INPUT_FILE = 'datasets/bank_customer_churn_data/Customer-Churn-Records.csv'
df = pd.read_csv(INPUT_FILE)
print(f"   Toplam kayıt: {len(df)}")
print(f"   Sütun sayısı: {len(df.columns)}")
print()

# --- 2. ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING) ---
print(">>> [2/6] Özellik mühendisliği yapılıyor...")

# Türetilmiş özellikler
df['Balance_per_Product'] = df['Balance'] / (df['NumOfProducts'] + 0.1)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
df['Credit_Score_Age_Ratio'] = df['CreditScore'] / (df['Age'] + 1)
df['Is_High_Value_Active'] = ((df['IsActiveMember'] == 1) & (df['Balance'] > df['Balance'].mean())).astype(int)

# Hedef ve özellikler
X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']

print(f"   Türetilmiş özellikler:")
print(f"     - Balance_per_Product: Ürün başına bakiye")
print(f"     - Age_Group: Yaş grubu (Young, Adult, Middle, Senior)")
print(f"     - Credit_Score_Age_Ratio: Kredi skoru / Yaş oranı")
print(f"     - Is_High_Value_Active: Yüksek değerli aktif müşteri (binary)")
print(f"   Toplam özellik sayısı: {X.shape[1]}")
print(f"   Hedef değişken dağılımı:")
print(f"     0 (Churn Yok): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
print(f"     1 (Churn Var): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
print()

# --- 3. VERİ ÖN İŞLEME (PREPROCESSING) ---
print(">>> [3/6] Veri ön işleme yapılıyor...")

# Sayısal ve kategorik değişkenleri ayır
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Balance_per_Product', 'Credit_Score_Age_Ratio', 'Is_High_Value_Active']

cat_cols = ['Geography', 'Gender', 'Age_Group']

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

print(f"   Sayısal değişkenler: {len(num_cols)} (StandardScaler ile ölçeklendirilecek)")
print(f"   Kategorik değişkenler: {len(cat_cols)} (OneHotEncoder ile kodlanacak)")
print()

# --- 4. VERİ BÖLME (TRAIN-TEST SPLIT) ---
print(">>> [4/6] Veri train-test split yapılıyor...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y
)

print(f"   Train set: {X_train.shape[0]} kayıt")
print(f"   Test set: {X_test.shape[0]} kayıt")
print()

# --- 5. MODEL TANIMLAMA ---
print(">>> [5/6] Model tanımlanıyor...")

# LightGBM modeli (optimize edilmiş parametreler)
lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    verbosity=-1,
    boosting_type='gbdt'
)

# Pipeline oluştur
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lgbm_model)
])

print("   LightGBM parametreleri:")
print(f"     n_estimators: 100")
print(f"     learning_rate: 0.1")
print(f"     max_depth: 5")
print(f"     subsample: 0.8")
print(f"     colsample_bytree: 0.8")
print(f"     boosting_type: gbdt")
print()

# --- 6. CROSS-VALIDATION EĞİTİMİ ---
print(">>> [6/6] 50-Fold Stratified Cross Validation eğitimi başlatılıyor...")
print()

CV_FOLDS = 50
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True)

# ROC-AUC ve Accuracy için cross-validation
print("   Cross-Validation skorları hesaplanıyor...")
start_time = time.time()

cv_auc_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
cv_acc_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)

elapsed_time = time.time() - start_time

print(f"   Tamamlandı! (Süre: {elapsed_time:.2f} saniye)")
print()

# --- 7. FINAL MODEL EĞİTİMİ ---
print("Final model tüm train seti ile eğitiliyor...")
pipeline.fit(X_train, y_train)
print("   Eğitim tamamlandı!")
print()

# --- 8. MODEL DEĞERLENDİRME ---
print("=" * 80)
print("MODEL DEĞERLENDİRME")
print("=" * 80)
print()

# Test set tahminleri
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Metrikler
test_accuracy = accuracy_score(y_test, y_pred)
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print("CROSS-VALIDATION SONUÇLARI (50-Fold):")
print("-" * 80)
print(f"ROC-AUC Ortalama: {cv_auc_scores.mean():.4f} ({cv_auc_scores.mean()*100:.2f}%)")
print(f"ROC-AUC Std Sapma: {cv_auc_scores.std():.4f} ({cv_auc_scores.std()*100:.2f}%)")
print(f"ROC-AUC Min: {cv_auc_scores.min():.4f} ({cv_auc_scores.min()*100:.2f}%)")
print(f"ROC-AUC Max: {cv_auc_scores.max():.4f} ({cv_auc_scores.max()*100:.2f}%)")
print()
print(f"Accuracy Ortalama: {cv_acc_scores.mean():.4f} ({cv_acc_scores.mean()*100:.2f}%)")
print(f"Accuracy Std Sapma: {cv_acc_scores.std():.4f} ({cv_acc_scores.std()*100:.2f}%)")
print(f"Accuracy Min: {cv_acc_scores.min():.4f} ({cv_acc_scores.min()*100:.2f}%)")
print(f"Accuracy Max: {cv_acc_scores.max():.4f} ({cv_acc_scores.max()*100:.2f}%)")
print()

print("TEST SET SONUÇLARI:")
print("-" * 80)
print(f"Accuracy:        {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"ROC-AUC:         {test_roc_auc:.4f} ({test_roc_auc*100:.2f}%)")
print(f"Precision:       {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"Recall:          {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"F1-Score:        {test_f1:.4f} ({test_f1*100:.2f}%)")
print()

# Classification Report
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=['Churn Yok', 'Churn Var']))
print()

# --- 9. MODEL KAYDETME ---
print("=" * 80)
print("MODEL KAYDEDİLİYOR...")
print("=" * 80)

# Model package
model_package = {
    'pipeline': pipeline,
    'preprocessor': preprocessor,
    'cv_auc_mean': cv_auc_scores.mean(),
    'cv_auc_std': cv_auc_scores.std(),
    'cv_acc_mean': cv_acc_scores.mean(),
    'cv_acc_std': cv_acc_scores.std(),
    'test_accuracy': test_accuracy,
    'test_roc_auc': test_roc_auc,
    'num_features': num_cols,
    'cat_features': cat_cols
}

output_file = 'churn_model_v1.pkl'
joblib.dump(model_package, output_file)
print(f"Model kaydedildi: {output_file}")
print()

print("=" * 80)
print("EĞİTİM TAMAMLANDI!")
print("=" * 80)

