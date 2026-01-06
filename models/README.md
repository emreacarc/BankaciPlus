# Model Eğitim Scriptleri

Bu klasör, BankaciPlus projesinde kullanılan tüm ML modellerinin eğitim scriptlerini içerir. Bu scriptler mülakatlarda gösterilmek üzere hazırlanmıştır ve app.py'de kullanılmaz.

## 📁 Dosyalar

### 1. `train_credit_risk_lite.py`
**Credit Risk Lite Model - XGBoost Classifier**

- **Değişken Sayısı:** 8 (7 temel + 1 türetilmiş)
- **Algoritma:** XGBoost Classifier
- **Optimizasyon:** RandomizedSearchCV (150 kombinasyon, 3-fold CV)
- **Özellikler:**
  - Veri yükleme ve temizleme
  - Özellik mühendisliği (`loan_to_income`)
  - One-Hot Encoding (kategorik değişkenler)
  - Hiperparametre optimizasyonu
  - Model eğitimi ve değerlendirme
  - Model kaydetme

**Kullanım:**
```bash
python models/train_credit_risk_lite.py
```

**Çıktı:** `credit_risk_lite_model.pkl`

---

### 2. `train_credit_risk_pro.py`
**Credit Risk Pro Model - XGBoost Classifier**

- **Değişken Sayısı:** 16 (13 temel + 3 türetilmiş)
- **Algoritma:** XGBoost Classifier
- **Optimizasyon:** RandomizedSearchCV (100 kombinasyon, 3-fold CV)
- **Özellikler:**
  - Veri yükleme ve temizleme
  - Özellik mühendisliği (`loan_to_income`, `installment_to_income`, `balance_income_ratio`)
  - One-Hot Encoding (kategorik değişkenler)
  - Hiperparametre optimizasyonu
  - Model eğitimi ve değerlendirme
  - Model kaydetme

**Kullanım:**
```bash
python models/train_credit_risk_pro.py
```

**Çıktı:** `credit_risk_model_20fold.pkl`

---

### 3. `train_churn_model.py`
**Churn Prediction Model - LightGBM Classifier**

- **Algoritma:** LightGBM (Light Gradient Boosting Machine)
- **Eğitim:** 50-Fold Stratified Cross Validation
- **Özellikler:**
  - Veri yükleme
  - Özellik mühendisliği (`Balance_per_Product`, `Age_Group`, `Credit_Score_Age_Ratio`, `Is_High_Value_Active`)
  - StandardScaler (sayısal değişkenler)
  - OneHotEncoder (kategorik değişkenler)
  - Pipeline yapısı
  - 50-fold CV ile model eğitimi
  - Detaylı performans metrikleri
  - Model kaydetme

**Kullanım:**
```bash
python models/train_churn_model.py
```

**Çıktı:** `churn_model_v1.pkl`

---

### 4. `train_nba_kmeans.py`
**NBA K-Means Clustering Model**

- **Algoritma:** K-Means Clustering (scikit-learn)
- **Küme Sayısı:** 6
- **Özellikler:**
  - Veri yükleme
  - Özellik seçimi (Balance, EstimatedSalary, NumOfProducts, Tenure, IsActiveMember)
  - MinMaxScaler normalizasyonu (0-1 arası)
  - K-Means clustering (`n_init=10000`)
  - Silhouette Score doğrulama
  - Segment analizi ve isimlendirme
  - Model kaydetme

**Kullanım:**
```bash
python models/train_nba_kmeans.py
```

**Çıktılar:**
- `kmeans_model.pkl`
- `scaler_model.pkl`
- `churn_processed_with_clusters.csv`

---

## 📋 Gereksinimler

Tüm scriptler aşağıdaki kütüphaneleri gerektirir:

```python
pandas
numpy
scikit-learn
xgboost
lightgbm
joblib
```

## 📊 Veri Setleri

Scriptlerin çalışması için aşağıdaki veri setlerinin mevcut olması gerekir:

1. **Credit Risk Modelleri:**
   - `lending_club_cleaned.csv` (proje kök dizininde)

2. **Churn ve NBA Modelleri:**
   - `datasets/bank_customer_churn_data/Customer-Churn-Records.csv`

## 🎯 Mülakat İçin Kullanım

Bu scriptler mülakatlarda şu amaçlarla kullanılabilir:

1. **Model Eğitim Süreci Gösterimi:** Her script, veri ön işlemeden model kaydetmeye kadar tüm süreci gösterir
2. **Teknik Detaylar:** Preprocessing, feature engineering, hyperparameter tuning gibi teknik detaylar açıkça görülebilir
3. **Performans Metrikleri:** Her model için detaylı performans metrikleri hesaplanır ve gösterilir
4. **Kod Kalitesi:** Temiz, yorumlanmış ve anlaşılır kod yapısı

## ⚠️ Notlar

- Bu scriptler **sadece eğitim amaçlıdır** ve app.py'de kullanılmaz
- Scriptler çalıştırıldığında mevcut model dosyalarını **üzerine yazar**
- Büyük veri setleri için eğitim süresi uzun olabilir
- Her script bağımsız olarak çalıştırılabilir

