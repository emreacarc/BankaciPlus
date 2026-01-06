# Bankacı Plus / Bankaci Plus

**[🇹🇷 Türkçe](#türkçe) | [🇬🇧 English](#english)**

---

## 🇹🇷 Türkçe

### 📋 Proje Hakkında

**Bankacı Plus**, üç temel bankacılık dikeyini bir araya getiren, Streamlit tabanlı kapsamlı bir karar destek uygulamasıdır:

1. **Kredi Risk Analizi** - XGBoost tabanlı Lite & Pro modeller
2. **Müşteri Kayıp (Churn) Önleme** - LightGBM tabanlı tahmin ve strateji motoru
3. **Akıllı Satış (Next Best Action - NBA)** - K-Means segmentasyonu ve kural tabanlı ürün önerileri

### 🎯 Özellikler

#### 1. Kredi Risk Modülü
- **Lite Model:** 8 değişken ile hızlı risk skorlama (XGBoost)
- **Pro Model:** 16 değişken ile detaylı risk analizi (XGBoost)
- Gerçek zamanlı risk skorlama
- Manuel giriş formu ile tek müşteri analizi
- Toplu risk listesi ve filtreleme
- İndirilebilir aksiyon planları

#### 2. Churn Önleme Modülü
- **LightGBM** tabanlı churn tahmin modeli
- 50-fold cross-validation ile eğitilmiş (%86+ ROC-AUC)
- Kişiselleştirilmiş strateji önerileri (8 farklı strateji)
- Risk seviyesine göre otomatik aksiyon planı
- Toplu churn risk analizi
- Segment bazlı kampanya yönetimi

#### 3. Akıllı Satış (NBA) Modülü
- **K-Means** ile 6 segmentli müşteri kümeleme
- Silhouette Score: 0.34 (model doğrulama)
- Finansal DNA analizi (5 boyutlu radar grafiği)
- Kural tabanlı ürün öneri sistemi
- Segment bazlı kampanya listesi oluşturma
- CSV export özelliği

### 🤖 Machine Learning Modelleri

#### Kredi Risk Modelleri

**Lite Model (XGBoost):**
- **Algoritma:** XGBoost Classifier
- **Değişken Sayısı:** 8 (7 temel + 1 türetilmiş)
- **Hiperparametreler:**
  - `n_estimators`: 100
  - `learning_rate`: 0.1
  - `max_depth`: 5
  - `subsample`: 0.8
  - `colsample_bytree`: 0.7
  - `min_child_weight`: 1
- **Optimizasyon:** RandomizedSearchCV (150 kombinasyon, 3-fold CV)
- **Performans:**
  - Test Accuracy: %65.29
  - Test ROC-AUC: %70.31
- **Kullanım:** Hızlı ön tarama, minimum bilgi gereksinimi

**Pro Model (XGBoost):**
- **Algoritma:** XGBoost Classifier
- **Değişken Sayısı:** 16 (13 temel + 3 türetilmiş)
- **Hiperparametreler:**
  - `n_estimators`: 350
  - `learning_rate`: 0.03
  - `max_depth`: 4
  - `subsample`: 0.75
  - `colsample_bytree`: 0.75
  - `min_child_weight`: 2
  - `gamma`: 0
- **Optimizasyon:** RandomizedSearchCV (100 kombinasyon, 3-fold CV)
- **Performans:**
  - Test Accuracy: %65.71
  - Test ROC-AUC: %71.24
- **Kullanım:** Büyük tutarlı krediler, detaylı risk analizi

#### Churn Tahmin Modeli

**LightGBM Classifier:**
- **Algoritma:** LightGBM (Light Gradient Boosting Machine)
- **Model Seçimi:** XGBoost, LightGBM ve CatBoost 50-fold CV ile karşılaştırıldı
- **Seçim Gerekçesi:** En hızlı eğitim süresi + yüksek performans kombinasyonu
- **Hiperparametreler:**
  - `n_estimators`: 100
  - `learning_rate`: 0.1
  - `max_depth`: 5
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `boosting_type`: gbdt
- **Eğitim:** 50-Fold Stratified Cross Validation
- **Performans:**
  - CV ROC-AUC Ortalama: %86.27 (Std: 3.51%)
  - CV Accuracy Ortalama: %86.39 (Std: 1.79%)
  - Test ROC-AUC: %87.42
  - Test Accuracy: %86.20
- **Avantajlar:**
  - XGBoost'a göre daha hızlı eğitim
  - Daha az bellek kullanımı
  - Yüksek performans-hız dengesi

#### NBA Segmentasyon Modeli

**K-Means Clustering:**
- **Algoritma:** K-Means (scikit-learn)
- **Küme Sayısı:** 6
- **Özellikler:** Balance, EstimatedSalary, NumOfProducts, Tenure, IsActiveMember
- **Optimizasyon:** `n_init=10000` (10,000 farklı başlangıç noktası)
- **Doğrulama:** Silhouette Score = 0.34
- **Normalizasyon:** MinMaxScaler (0-1 arası ölçeklendirme)
- **Kullanım:** Müşteri segmentasyonu ve finansal DNA analizi

### 🔧 Veri Önişleme (Data Preprocessing)

#### Kredi Risk Modülü
1. **Özellik Mühendisliği:**
   - `loan_to_income`: Kredi tutarı / Yıllık gelir oranı
   - `installment_to_income`: Aylık taksit / Aylık gelir oranı (PTI)
   - `balance_income_ratio`: Döner kredi bakiyesi / Yıllık gelir oranı

2. **Kategorik Veri İşleme:**
   - **One-Hot Encoding:** Ev Durumu, Amaç, Not, İstihdam Süresi, Doğrulama Durumu
   - Kategorik değişkenler sayısal forma dönüştürülmüştür

3. **Sayısal Veri İşleme:**
   - Eksik değer temizleme
   - Aykırı değer (outlier) kontrolü
   - Veri tipi dönüşümleri

#### Churn Modülü
1. **Özellik Mühendisliği:**
   - `Balance_per_Product`: Ürün Başına Bakiye
   - `Age_Group`: Yaş Grubu kategorilendirme (Young, Adult, Middle, Senior)
   - `Credit_Score_Age_Ratio`: Kredi Skoru / Yaş Oranı
   - `Is_High_Value_Active`: Yüksek Değerli Aktif Müşteri (binary)

2. **Preprocessing Pipeline:**
   - **Sayısal Değişkenler:** `StandardScaler` ile ölçeklendirme (ortalama=0, std=1)
   - **Kategorik Değişkenler:** `OneHotEncoder` ile kodlama
   - **Pipeline Yapısı:** Preprocessing ve model eğitimi birleştirilmiştir

3. **Veri Bölme:**
   - **Stratified Train-Test Split:** 80-20 oranında
   - Stratified yöntem ile sınıf dağılımı korunmuştur

#### NBA Modülü
1. **Özellik Seçimi:**
   - Balance, EstimatedSalary, NumOfProducts, Tenure, IsActiveMember

2. **Normalizasyon:**
   - **MinMaxScaler:** Tüm özellikler 0-1 arasına ölçeklendirilmiştir
   - K-Means algoritması için ölçeklendirme kritiktir

3. **Segment İsimlendirme:**
   - Centroid analizi ile finansal özelliklere göre segment isimlendirme
   - Her segment için ortalama bakiye, maaş, ürün sayısı hesaplanmıştır

### 💻 Tech Stack

#### Frontend & Framework
- **Streamlit:** Web uygulaması framework'ü
- **Custom CSS:** Dark mode, gradient başlıklar, glassmorphism efektleri
- **Google Fonts:** Outfit (body), Syne (headings)

#### Data Processing & Analysis
- **pandas:** Veri manipülasyonu ve analizi
- **numpy:** Sayısal hesaplamalar
- **scikit-learn:**
  - `StandardScaler`, `MinMaxScaler` - Veri ölçeklendirme
  - `OneHotEncoder` - Kategorik veri kodlama
  - `ColumnTransformer` - Pipeline preprocessing
  - `KMeans` - Kümeleme algoritması
  - `silhouette_score` - Kümeleme doğrulama
  - `train_test_split` - Veri bölme
  - `RandomizedSearchCV` - Hiperparametre optimizasyonu

#### Machine Learning
- **XGBoost:** Kredi risk modelleri (Lite & Pro)
- **LightGBM:** Churn tahmin modeli
- **CatBoost:** Model karşılaştırması (test edildi, seçilmedi)

#### Visualization
- **Plotly Express (`px`):** Bar, pie, scatter grafikleri
- **Plotly Graph Objects (`go`):** Radar grafikleri, özel grafikler
- **Streamlit Native:** Metrikler, tablolar, expander'lar

#### Model Persistence
- **joblib:** Model serialization ve deserialization (.pkl dosyaları)

#### Development Tools
- **Python 3.9+**
- **Git:** Versiyon kontrolü
- **GitHub:** Remote repository

### 📊 Veri Setleri

#### 1. Lending Club Dataset (Kredi Risk)
- **Kaynak:** Lending Club (2007-2015 P2P kredi verileri)
- **Kayıt Sayısı:** 40,000+
- **Kullanım:** Kredi risk modellerinin eğitimi
- **Özellikler:** Kredi tutarı, gelir, ev durumu, amaç, kredi geçmişi, vb.
- **Not:** Büyük CSV dosyaları repo'da bulunmamaktadır (`.gitignore`)

#### 2. Bank Customer Churn Dataset (Churn & NBA)
- **Kaynak:** Bank Customer Churn Modeling
- **Kayıt Sayısı:** 10,000
- **Kullanım:** Churn tahmin modeli ve NBA segmentasyonu
- **Özellikler:** Demografik bilgiler, finansal durum, ürün kullanımı
- **Not:** Büyük CSV dosyaları repo'da bulunmamaktadır (`.gitignore`)

### 🚀 Kurulum ve Çalıştırma

#### Gereksinimler
- Python 3.9 veya üzeri
- pip (Python paket yöneticisi)

#### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/emreacarc/BankaciPlus.git
cd BankaciPlus
```

2. **Sanal ortam oluşturun (önerilir):**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Veri setlerini ve modelleri hazırlayın:**
   - `lending_club_cleaned.csv` dosyasını `datasets/` klasörüne ekleyin
   - `bank_customer_churn_data/Customer-Churn-Records.csv` dosyasını `datasets/` klasörüne ekleyin
   - Eğitilmiş model dosyalarını (.pkl) proje kök dizinine ekleyin:
     - `credit_risk_model_20fold.pkl` (Pro model)
     - `credit_risk_lite_model.pkl` (Lite model)
     - `churn_model_v1.pkl` (Churn model)

5. **Uygulamayı çalıştırın:**
```bash
streamlit run app.py
```

6. **Tarayıcıda açın:**
   - Uygulama otomatik olarak `http://localhost:8501` adresinde açılacaktır

### 📁 Proje Yapısı

```
BankaciPlus/
├── app.py                          # Ana Streamlit uygulaması
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # Bu dosya
├── .gitignore                      # Git ignore kuralları
├── .streamlit/
│   └── config.toml                # Streamlit tema ayarları
├── datasets/                       # Veri setleri (git'te yok)
│   ├── lending_club_cleaned.csv
│   └── bank_customer_churn_data/
│       └── Customer-Churn-Records.csv
├── models/                         # Model eğitim scriptleri (sadece gösterim amaçlı)
│   ├── README.md                   # Model eğitim scriptleri açıklaması
│   ├── train_credit_risk_lite.py  # Credit Risk Lite Model eğitim scripti
│   ├── train_credit_risk_pro.py   # Credit Risk Pro Model eğitim scripti
│   ├── train_churn_model.py       # Churn Prediction Model eğitim scripti
│   └── train_nba_kmeans.py        # NBA K-Means Clustering Model eğitim scripti
├── compare_churn_models.py        # Model karşılaştırma scripti
└── model_comparison_log.txt        # Model karşılaştırma logu
```

### ⚠️ Önemli Notlar

- **Gizli Bilgiler:** `.streamlit/secrets.toml`, `.env`, API anahtarları gibi dosyaları commit etmeyin
- **Büyük Dosyalar:** CSV ve .pkl dosyaları `.gitignore`'da bulunmaktadır
- **Model Dosyaları:** Eğitilmiş modelleri harici olarak saklayın veya çalışma zamanında yükleyin
- **Veri Setleri:** Büyük veri setleri için indirme linkleri veya talimatlar ekleyin
- **`models/` Klasörü:** Bu klasördeki Python scriptleri **sadece gösterim amaçlıdır** ve mülakatlarda model eğitim sürecini göstermek için hazırlanmıştır. Bu scriptler `app.py`'de kullanılmaz ve uygulama çalıştırıldığında gerekli değildir.

### 📝 Lisans

Bu proje eğitim ve portföy amaçlıdır.

---

## 🇬🇧 English

### 📋 About the Project

**Bankaci Plus** is a comprehensive Streamlit-based decision support application that brings together three core banking verticals:

1. **Credit Risk Analysis** - XGBoost-based Lite & Pro models
2. **Customer Churn Prevention** - LightGBM-based prediction and strategy engine
3. **Smart Sales (Next Best Action - NBA)** - K-Means segmentation and rule-based product recommendations

### 🎯 Features

#### 1. Credit Risk Module
- **Lite Model:** Fast risk scoring with 8 variables (XGBoost)
- **Pro Model:** Detailed risk analysis with 16 variables (XGBoost)
- Real-time risk scoring
- Manual input form for single customer analysis
- Bulk risk list and filtering
- Downloadable action plans

#### 2. Churn Prevention Module
- **LightGBM**-based churn prediction model
- Trained with 50-fold cross-validation (86%+ ROC-AUC)
- Personalized strategy recommendations (8 different strategies)
- Automatic action plan based on risk level
- Bulk churn risk analysis
- Segment-based campaign management

#### 3. Smart Sales (NBA) Module
- **K-Means** clustering with 6 customer segments
- Silhouette Score: 0.34 (model validation)
- Financial DNA analysis (5-dimensional radar chart)
- Rule-based product recommendation system
- Segment-based campaign list generation
- CSV export feature

### 🤖 Machine Learning Models

#### Credit Risk Models

**Lite Model (XGBoost):**
- **Algorithm:** XGBoost Classifier
- **Number of Variables:** 8 (7 base + 1 derived)
- **Hyperparameters:**
  - `n_estimators`: 100
  - `learning_rate`: 0.1
  - `max_depth`: 5
  - `subsample`: 0.8
  - `colsample_bytree`: 0.7
  - `min_child_weight`: 1
- **Optimization:** RandomizedSearchCV (150 combinations, 3-fold CV)
- **Performance:**
  - Test Accuracy: 65.29%
  - Test ROC-AUC: 70.31%
- **Usage:** Fast preliminary screening, minimum information requirement

**Pro Model (XGBoost):**
- **Algorithm:** XGBoost Classifier
- **Number of Variables:** 16 (13 base + 3 derived)
- **Hyperparameters:**
  - `n_estimators`: 350
  - `learning_rate`: 0.03
  - `max_depth`: 4
  - `subsample`: 0.75
  - `colsample_bytree`: 0.75
  - `min_child_weight`: 2
  - `gamma`: 0
- **Optimization:** RandomizedSearchCV (100 combinations, 3-fold CV)
- **Performance:**
  - Test Accuracy: 65.71%
  - Test ROC-AUC: 71.24%
- **Usage:** Large amount loans, detailed risk analysis

#### Churn Prediction Model

**LightGBM Classifier:**
- **Algorithm:** LightGBM (Light Gradient Boosting Machine)
- **Model Selection:** XGBoost, LightGBM, and CatBoost compared with 50-fold CV
- **Selection Rationale:** Fastest training time + high performance combination
- **Hyperparameters:**
  - `n_estimators`: 100
  - `learning_rate`: 0.1
  - `max_depth`: 5
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `boosting_type`: gbdt
- **Training:** 50-Fold Stratified Cross Validation
- **Performance:**
  - CV ROC-AUC Average: 86.27% (Std: 3.51%)
  - CV Accuracy Average: 86.39% (Std: 1.79%)
  - Test ROC-AUC: 87.42%
  - Test Accuracy: 86.20%
- **Advantages:**
  - Faster training than XGBoost
  - Lower memory usage
  - High performance-speed balance

#### NBA Segmentation Model

**K-Means Clustering:**
- **Algorithm:** K-Means (scikit-learn)
- **Number of Clusters:** 6
- **Features:** Balance, EstimatedSalary, NumOfProducts, Tenure, IsActiveMember
- **Optimization:** `n_init=10000` (10,000 different starting points)
- **Validation:** Silhouette Score = 0.34
- **Normalization:** MinMaxScaler (0-1 scaling)
- **Usage:** Customer segmentation and financial DNA analysis

### 🔧 Data Preprocessing

#### Credit Risk Module
1. **Feature Engineering:**
   - `loan_to_income`: Loan amount / Annual income ratio
   - `installment_to_income`: Monthly installment / Monthly income ratio (PTI)
   - `balance_income_ratio`: Revolving credit balance / Annual income ratio

2. **Categorical Data Processing:**
   - **One-Hot Encoding:** Home ownership, Purpose, Grade, Employment length, Verification status
   - Categorical variables converted to numerical format

3. **Numerical Data Processing:**
   - Missing value cleaning
   - Outlier detection
   - Data type conversions

#### Churn Module
1. **Feature Engineering:**
   - `Balance_per_Product`: Balance per Product
   - `Age_Group`: Age group categorization (Young, Adult, Middle, Senior)
   - `Credit_Score_Age_Ratio`: Credit Score / Age Ratio
   - `Is_High_Value_Active`: High Value Active Customer (binary)

2. **Preprocessing Pipeline:**
   - **Numerical Variables:** Scaling with `StandardScaler` (mean=0, std=1)
   - **Categorical Variables:** Encoding with `OneHotEncoder`
   - **Pipeline Structure:** Preprocessing and model training combined

3. **Data Splitting:**
   - **Stratified Train-Test Split:** 80-20 ratio
   - Class distribution preserved with stratified method

#### NBA Module
1. **Feature Selection:**
   - Balance, EstimatedSalary, NumOfProducts, Tenure, IsActiveMember

2. **Normalization:**
   - **MinMaxScaler:** All features scaled to 0-1 range
   - Scaling is critical for K-Means algorithm

3. **Segment Naming:**
   - Segment naming based on centroid analysis of financial features
   - Average balance, salary, product count calculated for each segment

### 💻 Tech Stack

#### Frontend & Framework
- **Streamlit:** Web application framework
- **Custom CSS:** Dark mode, gradient headings, glassmorphism effects
- **Google Fonts:** Outfit (body), Syne (headings)

#### Data Processing & Analysis
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computations
- **scikit-learn:**
  - `StandardScaler`, `MinMaxScaler` - Data scaling
  - `OneHotEncoder` - Categorical data encoding
  - `ColumnTransformer` - Pipeline preprocessing
  - `KMeans` - Clustering algorithm
  - `silhouette_score` - Clustering validation
  - `train_test_split` - Data splitting
  - `RandomizedSearchCV` - Hyperparameter optimization

#### Machine Learning
- **XGBoost:** Credit risk models (Lite & Pro)
- **LightGBM:** Churn prediction model
- **CatBoost:** Model comparison (tested, not selected)

#### Visualization
- **Plotly Express (`px`):** Bar, pie, scatter charts
- **Plotly Graph Objects (`go`):** Radar charts, custom charts
- **Streamlit Native:** Metrics, tables, expanders

#### Model Persistence
- **joblib:** Model serialization and deserialization (.pkl files)

#### Development Tools
- **Python 3.9+**
- **Git:** Version control
- **GitHub:** Remote repository

### 📊 Datasets

#### 1. Lending Club Dataset (Credit Risk)
- **Source:** Lending Club (2007-2015 P2P loan data)
- **Number of Records:** 40,000+
- **Usage:** Credit risk model training
- **Features:** Loan amount, income, home ownership, purpose, credit history, etc.
- **Note:** Large CSV files are not in the repo (`.gitignore`)

#### 2. Bank Customer Churn Dataset (Churn & NBA)
- **Source:** Bank Customer Churn Modeling
- **Number of Records:** 10,000
- **Usage:** Churn prediction model and NBA segmentation
- **Features:** Demographic information, financial status, product usage
- **Note:** Large CSV files are not in the repo (`.gitignore`)

### 🚀 Installation and Running

#### Requirements
- Python 3.9 or higher
- pip (Python package manager)

#### Steps

1. **Clone the repository:**
```bash
git clone https://github.com/emreacarc/BankaciPlus.git
cd BankaciPlus
```

2. **Create virtual environment (recommended):**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Prepare datasets and models:**
   - Add `lending_club_cleaned.csv` to `datasets/` folder
   - Add `bank_customer_churn_data/Customer-Churn-Records.csv` to `datasets/` folder
   - Add trained model files (.pkl) to project root directory:
     - `credit_risk_model_20fold.pkl` (Pro model)
     - `credit_risk_lite_model.pkl` (Lite model)
     - `churn_model_v1.pkl` (Churn model)

5. **Run the application:**
```bash
streamlit run app.py
```

6. **Open in browser:**
   - Application will automatically open at `http://localhost:8501`

### 📁 Project Structure

```
BankaciPlus/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── .streamlit/
│   └── config.toml                # Streamlit theme settings
├── datasets/                       # Datasets (not in git)
│   ├── lending_club_cleaned.csv
│   └── bank_customer_churn_data/
│       └── Customer-Churn-Records.csv
├── models/                         # Model training scripts (for demonstration only)
│   ├── README.md                   # Model training scripts documentation
│   ├── train_credit_risk_lite.py  # Credit Risk Lite Model training script
│   ├── train_credit_risk_pro.py   # Credit Risk Pro Model training script
│   ├── train_churn_model.py       # Churn Prediction Model training script
│   └── train_nba_kmeans.py        # NBA K-Means Clustering Model training script
├── compare_churn_models.py        # Model comparison script
└── model_comparison_log.txt        # Model comparison log
```

### ⚠️ Important Notes

- **Secrets:** Do not commit files like `.streamlit/secrets.toml`, `.env`, API keys
- **Large Files:** CSV and .pkl files are in `.gitignore`
- **Model Files:** Store trained models externally or load at runtime
- **Datasets:** Add download links or instructions for large datasets
- **`models/` Folder:** The Python scripts in this folder are **for demonstration purposes only** and were prepared to showcase the model training process during interviews. These scripts are not used in `app.py` and are not required when running the application.

### 📝 License

This project is for educational and portfolio purposes.

---

**Developer:** Emre AÇAR  
**GitHub:** [emreacarc](https://github.com/emreacarc)  
**LinkedIn:** [LinkedIn Profilim](https://www.linkedin.com/in/emreacarc/)  
**Email:** ar.emreacar@gmail.com
