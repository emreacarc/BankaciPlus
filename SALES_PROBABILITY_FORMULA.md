# 📊 Satış İhtimali Hesaplama Formülü - Detaylı Açıklama

## 🎯 Genel Bakış

NBA (Next Best Action) modülünde satış ihtimali, müşterinin çeşitli özelliklerine göre **dinamik olarak** hesaplanmaktadır. Sabit değerler yerine, müşteri profilini analiz eden bir formül kullanılmaktadır.

---

## 📐 Hesaplama Formülü

### **Toplam Satış İhtimali = Base Probability + Tüm Faktörler**

```
Final_Probability = Base_Prob + Balance_Factor + Salary_Factor + Activity_Factor + 
                    Product_Factor + Age_Factor + Credit_Factor + Tenure_Factor + 
                    Product_Gap_Factor
```

**Normalizasyon:** Final değer 25% ile 95% arasına sınırlandırılır.

---

## 🔢 Faktörler ve Ağırlıkları

### 1. **Base Probability (Segment Bazlı Temel İhtimal)**

Her segment için farklı bir başlangıç değeri:

| Segment | Base Probability |
|---------|-------------------|
| 💎 Elit / Servet Yönetimi | 80% |
| 🚀 Dinamik / Aktif Müşteri | 75% |
| 💰 Güvenli / Birikimci | 70% |
| 📊 Standart Bankacılık | 65% |
| 🌱 Temel Mevduat / Giriş | 60% |
| ⚠️ Riskli / Pasif Müşteri | 55% |
| Segment bilinmiyorsa | 50% (varsayılan) |

**Mantık:** Yüksek değerli segmentler daha yüksek başlangıç ihtimaline sahiptir.

---

### 2. **Balance Factor (Bakiye Faktörü)**

Müşterinin vadesiz hesap bakiyesine göre:

| Bakiye Aralığı | Eklenen Puan |
|----------------|--------------|
| > 100,000$ | +12% |
| > 50,000$ | +8% |
| > 20,000$ | +5% |
| > 10,000$ | +2% |
| ≤ 10,000$ | 0% |

**Mantık:** Yüksek bakiye = daha fazla finansal güç = daha yüksek satış ihtimali

**Örnek:**
- Müşteri bakiyesi: 75,000$ → **+8%**

---

### 3. **Salary Factor (Gelir Faktörü)**

Müşterinin yıllık gelirine göre:

| Gelir Aralığı | Eklenen Puan |
|---------------|--------------|
| > 100,000$ | +10% |
| > 60,000$ | +7% |
| > 40,000$ | +4% |
| > 25,000$ | +2% |
| ≤ 25,000$ | 0% |

**Mantık:** Yüksek gelir = daha fazla harcama gücü = daha yüksek ürün alma ihtimali

**Örnek:**
- Müşteri maaşı: 85,000$ → **+7%**

---

### 4. **Activity Factor (Aktivite Faktörü)**

Müşterinin aktif/pasif durumuna göre:

| Durum | Eklenen/Azaltılan Puan |
|-------|------------------------|
| Aktif Üye (IsActiveMember = 1) | +8% |
| Pasif Üye (IsActiveMember = 0) | -5% |

**Mantık:** Aktif müşteriler daha sık bankacılık işlemi yapar, pasif müşteriler daha az ilgilenir

**Örnek:**
- Aktif müşteri → **+8%**
- Pasif müşteri → **-5%**

---

### 5. **Product Factor (Ürün Portföyü Faktörü)**

Müşterinin sahip olduğu ürün sayısına göre:

```
Product_Factor = min(NumOfProducts × 3, 10)
```

**Hesaplama:**
- 1 ürün → +3%
- 2 ürün → +6%
- 3 ürün → +9%
- 4+ ürün → +10% (maksimum)

**Mantık:** Daha fazla ürün = daha sadık müşteri = yeni ürün alma ihtimali daha yüksek

**Örnek:**
- Müşterinin 3 ürünü var → **+9%**

---

### 6. **Age Factor (Yaş Faktörü)**

Müşterinin yaşına göre:

| Yaş Aralığı | Eklenen/Azaltılan Puan |
|-------------|------------------------|
| 25-55 yaş | +5% |
| 20-25 veya 55-65 yaş | +2% |
| > 65 yaş | -3% |
| < 20 yaş | 0% |

**Mantık:** 25-55 yaş arası müşteriler en aktif ve ürün alma eğiliminde

**Örnek:**
- Müşteri yaşı: 42 → **+5%**

---

### 7. **Credit Factor (Kredi Skoru Faktörü)**

Müşterinin kredi skoruna göre:

| Kredi Skoru | Eklenen/Azaltılan Puan |
|-------------|------------------------|
| ≥ 750 | +6% |
| ≥ 700 | +4% |
| ≥ 650 | +2% |
| < 600 | -5% |
| 600-650 arası | 0% |

**Mantık:** Yüksek kredi skoru = daha güvenilir müşteri = daha yüksek onay ihtimali

**Örnek:**
- Müşteri kredi skoru: 720 → **+4%**

---

### 8. **Tenure Factor (Müşteri Sadakati Faktörü)**

Müşterinin bankada geçirdiği yıla göre:

```
Tenure_Factor = min(Tenure × 1.5, 8)
```

**Hesaplama:**
- 1 yıl → +1.5%
- 2 yıl → +3%
- 3 yıl → +4.5%
- 4 yıl → +6%
- 5 yıl → +7.5%
- 6+ yıl → +8% (maksimum)

**Mantık:** Daha uzun süreli müşteri = daha sadık = yeni ürün alma ihtimali daha yüksek

**Örnek:**
- Müşteri 7 yıldır müşteri → **+8%**

---

### 9. **Product Gap Factor (Ürün Eksikliği Faktörü)**

Önerilen ürünün müşteride eksik olması durumunda:

| Ürün Tipi | Eksikse Eklenen Puan |
|-----------|----------------------|
| Kredi Kartı | +8% |
| BES (Emeklilik) | +7% |
| Yatırım Ürünleri (Fon, Altın) | +6% |
| İhtiyaç Kredisi | +5% |

**Mantık:** Müşteride eksik olan ürünler için satış ihtimali daha yüksektir

**Örnek:**
- Önerilen ürün: "Premium Kredi Kartı"
- Müşterinin kredi kartı yok → **+8%**

---

## 🧮 Hesaplama Örneği

### Senaryo: Elit Segment Müşterisi

**Müşteri Özellikleri:**
- Segment: 💎 Elit / Servet Yönetimi
- Balance: 120,000$
- Salary: 85,000$
- IsActiveMember: 1 (Aktif)
- NumOfProducts: 3
- Age: 42
- CreditScore: 720
- Tenure: 7 yıl
- Önerilen Ürün: "Özel Yatırım Danışmanlığı" (Has_Yatirim = 0)

**Hesaplama:**

```
Base_Prob = 80% (Elit segment)

Balance_Factor = +12% (Balance > 100,000$)
Salary_Factor = +7% (Salary > 60,000$)
Activity_Factor = +8% (Aktif üye)
Product_Factor = +9% (3 ürün × 3 = 9, max 10)
Age_Factor = +5% (Yaş 25-55 arası)
Credit_Factor = +4% (CreditScore ≥ 700)
Tenure_Factor = +8% (7 yıl × 1.5 = 10.5, max 8)
Product_Gap_Factor = +6% (Yatırım ürünü eksik)

Toplam = 80 + 12 + 7 + 8 + 9 + 5 + 4 + 8 + 6 = 139%
```

**Normalizasyon:**
```
Ham_Skor = 139%
Normalize = min(139, 80) = 80% (ham skor max limit)
Random = -3% (örnek)
Final = 80 - 3 = 77%
```

**Sonuç:** Bu müşteri için satış ihtimali **77%** (her hesaplamada değişebilir)

---

### Senaryo: Standart Segment Müşterisi

**Müşteri Özellikleri:**
- Segment: 📊 Standart Bankacılık
- Balance: 15,000$
- Salary: 35,000$
- IsActiveMember: 0 (Pasif)
- NumOfProducts: 1
- Age: 28
- CreditScore: 680
- Tenure: 2 yıl
- Önerilen Ürün: "Standart Kredi Kartı" (HasCrCard = 0)

**Hesaplama:**

```
Base_Prob = 65% (Standart segment)

Balance_Factor = +2% (Balance > 10,000$)
Salary_Factor = +4% (Salary > 25,000$)
Activity_Factor = -5% (Pasif üye)
Product_Factor = +3% (1 ürün × 3 = 3)
Age_Factor = +5% (Yaş 25-55 arası)
Credit_Factor = +2% (CreditScore ≥ 650)
Tenure_Factor = +3% (2 yıl × 1.5 = 3)
Product_Gap_Factor = +8% (Kredi kartı eksik)

Toplam = 65 + 2 + 4 - 5 + 3 + 5 + 2 + 3 + 8 = 87%
```

**Normalizasyon:**
```
Ham_Skor = 87%
Normalize = 87% (zaten 25-80 arasında)
Random = +2% (örnek)
Final = 87 + 2 = 89%
```

**Sonuç:** Bu müşteri için satış ihtimali **89%** (her hesaplamada değişebilir)

---

## ⚙️ Normalizasyon Kuralları

1. **Ham Skor Normalizasyonu:** 25%-80% arası
   - Toplam faktörlerin toplamı 25-80 arasına sınırlandırılır
   - Bu, temel hesaplama sonucudur

2. **Random Varyasyon:** -5% ile +5% arası
   - Çeşitlilik için rastgele bir değer eklenir/çıkarılır
   - Her hesaplamada farklı sonuçlar üretir
   - Gerçekçi bir belirsizlik ekler

3. **Final Normalizasyon:** 25%-95% arası
   - Random varyasyon eklendikten sonra final sonuç 25-95 arasına sınırlandırılır
   - Random eklemesi sonucu 95'i aşabilir, bu durumda 95'e sabitlenir

4. **Yuvarlama:** Sonuç en yakın tam sayıya yuvarlanır

**Örnek:**
- Ham skor: 87% → Normalize: 80% (max limit)
- Random: +3% → 80 + 3 = 83%
- Final: 83%

---

## 📊 Formül Özeti

```
calculate_sales_probability(row, segment_name, product_type):

1. Base_Prob = segment_base_prob[segment_name] veya 50%

2. Balance_Factor = 
   - Balance > 100K → +12%
   - Balance > 50K → +8%
   - Balance > 20K → +5%
   - Balance > 10K → +2%
   - Diğer → 0%

3. Salary_Factor = 
   - Salary > 100K → +10%
   - Salary > 60K → +7%
   - Salary > 40K → +4%
   - Salary > 25K → +2%
   - Diğer → 0%

4. Activity_Factor = 
   - Aktif → +8%
   - Pasif → -5%

5. Product_Factor = min(NumOfProducts × 3, 10%)

6. Age_Factor = 
   - 25-55 yaş → +5%
   - 20-25 veya 55-65 → +2%
   - >65 yaş → -3%
   - Diğer → 0%

7. Credit_Factor = 
   - CreditScore ≥ 750 → +6%
   - CreditScore ≥ 700 → +4%
   - CreditScore ≥ 650 → +2%
   - CreditScore < 600 → -5%
   - Diğer → 0%

8. Tenure_Factor = min(Tenure × 1.5, 8%)

9. Product_Gap_Factor = 
   - Kredi Kartı eksik → +8%
   - BES eksik → +7%
   - Yatırım eksik → +6%
   - Kredi eksik → +5%
   - Diğer → 0%

10. Total = Base_Prob + Tüm Faktörler

11. Normalize = max(25, min(80, Total))  # Ham skor 25-80 arası

12. Random = random.randint(-5, 5)  # -5 ile +5 arası rastgele değer

13. Final = max(25, min(95, Normalize + Random))  # Final 25-95 arası
```

---

## 🎯 Avantajlar

1. **Dinamik:** Her müşteri için özel hesaplama
2. **Gerçekçi:** Müşteri özelliklerine dayalı
3. **Esnek:** Yeni faktörler eklenebilir
4. **Şeffaf:** Hesaplama mantığı açık ve anlaşılır
5. **Ölçeklenebilir:** Farklı segmentler için farklı base değerler

---

## 🔄 Önceki Sistem vs Yeni Sistem

### Önceki Sistem (Sabit Değerler):
```python
if segment == "Elit" and Has_Yatirim == 0:
    Prob = 92  # Sabit
```

### Yeni Sistem (Dinamik Hesaplama):
```python
Prob = calculate_sales_probability(row, segment_name, "Yatırım")
# Müşterinin tüm özelliklerine göre hesaplanır
```

---

## 📝 Notlar

- Formül, gerçek satış verileri olmadığı için **kural tabanlı** bir yaklaşımdır
- Gelecekte gerçek satış verileri ile bir ML modeli eğitilebilir
- Faktör ağırlıkları, domain knowledge ve iş mantığına göre belirlenmiştir
- Normalizasyon limitleri (25%-95%) gerçekçi bir aralık sağlar

