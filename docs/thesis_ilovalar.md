# ILOVALAR

---

## ILOVA A. Dasturiy kodning asosiy fragmentlari

### A.1-listing. Ko'p omilli tasniflash modelining arxitekturasi (MultiLabelClassifier)

Manba fayl: `src/improved_trainer.py`, 132-154 qatorlar.

```python
class MultiLabelClassifier(nn.Module):
    """XLM-RoBERTa with Multi-Label Classification Head"""
    
    def __init__(self, model_name: str, num_factors: int = 8, 
                 dropout: float = 0.3):
        super().__init__()
        self.xlm = AutoModel.from_pretrained(model_name)
        hidden_size = self.xlm.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_factors)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.xlm(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        # [CLS] token representation — 768 o'lchamli vektor
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
```

**Izoh:** Ushbu sinf `xlm-roberta-base` modelini yuklaydi va unga ikki bosqichli tasniflash boshi (768 → 256 → 8) qo'shadi. `forward()` metodi [CLS] tokenni ajratib, 8 ta omil uchun logitlar qaytaradi. Sigmoidli aktivatsiya va threshold qo'llanilishi tashqarida amalga oshiriladi.

---

### A.2-listing. Focal Loss funksiyasi

Manba fayl: `src/improved_trainer.py`, 161-203 qatorlar.

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    FL(p) = -α * (1-p)^γ * log(p)    for positive samples
    FL(p) = -(1-α) * p^γ * log(1-p)  for negative samples
    
    γ (gamma) = focusing parameter
    α (alpha) = per-factor weight from class distribution
    """
    
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha.to(DEVICE)  # Shape: (num_factors,)
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, num_factors) — raw, before sigmoid
        labels: (batch_size, num_factors) — binary 0/1
        """
        probs = torch.sigmoid(logits)
        
        # Raqamli barqarorlik uchun chegaralash
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
        
        # Focal modullovchi koeffitsiyent
        # Ijobiy uchun: (1-p)^γ — oson misollar past og'irlik oladi
        # Salbiy uchun: p^γ — oson salbiylar past og'irlik oladi
        pos_factor = (1.0 - probs) ** self.gamma
        neg_factor = probs ** self.gamma
        
        # Binary Cross-Entropy komponentlari
        pos_loss = -labels * pos_factor * torch.log(probs)
        neg_loss = -(1.0 - labels) * neg_factor * torch.log(1.0 - probs)
        
        # Alpha og'irliklari (har bir omil uchun alohida)
        loss = self.alpha.unsqueeze(0) * pos_loss + neg_loss
        
        return loss.mean()
```

**Izoh:** Focal Loss (Lin et al., 2017) sinf nomutanosibligi muammosini hal qiladi. γ = 2.0 "oson" misollarning og'irligini kamaytiradi. α koeffitsiyentlari har bir omil uchun alohida hisoblanadi va maksimal 3.0 ga cheklanadi (gradient portlashining oldini olish uchun).

---

### A.3-listing. Salomatlik ko'rsatkichini hisoblash funksiyasi

Manba fayl: `src/b2b_engine.py`, 495-685 qatorlar (qisqartirilgan asosiy qism).

```python
# Ta'sir koeffitsiyentlari (Random Forest + OLS regressiya natijasi)
# Rating = 4.036 + Σ(weight_i × factor_i)
IMPACT_WEIGHTS = {
    'product_defects': -0.6297,     # Eng kuchli salbiy ta'sir
    'product_quality': 0.2605,      # Eng kuchli ijobiy ta'sir
    'accuracy_expectation': -0.0435,
    'seller_service': 0.0218,
    'price_value': 0.0158,
    'logistics_delivery': 0.0108,
    'packaging_condition': -0.0090,
    'specifications': -0.0089
}

def calculate_health_score(self, reviews_data, actual_rating=0.0):
    """
    Sharhlar asosida mahsulot salomatlik ko'rsatkichini hisoblash.
    Qaytaradi: health_score (0-10 shkala)
    """
    # 1. Har bir sharh uchun XLM-RoBERTa orqali 8 ta omilni aniqlash
    all_factors = []
    for r in reviews_data:
        factor_scores = self.factor_extractor.extract_factors(
            r['content']
        )
        all_factors.append(factor_scores)
    
    # 2. Yulduzcha reyting bo'yicha sharhlarni ajratish
    #    Rating ≤ 3: salbiy eslatmalar
    #    Rating ≥ 4: ijobiy eslatmalar
    negative_factor_vals = {f: [] for f in SENTIMENT_FACTORS}
    positive_factor_vals = {f: [] for f in SENTIMENT_FACTORS}
    
    for i, r in enumerate(reviews_data):
        review_rating = r.get('rating', 0) or 0
        factors = all_factors[i]
        for f in SENTIMENT_FACTORS:
            if f in factors:
                score = factors[f]
                if review_rating <= 3:
                    negative_factor_vals[f].append(score)
                elif review_rating >= 4:
                    positive_factor_vals[f].append(score)
    
    # 3. O'rtacha omil qiymatlari
    avg_factors = {}
    for f in SENTIMENT_FACTORS:
        vals = all_factor_vals[f]
        val = np.mean(vals) if vals else 0
        if val < 0.20:  # Shovqin filtri
            val = 0.0
        avg_factors[f] = val
    
    # 4. Vaznli regressiya formulasi
    baseline = 4.0358
    predicted = baseline
    for f, weight in IMPACT_WEIGHTS.items():
        if f in avg_factors:
            predicted += weight * avg_factors[f]
    predicted = max(1.0, min(5.0, predicted))
    
    # 5. 10 ballik shkalaga o'tkazish
    health_score = (predicted / 5.0) * 10
    
    # 6. Jazo tizimi (defekt darajasiga asoslangan)
    if max_problem_rate > 0.15:
        health_score = min(health_score, 5.0)   # Tanqidiy
    elif max_problem_rate > 0.10:
        health_score = min(health_score, 6.0)   # Yuqori
    elif max_problem_rate > 0.05:
        health_score = min(health_score, 7.5)   # O'rtacha
    
    return {
        'health_score': round(health_score, 2),
        'predicted_rating': round(predicted, 2),
        'factor_breakdown': problem_rates,
        'top_problems': problems[:5]
    }
```

**Izoh:** Funksiya har bir sharhni XLM-RoBERTa orqali tahlil qilib, omillarni aniqlaydi. Keyin yulduzcha reyting bilan birlashtirib, ijobiy/salbiy eslatmalarni ajratadi. Vaznli regressiya formulasi ($\hat{R} = 4.036 + \sum w_i \cdot \bar{f}_i$) orqali bashorat qilingan reytingni hisoblab, 10 ballik shkalaga o'tkazadi. Jazo tizimi yuqori defekt darajasida ballni cheklaydi.

---

## ILOVA B. Sun'iy intellekt yordamida tasniflash uchun prompt matni

Quyida Llama 3.3-70B modeli uchun ishlatilgan to'liq prompt tizimi keltirilgan.

Manba fayl: `_archive/labeling_pipeline/prompt_templates_simple.py`.

### B.1. Tizim prompt (System Prompt)

```
Sen professional ma'lumotlar tahlilchisi va NLP mutaxassisisiz.

Vazifa: O'zbekiston bozorida xaridorlar sharhlarini tahlil qiling.
JAVOBDA BARCHA BERILGAN ID LAR BO'LISHI SHART. HECH QAYSI BIRINI 
TUSHIRIB QOLDIRMANG.

MUHIM QOIDALAR:
1. Javobni FAQAT JSON formatida qaytaring
2. Har bir faktor uchun: 1 (agar mention qilingan) yoki 0 (agar yo'q)
3. Uzbekcha va ruscha matnlarni tushunasiz
```

### B.2. Namunali misollar (Few-Shot Examples)

**1-misol (⭐5, ijobiy sharh):**

| Sharh matni | `"Sifat alo, juda yaxshi material. Narxi ham arzon. Kuryer tez yetkazib berdi."` |
|---|---|
| Rating | 5 ⭐ |
| product_quality | **1** |
| price_value | **1** |
| logistics_delivery | **1** |
| packaging_condition | 0 |
| accuracy_expectation | 0 |
| seller_service | 0 |
| specifications | 0 |
| product_defects | 0 |

**2-misol (⭐1, salbiy sharh):**

| Sharh matni | `"Rasmda boshqa rang edi. Qadoq ezilgan. Sotuvchi javob bermadi."` |
|---|---|
| Rating | 1 ⭐ |
| product_quality | 0 |
| price_value | 0 |
| logistics_delivery | 0 |
| packaging_condition | **1** |
| accuracy_expectation | **1** |
| seller_service | **1** |
| specifications | 0 |
| product_defects | **1** |

**3-misol (⭐4, aralash sharh):**

| Sharh matni | `"O'lcham kichik keldi, lekin sifat yaxshi."` |
|---|---|
| Rating | 4 ⭐ |
| product_quality | **1** |
| price_value | 0 |
| logistics_delivery | 0 |
| packaging_condition | 0 |
| accuracy_expectation | 0 |
| seller_service | 0 |
| specifications | **1** |
| product_defects | 0 |

**Izoh:** Misollar ataylab ijobiy (1-misol), salbiy (2-misol) va aralash (3-misol) holatlarni qamrab oladi. 3-misolda bitta sharh bir vaqtning o'zida ijobiy (sifat=1) va salbiy (xususiyatlar=1) omillarni o'z ichiga olishi mumkinligi ko'rsatilgan — bu ko'p omilli (multi-label) tasniflashning mohiyatini o'rgatadi.

### B.3. JSON chiqish formati (Output Schema)

```json
{
  "results": [
    {
      "id": "<sharh_id>",
      "product_quality": 0,
      "price_value": 0,
      "logistics_delivery": 0,
      "packaging_condition": 0,
      "accuracy_expectation": 0,
      "seller_service": 0,
      "specifications": 0,
      "product_defects": 0
    }
  ]
}
```

**Validatsiya qoidalari:**
- `"results"` kaliti mavjud va ro'yxat turi bo'lishi shart
- Har bir natijada `"id"` maydoni bo'lishi shart
- Barcha 8 ta omil maydoni mavjud bo'lishi shart
- Har bir omil qiymati faqat 0 yoki 1 bo'lishi shart
- Validatsiya muvaffaqiyatsiz bo'lsa → 3 martagacha qayta urinish

**API sozlamalari:**
- Model: Llama 3.3-70B Versatile
- API: Groq (yuqori tezlikli inference)
- Harorat (Temperature): 0.1 (determinizm uchun)
- Partiya hajmi (Batch size): 5 ta sharh
- JSON rejimi: `response_format={"type": "json_object"}`

---

## ILOVA C. Dasturiy ta'minotning ekran tasvirlari

*Eslatma: Quyidagi ekran tasvirlari tezisning bosma nusxasiga kiritilishi kerak. Ularni olish uchun dasturni ishga tushiring va quyidagi sahifalarning screenshot larini oling:*

### C.1. Mahsulot tahlilchisi (Product Analyzer) sahifasi

URL kiritish maydoni va asosiy tahlil natijalarini ko'rsatadigan sahifa. Bu sahifada sotuvchi Uzum mahsulot havolasini kiritadi va tizim real vaqtda tahlilni boshlaydi.

*Screenshot oling: `http://localhost:5173/product-analyzer`*

### C.2. Salomatlik kartochkasi (Health Scorecard)

10 ballik shkalada mahsulotning umumiy salomatlik ko'rsatkichini ko'rsatadigan komponent. Bashorat qilingan reyting va haqiqiy reyting taqqoslangan.

*Screenshot oling: HealthScorecard komponenti product-analyzer sahifasida ko'rinadi.*

### C.3. Omillar diagrammasi (Factor Breakdown Chart)

8 ta omilning har biri bo'yicha muammoli darajalarni vizual ko'rsatuvchi diagramma. Qaysi omillar tanqidiy darajada ekanligi rangli belgilar bilan ko'rsatilgan.

*Screenshot oling: FactorBreakdownChart komponenti product-analyzer sahifasida ko'rinadi.*

### C.4. AI maslahatchi (AI Consultant)

Sun'iy intellekt tomonidan tayyorlangan tavsiyalar — qaysi omillarni yaxshilash kerakligi va aniq harakatlar rejasi.

*Screenshot oling: AIConsultantCard komponenti product-analyzer sahifasida ko'rinadi.*

---

## ILOVA D. Manba kodining repositoriyasi

Ushbu bitiruv malakaviy ishida yaratilgan barcha dasturiy kodlar, ma'lumotlar to'plami, o'qitilgan model vaznlari va hujjatlar quyidagi GitHub repositoriyasida to'liq holda mavjud:

**Repositoriya havolasi:**

```
https://github.com/<USERNAME>/BMI_V4_NLP
```

*(Eslatma: `<USERNAME>` o'rniga haqiqiy GitHub foydalanuvchi nomingizni qo'ying. Agar repositoriya hali yaratilmagan bo'lsa, uni himoya oldidan GitHub ga yuklang.)*

**Repositoriya tuzilmasi:**

```
BMI_V4_NLP/
├── src/                          # Asosiy Python kodlar
│   ├── improved_trainer.py       # XLM-RoBERTa v2 trainer (Focal Loss)
│   ├── inference_api.py          # Model inference API
│   ├── b2b_engine.py             # B2B tahlil mexanizmi
│   ├── evaluate_v2.py            # Baholash va kalibrlash
│   ├── baseline_comparison.py    # TF-IDF bazaviy model
│   ├── gold_standard_validation.py  # Oltin standart validatsiyasi
│   └── market_analyzer.py        # Bozor tahlilchisi
│
├── frontend/                     # React frontend
│   └── src/
│       ├── App.tsx               # Asosiy ilova
│       ├── pages/                # Sahifalar (5 ta)
│       ├── components/           # UI komponentlar (72 ta)
│       └── i18n.ts               # Ko'p tillilik (UZ/RU/EN)
│
├── models/
│   └── uzum_nlp_v2/              # O'qitilgan model vaznlari
│       ├── model.pt              # PyTorch checkpoint (~1.1 GB)
│       ├── config.json           # Model konfiguratsiyasi
│       └── thresholds.json       # Optimal chegaralar
│
├── data/
│   ├── processed/                # Qayta ishlangan ma'lumotlar
│   │   ├── uzum_train.csv        # 12,399 ta sharh
│   │   ├── uzum_val.csv          # 1,550 ta sharh
│   │   └── uzum_test.csv         # 1,550 ta sharh
│   └── uzum_labeled.csv          # To'liq etiketlangan ma'lumotlar
│
├── docs/                         # Hujjatlar
│   ├── model_comparison.json     # Model natijalarini qiyoslash
│   └── labeling_methodology.md   # Etiketlash metodologiyasi
│
└── _archive/
    └── labeling_pipeline/        # AI etiketlash pipeline
        ├── ai_labeller_v3_simple.py
        ├── prompt_templates_simple.py
        └── config.py
```
