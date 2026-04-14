# Dataset Labeling Methodology

## Overview

23,987 Uzum marketplace reviews were labeled across **8 satisfaction factors** using an automated AI labeling pipeline powered by **Llama 3.3-70B** via the Groq API.

## The 8 Satisfaction Factors

| Factor | Description (Uzbek) | Examples |
|--------|---------------------|----------|
| `product_quality` | Mahsulot sifati, material, ishlab chiqarish | "Sifat alo, material yaxshi" / "Brak keldi, sifatsiz" |
| `price_value` | Narx-sifat munosabati, hamyonboplik | "Arzon, puliga arziydi" / "Qimmat, arzimaydi" |
| `logistics_delivery` | Yetkazib berish, kuryer xizmati | "Kuryer tez yetkazdi" / "5 kun kutdim" |
| `packaging_condition` | Qadoq holati, mahsulot butunligi | "Qadoq yaxshi" / "Korobka ezilgan" |
| `accuracy_expectation` | Rasmga mosligi, kutilgan vs haqiqat | "Rasmdagidek keldi" / "Boshqa rang" |
| `seller_service` | Sotuvchi munosabati, xizmat sifati | "Sotuvchi yordam berdi" / "Javob bermadi" |
| `specifications` | O'lcham, texnik parametrlar | "O'lcham to'g'ri keldi" / "Kichik keldi" |
| `product_defects` | Nuqsonlar, shikastlar, xavfli holatlar | "Singan keldi" / "Qiziyapti, xavfli" |

## Labeling Pipeline

### Model & Configuration
- **Model:** Llama 3.3-70B Versatile (via Groq API)
- **Temperature:** 0.1 (low for consistency)
- **Max tokens:** 1000 per batch
- **Batch size:** 5 reviews per API call
- **Output format:** Structured JSON with schema validation

### Prompt Design

The prompt uses **3 few-shot examples** in Uzbek and a structured system prompt:

```
System: "Sen professional ma'lumotlar tahlilchisi va NLP mutaxassisisiz.
Vazifa: O'zbekiston bozorida xaridorlar sharhlarini tahlil qiling.
MUHIM QOIDALAR:
1. Javobni FAQAT JSON formatida qaytaring
2. Har bir faktor uchun: 1 (agar mention qilingan) yoki 0 (agar yo'q)
3. Uzbekcha va ruscha matnlarni tushunasiz"
```

Each review is presented with its text and rating. The model outputs:
```json
{
  "results": [
    {
      "id": 123,
      "product_quality": 1,
      "price_value": 0,
      "logistics_delivery": 1,
      ...
    }
  ]
}
```

### Quality Control
1. **Schema validation** — Every response checked for correct JSON structure, valid IDs, and binary (0/1) values
2. **Retry logic** — Failed/invalid responses retried up to 3 times with API key rotation
3. **Anomaly detection** — Flagged reviews with >5 factors mentioned or confidence below 0.6
4. **Smart text truncation** — Cons (40% priority), Content (35%), Pros (25%)

### Why LLM Labeling?

Traditional keyword-based labeling cannot understand context. For example:
- "Sifat yaxshi emas" (Quality is **not** good) — keywords see "sifat" + "yaxshi" and might label quality=positive
- Llama 3.3-70B understands negation, sarcasm, and context in both Uzbek and Russian

## Class Distribution (Training Set — 15,499 reviews after filtering)

| Factor | Positive | Negative | Balance Ratio |
|--------|----------|----------|---------------|
| product_quality | 10,672 (69%) | 4,827 (31%) | 2.2:1 |
| price_value | 4,279 (28%) | 11,220 (72%) | 1:2.6 |
| logistics_delivery | 2,272 (15%) | 13,227 (85%) | 1:5.8 |
| seller_service | 2,924 (19%) | 12,575 (81%) | 1:4.3 |
| product_defects | 2,319 (15%) | 13,180 (85%) | 1:5.7 |
| packaging_condition | 1,363 (9%) | 14,136 (91%) | 1:10.4 |
| accuracy_expectation | 1,362 (9%) | 14,137 (91%) | 1:10.4 |
| specifications | 1,263 (8%) | 14,236 (92%) | 1:11.3 |

> **Note:** Weighted BCE loss with per-factor class weights was used during model training to handle this imbalance.

## Data Splits

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 12,399 | 80% |
| Validation | 1,550 | 10% |
| Test | 1,550 | 10% |

## Reference Files

The original labeling pipeline code is preserved in `_archive/labeling_pipeline/`:
- `ai_labeller_v3_simple.py` — Main labeler with API key rotation
- `prompt_templates_simple.py` — Prompt structure with few-shot examples
- `config.py` — Factor definitions, keywords, API configuration
- `validator.py` — Quality validation and anomaly detection
