# 🔍 Uzum B2B Seller Intelligence Platform

> AI-powered product analytics for Uzum marketplace sellers — real-time sentiment analysis, category benchmarking, and strategic business recommendations.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?logo=typescript&logoColor=white)

---

## ✨ Key Features

- **🧠 XLM-RoBERTa V3 NLP** — Multi-task deep learning model analyzes Uzbek/Russian product reviews
- **📊 8-Factor Sentiment Analysis** — Quality, Price, Delivery, Packaging, Accuracy, Service, Specs, Defects
- **❤️ Health Score** — Single 1-10 product health metric
- **📈 Category Benchmarking** — Compare against category averages
- **🤖 AI Business Consultant** — Groq AI generates actionable recommendations, ROI forecasts, and quick wins
- **🎯 Market Simulator** — Interactive "what-if" scenarios for rating improvements
- **🌐 Trilingual UI** — Full Uzbek, Russian, and English interface
- **🌗 Dark/Light Mode** — Theme toggle with polished aesthetics

---

## 📋 Project Structure

```
BMI_V4_NLP/
├── backend/                    # FastAPI REST API server
│   ├── main.py                 # API endpoints, scraping, NLP inference
│   ├── requirements.txt        # Python dependencies
│   └── data/b2b_cache/         # Product analysis cache (auto-generated)
│
├── frontend/                   # React + Vite + TypeScript SPA
│   ├── src/
│   │   ├── pages/              # MarketOverview, ProductAnalyzer, Simulation
│   │   ├── components/         # UI components (shadcn/ui based)
│   │   │   ├── dashboard/      # Charts, tornado, scatter matrix
│   │   │   ├── product-analyzer/ # Health card, flags, AI consultant
│   │   │   ├── simulation/     # Market simulator controls
│   │   │   └── ui/             # shadcn/ui primitives
│   │   ├── services/api.ts     # API client (React Query hooks)
│   │   ├── i18n.ts             # Translations (uz/ru/en)
│   │   └── types/              # TypeScript interfaces
│   ├── package.json
│   └── vite.config.ts
│
├── src/                        # Core ML & analytics modules
│   ├── b2b_engine.py           # Pipeline orchestrator (scrape → NLP → analytics)
│   ├── consultant.py           # Groq AI recommendation engine
│   ├── inference_api.py        # UzumBERT sentiment inference
│   ├── market_analyzer.py      # Category benchmarking & statistics
│   ├── satisfaction_formula.py # Health score calculation
│   ├── final_trainer.py        # Model training pipeline
│   └── verify_setup.py         # Setup verification utility
│
├── models/                     # Trained model weights (git-ignored, ~1.1GB)
├── data/                       # Training datasets (git-ignored)
├── docs/                       # Documentation
│   └── B2B_PORTAL_GUIDE.md
├── .env.example                # Environment template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Chromium browser binaries (for automated token extraction)
- Trained XLM-RoBERTa model weights placed in `models/uzum_nlp_v3/`

### 1. Clone & Setup Backend

```bash
git clone https://github.com/YOUR_USERNAME/BMI_V4_NLP.git
cd BMI_V4_NLP

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install Python dependencies
pip install -r backend/requirements.txt

# Install Playwright browser binaries (Required for Uzum Scraper)
playwright install chromium
```

### 2. Configure Credentials

Copy the template:

```bash
cp data.example.txt data.txt
```

Edit `data.txt`:
```
GROQ_API_KEY=gsk_your_api_key_goes_here
```
> **Note:** The `UZUM_TOKEN` is automatically extracted by the Playwright headless browser at runtime. You do not need to hunt for it manually!

### 3. Start Backend

```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Docs at `/docs`.

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5173`

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + TypeScript | Single-page application |
| **Styling** | Tailwind CSS + shadcn/ui | Component library & design system |
| **Build** | Vite | Fast dev server & bundler |
| **State** | React Query (TanStack) | Server state management |
| **i18n** | react-i18next | Trilingual interface (uz/ru/en) |
| **Backend** | FastAPI + Uvicorn | REST API server |
| **NLP Model** | UzumBERT (transformers) | 8-factor sentiment classification |
| **AI Engine** | Groq API (LLaMA) | Business recommendations |
| **Scraping** | Uzum GraphQL API | Live product data & reviews |
| **Data** | Pandas, NumPy | Data processing & analytics |

---

## 🔄 Data Pipeline

```
┌─────────────────┐
│  Product URL     │  User pastes Uzum product link
└────────┬────────┘
         ▼
┌─────────────────┐
│  Uzum GraphQL   │  Fetch product info, reviews, ratings
│  API Scraper    │  via authenticated GraphQL queries
└────────┬────────┘
         ▼
┌─────────────────┐
│  UzumBERT NLP   │  Classify each review into 8 sentiment
│  Inference      │  factors with confidence scores
└────────┬────────┘
         ▼
┌─────────────────┐
│  Health Score    │  Calculate 1-10 health score,
│  Calculator     │  predicted rating, factor breakdown
└────────┬────────┘
         ▼
┌─────────────────┐
│  Category       │  Compare against category averages,
│  Benchmarking   │  generate positive/warning flags
└────────┬────────┘
         ▼
┌─────────────────┐
│  Groq AI        │  Generate top problems, solution steps,
│  Consultant     │  ROI forecast, quick wins, success metrics
└────────┬────────┘
         ▼
┌─────────────────┐
│  React Frontend │  Interactive dashboard with charts,
│  Dashboard      │  scorecard, and recommendations
└─────────────────┘
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Analyze a product by URL |
| `POST` | `/api/consultant` | Get AI business recommendations |
| `GET` | `/api/market-overview` | Category-level market statistics |
| `GET` | `/api/health` | Server health check |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **401 Unauthorized** | Update `UZUM_TOKEN` in `data.txt` — tokens expire periodically |
| **No reviews found** | Verify the URL format: `https://uzum.uz/uz/product/{name}-{ID}` |
| **Groq API error** | Check API key validity at [console.groq.com](https://console.groq.com) |
| **Model not found** | Ensure the model binaries are inside `models/uzum_nlp_v3/` |
| **CORS error** | Backend must run on port 8000, frontend on 5173 |

---

## 📊 Model Training

To retrain the UzumBERT model:

```bash
# Prepare labeled data in data/uzum_labeled.csv
python src/final_trainer.py
```

The trainer will:
1. Preprocess & split data (train/val/test)
2. Fine-tune XLM-RoBERTa with custom satisfaction head
3. Save model to `models/uzum_nlp_v3/`
4. Output precision, recall, F1-score metrics

---

## 📝 License

This project is for educational and research purposes.

---

**Built with ❤️ for Uzum marketplace sellers**
