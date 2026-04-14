"""
=============================================================================
🚀 UZUM INTELLIGENCE HUB - FASTAPI BACKEND
=============================================================================

REST API backend for React frontend integration with ML models.

Endpoints:
    - GET  /api/health          - Health check
    - POST /api/analyze         - Analyze single review text
    - POST /api/analyze/batch   - Batch analysis
    - POST /api/simulate        - Rating simulation
    - GET  /api/market-overview - Dashboard statistics
    - GET  /api/coefficients    - Model quality coefficients
    - POST /api/analyze/product - Complete product analysis pipeline
    - POST /api/consultant      - AI business recommendations
    - POST /api/health-score    - Health score calculation
    - POST /api/simulate/impact - Impact simulator

Run:
    uvicorn backend.main:app --reload --port 8000

=============================================================================
"""

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime

# ── Structured logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("uzum.api")
# Suppress noisy third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
# ────────────────────────────────────────────────────────────────────

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Rate limiting (slowapi) ──────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    limiter = None
    RATE_LIMIT_AVAILABLE = False
    logger.warning("slowapi not installed — rate limiting disabled. Run: pip install slowapi")
# ────────────────────────────────────────────────────────────────────

# ── SQLite result cache ───────────────────────────────────────────────
from backend.cache import AnalysisCache, extract_product_id
analysis_cache = AnalysisCache(ttl_hours=24)
# ────────────────────────────────────────────────────────────────────

# Import ML inference API
try:
    from src.inference_api import  UZUMInferenceAPI, FACTORS
    ML_AVAILABLE = True
except Exception as e:
    print(f"⚠️ ML Model not available: {e}")
    ML_AVAILABLE = False
    FACTORS = [
        'product_quality', 'price_value', 'logistics_delivery', 'packaging_condition',
        'accuracy_expectation', 'seller_service', 'specifications', 'product_defects'
    ]

# Import B2B engine and consultant
try:
    from src.b2b_engine import B2BSellerEngine, ProductHealthAnalyzer, IMPACT_WEIGHTS
    from src.consultant import get_consultant_advice, build_analysis_context
    B2B_AVAILABLE = True
except Exception as e:
    print(f"⚠️ B2B Engine not available: {e}")
    B2B_AVAILABLE = False

# Import Market Analyzer
try:
    from src.market_analyzer import MarketAnalyzer
    MARKET_ANALYZER_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Market Analyzer not available: {e}")
    MARKET_ANALYZER_AVAILABLE = False


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Review text to analyze")
    confidence_threshold: float = Field(0.5, ge=0, le=1)

class AnalyzeBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    confidence_threshold: float = Field(0.5, ge=0, le=1)

class FactorPrediction(BaseModel):
    factor: str
    prediction: int
    confidence: float
    label: str  # 'positive', 'negative', 'neutral'

class AnalyzeResponse(BaseModel):
    text: str
    text_length: int
    script_type: str
    factors: List[FactorPrediction]
    positive_count: int
    total_factors: int
    overall_sentiment: str
    timestamp: str

class SimulationRequest(BaseModel):
    defect_rate: float = Field(0, ge=0, le=100)
    delivery_speed: int = Field(3, ge=1, le=7)
    response_time: int = Field(4, ge=1, le=24)

class SimulationResponse(BaseModel):
    projected_rating: float
    rating_change: float
    confidence: str
    factors_impact: dict

class StatCard(BaseModel):
    id: str
    title: str
    value: str
    trend: Optional[str] = None
    trend_type: Optional[str] = None
    badge: Optional[str] = None
    badge_type: Optional[str] = None
    icon: str

class MarketOverviewResponse(BaseModel):
    stats: List[StatCard]
    total_reviews: int
    avg_rating: float
    last_updated: str

# New request models for feature parity endpoints
class ProductAnalysisRequest(BaseModel):
    product_url: str = Field(..., description="Uzum product URL or product ID")
    use_cache: bool = Field(default=True, description="Whether to use cached results")
    force_refresh: bool = Field(default=False, description="Force fresh analysis ignoring cache")
    uzum_token: Optional[str] = Field(default=None, description="Uzum API token (optional if in data.txt)")

class ProductInfo(BaseModel):
    title: Optional[str]
    actual_rating: Optional[float]
    total_reviews: Optional[int]
    analyzed_reviews: Optional[int]
    review_count: Optional[int]
    price: Optional[float]
    seller_name: Optional[str]
    seller_rating: Optional[float]
    category: Optional[str]
    analysis_note: Optional[str] = None

class HealthAnalysis(BaseModel):
    health_score: Optional[float] = Field(None, alias="healthScore")
    predicted_rating: Optional[float] = None
    factor_breakdown: Optional[dict] = None
    top_problems: Optional[List[dict]] = None
    score_explanation: Optional[str] = None
    actual_rating: Optional[float] = None
    review_count: Optional[int] = None
    status: Optional[str] = None
    message: Optional[str] = None
    
    class Config:
        populate_by_name = True
        extra = "ignore"

class Benchmark(BaseModel):
    status: Optional[str] = None
    category: Optional[str] = None
    avg_health: Optional[float] = None
    avg_rating: Optional[float] = None
    comparison: Optional[str] = None
    percentile: Optional[Union[float, str]] = None
    performance: Optional[str] = None
    rating_diff: Optional[float] = None
    benchmark_message: Optional[str] = None
    
    class Config:
        extra = "ignore"

class ProductAnalysisResponse(BaseModel):
    status: str
    product_id: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    from_cache: Optional[bool] = None
    product_info: Optional[ProductInfo] = None
    health_analysis: Optional[HealthAnalysis] = None
    benchmark: Optional[Benchmark] = None
    raw_reviews: Optional[List[dict]] = None
    
    class Config:
        extra = "ignore" 

class HealthScoreRequest(BaseModel):
    factors: dict = Field(..., description="Dictionary of factor names to confidence values (0-1)")
    review_count: int = Field(default=100, description="Number of reviews analyzed")

class ImpactSimulationRequest(BaseModel):
    predicted_rating: float = Field(..., description="Current predicted rating")
    problems: List[dict] = Field(..., description="List of problems with factor, severity, impact_weight")
    improvements: dict = Field(..., description="Dictionary of factor to improvement percentage (0-100)")

class QualityCoefficient(BaseModel):
    id: str
    factor_name: str
    weight_impact: float
    baseline_coeff: float
    simulated_coeff: float
    status: str
    last_updated: str

    
class ConsultantRequest(BaseModel):
    analysis: dict
    language: str = "uz"

# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app):
    """Application lifespan: load ML model and warm up cache on startup"""
    logger.info("Starting Uzum Intelligence Hub API...")
    get_inference_api()
    # Clean up stale cache entries from previous runs
    deleted = analysis_cache.cleanup_expired()
    if deleted:
        logger.info(f"Startup cache cleanup: removed {deleted} expired entries")
    logger.info("API ready")
    yield

app = FastAPI(
    title="Uzum Intelligence Hub API",
    lifespan=lifespan,
    description="ML-powered review analysis and market intelligence API",
    version="2.0.0"   # bumped: V3 model + production hardening
)

# Attach rate limiter
if RATE_LIMIT_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — restrict to known origins (not wildcard *)
# Add your production domain here once deployed (e.g. https://uzum-intel.com)
ALLOWED_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:5174",
    "http://localhost:8080",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# =============================================================================
# GLOBAL MODEL INSTANCE
# =============================================================================

inference_api = None

def get_inference_api():
    global inference_api
    if inference_api is None and ML_AVAILABLE:
        try:
            # Use absolute path relative to project root
            project_root = Path(__file__).parent.parent
            model_path = project_root / 'models' / 'uzum_nlp_v3'   # ← V3: Multi-Task + Focal Loss
            print(f"Loading model from: {model_path}")
            inference_api = UZUMInferenceAPI(str(model_path))
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
    return inference_api

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_analyze_response(result: dict) -> AnalyzeResponse:
    """Format ML prediction result to API response"""
    factors = []
    positive_count = 0
    
    for factor, pred_info in result['predictions'].items():
        conf = pred_info['confidence']
        pred = pred_info['prediction']
        
        if pred == 1:
            positive_count += 1
            label = 'positive'
        elif conf < 0.3:
            label = 'negative'
        else:
            label = 'neutral'
        
        factors.append(FactorPrediction(
            factor=factor,
            prediction=pred,
            confidence=round(conf, 4),
            label=label
        ))
    
    # Determine overall sentiment
    ratio = positive_count / len(FACTORS)
    if ratio > 0.6:
        overall = "positive"
    elif ratio < 0.4:
        overall = "negative"
    else:
        overall = "neutral"
    
    return AnalyzeResponse(
        text=result['text'],
        text_length=result['text_length'],
        script_type=result['script_type'],
        factors=factors,
        positive_count=positive_count,
        total_factors=len(FACTORS),
        overall_sentiment=overall,
        timestamp=datetime.now().isoformat()
    )

def _mock_analyze_response(text: str) -> AnalyzeResponse:
    """Generate mock response when ML model is not available"""
    import random
    
    factors = []
    positive_count = 0
    
    for factor in FACTORS:
        conf = random.uniform(0.3, 0.9)
        pred = 1 if conf > 0.5 else 0
        if pred == 1:
            positive_count += 1
        
        factors.append(FactorPrediction(
            factor=factor,
            prediction=pred,
            confidence=round(conf, 4),
            label='positive' if pred == 1 else 'neutral'
        ))
    
    return AnalyzeResponse(
        text=text,
        text_length=len(text),
        script_type="latin" if all(ord(c) < 0x0400 for c in text if c.isalpha()) else "cyrillic",
        factors=factors,
        positive_count=positive_count,
        total_factors=len(FACTORS),
        overall_sentiment="positive" if positive_count > 4 else "neutral",
        timestamp=datetime.now().isoformat()
    )

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    api = get_inference_api()
    model_version = "none"
    if api is not None:
        model_version = "v3 (Multi-Task)" if api.is_v3 else ("v2 (Focal Loss)" if api.is_v2 else "v1")
    return {
        "status": "healthy",
        "version": "2.0.0",
        "ml_available": ML_AVAILABLE,
        "model_loaded": api is not None,
        "model_version": model_version,
        "rate_limiting": RATE_LIMIT_AVAILABLE,
        "cache": analysis_cache.stats(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/model-performance")
def model_performance():
    """Return model comparison metrics from docs/model_comparison.json"""
    import json as json_lib
    comparison_path = Path(__file__).parent.parent / "docs" / "model_comparison.json"
    if not comparison_path.exists():
        raise HTTPException(status_code=404, detail="Model comparison data not found. Run baseline_comparison.py first.")
    with open(comparison_path, 'r', encoding='utf-8') as f:
        return json_lib.load(f)

@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_review(request: AnalyzeRequest):
    """Analyze a single review text"""
    api = get_inference_api()
    
    if api is None:
        # Return mock data if model not available
        return _mock_analyze_response(request.text)
    
    try:
        result = api.predict(request.text, request.confidence_threshold)
        return _format_analyze_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/batch")
def analyze_batch(request: AnalyzeBatchRequest):
    """Analyze multiple reviews"""
    api = get_inference_api()
    
    if api is None:
        return {"results": [_mock_analyze_response(t) for t in request.texts]}
    
    try:
        results = api.predict_batch(request.texts, request.confidence_threshold)
        return {"results": [_format_analyze_response(r) for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/simulate", response_model=SimulationResponse)
def run_simulation(request: SimulationRequest):
    """Run rating simulation based on parameters"""
    base_rating = 4.6
    
    # Calculate impacts
    defect_impact = -request.defect_rate * 0.05
    delivery_impact = (4 - request.delivery_speed) * 0.05
    response_impact = (12 - request.response_time) * 0.01
    
    projected = max(1, min(5, base_rating + defect_impact + delivery_impact + response_impact))
    change = projected - base_rating
    
    return SimulationResponse(
        projected_rating=round(projected, 2),
        rating_change=round(change, 2),
        confidence="High" if abs(change) < 0.3 else "Medium",
        factors_impact={
            "defect_rate": round(defect_impact, 3),
            "delivery_speed": round(delivery_impact, 3),
            "response_time": round(response_impact, 3)
        }
    )

@app.get("/api/market-overview", response_model=MarketOverviewResponse)
def get_market_overview():
    """Get dashboard market overview statistics from real dataset"""
    analyzer = get_market_analyzer()

    # Try to pull live stats from MarketAnalyzer
    if analyzer is not None:
        try:
            stats_raw = analyzer.get_market_overview()
            total_reviews = stats_raw.get('total_reviews', 0)
            avg_rating = stats_raw.get('avg_rating', 0.0)

            # Build factor impact to find top strength / main problem
            factor_impact = analyzer.get_factor_impact()
            factors = factor_impact.get('factors', [])
            top_killer = factor_impact.get('top_killer', {}).get('name', 'Unknown')
            top_strength = factor_impact.get('top_strength', {}).get('name', 'Unknown')

            stats = [
                StatCard(
                    id="1", title="Total Reviews", value=f"{total_reviews:,}",
                    trend=f"+{stats_raw.get('satisfaction_rate', 0):.0f}% satisfaction",
                    trend_type="up", icon="FileOutput"
                ),
                StatCard(
                    id="2", title="Avg Rating", value=f"{avg_rating:.2f}",
                    trend="Across all products", trend_type="flat", icon="Star"
                ),
                StatCard(
                    id="3", title="Main Problem", value=top_killer.replace('_', ' ').title(),
                    badge="HIGH IMPACT", badge_type="destructive", icon="AlertTriangle"
                ),
                StatCard(
                    id="4", title="Top Strength", value=top_strength.replace('_', ' ').title(),
                    badge="Positive Sentiment", badge_type="success", icon="ThumbsUp"
                ),
            ]
            return MarketOverviewResponse(
                stats=stats,
                total_reviews=total_reviews,
                avg_rating=avg_rating,
                last_updated=datetime.now().isoformat()
            )
        except Exception as e:
            logger.warning(f"MarketAnalyzer failed for market-overview: {e}")

    # Fallback: clearly labelled as demo data so no one is misled
    logger.warning("Serving demo market-overview data — MarketAnalyzer unavailable")
    stats = [
        StatCard(id="1", title="Total Reviews", value="–",
                 badge="Data unavailable", badge_type="destructive", icon="FileOutput"),
        StatCard(id="2", title="Avg Rating", value="–",
                 badge="Data unavailable", badge_type="destructive", icon="Star"),
        StatCard(id="3", title="Main Problem", value="–",
                 badge="Data unavailable", badge_type="destructive", icon="AlertTriangle"),
        StatCard(id="4", title="Top Strength", value="–",
                 badge="Data unavailable", badge_type="destructive", icon="ThumbsUp"),
    ]
    return MarketOverviewResponse(
        stats=stats,
        total_reviews=0,
        avg_rating=0.0,
        last_updated=datetime.now().isoformat()
    )

@app.get("/api/coefficients")
def get_coefficients():
    """Get model quality coefficients"""
    coefficients = [
        QualityCoefficient(
            id="1", factor_name="Product Quality Score",
            weight_impact=0.85, baseline_coeff=0.72, simulated_coeff=0.78,
            status="Active", last_updated="2026-01-26"
        ),
        QualityCoefficient(
            id="2", factor_name="Shipping Reliability",
            weight_impact=0.72, baseline_coeff=0.65, simulated_coeff=0.71,
            status="Active", last_updated="2026-01-25"
        ),
        QualityCoefficient(
            id="3", factor_name="Customer Response Rate",
            weight_impact=0.58, baseline_coeff=0.82, simulated_coeff=0.85,
            status="Active", last_updated="2026-01-24"
        ),
        QualityCoefficient(
            id="4", factor_name="Price Elasticity",
            weight_impact=0.45, baseline_coeff=0.55, simulated_coeff=0.52,
            status="Pending", last_updated="2026-01-23"
        ),
        QualityCoefficient(
            id="5", factor_name="Return Rate Factor",
            weight_impact=-0.32, baseline_coeff=0.12, simulated_coeff=0.10,
            status="Active", last_updated="2026-01-22"
        ),
    ]
    return {"coefficients": [c.model_dump() for c in coefficients]}

@app.post("/api/consultant")
def get_consultant_recommendations(request: ConsultantRequest):
    """
    Get AI business recommendations from Groq
    
    Input: Complete analysis dict from product analysis
    Output: Groq AI verdict with top problems, solutions, ROI forecast
    """
    if not B2B_AVAILABLE:
        raise HTTPException(status_code=503, detail="B2B Engine not available")
    
    try:
        # Sanitize analysis data — None values from cached/live analysis cause crashes
        analysis = request.analysis or {}
        if 'health_analysis' in analysis and analysis['health_analysis']:
            ha = analysis['health_analysis']
            if ha.get('top_problems') is None:
                ha['top_problems'] = []
            if ha.get('factor_breakdown') is None:
                ha['factor_breakdown'] = {}
        else:
            analysis['health_analysis'] = {'top_problems': [], 'factor_breakdown': {}, 'health_score': 0, 'predicted_rating': 0}
        
        if analysis.get('product_info') is None:
            analysis['product_info'] = {}
        if analysis.get('benchmark') is None:
            analysis['benchmark'] = {}
        
        verdict = get_consultant_advice(analysis, language=request.language)
        if verdict is None:
            return {
                'status': 'fallback',
                'message': 'Consultant xizmati hozirda mavjud emas',
                'consultant_name': 'Uzum Business Consultant (Offline)',
                'top_problems': [],
                'overall_verdict': "Tahlil ma'lumotlari to'liq emas yoki Groq API ulanishi yo'q"
            }
        return verdict
    except Exception as e:
        logger.error(f"Consultant error: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'consultant_name': 'Uzum Business Consultant (Offline)',
            'top_problems': [],
            'overall_verdict': f"Xatolik yuz berdi: {str(e)}"
        }


@app.post("/api/analyze/product", response_model=ProductAnalysisResponse)
def analyze_product(request: ProductAnalysisRequest, lang: str = "uz"):
    """
    Complete product analysis pipeline.

    - Checks SQLite cache first (24h TTL) unless force_refresh=True
    - Runs full scrape + ML inference on cache miss
    - Writes result back to cache
    - Rate limited: 10 requests/minute per IP
    """
    if not B2B_AVAILABLE:
        raise HTTPException(status_code=503, detail="B2B Engine not available")

    product_id = extract_product_id(request.product_url)
    logger.info(f"Product analysis requested: {product_id} (url={request.product_url[:60]})")

    # ── Cache read ──────────────────────────────────────────────────
    if request.use_cache and not request.force_refresh:
        cached = analysis_cache.get(product_id)
        if cached is not None:
            logger.info(f"Returning cached result for {product_id}")
            try:
                return ProductAnalysisResponse(**cached)
            except Exception as e:
                logger.warning(f"Cached result schema mismatch for {product_id}: {e} — recomputing")

    # ── Token resolution ────────────────────────────────────────────
    uzum_token = request.uzum_token
    if not uzum_token:
        try:
            paths_to_check = [
                Path(__file__).parent.parent / 'data.txt',
                Path(__file__).parent.parent / '_archive' / 'data.txt',
            ]
            for data_path in paths_to_check:
                if data_path.exists():
                    with open(data_path, 'r') as f:
                        for line in f:
                            if line.startswith('UZUM_TOKEN='):
                                uzum_token = line.split('=', 1)[1].replace('Bearer ', '').strip()
                                break
                if uzum_token:
                    break
        except Exception as e:
            logger.warning(f"Error reading data.txt: {e}")

    # The B2BSellerEngine will auto-extract token via Playwright if uzum_token is None
    if not uzum_token:
        logger.info("No Uzum token provided in request or data.txt. Engine will use Playwright for auto-extraction.")

    # ── Full computation ────────────────────────────────────────────
    try:
        engine = B2BSellerEngine(uzum_token)
        use_engine_cache = request.use_cache and not request.force_refresh
        analysis = engine.analyze_product(
            request.product_url,
            use_cache=use_engine_cache,
            language=lang
        )

        # Write to SQLite cache (status=success only)
        if isinstance(analysis, dict) and analysis.get("status") == "success":
            analysis_cache.set(product_id, analysis)
            logger.info(f"Analysis cached for {product_id}")
        elif hasattr(analysis, 'status') and analysis.status == "success":
            analysis_cache.set(product_id, analysis.model_dump())
            logger.info(f"Analysis cached for {product_id}")

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Product analysis failed for {product_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)[:200]}"
        )


@app.post("/api/health-score")
def calculate_health_score(request: HealthScoreRequest):
    """
    Calculate health score from sentiment factors
    
    Input: 
      - factors: {factor_name: confidence_value} (0-1)
      - review_count: number of reviews analyzed
    Output:
      - health_score (1-10)
      - predicted_rating (1-5)
      - top_problems (sorted by impact)
    """
    if not B2B_AVAILABLE:
        raise HTTPException(status_code=503, detail="B2B Engine not available")
    
    try:
        # Calculate predicted rating using the same formula as Streamlit
        baseline_rating = 4.0358
        predicted_rating = baseline_rating
        
        for factor, value in request.factors.items():
            weight = IMPACT_WEIGHTS.get(factor, 0)
            predicted_rating += weight * value
        
        # Clamp to 1-5
        predicted_rating = max(1.0, min(5.0, predicted_rating))
        
        # Calculate health score (1-10)
        health_score = (predicted_rating / 5.0) * 10
        
        # Identify problems
        problems = []
        for factor, value in request.factors.items():
            if value > 0.5 and IMPACT_WEIGHTS.get(factor, 0) < 0:
                problems.append({
                    'factor': factor,
                    'severity': value,
                    'impact_weight': IMPACT_WEIGHTS.get(factor, 0)
                })
        
        # Sort by impact (most negative first)
        problems.sort(key=lambda x: x['impact_weight'])
        
        return {
            'health_score': round(health_score, 2),
            'predicted_rating': round(predicted_rating, 2),
            'top_problems': problems[:3],
            'review_count': request.review_count,
            'formula': 'baseline(4.0358) + Σ(factor × impact_weight)'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate/impact")
def simulate_impact(request: ImpactSimulationRequest):
    """
    Impact simulator matching Streamlit logic
    
    Input:
      - predicted_rating: Current predicted rating
      - problems: [{ factor, severity, impact_weight }]
      - improvements: { factor: improvement_percentage (0-100) }
    Output:
      - new_rating: Projected rating after improvements
      - total_improvement: Rating increase
      - sales_increase: Estimated sales increase %
    """
    try:
        # Calculate total improvement (matches Streamlit formula)
        total_improvement = 0
        for problem in request.problems:
            factor = problem.get('factor', '')
            impact_weight = problem.get('impact_weight', 0)
            improvement_pct = request.improvements.get(factor, 0)
            
            # Improvement = |weight| × (improvement% / 100)
            total_improvement += abs(impact_weight) * (improvement_pct / 100)
        
        # New rating
        new_rating = min(5.0, request.predicted_rating + total_improvement)
        
        # Sales increase estimate (0.1 star = 1.5% sales)
        sales_increase = total_improvement * 15
        
        return {
            'current_rating': round(request.predicted_rating, 2),
            'new_rating': round(new_rating, 2),
            'total_improvement': round(total_improvement, 2),
            'sales_increase_pct': round(sales_increase, 1),
            'confidence': 'High' if total_improvement < 0.5 else 'Medium'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# MARKET OVERVIEW ENDPOINTS
# =============================================================================

# Initialize market analyzer (lazy loading)
market_analyzer = None

def get_market_analyzer():
    """Get or create market analyzer instance"""
    global market_analyzer
    if market_analyzer is None and MARKET_ANALYZER_AVAILABLE:
        try:
            market_analyzer = MarketAnalyzer()
            print("✅ Market Analyzer initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Market Analyzer: {e}")
    return market_analyzer

@app.get("/api/market/categories")
async def get_categories():
    """
    Get list of unique categories sorted by review count
    
    Returns:
        - categories: List of unique category names with review counts
    """
    analyzer = get_market_analyzer()
    
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Market Analyzer not available. Check dataset path."
        )
    
    try:
        categories = analyzer.get_categories()
        return categories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/overview")
async def get_market_overview(category: str = None):
    """
    Get market-level statistics and aggregations
    
    Query Params:
        category: Optional category filter (e.g., 'Elektronika')
    
    Returns:
        - total_reviews: Total number of reviews in dataset
        - avg_rating: Average rating across all reviews
        - total_products: Number of unique products
        - total_sellers: Number of unique sellers
        - categories: Number of categories
        - date_range: Start and end dates of data
        - satisfaction_rate: Percentage of reviews with rating >= 4
        - selected_category: The category filter applied (if any)
    """
    analyzer = get_market_analyzer()
    
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Market Analyzer not available. Check dataset path."
        )
    
    try:
        stats = analyzer.get_market_overview(category=category)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/factor-impact")
async def get_factor_impact(category: str = None):
    """
    Get factor weights and impact analysis
    
    Query Params:
        category: Optional category filter
    
    Returns:
        - factors: List of factors with weights and classifications
        - top_killer: Most negative impact factor
        - top_strength: Most positive impact factor
        - selected_category: The category filter applied (if any)
    """
    analyzer = get_market_analyzer()
    
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Market Analyzer not available. Check dataset path."
        )
    
    try:
        impact = analyzer.get_factor_impact(category=category)
        return impact
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/matrix")
async def get_price_quality_matrix(category: str = None):
    """
    Get Price vs Quality scatter data for Strategic Positioning Matrix
    
    Query Params:
        category: Optional category filter
    
    Returns:
        - points: List of products with price, rating, title
        - median_price: Vertical reference line
        - rating_threshold: Horizontal reference line (4.0)
        - quadrant_counts: Products in each quadrant
    """
    analyzer = get_market_analyzer()
    
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Market Analyzer not available. Check dataset path."
        )
    
    try:
        matrix = analyzer.get_price_quality_matrix(category=category)
        return matrix
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/category-breakdown")
async def get_category_breakdown():
    """
    Get category-level statistics and factor scores
    
    Returns:
        - categories: List of categories with avg ratings and factor scores
    """
    analyzer = get_market_analyzer()
    
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Market Analyzer not available. Check dataset path."
        )
    
    try:
        breakdown = analyzer.get_category_breakdown()
        return breakdown
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/trends")
async def get_market_trends(period: str = 'monthly'):
    """
    Get time-series trends
    
    Args:
        period: 'daily', 'weekly', or 'monthly' (default: 'monthly')
    
    Returns:
        - trends: List of time periods with avg ratings and review counts
    """
    analyzer = get_market_analyzer()
    
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Market Analyzer not available. Check dataset path."
        )
    
    if period not in ['daily', 'weekly', 'monthly']:
        raise HTTPException(
            status_code=400,
            detail="Invalid period. Must be 'daily', 'weekly', or 'monthly'"
        )
    
    try:
        trends = analyzer.get_trends(period)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STARTUP
# =============================================================================

# Startup logic is now handled by the lifespan context manager (see above)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
