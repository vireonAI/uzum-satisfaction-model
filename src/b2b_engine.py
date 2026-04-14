"""
B2B Seller Analytics Engine
Live data pipeline: Scraping → Inference → Analytics → Benchmarking
====================================================================

Qayta ishlanayotgan:
1. Real-time scraper (Uzum API orqali mahsulot sharhlarini yig'ish)
2. Batch inference (UzBERT modelida 8 faktor ajratish)
3. Health score va benchmarking (Kategoriya bo'yicha tahlil)
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
import pickle
import os
import re
from pathlib import Path
import logging

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

UZUM_API_URL = "https://graphql.uzum.uz/"

# Kategoriya mappings
CATEGORIES = {
    "10020": "Elektronika",
    "10004": "Maishiy_Texnika",
    "10014": "Kiyim",
    "10003": "Aksessuarlar",
    "10018": "Uy_Rozgor",
    "10007": "Bolalar_Tovarlari",
    "1821": "Oziq_Ovqat",
    "10011": "Kitoblar"
}

CATEGORY_REVERSE = {v: k for k, v in CATEGORIES.items()}

# 8 ta sentiment factors (Model 1 output)
SENTIMENT_FACTORS = [
    'product_quality',
    'price_value',
    'logistics_delivery',
    'packaging_condition',
    'accuracy_expectation',
    'seller_service',
    'specifications',
    'product_defects'
]

# Impact coefficients from satisfaction_formula.py (Random Forest + OLS)
# These are the actual trained coefficients: Rating = 4.036 + Σ(weight_i × factor_i)
# Source: logs/satisfaction_formula.txt
IMPACT_WEIGHTS = {
    'product_defects': -0.6297,
    'product_quality': 0.2605,
    'accuracy_expectation': -0.0435,
    'seller_service': 0.0218,
    'price_value': 0.0158,
    'logistics_delivery': 0.0108,
    'packaging_condition': -0.0090,
    'specifications': -0.0089
}

# Category benchmarks (Global averages from training data)
def _compute_category_benchmarks() -> dict:
    """Dynamically compute category benchmarks from the actual labeled dataset"""
    import pandas as pd
    try:
        csv_path = Path(__file__).parent.parent / 'data' / 'uzum_labeled.csv'
        if not csv_path.exists():
            csv_path = Path('data/uzum_labeled.csv')
        
        df = pd.read_csv(csv_path)
        benchmarks = {}
        for category in df['Category'].unique():
            cat_df = df[df['Category'] == category]
            benchmarks[category] = {
                'avg_rating': round(float(cat_df['Rating'].mean()), 3),
                'review_count': int(len(cat_df)),
                'defect_rate': round(float(cat_df['product_defects'].mean()), 4) if 'product_defects' in cat_df.columns else 0,
                'quality_rate': round(float(cat_df['product_quality'].mean()), 4) if 'product_quality' in cat_df.columns else 0,
            }
        logger.info(f"✅ Category benchmarks computed dynamically from {len(df)} reviews ({len(benchmarks)} categories)")
        return benchmarks
    except Exception as e:
        logger.warning(f"⚠️ Could not compute benchmarks from dataset: {e}. Using fallback.")
        return {
            'Elektronika': {'avg_rating': 3.765, 'defect_rate': 0.8451},
            'Maishiy_Texnika': {'avg_rating': 3.806, 'defect_rate': 0.7135},
            'Kiyim': {'avg_rating': 3.985, 'quality_rate': 0.5915},
            'Aksessuarlar': {'avg_rating': 3.841, 'defect_rate': 0.5303},
            'Uy_Rozgor': {'avg_rating': 3.964, 'quality_rate': 0.4898},
            'Bolalar_Tovarlari': {'avg_rating': 4.138, 'defect_rate': 0.5796},
            'Oziq_Ovqat': {'avg_rating': 4.261, 'defect_rate': 0.4063},
            'Kitoblar': {'avg_rating': 4.529, 'defect_rate': 0.5119}
        }

CATEGORY_BENCHMARKS = _compute_category_benchmarks()

# Cache configuration
CACHE_DIR = Path('data/b2b_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 86400  # 24 hours


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def extract_product_id_from_url(url: str) -> Optional[str]:
    """
    URL'dan product ID'sini ajratib ol
    example: https://uzum.uz/uz/products/12345-nomlari -> "12345"
    """
    try:
        # Try /products/ first
        if '/products/' in url:
            parts = url.split('/products/')[1].split('-')[0]
            if parts.isdigit():
                return parts
        
        # Try /product/
        if '/product/' in url:
            path = url.split('/product/')[1]
            path = path.split('/')[0]
            match = re.search(r'(\d{5,})', path)
            if match:
                return match.group(1)
        
        # Try direct ID
        if url.isdigit():
            return url
        
        # Last resort
        match = re.search(r'(\d{5,})', url)
        if match:
            return match.group(1)
        
        return None
    except Exception as e:
        logger.error(f"URL parsing error: {e}")
        return None

def get_cache_path(product_id: str) -> Path:
    return CACHE_DIR / f"{product_id}.pkl"

def is_cache_valid(product_id: str) -> bool:
    cache_path = get_cache_path(product_id)
    if not cache_path.exists():
        return False
    file_age = datetime.now().timestamp() - cache_path.stat().st_mtime
    return file_age < CACHE_TTL

def save_cache(product_id: str, data: Dict):
    try:
        cache_path = get_cache_path(product_id)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"✅ Cache saved for product {product_id}")
    except Exception as e:
        logger.error(f"Cache save error: {e}")

def load_cache(product_id: str) -> Optional[Dict]:
    try:
        cache_path = get_cache_path(product_id)
        if not cache_path.exists():
            return None
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"✅ Cache loaded for product {product_id}")
        return data
    except Exception as e:
        logger.error(f"Cache load error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# LIVE DATA PIPELINE: STAGE 1 - SCRAPING
# ═══════════════════════════════════════════════════════════════════════════

class UzumScraper:
    """Uzum API orqali real-time ma'lumot yig'ish"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        # Load logic for token if not passed, handled in get_headers
        
    def get_headers(self):
        """Build headers with Browser emulation + Auth"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://uzum.uz",
            "Referer": "https://uzum.uz/",
            "apollographql-client-name": "web-customers"
        }
        
        # Try to get token from instance OR file
        final_token = self.token
        
        if not final_token:
            # Try load from data.txt
            try:
                paths_to_check = [
                    Path("data.txt"),
                    Path("../data.txt"),
                    Path(__file__).parent.parent / "data.txt",
                    Path(__file__).parent.parent / "_archive" / "data.txt"
                ]
                
                for p in paths_to_check:
                    if p.exists():
                        with open(p, 'r') as f:
                            for line in f:
                                if line.startswith('UZUM_TOKEN='):
                                    final_token = line.split('=', 1)[1].replace('Bearer ', '').strip()
                                    break
                    if final_token:
                        break
            except Exception as e:
                logger.warning(f"Error reading data.txt: {e}")
        
        # ── AUTO FETCH GUEST TOKEN IF MISSING ──
        if not final_token:
            logger.info("⚡ Automating token generation via Playwright...")
            try:
                import sys
                sys.path.append(str(Path(__file__).parent))
                from uzum_token_extractor import get_guest_token
                
                new_token = get_guest_token()
                if new_token:
                    final_token = new_token
                    self.token = new_token
                    # Save it for future usage non-destructively
                    token_cache_path = Path(__file__).parent.parent / "data.txt"
                    try:
                        lines = []
                        if token_cache_path.exists():
                            with open(token_cache_path, "r", encoding="utf-8") as f:
                                lines = f.readlines()
                        
                        # Replace existing token or append
                        token_found_in_file = False
                        for i, line in enumerate(lines):
                            if line.startswith("UZUM_TOKEN="):
                                lines[i] = f"UZUM_TOKEN=Bearer {final_token}\n"
                                token_found_in_file = True
                                break
                        
                        if not token_found_in_file:
                            lines.append(f"UZUM_TOKEN=Bearer {final_token}\n")
                            
                        with open(token_cache_path, "w", encoding="utf-8") as f:
                            f.writelines(lines)
                            
                        logger.info(f"✅ Auto-generated token saved to {token_cache_path}")
                    except Exception:
                        pass
                else:
                    logger.warning("⚠️ Failed to extract token automatically.")
            except ImportError as e:
                logger.warning(f"⚠️ Playwright token extractor missing/failed: {e}. Check if playwright is installed.")
            except Exception as e:
                logger.warning(f"⚠️ Token generation error: {e}")

        if final_token:
            headers["Authorization"] = f"Bearer {final_token}"
        else:
            logger.warning("⚠️ No UZUM_TOKEN found! Using Guest Mode (likely 401)")
            
        return headers
    
    def get_product_details(self, product_id: str) -> Dict:
        """Fetch real product metadata using GraphQL"""
        query = """
        query ProductPage($productId: Int!) {
          productPage(id: $productId) {
            product {
              id
              title
              rating
              feedbackQuantity
              category {
                id
                title
              }
              shop {
                id
                title
                rating
              }
              skuList {
                sellPrice
              }
            }
          }
        }
        """
        
        try:
            response = requests.post(
                UZUM_API_URL,
                json={
                    "operationName": "ProductPage",
                    "query": query,
                    "variables": {"productId": int(product_id)}
                },
                headers=self.get_headers(),
                timeout=10
            )
            
            if response.status_code == 401:
                logger.error("🛑 UZUM API 401 UNAUTHORIZED. PLEASE UPDATE TOKEN IN data.txt")
                return None

            if response.status_code == 200:
                data = response.json().get('data', {}).get('productPage', {}).get('product', {})
                if data:
                    cat = data.get('category', {})
                    shop = data.get('shop', {})
                    skus = data.get('skuList', [])
                    price = skus[0].get('sellPrice', 0) if skus else 0
                    
                    return {
                        'title': data.get('title', 'Unknown'),
                        'rating': data.get('rating', 0),
                        'review_count': data.get('feedbackQuantity', 0),
                        'price': price,
                        'category_id': cat.get('id', 'unknown'),
                        'category': cat.get('title', 'Unknown'),
                        'seller_title': shop.get('title', 'Unknown'),
                        'seller_rating': shop.get('rating', 0)
                    }
                    
            logger.warning(f"Product details fetch failed: {response.status_code}")
            return self._fallback(product_id)
            
        except Exception as e:
            logger.error(f"Error fetching product details: {e}")
            return self._fallback(product_id)

    def _fallback(self, product_id):
        return {
            'title': f'Product {product_id}',
            'rating': 0, 'review_count': 0, 'price': 0,
            'category': 'Unknown', 'seller_title': 'Unknown'
        }

    def get_reviews(self, product_id: str, limit: int = 2000) -> List[Dict]:
        """Fetch reviews with aggressive pagination and content filtering"""
        import time
        
        query = """
        query Feedbacks($productPageId: Int!, $page: Int!, $size: Int!, $sort: FeedbackSortType!) {
          productPage(id: $productPageId) {
            product { feedbackQuantity }
            feedbacks(page: $page, size: $size, sort: $sort) {
              id
              rating
              content
              dateCreated
              sku { sellPrice }
            }
          }
        }
        """
        
        print(f"\n🔍 FETCHING REVIEWS FOR {product_id}...")
        all_reviews = []
        page = 0
        size = 100
        
        headers = self.get_headers()
        
        try:
            # Step 1: Get Total Count
            init_resp = requests.post(
                UZUM_API_URL,
                json={
                    "operationName": "Feedbacks",
                    "query": query,
                    "variables": {"productPageId": int(product_id), "page": 0, "size": 1, "sort": "DATE_DESC"}
                },
                headers=headers,
                timeout=10
            )
            
            if init_resp.status_code == 401:
                logger.error("🛑 401 UNAUTHORIZED - CANNOT FETCH REVIEWS. UPDATE TOKEN.")
                return []
                
            total_reviews = 0
            if init_resp.status_code == 200:
                total_reviews = init_resp.json().get('data', {}).get('productPage', {}).get('product', {}).get('feedbackQuantity', 0)
                print(f"✅ Found {total_reviews} total reviews on platform")
                
            # Step 2: Fetch loop
            # Uncap loop: continue until we hit total_reviews or limit
            max_pages = (limit // size) + 2
            
            while page < max_pages and len([r for r in all_reviews if r['content']]) < limit:
                print(f"📄 Fetching page {page}...", end='\r')
                
                resp = requests.post(
                    UZUM_API_URL,
                    json={
                        "operationName": "Feedbacks",
                        "query": query,
                        "variables": {"productPageId": int(product_id), "page": page, "size": size, "sort": "DATE_DESC"}
                    },
                    headers=headers,
                    timeout=10
                )
                
                if resp.status_code != 200:
                    break
                    
                data = resp.json().get('data', {}).get('productPage', {})
                feedbacks = data.get('feedbacks', [])
                
                if not feedbacks:
                    break
                    
                for fb in feedbacks:
                    content = (fb.get('content') or '').strip()
                    all_reviews.append({
                        'id': fb.get('id'),
                        'rating': fb.get('rating'),
                        'content': content,
                        'has_content': bool(content),
                        'date': fb.get('dateCreated'),
                        'price': fb.get('sku', {}).get('sellPrice', 0)
                    })
                
                page += 1
                time.sleep(0.1)  # Aggressive: only 0.1s sleep
                
        except Exception as e:
            logger.error(f"Review fetch error: {e}")
            
        text_reviews = [r for r in all_reviews if r['has_content']]
        print(f"\n✅ Fetched {len(all_reviews)} total, {len(text_reviews)} with text")
        return text_reviews[:limit]


# ═══════════════════════════════════════════════════════════════════════════
# LIVE DATA PIPELINE: STAGE 2 - INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

class FactorExtractor:
    """Extract 8 sentiment factors using XLM-RoBERTa"""
    
    def __init__(self, model_path: str = None):
        # Use absolute path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = model_path or os.path.join(project_root, "models", "uzum_nlp_v2")
        self.model_api = None
        
        try:
            from src.inference_api import UZUMInferenceAPI
            print(f"🔄 Loading ML model from {self.model_path}...")
            
            # Check if path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
                
            self.model_api = UZUMInferenceAPI(self.model_path)
            print("✅ ML model loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️  Could not load ML model: {e}")
            logger.warning("⚠️  Falling back to keyword-based extraction")
            self.model_api = None
            self._init_keyword_patterns()
    
    def _init_keyword_patterns(self):
        self.patterns = {
            'product_quality': ['yaxshi', 'ajoyib', 'zo\'r', 'sifatli'],
            'price_value': ['arzon', 'qimmat', 'narx'],
            'logistics_delivery': ['yetkazish', 'tez', 'delivery'],
            'packaging_condition': ['qadoq', 'o\'ralgan', 'buzilgan'],
            'accuracy_expectation': ['mos', 'rasm', 'foto'],
            'seller_service': ['sotuvchi', 'xizmat', 'muloqot'],
            'specifications': ['razmer', 'o\'lcham', 'rang'],
            'product_defects': ['buzuq', 'shikast', 'nosoz', 'ishlamaydi']
        }
    
    def extract_factors(self, review_text: str) -> Dict[str, float]:
        if not review_text or not isinstance(review_text, str):
            return {f: 0.0 for f in SENTIMENT_FACTORS}
            
        if self.model_api:
            try:
                result = self.model_api.predict(review_text)
                preds = result.get('predictions', {})
                return {f: float(preds.get(f, {}).get('confidence', 0.0)) for f in SENTIMENT_FACTORS}
            except Exception as e:
                logger.error(f"Inference failed: {e}")
        
        # Fallback
        text = review_text.lower()
        factors = {}
        for factor, kws in self.patterns.items():
            match = any(k in text for k in kws)
            if match:
                factors[factor] = 0.8 
            else:
                # Add random noise to avoid 0.0 flatlines
                factors[factor] = np.random.uniform(0.05, 0.25)
        return factors


# ═══════════════════════════════════════════════════════════════════════════
# LIVE DATA PIPELINE: STAGE 3 - ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

class ProductHealthAnalyzer:
    """Calculate Health Score (1-10) using Model 2 weights"""
    
    def __init__(self):
        self.factor_extractor = FactorExtractor()
    
    # Minimum characters for a review to be worth running ML inference on.
    # Single words like "yaxshi", "yaxshi" or emoji-only reviews produce pure noise.
    MIN_REVIEW_LENGTH = 15

    def calculate_health_score(self, reviews_data: List[Dict], actual_rating: float = 0.0) -> Dict:
        # ── Gate: drop reviews that are too short to carry signal ──────────────
        valid_reviews = [
            r for r in reviews_data
            if len((r.get('content') or '').strip()) >= self.MIN_REVIEW_LENGTH
        ]
        skipped = len(reviews_data) - len(valid_reviews)
        if skipped > 0:
            logger.info(f"⚡ Skipped {skipped} short reviews (< {self.MIN_REVIEW_LENGTH} chars)")

        if not valid_reviews or len(valid_reviews) < 5:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least 5 meaningful text reviews (got {len(valid_reviews)} after filtering {skipped} short ones)',
                'health_score': 0,
                'predicted_rating': 0
            }

        # Work on filtered set from here on
        reviews_data = valid_reviews

        # 1. Extract & Tag Reviews
        all_factors = []

        # Hybrid keyword anchors for common issues
        quality_keywords = ['протека', 'сифатсиз', 'ёмон', 'ploxoy', 'defect', 'broken', 'torn', 'yomon']
        defect_keywords = ['дефект', 'nuqson', 'buzilgan', 'defekt', 'broken']

        for r in reviews_data:
            factor_scores = self.factor_extractor.extract_factors(r['content'])
            all_factors.append(factor_scores)
            
            # ═══════════════════════════════════════════════════════════════════
            # PER-REVIEW TAGGING: Attach detected_factors for frontend filtering
            # ═══════════════════════════════════════════════════════════════════
            detected = []
            content_lower = r.get('content', '').lower()
            
            # HYBRID APPROACH: ML scores + keyword detection
            for factor, score in factor_scores.items():
                # Tag if ML confidence > 25%
                if score >= 0.25:
                    detected.append(factor)
            
            # Keyword-based detection for critical factors (ensures we catch obvious cases)
            if any(keyword in content_lower for keyword in quality_keywords):
                if 'product_quality' not in detected:
                    detected.append('product_quality')
                    
            if any(keyword in content_lower for keyword in defect_keywords):
                if 'product_defects' not in detected:
                    detected.append('product_defects')
            
            r['detected_factors'] = detected
            
        # ═══════════════════════════════════════════════════════════════════
        # 2. STAR-CONDITIONED FACTOR ANALYSIS
        # Split reviews by star rating to distinguish complaints vs praise
        # ═══════════════════════════════════════════════════════════════════
        negative_factor_vals = {f: [] for f in SENTIMENT_FACTORS}  # From 1-3 star reviews
        positive_factor_vals = {f: [] for f in SENTIMENT_FACTORS}  # From 4-5 star reviews
        all_factor_vals = {f: [] for f in SENTIMENT_FACTORS}       # All reviews (for formula)
        
        for i, r in enumerate(reviews_data):
            review_rating = r.get('rating', 0) or 0
            factors = all_factors[i]
            
            for f in SENTIMENT_FACTORS:
                if f in factors:
                    score = factors[f]
                    all_factor_vals[f].append(score)
                    
                    if review_rating <= 3:
                        negative_factor_vals[f].append(score)
                    elif review_rating >= 4:
                        positive_factor_vals[f].append(score)
        
        # Compute averages for ALL reviews (used for predicted rating formula)
        avg_factors = {}
        for f in SENTIMENT_FACTORS:
            vals = all_factor_vals[f]
            val = np.mean(vals) if vals else 0
            if val < 0.20:  # NOISE GATE
                val = 0.0
            avg_factors[f] = val
        
        # Compute PROBLEM RATES: factor detection rate among NEGATIVE reviews only
        total_reviews = len(reviews_data)
        negative_count = sum(1 for r in reviews_data if (r.get('rating', 0) or 0) <= 3)
        positive_count = total_reviews - negative_count
        
        problem_rates = {}
        for f in SENTIMENT_FACTORS:
            neg_vals = negative_factor_vals[f]
            if neg_vals:
                # What % of negative reviews mention this factor?
                neg_avg = np.mean(neg_vals)
                # Scale by ratio of negative reviews to total
                problem_rate = neg_avg * (negative_count / total_reviews) if total_reviews > 0 else 0
            else:
                problem_rate = 0.0
            problem_rates[f] = round(problem_rate, 4)
            
        # SAFETY NET: If all factors are zero, return insufficient signal
        if sum(avg_factors.values()) == 0:
            logger.warning("⚠️ All factors zero — model could not detect any signals.")
            return {
                'status': 'insufficient_signal',
                'message': 'ML model detected no satisfaction factors in the reviews. '
                           'Reviews may be too short or in an unsupported language.',
                'health_score': 0,
                'predicted_rating': 0,
                'review_count': total_reviews,
                'factor_breakdown': avg_factors
            }
        
        # 3. Apply Weights — use actual formula: Rating = 4.036 + Σ(weight_i × factor_i)
        baseline = 4.0358  # From satisfaction formula derivation
        predicted = baseline
        
        for f, weight in IMPACT_WEIGHTS.items():
            if f in avg_factors:
                val = avg_factors[f]
                impact = weight * val
                predicted += impact
        
        predicted = max(1.0, min(5.0, predicted))
        
        # 4. Scale to 10
        health_score = (predicted / 5.0) * 10
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PENALTY OVERRIDE: Cap score based on PROBLEM RATES (not raw mentions)
        #
        # Thresholds raised and guarded by a minimum negative review count.
        # Without the guard, even 2 bad reviews on a 150-review product could
        # trigger "CRITICAL" when negative_count/total is still tiny.
        # ═══════════════════════════════════════════════════════════════════════════
        defect_problem = problem_rates.get('product_defects', 0)
        quality_problem = problem_rates.get('product_quality', 0)
        accuracy_problem = problem_rates.get('accuracy_expectation', 0)

        max_problem_rate = max(defect_problem, quality_problem, accuracy_problem)

        max_possible_score = 10.0
        penalty_reason = None

        # Only apply penalty caps when there's statistically enough negative evidence.
        # Minimum 20 negative reviews avoids over-penalising products where 3 bad
        # reviews happen to mention defects out of 200 total.
        MIN_NEGATIVE_FOR_PENALTY = 20

        if negative_count >= MIN_NEGATIVE_FOR_PENALTY:
            if max_problem_rate > 0.25:   # was 0.15 — 25%+ of ALL reviews are negative+mentioned
                max_possible_score = 5.0
                penalty_reason = f"CRITICAL: {int(max_problem_rate*100)}% complaint rate"
            elif max_problem_rate > 0.18:  # was 0.10
                max_possible_score = 6.5
                penalty_reason = f"HIGH: {int(max_problem_rate*100)}% complaint rate"
            elif max_problem_rate > 0.10:  # was 0.05
                max_possible_score = 7.5
                penalty_reason = f"MODERATE: {int(max_problem_rate*100)}% complaint rate"

        # Apply cap
        if health_score > max_possible_score:
            logger.warning(f"🚨 PENALTY OVERRIDE: Score capped from {health_score:.1f} to {max_possible_score} ({penalty_reason})")
            health_score = max_possible_score

        
        # 5. Problems & Flags — use PROBLEM RATES (not raw avg_factors)
        problems = []
        
        for f in SENTIMENT_FACTORS:
            pr = problem_rates.get(f, 0)
            # Flag factors with >5% complaint rate
            if pr >= 0.05:
                problems.append({
                    'factor': f, 
                    'severity': pr,
                    'impact_weight': IMPACT_WEIGHTS.get(f, 0)
                })
        
        # LOGIC FLAGS
        if health_score < 6.0 and actual_rating > 4.5:
            problems.append({
                'factor': 'logic_mismatch',
                'severity': 1.0,
                'impact_weight': -0.1,
                'message': "⚠️ Low Health Score vs High Actual Rating"
            })
            
        if predicted < (actual_rating - 1.0) and actual_rating > 0:
            problems.append({
                'factor': 'sentiment_gap',
                'severity': 1.0,
                'impact_weight': -0.1,
                'message': "📉 AI Sentiment Disagrees with Stars"
            })
            
        problems.sort(key=lambda x: x['impact_weight'])
        
        return {
            'status': 'success',
            'health_score': round(health_score, 2),
            'predicted_rating': round(predicted, 2),
            'actual_rating': actual_rating, 
            'review_count': total_reviews,
            'negative_review_count': negative_count,
            'positive_review_count': positive_count,
            'factor_breakdown': {k: round(v, 3) for k, v in problem_rates.items()},
            'factor_mentions': {k: round(v, 3) for k, v in avg_factors.items()},
            'top_problems': problems[:5],
            'score_explanation': f"Based on {total_reviews} reviews ({negative_count} negative, {positive_count} positive)" + (f" | {penalty_reason}" if penalty_reason else "")
        }
        
    def benchmark_against_category(self, health_analysis: Dict, category: str) -> Dict:
        if category not in CATEGORY_BENCHMARKS:
            return {'status': 'category_not_found'}
            
        benchmark = CATEGORY_BENCHMARKS[category]
        pred = health_analysis.get('predicted_rating', 0)
        cat_avg = benchmark.get('avg_rating', 4.0)
        diff = pred - cat_avg
        
        # Calculate real percentile estimate using normal distribution approximation
        # Based on observed rating std ~0.5 across categories
        import math
        z_score = diff / 0.5  # Approximate std of ratings
        percentile = min(99, max(1, int(50 * (1 + math.erf(z_score / math.sqrt(2))))))
        
        if percentile >= 75:
            comparison = f'Top {100 - percentile}%'
        elif percentile <= 25:
            comparison = f'Bottom {percentile}%'
        else:
            comparison = f'Middle {percentile}%'
        
        return {
            'status': 'success',
            'category': category,
            'rating_diff': round(diff, 2),
            'performance': 'above_average' if diff > 0 else 'below_average',
            'avg_rating': cat_avg,
            'avg_health': (cat_avg / 5) * 10,
            'percentile': percentile,
            'comparison': comparison,
            'benchmark_message': f"Category average: {cat_avg}"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class B2BSellerEngine:
    def __init__(self, token: str):
        self.scraper = UzumScraper(token)
        self.analyzer = ProductHealthAnalyzer()
        
    def analyze_product(self, product_url: str, use_cache: bool = True, language: str = 'uz') -> Dict:
        product_id = extract_product_id_from_url(product_url)
        if not product_id:
            return {'status': 'error', 'message': 'Invalid URL'}
            
        # Cache check
        if use_cache and is_cache_valid(product_id):
            cached = load_cache(product_id)
            if cached:
                cached['from_cache'] = True
                return cached
        
        logger.info(f"🚀 Analyzing {product_id}...")
        
        # 1. Details
        details = self.scraper.get_product_details(product_id)
        if not details: 
            return {'status': 'error', 'message': 'Product not found or API blocked'}
            
        # 2. Reviews
        reviews = self.scraper.get_reviews(product_id, limit=2000)
        
        # 3. Analytics
        health = self.analyzer.calculate_health_score(reviews, actual_rating=float(details.get('rating', 0)))
        
        # 4. Benchmark
        bench = self.analyzer.benchmark_against_category(health, details.get('category', 'Unknown'))
        
        # Construct product_info strictly matching Pydantic model
        product_info_formatted = {
            "title": details.get('title'),
            "actual_rating": float(details.get('rating', 0)),
            "total_reviews": details.get('review_count', 0),
            "analyzed_reviews": len(reviews),
            "seller_name": details.get('seller_title'),
            "price": details.get('price'),
            "category": details.get('category'),
            "review_count": details.get('review_count', 0), # Redundant but kept for safety
            "seller_rating": details.get('seller_rating', 0)
        }
        
        result = {
            'status': 'success',
            'product_id': product_id,
            'timestamp': datetime.now().isoformat(),
            'from_cache': False,
            'product_info': product_info_formatted,
            'health_analysis': health,
            'benchmark': bench,
            'raw_reviews': self._get_sample_reviews(reviews, sample_size=20)
        }
        
        save_cache(product_id, result)
        return result
    
    def _get_sample_reviews(self, reviews: List[Dict], sample_size: int = 20) -> List[Dict]:
        """
        Return sample reviews, prioritizing those with detected_factors
        """
        if len(reviews) <= sample_size:
            return reviews
            
        # Separate reviews with and without detected factors
        tagged_reviews = [r for r in reviews if r.get('detected_factors')]
        untagged_reviews = [r for r in reviews if not r.get('detected_factors')]
        
        # Prioritize tagged reviews (show evidence first)
        result = tagged_reviews[:sample_size]
        
        # Fill remaining slots with untagged reviews
        remaining_slots = sample_size - len(result)
        if remaining_slots > 0:
            result.extend(untagged_reviews[:remaining_slots])
        
        return result

