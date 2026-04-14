"""
Groq AI Business Consultant Integration
Real-time strategic recommendations for B2B sellers
=====================================================

Ushbu modul:
1. Tahlil natijalasini "konsentrat" (JSON) ko'rinishida Groqga yubora
2. Groq AI professional maslahat beradi
3. ROI prognozi hisoblanadi
"""

import json
import os
import re
from typing import Dict, Optional, List
from pathlib import Path
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# GROQ API CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _load_groq_keys() -> list:
    """
    Load Groq API keys from _archive/data.txt or environment variables.
    Never hardcode keys in source code.
    """
    keys = []

    # Strategy 1: Load from data.txt or _archive/data.txt
    project_root = Path(__file__).resolve().parent.parent
    
    paths_to_check = [
        project_root / "data.txt",
        project_root / "_archive" / "data.txt"
    ]

    for data_txt_path in paths_to_check:
        if data_txt_path.exists():
            try:
                content = data_txt_path.read_text(encoding="utf-8")
                # Match keys in format: GROQ_API_KEY=gsk_... or "gsk_..."
                found = re.findall(r'(gsk_[A-Za-z0-9]+)', content)
                if found:
                    keys.extend(found)
                    logger.info(f"✓ Loaded {len(found)} Groq API keys from {data_txt_path}")
                    break  # Stop checking if we found keys
            except Exception as e:
                logger.warning(f"⚠️ Could not read {data_txt_path}: {e}")

    # Strategy 2: Fallback to environment variable
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key and env_key not in keys:
        keys.append(env_key)
        logger.info("✓ Loaded Groq API key from GROQ_API_KEY env var")

    if not keys:
        logger.error("❌ No Groq API keys found! Set GROQ_API_KEY env var or add keys to _archive/data.txt")

    return keys


GROQ_KEYS = _load_groq_keys()

GROQ_MODEL = "llama-3.3-70b-versatile"

# System prompt for Groq
SYSTEM_PROMPT = """Siz Uzum bozorida sotuvchilarga yordam beradigan professional biznes maslahatchisiz.

Sizga mahsulot tahlili natijalari (JSON) beriladi. Qo'llanuvchi ma'lumotlar:
- Mahsulot nomi va kategoriyasi
- Health Score (1-10)
- Predicted Rating (1-5 stars)
- Kategoriya o'rtachasi bilan solishtirish
- Eng muhim muammolar ro'yxati (factors)

Vazifangiz:
1. **Top 2 Muammoni Aniqlash**: Eng katta ta'siri bo'lgan 2 ta muammoni belgilang
2. **Hal Qilish Rejasi**: Har bir muammo uchun aniq, amaliy 3-4 qadamli rejani yozing
3. **ROI Prognozi**: Agar muammolar hal bo'lsa, reyting va sotuv qanchaga o'sishi (%)

Javobni FAQAT JSON formatida qaytaring:
{
  "consultant_name": "Uzum Business Consultant",
  "confidence_level": 0.95,
  "top_problems": [
    {
      "rank": 1,
      "problem_name": "...",
      "current_impact": "...",
      "solution_steps": ["qadam 1", "qadam 2", "qadam 3"],
      "estimated_improvement": "reyting +0.5 stars"
    },
    {
      "rank": 2,
      "problem_name": "...",
      "current_impact": "...",
      "solution_steps": ["qadam 1", "qadam 2"],
      "estimated_improvement": "reyting +0.2 stars"
    }
  ],
  "roi_forecast": {
    "timeline": "2 hafta",
    "rating_increase": "+0.7 stars",
    "sales_increase_estimate": "+15-25%",
    "confidence": "High"
  },
  "success_metrics": ["Sharh soni +40%", "4-5 yulduz sharhlar +60%", "Tiklanish vaqti 2-3 kun"],
  "quick_wins": ["Tezlikni oshirish", "Qadoqni yaxshilash"],
  "overall_verdict": "Muammolar hal bo'lsa, bu mahsulot kategoryasida TOP 10'ga kirish imkoniga ega"
}

Javobda faqat JSON bo'lsin, boshqa matn bo'lmasin!"""


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_analysis_context(analysis: Dict) -> Dict:
    """
    Tahlil natijalasidan Groqga yuboriladigan "konsentrat" tuzish
    Minimal, but contextually rich data
    """
    
    if analysis.get('status') != 'success':
        return None
    
    product = analysis.get('product_info') or {}
    health = analysis.get('health_analysis') or {}
    benchmark = analysis.get('benchmark') or {}
    
    # Extract top 2 problems
    problems = health.get('top_problems') or []
    top_problems = []
    
    for problem in problems[:2]:
        factor = problem.get('factor', '')
        severity = problem.get('severity', 0)
        
        factor_display = {
            'product_defects': 'Tovar Nuqsonlari',
            'product_quality': 'Tovar Sifati',
            'accuracy_expectation': 'Kutilish Moslik',
            'logistics_delivery': 'Yetkazib Berish',
            'packaging_condition': 'Qadoqlash',
            'price_value': 'Narx-Qiymat Nisbati',
            'seller_service': 'Sotuvchi Xizmati',
            'specifications': 'Spesifikasiya'
        }.get(factor, factor)
        
        top_problems.append({
            'factor': factor_display,
            'severity': round(severity * 100),
            'impact_weight': problem.get('impact_weight', 0)
        })
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FORCE-FEED DETECTED ISSUES (From factor_breakdown)
    # ═══════════════════════════════════════════════════════════════════════════
    factor_breakdown = health.get('factor_breakdown') or {}
    detected_issues = []
    
    factor_names_uz = {
        'product_defects': 'Mahsulot Nuqsonlari',
        'product_quality': 'Mahsulot Sifati',
        'accuracy_expectation': 'Kutilgan Moslik',
        'logistics_delivery': 'Yetkazib Berish',
        'packaging_condition': 'Qadoqlash',
        'price_value': 'Narx-Qiymat',
        'seller_service': 'Sotuvchi Xizmati',
        'specifications': 'Spesifikasiya'
    }
    
    # Iterate and detect issues above 5% threshold
    for factor, value in factor_breakdown.items():
        if value >= 0.05:  # 5% threshold
            percentage = int(value * 100)
            factor_name_uz = factor_names_uz.get(factor, factor)
            
            # Severity classification
            if value >= 0.30:
                severity_label = "KRITIK"
            elif value >= 0.20:
                severity_label = "YUQORI"
            elif value >= 0.10:
                severity_label = "O'RTACHA"
            else:
                severity_label = "PAST"
            
            detected_issues.append(
                f"{factor_name_uz}: {severity_label} ({percentage}% sharhlar bu muammo haqida shikoyat qiladi)"
            )
    
    context = {
        'product_name': product.get('title', 'N/A'),
        'category': product.get('category', 'N/A'),
        'health_score': health.get('health_score'),
        'predicted_rating': health.get('predicted_rating'),
        'actual_rating': product.get('actual_rating'),
        'review_count': product.get('review_count'),
        'category_average_rating': benchmark.get('category_avg_rating'),
        'performance_vs_category': benchmark.get('rating_diff'),
        'percentile': benchmark.get('percentile'),
        'top_problems': top_problems,
        'seller_name': product.get('seller_name'),
        'price': product.get('price'),
        'analysis_timestamp': analysis.get('timestamp'),
        'factor_breakdown': factor_breakdown,  # Include raw data
        'detected_issues': detected_issues  # Force-feed list
    }
    
    return context


# ═══════════════════════════════════════════════════════════════════════════
# GROQ API CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class GroqConsultant:
    """Groq AI orqali professional biznes maslahat berish"""
    
    def __init__(self, api_key: str = None):
        # Dynamically reload keys if empty (to catch live updates to data.txt)
        keys = _load_groq_keys()
        if not keys and not api_key:
            logger.warning("No Groq Keys available. API calls will fail.")
            self.api_key = ""
        else:
            self.api_key = api_key or keys[0]
            
        self.model = GROQ_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def get_consultant_verdict(self, analysis_context: Dict, language: str = 'uz', timeout: int = 30) -> Dict:
        """
        Groqdan professional maslahat olish
        
        Input: Analysis context (compact JSON)
        Output: Consultant verdict with recommendations
        """
        
        if not analysis_context:
            return {
                'status': 'error',
                'message': 'Analysis context topilmadi'
            }
        
        # ═══════════════════════════════════════════════════════════════════════════
        # STEP 1: CLASSIFY PRODUCT SCALE (Uzum Market Context)
        # ═══════════════════════════════════════════════════════════════════════════
        review_count = analysis_context.get('review_count', 0) or 0
        
        if review_count >= 1000:
            product_scale = "Market Leader"
            scale_note = "2000+ reviews is EXTREMELY HIGH for Uzum (Uzbekistan market). This is a top-selling product."
        elif review_count >= 100:
            product_scale = "Established Product"
            scale_note = "100+ reviews is a well-established product on Uzum."
        elif review_count >= 50:
            product_scale = "Growing Product"
            scale_note = "50+ reviews shows decent market traction."
        else:
            product_scale = "New Entrant"
            scale_note = "Under 50 reviews indicates a new or niche product."
        
        analysis_context['product_scale'] = product_scale
        analysis_context['scale_note'] = scale_note
        
        # ═══════════════════════════════════════════════════════════════════════════
        # STEP 2: IDENTIFY TOP PROBLEM FACTOR (Data-Driven)
        # ═══════════════════════════════════════════════════════════════════════════
        top_problems = analysis_context.get('top_problems', [])
        
        primary_factor = None
        primary_severity = 0
        
        if top_problems:
            # Find the factor with highest severity
            for problem in top_problems:
                severity = problem.get('severity', 0)
                if severity > primary_severity:
                    primary_severity = severity
                    primary_factor = problem.get('factor', '')
        
        analysis_context['primary_problem_factor'] = primary_factor
        analysis_context['primary_problem_severity'] = primary_severity
        
        # ═══════════════════════════════════════════════════════════════════════════
        # STEP 3: BUILD CONTEXT-AWARE SYSTEM PROMPT
        # ═══════════════════════════════════════════════════════════════════════════
        
        # Get detected issues for force-feeding
        detected_issues = analysis_context.get('detected_issues', [])
        issues_text = "\n".join([f"   - {issue}" for issue in detected_issues]) if detected_issues else "   - Hech qanday jiddiy muammo topilmadi"
        
        context_aware_prompt = f"""Siz Uzum bozorida (O'zbekiston) sotuvchilarga yordam beradigan professional biznes maslahatchisiz.
JAVOB TILI: {language.upper()} (Barcha javoblar, tahlillar va maslahatlar shu tilda bo'lishi SHART).

EARTH MARKET CONTEXT:
- Uzum - O'zbekiston e-commerce platformasi
- 1000+ sharh = MARKET LEADER (juda ko'p)
- 100-999 sharh = Established product
- 50-99 sharh = Growing product
- <50 sharh = New entrant

🚨 ANIQLANGAN MUAMMOLAR (ML Model tahlili):
{issues_text}

⚠️ CRITICAL RULES:
1. **Agar mahsulot "{product_scale}" bo'lsa:**
   {scale_note}
   - "Ko'proq sharh oling" deb HECH QACHON tavsiya BERMASLIK
   - Focus on RETENTION va QUALITY improvement

2. **DATA-DRIVEN MASLAHAT (MAJBURIY):**
   - Yuqorida ko'rsatilgan "ANIQLANGAN MUAMMOLAR" ro'yxati - BU FAKTLAR
   - Siz bu muammolarni QABUL QILISHINGIZ SHART
   - Health Score'ga ISHONMASLIK (u noto'g'ri bo'lishi mumkin)
   - Sizning "Quick Wins" ALBATTA yuqoridagi muammolarni hal qilishi kerak
   - Agar "Mahsulot Sifati: YUQORI (22%)" ko'rsatilgan bo'lsa, sizning barcha maslahatingiz SIFAT haqida bo'lishi kerak
   - Agar biror muammo 0% yoki ro'yxatda yo'q bo'lsa, u haqida gapirmaslik!

3. **QUICK WINS QOIDASI:**
   - Faqat "ANIQLANGAN MUAMMOLAR" ro'yxatidagi eng yuqori muammoni hal qilish
   - Agar Quality 22%, Logistics 0% bo'lsa → "Sifatni yaxshilash" (CORRECT) ✅
   - "Yetkazib berishni tezlashtirish" (WRONG) ❌

Sizga mahsulot tahlili natijalari (JSON) beriladi. Qo'llanuvchi ma'lumotlar:
- Mahsulot nomi va kategoriyasi
- Health Score (1-10) - BU RAQAMGA ISHONMASLIK, faqat ANIQLANGAN MUAMMOLAR'ga qarang
- Predicted Rating (1-5 stars)
- Kategoriya o'rtachasi bilan solishtirish
- ANIQLANGAN MUAMMOLAR (Yuqorida ko'rsatilgan) - BU ASOSIY MANBA

Vazifangiz:
1. **Top 2 Muammoni Aniqlash**: "ANIQLANGAN MUAMMOLAR" ro'yxatidan eng yuqori 2 tasini tanlang
2. **Hal Qilish Rejasi**: Har bir muammo uchun aniq, amaliy 3-4 qadamli rejani yozing
3. **ROI Prognozi**: Agar muammolar hal bo'lsa, reyting va sotuv qanchaga o'sishi (%)

Javobni FAQAT JSON formatida qaytaring:
{{
  "consultant_name": "Uzum Business Consultant",
  "confidence_level": 0.95,
  "top_problems": [
    {{
      "rank": 1,
      "problem_name": "ANIQLANGAN MUAMMOLAR'dan birinchi muammo",
      "current_impact": "XX% mijozlar shikoyat qilishmoqda",
      "solution_steps": ["qadam 1", "qadam 2", "qadam 3"],
      "estimated_improvement": "reyting +0.5 stars"
    }},
    {{
      "rank": 2,
      "problem_name": "ANIQLANGAN MUAMMOLAR'dan ikkinchi muammo",
      "current_impact": "...",
      "solution_steps": ["qadam 1", "qadam 2"],
      "estimated_improvement": "reyting +0.2 stars"
    }}
  ],
  "roi_forecast": {{
    "timeline": "2 hafta",
    "rating_increase": "+0.7 stars",
    "sales_increase_estimate": "+15-25%",
    "confidence": "High"
  }},
  "success_metrics": ["Sharh soni +40%", "4-5 yulduz sharhlar +60%", "Tiklanish vaqti 2-3 kun"],
  "quick_wins": ["Birinchi muammoni hal qilish uchun ANIQ qadam", "Ikkinchi muammoni hal qilish"],
  "overall_verdict": "Muammolar hal bo'lsa, bu mahsulot kategoryasida TOP 10'ga kirish imkoniga ega"
}}

Javobda faqat JSON bo'lsin, boshqa matn bo'lmasin!"""
        
        # Prepare context message
        context_str = json.dumps(analysis_context, indent=2, ensure_ascii=False)
        
        user_message = f"""Quyidagi mahsulot tahlili natijalari asosida professional maslahat bering:

PRODUCT SCALE: {product_scale} ({review_count} reviews)
{scale_note}

PRIMARY PROBLEM: {primary_factor} ({primary_severity}%)

{context_str}

Javobni FAQAT JSON formatida qaytaring, boshqa matn bo'lmasin."""
        
        # Prepare request
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": context_aware_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.9
        }
        
        try:
            logger.info(f"🤖 Consulting with Groq AI... (Product Scale: {product_scale}, Primary Issue: {primary_factor})")
            
            response = requests.post(
                GROQ_API_URL,
                json=payload,
                headers=self.headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                choices = result.get('choices')
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    logger.error(f"Groq API returned no choices: {result}")
                    return {
                        'status': 'error',
                        'message': 'Groq API returned empty response',
                        'fallback': self._generate_fallback_verdict(analysis_context)
                    }
                content = choices[0].get('message', {}).get('content', '')
                
                # CRITICAL FIX: Strip markdown code blocks that Groq often adds
                # Example: ```json\n{...}\n``` → {...}
                clean_content = content.strip()
                
                # Remove markdown code blocks
                if clean_content.startswith('```'):
                    # Remove opening ```json or ```
                    clean_content = clean_content.split('\n', 1)[1] if '\n' in clean_content else clean_content[3:]
                    # Remove closing ```
                    if clean_content.endswith('```'):
                        clean_content = clean_content.rsplit('```', 1)[0]
                
                # Additional cleanup
                clean_content = clean_content.strip()
                
                # Parse JSON response
                try:
                    verdict = json.loads(clean_content)
                    verdict['status'] = 'success'
                    verdict['context'] = analysis_context
                    verdict['timestamp'] = datetime.now().isoformat()
                    logger.info("✅ Groq verdict received and parsed successfully")
                    return verdict
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    logger.error(f"Raw content (first 300 chars): {content[:300]}")
                    logger.error(f"Cleaned content (first 300 chars): {clean_content[:300]}")
                    return {
                        'status': 'parse_error',
                        'message': 'Groq javobini parse qilishda xatolik',
                        'raw_response': content,
                        'cleaned_response': clean_content,
                        'error_details': str(e)
                    }
            else:
                logger.error(f"Groq API error: {response.status_code}")
                return {
                    'status': 'api_error',
                    'message': f"Groq API error: {response.status_code}",
                    'details': response.text
                }
        
        except requests.exceptions.Timeout:
            logger.error("Groq API timeout")
            return {
                'status': 'timeout',
                'message': 'Groq API so\'rovida vaqt tugadi (30 sec)',
                'fallback': self._generate_fallback_verdict(analysis_context)
            }
        
        except Exception as e:
            logger.error(f"Groq consultation error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'fallback': self._generate_fallback_verdict(analysis_context)
            }
    
    def _generate_fallback_verdict(self, context: Dict) -> Dict:
        """Fallback recommendation if Groq fails"""
        
        problems = context.get('top_problems', [])
        predicted_rating = context.get('predicted_rating', 0)
        
        top_problems_verdict = []
        
        for i, problem in enumerate(problems[:2], 1):
            factor = problem.get('factor', 'Unknown')
            severity = problem.get('severity', 50)
            
            solution_steps = self._generate_solution_steps(factor, severity)
            
            improvement = self._estimate_improvement(factor)
            
            top_problems_verdict.append({
                'rank': i,
                'problem_name': factor,
                'current_impact': f"{severity}% ta'siri",
                'solution_steps': solution_steps,
                'estimated_improvement': improvement
            })
        
        return {
            'consultant_name': 'Uzum Business Consultant (Fallback)',
            'confidence_level': 0.70,
            'top_problems': top_problems_verdict,
            'roi_forecast': {
                'timeline': '2-3 hafta',
                'rating_increase': f"+{0.5 * len(problems)} stars",
                'sales_increase_estimate': f"+{10 * len(problems)}%",
                'confidence': 'Medium'
            },
            'overall_verdict': f"Muammolarni hal qilish {len(problems)} ta masala bo'yicha reyting oshishiga olib keladi",
            'fallback': True
        }
    
    def _generate_solution_steps(self, factor: str, severity: int) -> List[str]:
        """Generate solution steps for a factor"""
        
        solutions = {
            'Tovar Nuqsonlari': [
                'QA jarayonini kuchaytirishni boshlang',
                'Sotuvchini defektli mahsulotlarni qaytarish uchun o\'rgiting',
                'Yetkazuvchilar bilan birga ishlang (lokal/xorijiy)'
            ],
            'Tovar Sifati': [
                'Tavsiflar va rasmlarni aniqroq qiling',
                'Katta brend bilan hamkorlik qilishni yoki supplier o\'zgartirashni ko\'rib chiqing',
                'Buyerlar uchun vositalar/aksesuarlarni taklif qiling (quality guarantee)'
            ],
            'Kutilish Moslik': [
                'Mahsulot tavsifini yangilang (batafsil xususiyatlar qo\'shing)',
                'Variantlar bo\'yicha rasmlarni qo\'shing (rang, o\'lchami)',
                'FAQ ni yangilang'
            ],
            'Yetkazib Berish': [
                'Yetkazib berish vaqtini qisqartiring (express yetkazish)',
                'Xarita bo\'yicha faqatgina ro\'yxatdan o\'tgan ko\'cha va hudud',
                'Logistika ortakclari bilan shartnoma yangilang'
            ],
            'Qadoqlash': [
                'Qadoqlash materiallarini yaxshilang (stronger, protective)',
                'Mahsulotlar uchun anti-shock paddings qo\'shing',
                'Branding/logo bilan birga premium qadoqlash qiling'
            ],
            'Narx-Qiymat Nisbati': [
                'Narxni bozor reytingiga mos keladigan darajaga qisqartiring',
                'Eng ko\'p sotuvciga miqyosli chegirma bering',
                'Bundle/paket taklif qiling'
            ],
            'Sotuvchi Xizmati': [
                'Savdo xizmatini tezlashtiring (24 soat ichida javob berish)',
                'Savdo bilan bog\'liq muammolarni hal qilishning SOP\'sini tuzing',
                'Reklama shunoslarni oʻrgiting'
            ],
            'Spesifikasiya': [
                'Spesifikasiyalarni to\'liq va aniq qiling',
                'Mahsulot o\'lchami/vazni rasmlar bilan tasdiqla',
                'Sertifikat va to\'g\'ri hujjatlarni taqdim eting'
            ]
        }
        
        return solutions.get(factor, ['Muammoni hal qilish rejasini tuzing'])
    
    def _estimate_improvement(self, factor: str) -> str:
        """Estimate rating improvement"""
        
        improvements = {
            'Tovar Nuqsonlari': 'reyting +0.6 stars',
            'Tovar Sifati': 'reyting +0.3 stars',
            'Kutilish Moslik': 'reyting +0.2 stars',
            'Yetkazib Berish': 'reyting +0.1 stars',
            'Qadoqlash': 'reyting +0.1 stars',
            'Narx-Qiymat Nisbati': 'reyting +0.15 stars',
            'Sotuvchi Xizmati': 'reyting +0.2 stars',
            'Spesifikasiya': 'reyting +0.1 stars'
        }
        
        return improvements.get(factor, 'reyting +0.2 stars')


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_consultant_advice(analysis: Dict, language: str = 'uz') -> Dict:
    """
    Complete consultation pipeline:
    Analysis → Context → Groq → Verdict
    """
    
    # Build context
    context = build_analysis_context(analysis)
    
    if not context:
        return {
            'status': 'error',
            'message': 'Tahlil ma\'lumotlari topilmadi'
        }
    
    # Get consultant verdict
    consultant = GroqConsultant()
    verdict = consultant.get_consultant_verdict(context, language=language)
    
    return verdict
