"""
═════════════════════════════════════════════════════════════════════════════
MARKET ANALYZER - Market-Level Analytics Engine
═════════════════════════════════════════════════════════════════════════════

Purpose: Aggregate statistics and insights from historical dataset
Data Source: data/uzum_labeled.csv (23,987 reviews)
Use Case: Power the Market Overview dashboard

Key Features:
- Market-level statistics (total reviews, avg rating, etc.)
- Factor impact analysis (using trained model weights)
- Category breakdown and comparisons
- Trend analysis over time

Author: BMI_V4_NLP System
Date: 2026-02-09
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Import trained model weights
from src.b2b_engine import IMPACT_WEIGHTS

logger = logging.getLogger(__name__)

# Factor display names with emoji
FACTOR_DISPLAY_NAMES = {
    'product_quality': '📦 Product Quality',
    'product_defects': '⚠️ Product Defects',
    'price_value': '💰 Price Value',
    'logistics_delivery': '🚚 Logistics & Delivery',
    'packaging_condition': '📮 Packaging Condition',
    'accuracy_expectation': '✅ Accuracy & Expectations',
    'seller_service': '🤝 Seller Service',
    'specifications': '📋 Specifications'
}


class MarketAnalyzer:
    """
    Market-level analytics engine for Uzum marketplace
    """
    
    def __init__(self, data_path: str = 'data/uzum_labeled.csv'):
        """
        Initialize analyzer with dataset
        
        Args:
            data_path: Path to labeled dataset CSV
        """
        self.data_path = Path(data_path)
        self.df = None
        self.weights = IMPACT_WEIGHTS
        self._load_dataset()
    
    
    def _load_dataset(self):
        """Load and validate dataset with absolute path resolution"""
        import os
        
        # Resolve project root from this file's location
        # src/market_analyzer.py → src/ → project_root/
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        data_path = project_root / "data" / "uzum_labeled.csv"
        
        logger.debug(f"Project root: {project_root}")
        logger.debug(f"Target dataset: {data_path} (exists: {data_path.exists()})")
        
        # Load or fallback gracefully
        if not data_path.exists():
            logger.warning(f"⚠️ Dataset not found at {data_path}. This is expected if the repository was cloned without the huge dataset CSV. Falling back to empty market baseline.")
            self.df = pd.DataFrame()
            return
        
        # Load the dataset
        try:
            self.df = pd.read_csv(data_path)
            success_msg = f"✅ Market Analyzer: Loaded {len(self.df):,} reviews from dataset"
            logger.info(success_msg)
            
            # Validate required columns
            required_cols = ['Rating', 'Category']
            missing = [col for col in required_cols if col not in self.df.columns]
            if missing:
                logger.warning(f"⚠️ Missing columns: {missing}")
                
        except Exception as e:
            error_msg = f"❌ FAILED to read CSV from {data_path}: {e}"
            logger.error(error_msg)
            self.df = pd.DataFrame()


    
    def get_categories(self) -> Dict:
        """
        Get list of unique categories sorted by review count
        
        Returns:
            {
                'categories': [
                    {'name': str, 'review_count': int},
                    ...
                ]
            }
        """
        if self.df.empty:
            return {'categories': []}
        
        try:
            category_counts = self.df['Category'].value_counts().to_dict()
            categories = [
                {'name': str(cat), 'review_count': int(count)}
                for cat, count in category_counts.items()
            ]
            # Already sorted by count (descending) from value_counts()
            logger.info(f"📊 Found {len(categories)} categories")
            return {'categories': categories}
        except Exception as e:
            logger.error(f"❌ Error getting categories: {e}")
            return {'categories': []}
    
    def get_market_overview(self, category: Optional[str] = None) -> Dict:
        """
        Get high-level market statistics
        
        Args:
            category: Optional category filter (e.g., 'Elektronika')
        
        Returns:
            {
                'total_reviews': int,
                'avg_rating': float,
                'total_products': int,
                'total_sellers': int,
                'categories': int,
                'date_range': {'start': str, 'end': str},
                'satisfaction_rate': float,  # % with rating >= 4
                'selected_category': str | None
            }
        """
        if self.df.empty:
            return self._empty_stats()
        
        # Apply category filter if specified
        df_filtered = self.df[self.df['Category'] == category] if category else self.df
        
        if len(df_filtered) == 0:
            logger.warning(f"⚠️ No data found for category: {category}")
            return self._empty_stats()
        
        try:
            stats = {
                'total_reviews': int(len(df_filtered)),
                'avg_rating': float(df_filtered['Rating'].mean()),
                'total_products': int(df_filtered['Product_ID'].nunique()) if 'Product_ID' in df_filtered.columns else 0,
                'total_sellers': int(df_filtered['Seller'].nunique()) if 'Seller' in df_filtered.columns else 0,
                'categories': int(df_filtered['Category'].nunique()),
                'satisfaction_rate': float((df_filtered['Rating'] >= 4).mean() * 100),
                'selected_category': category
            }
            
            # Date range if available
            if 'Date' in df_filtered.columns:
                try:
                    dates = pd.to_datetime(df_filtered['Date'], errors='coerce')
                    stats['date_range'] = {
                        'start': dates.min().strftime('%Y-%m-%d') if pd.notna(dates.min()) else None,
                        'end': dates.max().strftime('%Y-%m-%d') if pd.notna(dates.max()) else None
                    }
                except:
                    stats['date_range'] = {'start': None, 'end': None}
            else:
                stats['date_range'] = {'start': None, 'end': None}
            
            cat_info = f" (category: {category})" if category else ""
            logger.info(f"📊 Market overview generated: {stats['total_reviews']:,} reviews{cat_info}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error computing market overview: {e}")
            return self._empty_stats()
    
    def get_factor_impact(self, category: Optional[str] = None) -> Dict:
        """
        Get factor weights and impact analysis
        
        Args:
            category: Optional category filter for recalculating weights
        
        Returns:
            {
                'factors': [
                    {'name': str, 'display_name': str, 'weight': float, 'type': 'positive'|'negative'},
                    ...
                ],
                'top_killer': {'name': str, 'weight': float},
                'top_strength': {'name': str, 'weight': float},
                'selected_category': str | None
            }
        """
        try:
            # Use global weights (from trained model) - these are static
            # NOTE: For MVP, we use the same weights across all categories
            # In future, could train category-specific models
            factors = []
            for factor_key, weight in self.weights.items():
                factors.append({
                    'name': factor_key,
                    'display_name': FACTOR_DISPLAY_NAMES.get(factor_key, factor_key),
                    'weight': round(weight, 3),
                    'type': 'positive' if weight > 0 else 'negative',
                    'impact_level': self._classify_impact(abs(weight))
                })
            
            # Sort by absolute weight (descending)
            factors.sort(key=lambda x: abs(x['weight']), reverse=True)
            
            # Find top killer (most negative) and top strength (most positive)
            negative_factors = [f for f in factors if f['type'] == 'negative']
            positive_factors = [f for f in factors if f['type'] == 'positive']
            
            top_killer = negative_factors[0] if negative_factors else None
            top_strength = positive_factors[0] if positive_factors else None
            
            return {
                'factors': factors,
                'top_killer': {
                    'name': top_killer['name'],
                    'display_name': top_killer['display_name'],
                    'weight': top_killer['weight']
                } if top_killer else None,
                'top_strength': {
                    'name': top_strength['name'],
                    'display_name': top_strength['display_name'],
                    'weight': top_strength['weight']
                } if top_strength else None,
                'selected_category': category
            }
            
        except Exception as e:
            logger.error(f"❌ Error computing factor impact: {e}")
            return {'factors': [], 'top_killer': None, 'top_strength': None}
    
    def get_category_breakdown(self) -> Dict:
        """
        Get category-level statistics and factor scores
        
        Returns:
            {
                'categories': [
                    {
                        'name': str,
                        'avg_rating': float,
                        'review_count': int,
                        'factor_scores': {factor: score, ...}
                    },
                    ...
                ]
            }
        """
        if self.df.empty:
            return {'categories': []}
        
        try:
            categories = []
            factor_columns = [f for f in self.weights.keys() if f in self.df.columns]
            
            for category in self.df['Category'].unique():
                cat_df = self.df[self.df['Category'] == category]
                
                category_data = {
                    'name': str(category),
                    'avg_rating': float(cat_df['Rating'].mean()),
                    'review_count': int(len(cat_df))
                }
                
                # Compute average factor scores for this category
                factor_scores = {}
                for factor in factor_columns:
                    factor_scores[factor] = float(cat_df[factor].mean())
                
                category_data['factor_scores'] = factor_scores
                categories.append(category_data)
            
            # Sort by review count (descending)
            categories.sort(key=lambda x: x['review_count'], reverse=True)
            
            logger.info(f"📊 Category breakdown generated for {len(categories)} categories")
            return {'categories': categories}
        except Exception as e:
            logger.error(f"❌ Error computing category breakdown: {e}")
            return {'categories': []}
    

    
    def get_trends(self, period: str = 'monthly') -> Dict:
        """
        Get time-series trends
        
        Args:
            period: 'daily', 'weekly', or 'monthly'
        
        Returns:
            {
                'trends': [
                    {'period': str, 'avg_rating': float, 'review_count': int},
                    ...
                ]
            }
        """
        if self.df.empty or 'Date' not in self.df.columns:
            return {'trends': []}
        
        try:
            df_with_dates = self.df.copy()
            df_with_dates['Date'] = pd.to_datetime(df_with_dates['Date'], errors='coerce')
            df_with_dates = df_with_dates.dropna(subset=['Date'])
            
            if len(df_with_dates) == 0:
                return {'trends': []}
            
            # Group by period
            freq_map = {
                'daily': 'D',
                'weekly': 'W',
                'monthly': 'M'
            }
            freq = freq_map.get(period, 'M')
            
            df_with_dates.set_index('Date', inplace=True)
            grouped = df_with_dates.resample(freq).agg({
                'Rating': 'mean',
                'Product_ID': 'count'
            }).reset_index()
            
            trends = []
            for _, row in grouped.iterrows():
                trends.append({
                    'period': row['Date'].strftime('%Y-%m-%d'),
                    'avg_rating': float(row['Rating']),
                    'review_count': int(row['Product_ID'])
                })
            
            return {'trends': trends}
            
        except Exception as e:
            logger.error(f"❌ Error computing trends: {e}")
            return {'trends': []}
    
    # Helper methods
    
    def _classify_impact(self, weight: float) -> str:
        """Classify impact level based on weight magnitude (calibrated to IMPACT_WEIGHTS range)"""
        if weight >= 1.5:
            return 'critical'
        elif weight >= 0.8:
            return 'high'
        elif weight >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics structure"""
        return {
            'total_reviews': 0,
            'avg_rating': 0.0,
            'total_products': 0,
            'total_sellers': 0,
            'categories': 0,
            'date_range': {'start': None, 'end': None},
            'satisfaction_rate': 0.0
        }
    
    def get_price_quality_matrix(self, category: str = None) -> Dict:
        """
        Generate Price vs Rating scatter data for Strategic Positioning Matrix
        
        Strategy:
        1. Aggregate by Product_ID (avg rating per product)
        2. Remove top 1% price outliers for readability
        3. Sample down to ~500 points for frontend performance
        
        Args:
            category: Optional category filter
            
        Returns:
            Dict with scatter data points and quadrant reference lines
        """
        if self.df is None or self.df.empty:
            return {
                'points': [],
                'median_price': 0,
                'rating_threshold': 4.0,
                'total_products': 0,
                'quadrant_counts': {'premium': 0, 'overpriced': 0, 'hidden_gem': 0, 'budget': 0}
            }
        
        try:
            df = self.df.copy()
            
            # Filter by category if specified
            if category and 'Category' in df.columns:
                df = df[df['Category'] == category]
            
            # Ensure required columns exist
            if 'Price' not in df.columns or 'Rating' not in df.columns:
                logger.warning("Missing Price or Rating columns")
                return self._empty_matrix()
            
            # Convert Price to numeric, drop NaN
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
            df = df.dropna(subset=['Price', 'Rating'])
            df = df[df['Price'] > 0]  # Remove free/zero-priced items
            
            # Aggregate by Product_ID (average rating per product)
            if 'Product_ID' in df.columns:
                product_agg = df.groupby('Product_ID').agg({
                    'Price': 'first',  # Price is same per product
                    'Rating': 'mean',
                    'Title': 'first',
                    'Category': 'first'
                }).reset_index()
                
                # Count reviews per product
                review_counts = df.groupby('Product_ID').size().reset_index(name='review_count')
                product_agg = product_agg.merge(review_counts, on='Product_ID', how='left')
            else:
                product_agg = df[['Price', 'Rating', 'Title', 'Category']].copy()
                product_agg['Product_ID'] = range(len(product_agg))
                product_agg['review_count'] = 1
            
            # Remove price outliers (top 1%)
            price_99th = product_agg['Price'].quantile(0.99)
            product_agg = product_agg[product_agg['Price'] <= price_99th]
            
            # Calculate median price for reference line
            median_price = float(product_agg['Price'].median())
            rating_threshold = 4.0
            
            # Sample if too many products (max 500)
            if len(product_agg) > 500:
                product_agg = product_agg.sample(n=500, random_state=42)
            
            # Classify quadrants
            product_agg['quadrant'] = product_agg.apply(
                lambda row: self._classify_quadrant(row['Price'], row['Rating'], median_price, rating_threshold),
                axis=1
            )
            
            # Build scatter points
            points = []
            for _, row in product_agg.iterrows():
                title = str(row.get('Title', 'Unknown'))
                # Truncate title for tooltip
                if len(title) > 60:
                    title = title[:57] + '...'
                    
                points.append({
                    'product_id': str(row.get('Product_ID', '')),
                    'title': title,
                    'price': float(row['Price']),
                    'rating': round(float(row['Rating']), 2),
                    'category': str(row.get('Category', '')),
                    'review_count': int(row.get('review_count', 1)),
                    'quadrant': row['quadrant']
                })
            
            # Count quadrants
            quadrant_counts = product_agg['quadrant'].value_counts().to_dict()
            
            result = {
                'points': points,
                'median_price': median_price,
                'rating_threshold': rating_threshold,
                'total_products': len(points),
                'quadrant_counts': {
                    'premium': quadrant_counts.get('premium', 0),
                    'overpriced': quadrant_counts.get('overpriced', 0),
                    'hidden_gem': quadrant_counts.get('hidden_gem', 0),
                    'budget': quadrant_counts.get('budget', 0)
                },
                'price_range': {
                    'min': float(product_agg['Price'].min()),
                    'max': float(product_agg['Price'].max())
                },
                'selected_category': category
            }
            
            logger.info(f"📊 Price-Quality Matrix: {len(points)} products, median price: {median_price:,.0f} UZS")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating price-quality matrix: {e}")
            return self._empty_matrix()
    
    def _classify_quadrant(self, price: float, rating: float, median_price: float, rating_threshold: float) -> str:
        """Classify a product into a positioning quadrant"""
        if price >= median_price and rating >= rating_threshold:
            return 'premium'       # Top-Right: Expensive + High Quality
        elif price >= median_price and rating < rating_threshold:
            return 'overpriced'    # Bottom-Right: Expensive + Low Quality (JUNK)
        elif price < median_price and rating >= rating_threshold:
            return 'hidden_gem'    # Top-Left: Cheap + High Quality (GEM)
        else:
            return 'budget'        # Bottom-Left: Cheap + Low Quality
    
    def _empty_matrix(self) -> Dict:
        """Return empty matrix structure"""
        return {
            'points': [],
            'median_price': 0,
            'rating_threshold': 4.0,
            'total_products': 0,
            'quadrant_counts': {'premium': 0, 'overpriced': 0, 'hidden_gem': 0, 'budget': 0},
            'price_range': {'min': 0, 'max': 0},
            'selected_category': None
        }
