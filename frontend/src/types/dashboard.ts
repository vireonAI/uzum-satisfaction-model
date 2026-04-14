// Types for the Uzum Intel Dashboard

export interface StatCardData {
  id: string;
  title: string;
  value: string | number;
  trend?: string;
  trendType?: 'up' | 'down' | 'flat';
  badge?: string;
  badgeType?: 'success' | 'warning' | 'destructive' | 'default';
  icon: string;
}

export interface ImpactData {
  category: string;
  importance: number;
  satisfaction: number;
}

export interface AlertItem {
  id: string;
  message: string;
  type: 'error' | 'warning' | 'success';
  timestamp: string;
}

export interface PredictionMetric {
  id: string;
  label: string;
  value: number;
  maxValue: number;
  unit?: string;
  colorType: 'success' | 'primary' | 'warning' | 'destructive';
}

export interface StrategyAction {
  id: string;
  icon: string;
  text: string;
}

export interface QualityCoefficient {
  id: string;
  factorName: string;
  weightImpact: number;
  baselineCoeff: number;
  simulatedCoeff: number;
  status: 'Active' | 'Pending' | 'Inactive';
  lastUpdated: string;
}

export interface SimulationState {
  defectRate: number;
  deliverySpeed: number;
  responseTime: number;
}

export interface DetectedFactor {
  id: string;
  label: string;
  type: 'positive' | 'negative' | 'neutral';
}

export interface Problem {
  factor: string;
  severity: number;
  impact_weight: number;
}

export interface HealthAnalysis {
  health_score: number;
  healthScore?: number; // Legacy compatibility
  predicted_rating: number;
  factor_breakdown: Record<string, number>;
  top_problems?: Problem[];
  score_explanation?: string;
  actual_rating?: number;
  review_count?: number;
}

export interface Benchmark {
  status?: string;
  category: string;
  avg_health: number;
  avg_rating: number;
  comparison: string;
  percentile: number;
  performance?: 'above_average' | 'below_average';
  rating_diff?: number;
  benchmark_message?: string;
}

export interface ProductAnalysisResult {
  status: string;
  product_id?: string;
  message?: string;
  timestamp?: string;
  from_cache?: boolean;
  product_info?: {
    title?: string;
    actual_rating?: number;
    total_reviews?: number;
    analyzed_reviews?: number;
    review_count?: number;
    price?: number;
    seller_name?: string;
    seller_rating?: number;
    category?: string;
    analysis_note?: string;
  };
  health_analysis?: HealthAnalysis;
  benchmark?: Benchmark;
  raw_reviews?: any[];
}
