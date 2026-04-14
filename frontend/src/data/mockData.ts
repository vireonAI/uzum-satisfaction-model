import type {
  StatCardData,
  ImpactData,
  AlertItem,
  PredictionMetric,
  StrategyAction,
  QualityCoefficient,
  DetectedFactor,
} from '@/types/dashboard';

export const marketOverviewStats: StatCardData[] = [
  {
    id: '1',
    title: 'Total Reviews',
    value: '1,240',
    trend: '+12%',
    trendType: 'up',
    icon: 'FileOutput',
  },
  {
    id: '2',
    title: 'Avg Rating',
    value: '4.2',
    trend: 'Flat',
    trendType: 'flat',
    icon: 'Star',
  },
  {
    id: '3',
    title: 'Main Problem',
    value: 'Shipping Delays',
    badge: 'HIGH IMPACT',
    badgeType: 'destructive',
    icon: 'AlertTriangle',
  },
  {
    id: '4',
    title: 'Top Strength',
    value: 'Durability',
    badge: 'Positive Sentiment',
    badgeType: 'success',
    icon: 'ThumbsUp',
  },
];

export const impactAnalysisData: ImpactData[] = [
  { category: 'Product Quality', importance: 85, satisfaction: 78 },
  { category: 'Delivery Speed', importance: 72, satisfaction: 45 },
  { category: 'Customer Service', importance: 65, satisfaction: 82 },
  { category: 'Price Competitiveness', importance: 58, satisfaction: 68 },
];

export const recentAlerts: AlertItem[] = [
  {
    id: '1',
    message: 'Negative sentiment spike detected in reviews',
    type: 'error',
    timestamp: '2 hours ago',
  },
  {
    id: '2',
    message: 'Competitor price drop on similar products',
    type: 'warning',
    timestamp: '5 hours ago',
  },
  {
    id: '3',
    message: 'Review response rate improved to 94%',
    type: 'success',
    timestamp: '1 day ago',
  },
  {
    id: '4',
    message: 'New market trend identified: eco-friendly packaging',
    type: 'warning',
    timestamp: '2 days ago',
  },
];

export const predictionMetrics: PredictionMetric[] = [
  { id: '1', label: 'Viral Potential', value: 92, maxValue: 100, colorType: 'success' },
  { id: '2', label: 'Price Competitiveness', value: 78, maxValue: 100, colorType: 'primary' },
  { id: '3', label: 'Review Sentiment', value: 88, maxValue: 100, colorType: 'primary' },
  { id: '4', label: 'Stock Velocity', value: 64, maxValue: 100, colorType: 'warning' },
  { id: '5', label: 'Seasonality Index', value: 45, maxValue: 100, colorType: 'warning' },
  { id: '6', label: 'Ad Click-Through Est.', value: 3.2, maxValue: 10, unit: '%', colorType: 'success' },
];

export const strategyActions: StrategyAction[] = [
  { id: '1', icon: 'TrendingUp', text: 'Increase ad spend during peak hours (2-6 PM)' },
  { id: '2', icon: 'Package', text: 'Bundle with complementary products for +15% AOV' },
  { id: '3', icon: 'MessageSquare', text: 'Respond to negative reviews within 24 hours' },
];

export const qualityCoefficients: QualityCoefficient[] = [
  {
    id: '1',
    factorName: 'Product Quality Score',
    weightImpact: 0.85,
    baselineCoeff: 0.72,
    simulatedCoeff: 0.78,
    status: 'Active',
    lastUpdated: '2024-01-15',
  },
  {
    id: '2',
    factorName: 'Shipping Reliability',
    weightImpact: 0.72,
    baselineCoeff: 0.65,
    simulatedCoeff: 0.71,
    status: 'Active',
    lastUpdated: '2024-01-14',
  },
  {
    id: '3',
    factorName: 'Customer Response Rate',
    weightImpact: 0.58,
    baselineCoeff: 0.82,
    simulatedCoeff: 0.85,
    status: 'Active',
    lastUpdated: '2024-01-13',
  },
  {
    id: '4',
    factorName: 'Price Elasticity',
    weightImpact: 0.45,
    baselineCoeff: 0.55,
    simulatedCoeff: 0.52,
    status: 'Pending',
    lastUpdated: '2024-01-12',
  },
  {
    id: '5',
    factorName: 'Return Rate Factor',
    weightImpact: -0.32,
    baselineCoeff: 0.12,
    simulatedCoeff: 0.10,
    status: 'Active',
    lastUpdated: '2024-01-11',
  },
];

export const detectedFactors: DetectedFactor[] = [
  { id: '1', label: 'Product Quality', type: 'positive' },
  { id: '2', label: 'Shipping Speed', type: 'negative' },
  { id: '3', label: 'Customer Service', type: 'positive' },
  { id: '4', label: 'Value for Money', type: 'neutral' },
  { id: '5', label: 'Packaging', type: 'positive' },
];

export const aiInsightText = `Based on 1,240 reviews analyzed, your products show strong performance in durability (+23% above category average). However, shipping delays are causing 34% of negative sentiment. Addressing logistics could improve overall rating by 0.3 points.`;

export const aiStrategyText = `Your product shows high viral potential with a 92% score. Current market conditions favor aggressive pricing strategy. Expected ROI: +47% with recommended optimizations.`;
