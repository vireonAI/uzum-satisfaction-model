// =============================================================================
// 🔗 UZUM INTELLIGENCE HUB - API SERVICE LAYER
// =============================================================================
// 
// React Query hooks and API client for backend communication.
// Replace mock data imports with these hooks for real-time ML inference.
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// =============================================================================
// TYPES
// =============================================================================

export interface FactorPrediction {
    factor: string;
    prediction: number;
    confidence: number;
    label: 'positive' | 'negative' | 'neutral';
}

export interface AnalyzeResponse {
    text: string;
    text_length: number;
    script_type: string;
    factors: FactorPrediction[];
    positive_count: number;
    total_factors: number;
    overall_sentiment: 'positive' | 'negative' | 'neutral';
    timestamp: string;
}

export interface SimulationRequest {
    defect_rate: number;
    delivery_speed: number;
    response_time: number;
}

export interface SimulationResponse {
    projected_rating: number;
    rating_change: number;
    confidence: string;
    factors_impact: Record<string, number>;
}

export interface StatCard {
    id: string;
    title: string;
    value: string;
    trend?: string;
    trend_type?: 'up' | 'down' | 'flat';
    badge?: string;
    badge_type?: 'success' | 'warning' | 'destructive' | 'default';
    icon: string;
}

export interface MarketOverviewResponse {
    stats: StatCard[];
    total_reviews: number;
    avg_rating: number;
    last_updated: string;
}

export interface HealthResponse {
    status: string;
    ml_available: boolean;
    model_loaded: boolean;
    timestamp: string;
}

// Market Overview Types
export interface MarketStats {
    total_reviews: number;
    avg_rating: number;
    total_products: number;
    total_sellers: number;
    categories: number;
    date_range: {
        start: string | null;
        end: string | null;
    };
    satisfaction_rate: number;
}

export interface Factor {
    name: string;
    display_name: string;
    weight: number;
    type: 'positive' | 'negative';
    impact_level: 'critical' | 'high' | 'medium' | 'low';
}

export interface FactorImpactResponse {
    factors: Factor[];
    top_killer: {
        name: string;
        display_name: string;
        weight: number;
    } | null;
    top_strength: {
        name: string;
        display_name: string;
        weight: number;
    } | null;
}

export interface CategoryData {
    name: string;
    avg_rating: number;
    review_count: number;
    factor_scores: Record<string, number>;
}

export interface CategoryBreakdownResponse {
    categories: CategoryData[];
}

export interface TrendData {
    period: string;
    avg_rating: number;
    review_count: number;
}

export interface TrendsResponse {
    trends: TrendData[];
}


// =============================================================================
// API CLIENT
// =============================================================================

class ApiClient {
    private baseUrl: string;

    constructor(baseUrl: string = API_BASE_URL) {
        this.baseUrl = baseUrl;
    }

    private async request<T>(
        endpoint: string,
        options: RequestInit = {}
    ): Promise<T> {
        const url = `${this.baseUrl}${endpoint}`;

        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `API Error: ${response.status}`);
        }

        return response.json();
    }

    // Health Check
    async healthCheck(): Promise<HealthResponse> {
        return this.request<HealthResponse>('/api/health');
    }

    // Analyze single review
    async analyzeReview(
        text: string,
        confidenceThreshold: number = 0.5
    ): Promise<AnalyzeResponse> {
        return this.request<AnalyzeResponse>('/api/analyze', {
            method: 'POST',
            body: JSON.stringify({
                text,
                confidence_threshold: confidenceThreshold,
            }),
        });
    }

    // Analyze batch of reviews
    async analyzeBatch(
        texts: string[],
        confidenceThreshold: number = 0.5
    ): Promise<{ results: AnalyzeResponse[] }> {
        return this.request('/api/analyze/batch', {
            method: 'POST',
            body: JSON.stringify({
                texts,
                confidence_threshold: confidenceThreshold,
            }),
        });
    }

    // Run simulation
    async runSimulation(params: SimulationRequest): Promise<SimulationResponse> {
        return this.request<SimulationResponse>('/api/simulate', {
            method: 'POST',
            body: JSON.stringify(params),
        });
    }

    // Get market overview
    async getMarketOverview(): Promise<MarketOverviewResponse> {
        return this.request<MarketOverviewResponse>('/api/market-overview');
    }

    // Get coefficients
    async getCoefficients(): Promise<{ coefficients: any[] }> {
        return this.request('/api/coefficients');
    }

    // Analyze complete product
    async analyzeProduct(
        productUrl: string,
        language: string = 'uz'
    ): Promise<any> {
        return this.request(`/api/analyze/product?lang=${language}`, {
            method: 'POST',
            body: JSON.stringify({
                product_url: productUrl,
                force_refresh: false
            }),
        });
    }

    // Get AI consultant recommendations (new endpoint)
    async getConsultantAdvice(analysis: any, language: string = 'uz'): Promise<any> {
        return this.request('/api/consultant', {
            method: 'POST',
            body: JSON.stringify({ analysis, language }),
        });
    }

    // =========================================================================
    // MARKET ANALYZER ENDPOINTS
    // =========================================================================

    // Get list of categories
    async getCategories(): Promise<{ categories: Array<{ name: string; review_count: number }> }> {
        return this.request('/api/market/categories');
    }

    // Get market overview statistics
    async getMarketStats(category?: string): Promise<MarketStats> {
        const params = category ? `?category=${encodeURIComponent(category)}` : '';
        return this.request<MarketStats>(`/api/market/overview${params}`);
    }

    // Get factor impact analysis
    async getFactorImpact(category?: string): Promise<FactorImpactResponse> {
        const params = category ? `?category=${encodeURIComponent(category)}` : '';
        return this.request<FactorImpactResponse>(`/api/market/factor-impact${params}`);
    }

    // Get price-quality matrix scatter data
    async getPriceQualityMatrix(category?: string): Promise<any> {
        const params = category ? `?category=${encodeURIComponent(category)}` : '';
        return this.request(`/api/market/matrix${params}`);
    }

    // Get category breakdown
    async getCategoryBreakdown(): Promise<CategoryBreakdownResponse> {
        return this.request<CategoryBreakdownResponse>('/api/market/category-breakdown');
    }

    // Get market trends
    async getMarketTrends(period: 'daily' | 'weekly' | 'monthly' = 'monthly'): Promise<TrendsResponse> {
        return this.request<TrendsResponse>(`/api/market/trends?period=${period}`);
    }
}

// Export singleton instance
export const apiClient = new ApiClient();

// =============================================================================
// REACT QUERY HOOKS
// =============================================================================

import { useQuery, useMutation, UseQueryOptions } from '@tanstack/react-query';

// Query Keys
export const queryKeys = {
    health: ['health'] as const,
    marketOverview: ['market-overview'] as const,
    categories: ['categories'] as const,
    marketStats: (category?: string) => ['market-stats', category] as const,
    factorImpact: (category?: string) => ['factor-impact', category] as const,
    priceQualityMatrix: (category?: string) => ['price-quality-matrix', category] as const,
    categoryBreakdown: ['category-breakdown'] as const,
    marketTrends: (period: string) => ['market-trends', period] as const,
    coefficients: ['coefficients'] as const,
    analyze: (text: string) => ['analyze', text] as const,
};

// Health check hook
export function useHealthCheck(options?: Omit<UseQueryOptions<HealthResponse>, 'queryKey' | 'queryFn'>) {
    return useQuery({
        queryKey: queryKeys.health,
        queryFn: () => apiClient.healthCheck(),
        staleTime: 30000, // 30 seconds
        ...options,
    });
}

// Categories hook
export function useCategories(options?: Omit<UseQueryOptions<{ categories: Array<{ name: string; review_count: number }> }>, 'queryKey' | 'queryFn'>) {
    return useQuery({
        queryKey: queryKeys.categories,
        queryFn: () => apiClient.getCategories(),
        staleTime: 300000, // 5 minutes - categories don't change often
        ...options,
    });
}


// Market stats hook
export function useMarketStats(category?: string, options?: Omit<UseQueryOptions<MarketStats>, 'queryKey' | 'queryFn'>) {
    return useQuery({
        queryKey: queryKeys.marketStats(category),
        queryFn: () => apiClient.getMarketStats(category),
        staleTime: 60000, // 1 minute - data doesn't change often
        ...options,
    });
}

// Factor impact hook
export function useFactorImpact(category?: string, options?: Omit<UseQueryOptions<FactorImpactResponse>, 'queryKey' | 'queryFn'>) {
    return useQuery({
        queryKey: queryKeys.factorImpact(category),
        queryFn: () => apiClient.getFactorImpact(category),
        staleTime: 300000, // 5 minutes - static model weights
        ...options,
    });
}

// Price-quality matrix hook
export function usePriceQualityMatrix(category?: string, options?: Omit<UseQueryOptions<any>, 'queryKey' | 'queryFn'>) {
    return useQuery({
        queryKey: queryKeys.priceQualityMatrix(category),
        queryFn: () => apiClient.getPriceQualityMatrix(category),
        staleTime: 300000, // 5 minutes
        ...options,
    });
}

// Category breakdown hook
export function useCategoryBreakdown(options?: Omit<UseQueryOptions<CategoryBreakdownResponse>, 'queryKey' | 'queryFn'>) {
    return useQuery({
        queryKey: queryKeys.categoryBreakdown,
        queryFn: () => apiClient.getCategoryBreakdown(),
        staleTime: 60000, // 1 minute
        ...options,
    });
}

// Market trends hook
export function useMarketTrends(
    period: 'daily' | 'weekly' | 'monthly' = 'monthly',
    options?: Omit<UseQueryOptions<TrendsResponse>, 'queryKey' | 'queryFn'>
) {
    return useQuery({
        queryKey: queryKeys.marketTrends(period),
        queryFn: () => apiClient.getMarketTrends(period),
        staleTime: 60000, // 1 minute
        ...options,
    });
}


// Market overview hook
export function useMarketOverview(options?: Omit<UseQueryOptions<MarketOverviewResponse>, 'queryKey' | 'queryFn'>) {
    return useQuery({
        queryKey: queryKeys.marketOverview,
        queryFn: () => apiClient.getMarketOverview(),
        staleTime: 60000, // 1 minute
        ...options,
    });
}

// Coefficients hook
export function useCoefficients() {
    return useQuery({
        queryKey: queryKeys.coefficients,
        queryFn: () => apiClient.getCoefficients(),
        staleTime: 300000, // 5 minutes
    });
}

// Analyze review mutation
export function useAnalyzeReview() {
    return useMutation({
        mutationFn: ({ text, threshold }: { text: string; threshold?: number }) =>
            apiClient.analyzeReview(text, threshold),
    });
}

// Simulation mutation
export function useSimulation() {
    return useMutation({
        mutationFn: (params: SimulationRequest) => apiClient.runSimulation(params),
    });
}

// Product analysis mutation
export const useProductAnalysis = () => {
    return useMutation({
        mutationFn: ({ url, language }: { url: string; language: string }) => apiClient.analyzeProduct(url, language),
    });
};

// Consultant advice mutation
export function useConsultantAdvice() {
    return useMutation({
        mutationFn: ({ analysis, language }: { analysis: any; language: string }) =>
            apiClient.getConsultantAdvice(analysis, language),
    });
}
