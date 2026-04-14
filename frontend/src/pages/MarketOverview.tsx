import { SidebarLayout } from '@/components/layout/SidebarLayout';
import { StatCard } from '@/components/dashboard/StatCard';
import { ImpactAnalysisChart } from '@/components/dashboard/ImpactAnalysisChart';
import { MarketSimulator } from '@/components/dashboard/MarketSimulator';
import { PriceQualityMatrix } from '@/components/dashboard/PriceQualityMatrix';
import { useMarketStats, useFactorImpact, useCategories, usePriceQualityMatrix } from '@/services/api';
import { Loader2, AlertCircle, Filter } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardDescription } from '@/components/ui/card';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

export default function MarketOverview() {
    // Category selection state
    const [selectedCategory, setSelectedCategory] = useState<string | undefined>(undefined);
    const { t, i18n } = useTranslation();

    // Fetch data
    const { data: categoriesData, isLoading: categoriesLoading } = useCategories();
    const { data: marketStats, isLoading: statsLoading, error: statsError } = useMarketStats(selectedCategory);
    const { data: factorData, isLoading: factorLoading, error: factorError } = useFactorImpact(selectedCategory);
    const { data: matrixData, isLoading: matrixLoading } = usePriceQualityMatrix(selectedCategory);

    const isLoading = statsLoading || factorLoading;
    const hasError = Boolean(statsError || factorError);

    // Transform market stats into StatCard format
    const statsCards = useMemo(() => {
        if (!marketStats) return [];

        return [
            {
                id: 'total-reviews',
                title: t('dashboard.totalReviews'),
                value: marketStats.total_reviews.toLocaleString(),
                trend: selectedCategory ? `${selectedCategory}` : `${marketStats.categories} ${t('dashboard.category').toLowerCase()}s`,
                trend_type: 'flat' as const,
                badge: selectedCategory ? t('common.filtered') : t('common.dataset'),
                badge_type: 'default' as const,
                icon: 'FileText'
            },
            {
                id: 'avg-rating',
                title: selectedCategory ? t('dashboard.categoryAverage') : t('dashboard.marketAverage'),
                value: `${marketStats.avg_rating.toFixed(2)}⭐`,
                trend: `${marketStats.satisfaction_rate.toFixed(1)}% ${t('common.satisfied')}`,
                trend_type: marketStats.avg_rating >= 4.0 ? 'up' as const : 'down' as const,
                badge: t('common.live'),
                badge_type: 'success' as const,
                icon: 'TrendingUp'
            },
            {
                id: 'top-strength',
                title: t('dashboard.topStrength'),
                value: factorData?.top_strength?.name ? t(`factors.${factorData.top_strength.name}`) : 'Loading...',
                trend: `+${Math.abs(factorData?.top_strength?.weight || 0).toFixed(2)} ${t('common.impact')}`,
                trend_type: 'up' as const,
                badge: t('common.positive'),
                badge_type: 'success' as const,
                icon: 'TrendingUp'
            },
            {
                id: 'main-problem',
                title: t('dashboard.mainProblem'),
                value: factorData?.top_killer?.name ? t(`factors.${factorData.top_killer.name}`) : 'None',
                trend: `${Math.abs(factorData?.top_killer?.weight || 0).toFixed(2)} ${t('common.negative_impact')}`,
                trend_type: 'down' as const,
                badge: t('common.critical'),
                badge_type: 'destructive' as const,
                icon: 'AlertTriangle'
            }
        ];
    }, [marketStats, factorData, selectedCategory, t, i18n.language]);

    // Transform factor data for Tornado Chart
    const impactChartData = useMemo(() => {
        if (!factorData?.factors) return [];
        return factorData.factors.map(factor => ({
            factor: factor.display_name,
            weight: factor.weight,
            type: factor.type
        }));
    }, [factorData]);

    return (
        <SidebarLayout>
            <div className="space-y-6">
                {/* Page Header with Category Filter */}
                <div className="flex items-center justify-between gap-4">
                    <div>
                        <h1 className="text-2xl font-bold text-foreground">{t('dashboard.title')}</h1>
                        <CardDescription>
                            {t('dashboard.subtitle', { count: marketStats?.total_reviews || 0 })}
                        </CardDescription>
                    </div>

                    {/* Category Filter */}
                    <Card className="p-3 min-w-[200px]">
                        <div className="flex items-center gap-2 mb-2">
                            <Filter className="h-4 w-4 text-muted-foreground" />
                            <span className="text-xs font-medium text-muted-foreground">{t('dashboard.category')}</span>
                        </div>
                        <Select
                            value={selectedCategory || 'all'}
                            onValueChange={(val) => setSelectedCategory(val === 'all' ? undefined : val)}
                            disabled={categoriesLoading}
                        >
                            <SelectTrigger className="w-full h-8 text-sm">
                                <SelectValue placeholder={t('dashboard.category')} />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="all">All Categories</SelectItem>
                                {categoriesData?.categories.map((cat) => (
                                    <SelectItem key={cat.name} value={cat.name}>
                                        {cat.name} ({cat.review_count.toLocaleString()})
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </Card>
                </div>

                {/* Error State */}
                {hasError && (
                    <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>
                            Failed to load market data. Ensure backend is running at localhost:8000
                        </AlertDescription>
                    </Alert>
                )}

                {/* Loading State */}
                {isLoading && !hasError && (
                    <div className="flex items-center justify-center py-16">
                        <Loader2 className="h-8 w-8 animate-spin text-primary" />
                        <span className="ml-3 text-muted-foreground">Loading market analysis...</span>
                    </div>
                )}

                {/* Main Content */}
                {!isLoading && !hasError && (
                    <>
                        {/* Stats Row */}
                        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                            {statsCards.map((stat) => (
                                <StatCard key={stat.id} data={stat} />
                            ))}
                        </div>

                        {/* Mission Control: Chart + Simulator Side-by-Side */}
                        <div className="grid gap-6 lg:grid-cols-5">
                            {/* Left: Tornado Chart (60% = 3 cols) */}
                            <div className="lg:col-span-3">
                                <ImpactAnalysisChart
                                    data={impactChartData}
                                    onExport={() => console.log('Exporting report...')}
                                />
                            </div>

                            {/* Right: Simulator (40% = 2 cols) - with emphasis styling */}
                            <div className="lg:col-span-2">
                                {marketStats && factorData && (
                                    <div className="relative">
                                        {/* Subtle glow effect */}
                                        <div className="absolute -inset-1 bg-gradient-to-r from-primary/20 via-primary/10 to-primary/20 rounded-xl blur-sm opacity-70" />
                                        <div className="relative">
                                            <MarketSimulator
                                                currentRating={marketStats.avg_rating}
                                                factors={factorData.factors}
                                            />
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Price-Quality Matrix — Full Width */}
                        {matrixData && matrixData.points && matrixData.points.length > 0 && (
                            <PriceQualityMatrix data={matrixData} />
                        )}

                        {/* Footer */}
                        <div className="text-center text-xs text-muted-foreground pt-4 border-t border-border space-y-1">
                            <p>Elektron tijorat bozorida mijozlar qoniqishining asosiy omillarini aniqlash modeli</p>
                            <p className="opacity-60">Data: uzum_labeled.csv • Model: Regression Coefficients • {new Date().getFullYear()}</p>
                        </div>
                    </>
                )}
            </div>
        </SidebarLayout>
    );
}
