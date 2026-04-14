import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { SidebarLayout } from '@/components/layout/SidebarLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Search, Loader2, AlertCircle, Sparkles, BarChart3, Brain, Activity } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

// Import new components
import HealthScorecard from '@/components/product-analyzer/HealthScorecard';
import FlagsPanel from '@/components/product-analyzer/FlagsPanel';
import FactorBreakdownChart from '@/components/product-analyzer/FactorBreakdownChart';
import AIConsultantCard from '@/components/product-analyzer/AIConsultantCard';
import { ProductAnalysisResult } from '@/types/dashboard';

// Import hooks
import { useProductAnalysis, useConsultantAdvice } from '@/services/api';

export default function ProductAnalyzer() {
  const { t, i18n } = useTranslation();
  const [productUrl, setProductUrl] = useState('');
  const [analysis, setAnalysis] = useState<ProductAnalysisResult | null>(null);
  const [consultantVerdict, setConsultantVerdict] = useState<any>(null);

  const productAnalysisMutation = useProductAnalysis();
  const consultantMutation = useConsultantAdvice();

  const handleAnalyze = async () => {
    if (!productUrl.trim()) return;

    try {
      const result = await productAnalysisMutation.mutateAsync({
        url: productUrl,
        language: i18n.language
      });

      console.log("🔥 RAW API RESPONSE:", result);
      setAnalysis(result);

      if (result.status === 'success') {
        // Consultant call is separate — don't fail the whole analysis if it errors
        try {
          const verdict = await consultantMutation.mutateAsync({
            analysis: result,
            language: i18n.language
          });
          setConsultantVerdict(verdict);
        } catch (consultantError) {
          console.warn('Consultant unavailable:', consultantError);
          setConsultantVerdict({
            status: 'fallback',
            consultant_name: 'Uzum Business Consultant (Offline)',
            top_problems: [],
            overall_verdict: 'AI maslahatchi hozirda mavjud emas. Tahlil natijalari quyida ko\'rsatilgan.'
          });
        }
      }
    } catch (error) {
      console.error('Analysis failed:', error);
    }
  };

  const isLoading = productAnalysisMutation.isPending || consultantMutation.isPending;
  const hasError = productAnalysisMutation.isError || consultantMutation.isError;

  // Get current time-based greeting
  const getGreetingEmoji = () => {
    const hour = new Date().getHours();
    if (hour < 12) return '☀️';
    if (hour < 18) return '🌤️';
    return '🌙';
  };

  return (
    <SidebarLayout>
      <div className="space-y-6">
        {/* Hero Welcome Header */}
        <div className="relative overflow-hidden rounded-xl bg-gradient-to-r from-[hsl(186,80%,38%)] via-[hsl(220,70%,50%)] to-[hsl(265,60%,52%)] p-6 text-white shadow-lg">
          {/* Animated background pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="absolute -top-4 -right-4 h-32 w-32 rounded-full bg-white/20 blur-2xl animate-pulse" />
            <div className="absolute bottom-0 left-10 h-24 w-24 rounded-full bg-white/15 blur-xl animate-pulse" style={{ animationDelay: '1s' }} />
            <div className="absolute top-6 left-1/3 h-16 w-16 rounded-full bg-white/10 blur-lg animate-pulse" style={{ animationDelay: '2s' }} />
          </div>

          <div className="relative z-10 flex items-start justify-between">
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <span className="text-2xl">{getGreetingEmoji()}</span>
                <h1 className="text-2xl font-bold tracking-tight">
                  {t('product.welcome_greeting')}, <span className="text-cyan-200">Doniyor</span>
                </h1>
              </div>
              <p className="text-white/80 text-sm max-w-md">
                {t('product.welcome_sub')}
              </p>
            </div>

            {/* Stats chips */}
            <div className="hidden md:flex items-center gap-3">
              <div className="flex items-center gap-2 bg-white/15 backdrop-blur-sm rounded-full px-4 py-2 text-sm font-medium">
                <Brain className="h-4 w-4" />
                <span>UzumBERT</span>
              </div>
              <div className="flex items-center gap-2 bg-white/15 backdrop-blur-sm rounded-full px-4 py-2 text-sm font-medium">
                <Sparkles className="h-4 w-4" />
                <span>Groq AI</span>
              </div>
              <div className="flex items-center gap-2 bg-white/15 backdrop-blur-sm rounded-full px-4 py-2 text-sm font-medium">
                <Activity className="h-4 w-4" />
                <span>8-Factor</span>
              </div>
            </div>
          </div>
        </div>

        {/* URL Input Section */}
        <Card className="shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              {t('product.card_title')}
            </CardTitle>
            <CardDescription>
              {t('product.card_desc')}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-3">
              <Input
                placeholder={t('product.input_placeholder')}
                value={productUrl}
                onChange={(e) => setProductUrl(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
                className="flex-1"
              />
              <Button
                onClick={handleAnalyze}
                disabled={isLoading || !productUrl.trim()}
                className="bg-gradient-cyan text-white shadow-md hover:shadow-lg transition-shadow"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {t('product.analyzing')}
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-4 w-4" />
                    {t('product.analyze_btn')}
                  </>
                )}
              </Button>
            </div>

            {hasError && (
              <Alert variant="destructive" className="mt-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {t('product.error_msg')}
                  {productAnalysisMutation.error && (
                    <div className="mt-2 text-xs">
                      {String(productAnalysisMutation.error)}
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Loading State */}
        {isLoading && !analysis && (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
              <p className="text-muted-foreground">
                {productAnalysisMutation.isPending
                  ? t('product.analyzing_reviews')
                  : t('product.generating_ai')}
              </p>
            </CardContent>
          </Card>
        )}

        {/* Analysis Results */}
        {analysis && analysis.status === 'success' && (
          <div className="space-y-6">
            {/* Product Info Card */}
            <Card>
              <CardHeader>
                <CardTitle>{t('product.info_header')}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">{t('product.info_header')}</p>
                    <p className="font-medium truncate">
                      {analysis.product_info?.title || 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">{t('product.price')}</p>
                    <p className="font-medium">
                      {analysis.product_info?.price?.toLocaleString() || '0'} UZS
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">{t('product.seller')}</p>
                    <p className="font-medium truncate">
                      {analysis.product_info?.seller_name || 'Unknown'}
                    </p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">{t('product.reviews_analyzed')}</p>
                    <div className="flex flex-col">
                      <p className="font-medium">
                        {analysis.product_info?.analyzed_reviews || analysis.product_info?.review_count || 0} {t('product.text_reviews')}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        ({t('product.total_ratings')}: {analysis.product_info?.total_reviews || 0})
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Predicted Rating */}
            <HealthScorecard
              healthScore={
                analysis.health_analysis?.health_score ??
                analysis.health_analysis?.healthScore ??
                0
              }
              predictedRating={analysis.health_analysis?.predicted_rating || 0}
              actualRating={analysis.product_info?.actual_rating}
              category={analysis.product_info?.category || 'Unknown'}
              reviewCount={analysis.product_info?.review_count}
            />

            {/* Flags Panel */}
            <FlagsPanel
              healthAnalysis={analysis.health_analysis}
              benchmark={analysis.benchmark}
            />

            {/* Factor Breakdown Chart */}
            {analysis.health_analysis?.factor_breakdown && (
              <FactorBreakdownChart
                factors={Object.entries(analysis.health_analysis.factor_breakdown).map(([key, value]) => ({
                  name: t(`factors.${key}`, key.replace(/_/g, ' ')),
                  value: Number((Number(value) * 100).toFixed(1)),
                  fill: Number(value) > 0.05 ? '#ef4444' : '#10b981'
                }))}
              />
            )}

            {/* AI Consultant Card */}
            <AIConsultantCard
              verdict={consultantVerdict}
              isLoading={consultantMutation.isPending}
              onRefresh={handleAnalyze}
            />

            {/* Sample Reviews */}
            {analysis.raw_reviews && analysis.raw_reviews.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>{t('product.sample_reviews')}</CardTitle>
                  <CardDescription>
                    {t('product.sample_reviews_desc', { count: analysis.raw_reviews.length })}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {analysis.raw_reviews.map((review: any, idx: number) => (
                    <Card key={idx} className="bg-background/50">
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium">
                              {review.rating || 'N/A'}⭐
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {review.date || ''}
                            </span>
                          </div>
                        </div>
                        <p className="text-sm text-foreground">
                          {review.content || ''}
                        </p>
                      </CardContent>
                    </Card>
                  ))}
                </CardContent>
              </Card>
            )}

            {/* Cache Info */}
            {analysis.from_cache && (
              <Alert>
                <AlertDescription className="text-xs">
                  💾 {t('product.cache_info')}{' '}
                  {new Date(analysis.timestamp).toLocaleString()}
                </AlertDescription>
              </Alert>
            )}
          </div>
        )}

        {/* Error State */}
        {analysis && analysis.status !== 'success' && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {analysis.message || t('product.error_msg')}
            </AlertDescription>
          </Alert>
        )}

        {/* Welcome Section */}
        {!analysis && !isLoading && (
          <Card className="border-dashed border-2 border-primary/20">
            <CardContent className="py-12 text-center space-y-4">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-mixed text-white text-2xl shadow-lg">
                🏪
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-2">
                  {t('product.welcome_title')}
                </h3>
                <p className="text-sm text-muted-foreground max-w-md mx-auto">
                  {t('product.welcome_desc')}
                </p>
                <ul className="text-sm text-muted-foreground mt-3 space-y-1.5 inline-block text-left">
                  <li className="flex items-center gap-2">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-green-500/10 flex items-center justify-center text-xs">✅</span>
                    {t('product.welcome_1')}
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-green-500/10 flex items-center justify-center text-xs">✅</span>
                    {t('product.welcome_2')}
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-green-500/10 flex items-center justify-center text-xs">✅</span>
                    {t('product.welcome_3')}
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-green-500/10 flex items-center justify-center text-xs">✅</span>
                    {t('product.welcome_4')}
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-green-500/10 flex items-center justify-center text-xs">✅</span>
                    {t('product.welcome_5')}
                  </li>
                </ul>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </SidebarLayout>
  );
}
