import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { CheckCircle2, AlertTriangle } from 'lucide-react';
import { HealthAnalysis, Benchmark } from '@/types/dashboard';
import { useTranslation } from 'react-i18next';

interface Props {
    healthAnalysis?: HealthAnalysis;
    benchmark?: Benchmark;
}

const FlagsPanel = ({ healthAnalysis, benchmark }: Props) => {
    const { t } = useTranslation();

    if (!healthAnalysis || !benchmark) return null;

    const positiveFlags: string[] = [];
    const negativeFlags: string[] = [];
    const factorBreakdown = healthAnalysis.factor_breakdown || {};

    // Positive flags
    if (benchmark.performance === 'above_average') {
        positiveFlags.push(benchmark.benchmark_message || t('product.above_category'));
    }

    if (healthAnalysis.predicted_rating >= 4.0) {
        positiveFlags.push(t('product.rating_at_target'));
    }

    // Use translated factor names
    const factorKeys: Record<string, string> = {
        product_defects: 'factors.product_defects',
        product_quality: 'factors.product_quality',
        accuracy_expectation: 'factors.accuracy_expectation',
        logistics_delivery: 'factors.logistics_delivery',
        packaging_condition: 'factors.packaging_condition',
        price_value: 'factors.price_value',
        seller_service: 'factors.seller_service',
        specifications: 'factors.specifications'
    };

    // Detect issues from factor_breakdown (problem rates from negative reviews)
    Object.entries(factorBreakdown).forEach(([factor, value]) => {
        const numValue = typeof value === 'number' ? value : 0;

        if (numValue >= 0.03) {
            const percentage = (numValue * 100).toFixed(1);
            const displayName = t(factorKeys[factor] || `factors.${factor}`, factor);

            let severity = '';
            if (numValue >= 0.15) severity = t('product.severity_critical');
            else if (numValue >= 0.08) severity = t('product.severity_high');
            else if (numValue >= 0.03) severity = t('product.severity_moderate');

            negativeFlags.push(`${severity}: ${displayName} ${t('product.issues')} (${percentage}%)`);
        }
    });

    // Add benchmark-based flags
    if (benchmark.performance === 'below_average' && benchmark.rating_diff) {
        negativeFlags.push(
            `${Math.abs(benchmark.rating_diff).toFixed(2)}⭐ ${t('product.below_average')}`
        );
    }

    // Update positive flags based on what we found
    if (negativeFlags.length === 0) {
        positiveFlags.push(t('product.no_critical'));
    }

    return (
        <div className="grid gap-4 md:grid-cols-2">
            {/* Positive Indicators */}
            <Card className="border-green-500/20 bg-green-500/5">
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2 text-green-500">
                        <CheckCircle2 className="h-4 w-4" />
                        {t('product.positive_indicators')}
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                    {positiveFlags.length > 0 ? (
                        positiveFlags.map((flag, idx) => (
                            <div key={idx} className="flex items-start gap-2">
                                <Badge variant="outline" className="border-green-500/50 text-green-500 shrink-0">
                                    ✅
                                </Badge>
                                <span className="text-sm text-foreground">{flag}</span>
                            </div>
                        ))
                    ) : (
                        <p className="text-sm text-muted-foreground italic">
                            {t('product.no_positive')}
                        </p>
                    )}
                </CardContent>
            </Card>

            {/* Negative Warnings */}
            <Card className="border-destructive/20 bg-destructive/5">
                <CardHeader className="pb-3">
                    <CardTitle className="text-base flex items-center gap-2 text-destructive">
                        <AlertTriangle className="h-4 w-4" />
                        {t('product.warning_signals')}
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                    {negativeFlags.length > 0 ? (
                        negativeFlags.map((flag, idx) => (
                            <div key={idx} className="flex items-start gap-2">
                                <Badge variant="destructive" className="shrink-0 text-xs">
                                    ⚠️
                                </Badge>
                                <span className="text-sm text-foreground">{flag}</span>
                            </div>
                        ))
                    ) : (
                        <p className="text-sm text-muted-foreground italic">
                            {t('product.no_issues')}
                        </p>
                    )}
                </CardContent>
            </Card>
        </div>
    );
};

export default FlagsPanel;
