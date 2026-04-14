import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { useState, useMemo } from 'react';
import { TrendingUp, Lightbulb } from 'lucide-react';
import { Factor } from '@/services/api';
import { useTranslation } from 'react-i18next';

interface MarketSimulatorProps {
    currentRating: number;
    factors: Factor[];
}

interface FactorImprovement {
    factor: Factor;
    improvementPercent: number;
}

export function MarketSimulator({ currentRating, factors }: MarketSimulatorProps) {
    const { t } = useTranslation();
    // Get top 3 negative factors
    const topNegativeFactors = useMemo(() => {
        return factors
            .filter(f => f.type === 'negative')
            .sort((a, b) => a.weight - b.weight) // Most negative first
            .slice(0, 3);
    }, [factors]);

    // State: improvement percentage for each factor (0-50%)
    const [improvements, setImprovements] = useState<Record<string, number>>(
        Object.fromEntries(topNegativeFactors.map(f => [f.name, 0]))
    );

    // Calculate projected rating based on improvements
    const { projectedRating, totalImpact, breakdown } = useMemo(() => {
        let totalImpact = 0;
        const breakdown: FactorImprovement[] = [];

        topNegativeFactors.forEach(factor => {
            const improvementPct = improvements[factor.name] || 0;

            // Formula: Reducing a negative factor = -1 * weight * (improvement% / 100)
            // Example: Defects weight = -0.63, improvement = 20%
            // Impact = -1 * (-0.63) * 0.20 = +0.126
            const impact = -1 * factor.weight * (improvementPct / 100);
            totalImpact += impact;

            breakdown.push({
                factor,
                improvementPercent: improvementPct
            });
        });

        const projectedRating = Math.min(5.0, currentRating + totalImpact);

        return {
            projectedRating,
            totalImpact,
            breakdown
        };
    }, [improvements, topNegativeFactors, currentRating]);

    const handleSliderChange = (factorName: string, value: number[]) => {
        setImprovements(prev => ({
            ...prev,
            [factorName]: value[0]
        }));
    };

    return (
        <Card className="border-primary/20 bg-gradient-to-br from-primary/5 to-primary/10">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Lightbulb className="h-5 w-5 text-primary" />
                        <CardTitle className="text-lg">{t('simulator.title')}</CardTitle>
                    </div>
                    <Badge variant="outline" className="bg-background">
                        <TrendingUp className="h-3 w-3 mr-1" />
                        {t('sidebar.platform')}
                    </Badge>
                </div>
                <CardDescription>
                    {t('simulator.subtitle')}
                </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6">
                {/* Current vs Projected */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-background/80 rounded-lg border">
                        <div className="text-sm text-muted-foreground mb-1">{t('simulator.currentRating')}</div>
                        <div className="text-2xl font-bold text-foreground">
                            {currentRating.toFixed(2)}⭐
                        </div>
                    </div>

                    <div className="p-4 bg-primary/10 rounded-lg border border-primary/30">
                        <div className="text-sm text-primary mb-1 flex items-center gap-1">
                            <TrendingUp className="h-3 w-3" />
                            {t('simulator.projectedRating')}
                        </div>
                        <div className="text-2xl font-bold text-primary">
                            {projectedRating.toFixed(2)}⭐
                        </div>
                        {totalImpact > 0 && (
                            <div className="text-xs text-primary/70 mt-1">
                                +{totalImpact.toFixed(3)} {t('simulator.improvement')}
                            </div>
                        )}
                    </div>
                </div>

                {/* Factor Improvement Sliders */}
                <div className="space-y-4">
                    <div className="text-sm font-medium text-foreground">
                        {t('simulator.targets')}
                    </div>

                    {topNegativeFactors.map((factor) => {
                        const currentValue = improvements[factor.name] || 0;
                        const impact = -1 * factor.weight * (currentValue / 100);

                        return (
                            <div key={factor.name} className="space-y-2">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <span className="text-sm font-medium">{t(`factors.${factor.name}`) || factor.display_name}</span>
                                        <Badge variant="destructive" className="text-xs px-1.5 py-0">
                                            {factor.weight.toFixed(2)}
                                        </Badge>
                                    </div>
                                    <span className="text-sm text-muted-foreground">
                                        {currentValue}% {t('simulator.reduction')}
                                    </span>
                                </div>

                                <Slider
                                    value={[currentValue]}
                                    onValueChange={(value) => handleSliderChange(factor.name, value)}
                                    max={50}
                                    step={5}
                                    className="w-full"
                                />

                                {currentValue > 0 && (
                                    <div className="text-xs text-muted-foreground">
                                        {t('simulator.impact')}: +{impact.toFixed(3)} ⭐
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>

                {/* Insight */}
                {totalImpact > 0.1 && (
                    <div className="p-3 bg-primary/5 border border-primary/20 rounded-md">
                        <div className="flex items-start gap-2">
                            <Lightbulb className="h-4 w-4 text-primary mt-0.5" />
                            <div className="text-sm text-foreground">
                                <span className="font-medium">Strategy Insight:</span> By reducing{' '}
                                {topNegativeFactors[0]?.display_name} by{' '}
                                {improvements[topNegativeFactors[0]?.name]}%, you could improve the market rating from{' '}
                                {currentRating.toFixed(2)}⭐ to {projectedRating.toFixed(2)}⭐.
                                {projectedRating >= 4.5 && (
                                    <span className="text-primary font-medium"> This would move the market into premium territory!</span>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* Methodology Note */}
                <div className="pt-2 border-t text-xs text-muted-foreground">
                    <strong>{t('simulator.calculation')}:</strong> {t('common.impact')} = -1 × {t('common.impact_coef')} × ({t('simulator.improvement')}% ÷ 100)
                </div>
            </CardContent>
        </Card>
    );
}
