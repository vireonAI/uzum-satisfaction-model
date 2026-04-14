import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
    Sparkles,
    TrendingUp,
    Clock,
    Target,
    ChevronDown,
    ChevronRight,
    RefreshCw,
    Zap,
    Star
} from 'lucide-react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { useTranslation } from 'react-i18next';

interface Problem {
    rank: number;
    problem_name: string;
    current_impact: string;
    solution_steps: string[];
    estimated_improvement: string;
}

interface ROIForecast {
    timeline: string;
    rating_increase: string;
    sales_increase_estimate: string;
    confidence: 'High' | 'Medium' | 'Low';
}

interface ConsultantVerdict {
    status: string;
    consultant_name?: string;
    confidence_level?: number;
    top_problems?: Problem[];
    roi_forecast?: ROIForecast;
    overall_verdict?: string;
    success_metrics?: string[];
    quick_wins?: string[];
    message?: string;
}

interface Props {
    verdict: ConsultantVerdict | null;
    isLoading?: boolean;
    onRefresh?: () => void;
}

const AIConsultantCard = ({ verdict, isLoading = false, onRefresh }: Props) => {
    const { t } = useTranslation();
    const [expandedProblems, setExpandedProblems] = useState<Set<number>>(new Set());

    const toggleProblem = (rank: number) => {
        const newExpanded = new Set(expandedProblems);
        if (newExpanded.has(rank)) {
            newExpanded.delete(rank);
        } else {
            newExpanded.add(rank);
        }
        setExpandedProblems(newExpanded);
    };

    if (isLoading) {
        return (
            <Card className="border-gradient-cyan bg-card/50 backdrop-blur">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Sparkles className="h-5 w-5 text-gradient-cyan" />
                        {t('product.ai_consultant')}
                    </CardTitle>
                    <CardDescription>{t('product.ai_generating')}</CardDescription>
                </CardHeader>
                <CardContent className="flex items-center justify-center py-12">
                    <RefreshCw className="h-8 w-8 animate-spin text-gradient-cyan" />
                </CardContent>
            </Card>
        );
    }

    if (!verdict || verdict.status !== 'success') {
        return (
            <Card className="border-destructive/50">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-destructive">
                        <Sparkles className="h-5 w-5" />
                        {t('product.ai_unavailable')}
                    </CardTitle>
                    <CardDescription>
                        {verdict?.message || t('product.ai_no_recs')}
                    </CardDescription>
                </CardHeader>
                {onRefresh && (
                    <CardContent>
                        <Button onClick={onRefresh} variant="outline" size="sm">
                            <RefreshCw className="mr-2 h-4 w-4" />
                            {t('product.retry')}
                        </Button>
                    </CardContent>
                )}
            </Card>
        );
    }

    const roi = verdict.roi_forecast;
    const confidenceColor = roi?.confidence === 'High'
        ? 'text-green-500'
        : roi?.confidence === 'Medium'
            ? 'text-yellow-500'
            : 'text-red-500';

    return (
        <Card className="border-gradient-cyan bg-gradient-to-br from-card to-card/50 backdrop-blur">
            <CardHeader>
                <div className="flex items-start justify-between">
                    <div className="space-y-1">
                        <CardTitle className="flex items-center gap-2">
                            <Sparkles className="h-5 w-5 text-gradient-cyan" />
                            {verdict.consultant_name || t('product.ai_consultant')}
                        </CardTitle>
                        <CardDescription>
                            {t('product.powered_by')} {' '}
                            <span className={confidenceColor}>
                                {((verdict.confidence_level || 0) * 100).toFixed(0)}%
                            </span>
                        </CardDescription>
                    </div>
                    {onRefresh && (
                        <Button onClick={onRefresh} variant="ghost" size="icon">
                            <RefreshCw className="h-4 w-4" />
                        </Button>
                    )}
                </div>
            </CardHeader>

            <CardContent className="space-y-6">
                {/* Top Problems */}
                {verdict.top_problems && verdict.top_problems.length > 0 && (
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                            <Target className="h-4 w-4 text-destructive" />
                            {t('product.top_issues', { count: verdict.top_problems.length })}
                        </h3>

                        <div className="space-y-2">
                            {verdict.top_problems.map((problem) => (
                                <Collapsible
                                    key={problem.rank}
                                    open={expandedProblems.has(problem.rank)}
                                    onOpenChange={() => toggleProblem(problem.rank)}
                                >
                                    <Card className="border-destructive/20">
                                        <CollapsibleTrigger className="w-full">
                                            <CardHeader className="pb-3">
                                                <div className="flex items-start justify-between">
                                                    <div className="flex-1 text-left">
                                                        <div className="flex items-center gap-2">
                                                            <Badge variant="destructive" className="text-xs">
                                                                #{problem.rank}
                                                            </Badge>
                                                            <CardTitle className="text-base">
                                                                {problem.problem_name}
                                                            </CardTitle>
                                                            {expandedProblems.has(problem.rank) ? (
                                                                <ChevronDown className="h-4 w-4 text-muted-foreground" />
                                                            ) : (
                                                                <ChevronRight className="h-4 w-4 text-muted-foreground" />
                                                            )}
                                                        </div>
                                                        <CardDescription className="mt-1">
                                                            {problem.current_impact}
                                                        </CardDescription>
                                                    </div>
                                                </div>
                                            </CardHeader>
                                        </CollapsibleTrigger>

                                        <CollapsibleContent>
                                            <CardContent className="pt-0 space-y-3">
                                                <div>
                                                    <h4 className="text-sm font-medium mb-2">{t('product.solution_steps')}</h4>
                                                    <ol className="list-decimal list-inside space-y-1 text-sm text-muted-foreground">
                                                        {problem.solution_steps.map((step, idx) => (
                                                            <li key={idx}>{step}</li>
                                                        ))}
                                                    </ol>
                                                </div>
                                                <div className="flex items-center gap-2 p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                                                    <TrendingUp className="h-4 w-4 text-green-500" />
                                                    <span className="text-sm font-medium text-green-500">
                                                        {problem.estimated_improvement}
                                                    </span>
                                                </div>
                                            </CardContent>
                                        </CollapsibleContent>
                                    </Card>
                                </Collapsible>
                            ))}
                        </div>
                    </div>
                )}

                {/* ROI Forecast */}
                {roi && (
                    <div className="space-y-3">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                            <TrendingUp className="h-4 w-4 text-green-500" />
                            {t('product.roi_forecast')}
                        </h3>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <Card className="bg-background/50">
                                <CardContent className="p-3 text-center">
                                    <Clock className="h-4 w-4 mx-auto mb-1 text-muted-foreground" />
                                    <p className="text-xs text-muted-foreground">{t('product.timeline')}</p>
                                    <p className="text-sm font-semibold">{roi.timeline}</p>
                                </CardContent>
                            </Card>

                            <Card className="bg-green-500/10 border-green-500/20">
                                <CardContent className="p-3 text-center">
                                    <Star className="h-4 w-4 mx-auto mb-1 text-green-500" />
                                    <p className="text-xs text-muted-foreground">{t('product.rating_boost')}</p>
                                    <p className="text-sm font-semibold text-green-500">{roi.rating_increase}</p>
                                </CardContent>
                            </Card>

                            <Card className="bg-blue-500/10 border-blue-500/20">
                                <CardContent className="p-3 text-center">
                                    <TrendingUp className="h-4 w-4 mx-auto mb-1 text-blue-500" />
                                    <p className="text-xs text-muted-foreground">{t('product.sales_increase')}</p>
                                    <p className="text-sm font-semibold text-blue-500">{roi.sales_increase_estimate}</p>
                                </CardContent>
                            </Card>

                            <Card className="bg-background/50">
                                <CardContent className="p-3 text-center">
                                    <Target className="h-4 w-4 mx-auto mb-1 text-muted-foreground" />
                                    <p className="text-xs text-muted-foreground">{t('product.confidence')}</p>
                                    <p className={`text-sm font-semibold ${confidenceColor}`}>{roi.confidence}</p>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                )}

                {/* Quick Wins */}
                {verdict.quick_wins && verdict.quick_wins.length > 0 && (
                    <div className="space-y-2">
                        <h3 className="text-sm font-semibold flex items-center gap-2">
                            <Zap className="h-4 w-4 text-yellow-500" />
                            {t('product.quick_wins')}
                        </h3>
                        <div className="flex flex-wrap gap-2">
                            {verdict.quick_wins.map((win, idx) => (
                                <Badge key={idx} variant="outline" className="border-yellow-500/50 text-yellow-500">
                                    ⚡ {win}
                                </Badge>
                            ))}
                        </div>
                    </div>
                )}

                {/* Success Metrics */}
                {verdict.success_metrics && verdict.success_metrics.length > 0 && (
                    <div className="space-y-2">
                        <h3 className="text-sm font-semibold">{t('product.success_metrics')}</h3>
                        <ul className="space-y-1 text-sm text-muted-foreground">
                            {verdict.success_metrics.map((metric, idx) => (
                                <li key={idx} className="flex items-center gap-2">
                                    <div className="h-1.5 w-1.5 rounded-full bg-green-500" />
                                    {metric}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}

                {/* Overall Verdict */}
                {verdict.overall_verdict && (
                    <Card className="bg-gradient-cyan/10 border-gradient-cyan/50">
                        <CardContent className="p-4">
                            <p className="text-sm leading-relaxed">{verdict.overall_verdict}</p>
                        </CardContent>
                    </Card>
                )}
            </CardContent>
        </Card>
    );
};

export default AIConsultantCard;
