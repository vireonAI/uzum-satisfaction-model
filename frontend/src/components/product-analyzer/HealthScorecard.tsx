import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Heart, Star, TrendingUp, FolderOpen } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface Props {
    healthScore: number; // 1-10
    predictedRating: number; // 1-5
    actualRating?: number; // 1-5
    category: string;
    reviewCount?: number;
}

const HealthScorecard = ({
    healthScore,
    predictedRating,
    actualRating,
    category,
    reviewCount
}: Props) => {
    const { t } = useTranslation();

    const getHealthColor = (score: number) => {
        if (score >= 7) return { icon: '🟢', color: 'text-green-500', bg: 'bg-green-500/10', border: 'border-green-500/20', label: t('product.excellent') };
        if (score >= 5) return { icon: '🟡', color: 'text-yellow-500', bg: 'bg-yellow-500/10', border: 'border-yellow-500/20', label: t('product.good') };
        return { icon: '🔴', color: 'text-red-500', bg: 'bg-red-500/10', border: 'border-red-500/20', label: t('product.needs_work') };
    };

    const healthDisplay = getHealthColor(healthScore);
    const ratingDiff = actualRating ? (predictedRating - actualRating).toFixed(2) : null;

    return (
        <div className="grid gap-4 grid-cols-2 md:grid-cols-4">
            {/* Health Score */}
            <Card className={`${healthDisplay.bg} ${healthDisplay.border} border`}>
                <CardContent className="p-4">
                    <div className="flex items-center gap-2 mb-1">
                        <Heart className={`h-4 w-4 ${healthDisplay.color}`} />
                        <span className="text-xs font-medium text-muted-foreground">{t('product.health_score')}</span>
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className={`text-3xl font-bold ${healthDisplay.color}`}>
                            {healthScore.toFixed(1)}
                        </span>
                        <span className="text-sm text-muted-foreground">/10</span>
                    </div>
                    <Badge variant="outline" className="mt-2 text-xs">
                        {healthDisplay.icon} {healthDisplay.label}
                    </Badge>
                </CardContent>
            </Card>

            {/* Predicted Rating */}
            <Card className="bg-blue-500/10 border-blue-500/20 border">
                <CardContent className="p-4">
                    <div className="flex items-center gap-2 mb-1">
                        <Star className="h-4 w-4 text-blue-500" />
                        <span className="text-xs font-medium text-muted-foreground">{t('product.predicted_rating')}</span>
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-bold text-blue-500">
                            {predictedRating.toFixed(2)}
                        </span>
                        <span className="text-sm text-muted-foreground">⭐</span>
                    </div>
                    {reviewCount && (
                        <Badge variant="secondary" className="mt-2 text-xs">
                            {t('product.based_on_reviews', { count: reviewCount })}
                        </Badge>
                    )}
                </CardContent>
            </Card>

            {/* Actual Rating */}
            <Card className="bg-purple-500/10 border-purple-500/20 border">
                <CardContent className="p-4">
                    <div className="flex items-center gap-2 mb-1">
                        <Star className="h-4 w-4 text-purple-500 fill-purple-500" />
                        <span className="text-xs font-medium text-muted-foreground">{t('product.actual_rating')}</span>
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-3xl font-bold text-purple-500">
                            {actualRating?.toFixed(2) || 'N/A'}
                        </span>
                        {actualRating && <span className="text-sm text-muted-foreground">⭐</span>}
                    </div>
                    {ratingDiff && (
                        <Badge
                            variant={parseFloat(ratingDiff) >= 0 ? 'default' : 'destructive'}
                            className="mt-2 text-xs"
                        >
                            {parseFloat(ratingDiff) >= 0 ? '↑' : '↓'} {Math.abs(parseFloat(ratingDiff))} {t('product.from_predicted')}
                        </Badge>
                    )}
                </CardContent>
            </Card>

            {/* Category */}
            <Card className="bg-background/50 border">
                <CardContent className="p-4">
                    <div className="flex items-center gap-2 mb-1">
                        <FolderOpen className="h-4 w-4 text-muted-foreground" />
                        <span className="text-xs font-medium text-muted-foreground">{t('product.category')}</span>
                    </div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-lg font-semibold text-foreground truncate">
                            {category || 'Unknown'}
                        </span>
                    </div>
                    <Badge variant="outline" className="mt-2 text-xs">
                        <TrendingUp className="h-3 w-3 mr-1" />
                        {t('product.market_analysis')}
                    </Badge>
                </CardContent>
            </Card>
        </div>
    );
};

export default HealthScorecard;
