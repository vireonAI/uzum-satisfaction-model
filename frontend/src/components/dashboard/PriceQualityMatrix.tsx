import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    CartesianGrid,
    ResponsiveContainer,
    Tooltip,
    ReferenceLine,
    Cell,
} from 'recharts';
import { useTranslation } from 'react-i18next';

interface MatrixPoint {
    product_id: string;
    title: string;
    price: number;
    rating: number;
    category: string;
    review_count: number;
    quadrant: 'premium' | 'overpriced' | 'hidden_gem' | 'budget';
}

interface PriceQualityMatrixProps {
    data: {
        points: MatrixPoint[];
        median_price: number;
        rating_threshold: number;
        total_products: number;
        quadrant_counts: {
            premium: number;
            overpriced: number;
            hidden_gem: number;
            budget: number;
        };
    };
}

// Format price for axis: 9000 → "9K", 190000 → "190K", 1500000 → "1.5M"
function formatPrice(value: number): string {
    if (value >= 1_000_000) {
        return `${(value / 1_000_000).toFixed(1)}M`;
    }
    if (value >= 1_000) {
        return `${Math.round(value / 1_000)}K`;
    }
    return String(value);
}

// Color by rating tier
function getDotColor(rating: number): string {
    if (rating >= 4.5) return 'hsl(142 76% 46%)';   // Bright green
    if (rating >= 3.5) return 'hsl(45 93% 55%)';    // Warm amber
    return 'hsl(0 84% 55%)';                         // Red
}

export function PriceQualityMatrix({ data }: PriceQualityMatrixProps) {
    const { points, median_price, rating_threshold, quadrant_counts } = data;
    const { t } = useTranslation();

    // Custom tooltip
    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const point = payload[0].payload as MatrixPoint;
            return (
                <div className="bg-popover border border-border rounded-lg shadow-xl p-3 max-w-[280px]">
                    <p className="font-semibold text-foreground text-sm leading-tight mb-2">
                        {point.title}
                    </p>
                    <div className="space-y-1 text-xs">
                        <div className="flex justify-between gap-4">
                            <span className="text-muted-foreground">Price:</span>
                            <span className="font-mono font-medium text-foreground">
                                {point.price.toLocaleString()} UZS
                            </span>
                        </div>
                        <div className="flex justify-between gap-4">
                            <span className="text-muted-foreground">Rating:</span>
                            <span className="font-mono font-medium text-foreground">
                                {point.rating.toFixed(2)} ⭐
                            </span>
                        </div>
                        <div className="flex justify-between gap-4">
                            <span className="text-muted-foreground">Reviews:</span>
                            <span className="font-mono font-medium text-foreground">
                                {point.review_count}
                            </span>
                        </div>
                        <div className="flex justify-between gap-4">
                            <span className="text-muted-foreground">Category:</span>
                            <span className="font-medium text-foreground truncate">
                                {point.category}
                            </span>
                        </div>
                    </div>
                    <div className={`mt-2 pt-2 border-t border-border text-xs font-medium ${point.quadrant === 'hidden_gem' ? 'text-green-400' :
                        point.quadrant === 'overpriced' ? 'text-red-400' :
                            point.quadrant === 'premium' ? 'text-blue-400' :
                                'text-muted-foreground'
                        }`}>
                        {point.quadrant === 'hidden_gem' ? `💎 ${t('matrix.gems')}` :
                            point.quadrant === 'overpriced' ? `⚠️ ${t('matrix.overpriced')}` :
                                point.quadrant === 'premium' ? `👑 ${t('matrix.premium')}` :
                                    `📦 ${t('matrix.budget')}`}
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <Card className="border-border bg-card">
            <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="text-lg font-semibold">
                            {t('matrix.title')}
                        </CardTitle>
                        <p className="text-xs text-muted-foreground mt-1">
                            {t('matrix.subtitle', { count: data.total_products })}
                        </p>
                    </div>

                    {/* Quadrant Summary Badges */}
                    <div className="flex gap-2 flex-wrap justify-end">
                        <span className="inline-flex items-center gap-1 rounded-full bg-green-500/10 px-2 py-0.5 text-xs font-medium text-green-400 border border-green-500/20">
                            💎 {quadrant_counts.hidden_gem} {t('matrix.gems')}
                        </span>
                        <span className="inline-flex items-center gap-1 rounded-full bg-red-500/10 px-2 py-0.5 text-xs font-medium text-red-400 border border-red-500/20">
                            ⚠️ {quadrant_counts.overpriced} {t('matrix.overpriced')}
                        </span>
                        <span className="inline-flex items-center gap-1 rounded-full bg-blue-500/10 px-2 py-0.5 text-xs font-medium text-blue-400 border border-blue-500/20">
                            👑 {quadrant_counts.premium} {t('matrix.premium')}
                        </span>
                        <span className="inline-flex items-center gap-1 rounded-full bg-zinc-500/10 px-2 py-0.5 text-xs font-medium text-zinc-400 border border-zinc-500/20">
                            📦 {quadrant_counts.budget} {t('matrix.budget')}
                        </span>
                    </div>
                </div>
            </CardHeader>
            <CardContent>
                <div className="h-[500px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 40, left: 20, bottom: 40 }}>
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="hsl(222 47% 18%)"
                            />

                            {/* X-Axis: Price */}
                            <XAxis
                                type="number"
                                dataKey="price"
                                name="Price"
                                tickFormatter={formatPrice}
                                tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }}
                                axisLine={{ stroke: 'hsl(222 47% 25%)' }}
                                label={{
                                    value: t('matrix.price_axis'),
                                    position: 'insideBottom',
                                    offset: -25,
                                    fill: 'hsl(215 20% 65%)',
                                    fontSize: 11,
                                }}
                            />

                            {/* Y-Axis: Rating */}
                            <YAxis
                                type="number"
                                dataKey="rating"
                                name="Rating"
                                domain={[1, 5]}
                                ticks={[1, 2, 3, 4, 5]}
                                tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }}
                                axisLine={{ stroke: 'hsl(222 47% 25%)' }}
                                label={{
                                    value: t('matrix.rating_axis'),
                                    angle: -90,
                                    position: 'insideLeft',
                                    offset: 5,
                                    fill: 'hsl(215 20% 65%)',
                                    fontSize: 11,
                                }}
                            />

                            {/* Vertical Reference: Median Price */}
                            <ReferenceLine
                                x={median_price}
                                stroke="hsl(215 20% 45%)"
                                strokeWidth={2}
                                strokeDasharray="8 4"
                                label={{
                                    value: `${t('matrix.median')}: ${formatPrice(median_price)} UZS`,
                                    position: 'top',
                                    fill: 'hsl(215 20% 55%)',
                                    fontSize: 10,
                                }}
                            />

                            {/* Horizontal Reference: 4.0 Star Rating */}
                            <ReferenceLine
                                y={rating_threshold}
                                stroke="hsl(215 20% 45%)"
                                strokeWidth={2}
                                strokeDasharray="8 4"
                                label={{
                                    value: '4.0 ⭐ threshold',
                                    position: 'right',
                                    fill: 'hsl(215 20% 55%)',
                                    fontSize: 10,
                                }}
                            />

                            <Tooltip content={<CustomTooltip />} />

                            <Scatter data={points} fillOpacity={0.7}>
                                {points.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={getDotColor(entry.rating)}
                                        r={Math.min(3 + Math.log2(entry.review_count + 1), 10)} // Size by review count
                                    />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>

                {/* Quadrant Legend */}
                <div className="grid grid-cols-2 gap-3 mt-4 pt-4 border-t border-border">
                    <div className="flex items-start gap-2 text-xs">
                        <span className="text-green-400 text-base leading-none mt-0.5">●</span>
                        <div>
                            <span className="text-foreground font-medium">Top-Left: 💎 {t('matrix.gems')}</span>
                            <p className="text-muted-foreground">{t('matrix.gems_desc')}</p>
                        </div>
                    </div>
                    <div className="flex items-start gap-2 text-xs">
                        <span className="text-blue-400 text-base leading-none mt-0.5">●</span>
                        <div>
                            <span className="text-foreground font-medium">Top-Right: 👑 {t('matrix.premium')}</span>
                            <p className="text-muted-foreground">{t('matrix.premium_desc')}</p>
                        </div>
                    </div>
                    <div className="flex items-start gap-2 text-xs">
                        <span className="text-zinc-400 text-base leading-none mt-0.5">●</span>
                        <div>
                            <span className="text-foreground font-medium">Bottom-Left: 📦 {t('matrix.budget')}</span>
                            <p className="text-muted-foreground">{t('matrix.budget_desc')}</p>
                        </div>
                    </div>
                    <div className="flex items-start gap-2 text-xs">
                        <span className="text-red-400 text-base leading-none mt-0.5">●</span>
                        <div>
                            <span className="text-foreground font-medium">Bottom-Right: ⚠️ {t('matrix.overpriced')}</span>
                            <p className="text-muted-foreground">{t('matrix.overpriced_desc')}</p>
                        </div>
                    </div>
                </div>

                {/* Color Legend */}
                <div className="flex items-center justify-center gap-6 mt-3 pt-3 border-t border-border">
                    <div className="flex items-center gap-1.5 text-xs">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(142 76% 46%)' }}></div>
                        <span className="text-muted-foreground">Rating ≥ 4.5</span>
                    </div>
                    <div className="flex items-center gap-1.5 text-xs">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(45 93% 55%)' }}></div>
                        <span className="text-muted-foreground">Rating 3.5–4.5</span>
                    </div>
                    <div className="flex items-center gap-1.5 text-xs">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'hsl(0 84% 55%)' }}></div>
                        <span className="text-muted-foreground">Rating &lt; 3.5</span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
