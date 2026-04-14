import { Download } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
  Cell,
} from 'recharts';
import { useTranslation } from 'react-i18next'; // Import useTranslation

interface ImpactAnalysisChartProps {
  data: Array<{
    factor: string;
    weight: number;
    type: 'positive' | 'negative';
  }>;
  onExport?: () => void;
}

export function ImpactAnalysisChart({ data, onExport }: ImpactAnalysisChartProps) {
  const { t } = useTranslation();
  // Calculate domain based on actual data range
  const weights = data.map(d => d.weight);
  const maxAbsValue = Math.max(...weights.map(Math.abs));
  const domainMax = Math.ceil(maxAbsValue * 1.2); // Add 20% padding
  const domain = [-domainMax, domainMax];

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const value = data.weight;
      const isNegative = value < 0;

      return (
        <div className="bg-popover border border-border rounded-lg shadow-lg p-3">
          <p className="font-semibold text-foreground text-sm mb-1">
            {data.factor}
          </p>
          <p className={`text-sm font-medium ${isNegative ? 'text-destructive' : 'text-primary'}`}>
            {isNegative ? t('chart.creates_dissatisfaction') : t('chart.drives_satisfaction')}
          </p>
          <p className={`text-lg font-bold ${isNegative ? 'text-destructive' : 'text-primary'}`}>
            {value > 0 ? '+' : ''}{value.toFixed(2)} {t('common.impact')}
          </p>
        </div>
      );
    }
    return null;
  };

  // Custom Y-axis tick to truncate long labels
  const CustomYAxisTick = ({ x, y, payload }: any) => {
    const maxLength = 25;
    let displayText = payload.value;
    if (displayText.length > maxLength) {
      displayText = displayText.substring(0, maxLength) + '...';
    }

    return (
      <text
        x={x - 8}
        y={y}
        textAnchor="end"
        fill="hsl(210 40% 98%)"
        fontSize={11}
        dominantBaseline="middle"
      >
        {displayText}
      </text>
    );
  };

  return (
    <Card className="border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-lg font-semibold">{t('dashboard.impactAnalysis')}</CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            {t('dashboard.subtitle', { count: data.length })}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={onExport}
          className="border-border text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <Download className="mr-2 h-4 w-4" />
          Export Report
        </Button>
      </CardHeader>
      <CardContent>
        <div className="h-[500px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={data}
              margin={{ top: 10, right: 30, left: 120, bottom: 20 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(222 47% 20%)"
                horizontal={false}
              />

              {/* X-Axis: Impact Score */}
              <XAxis
                type="number"
                domain={domain}
                tick={{ fill: 'hsl(215 20% 65%)', fontSize: 11 }}
                axisLine={{ stroke: 'hsl(222 47% 30%)' }}
                label={{
                  value: t('chart.impact_coef'),
                  position: 'insideBottom',
                  offset: -10,
                  fill: 'hsl(215 20% 65%)',
                  fontSize: 11
                }}
              />

              {/* Y-Axis: Factor Names */}
              <YAxis
                type="category"
                dataKey="factor"
                tick={<CustomYAxisTick />}
                axisLine={{ stroke: 'hsl(222 47% 30%)' }}
                width={115}
              />

              {/* Reference Line at 0 (Neutral) */}
              <ReferenceLine
                x={0}
                stroke="hsl(215 20% 65%)"
                strokeWidth={2}
                strokeDasharray="5 5"
              />

              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'hsl(222 47% 15%)' }} />

              {/* Single Bar with Dynamic Colors */}
              <Bar
                dataKey="weight"
                radius={[0, 4, 4, 0]}
                barSize={20}
              >
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.type === 'negative'
                      ? 'hsl(0 84% 60%)'    // Red for negative (barriers)
                      : 'hsl(142 76% 56%)'  // Green for positive (drivers)
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-4 pt-4 border-t border-border">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-[hsl(142,76%,56%)]"></div>
            <span className="text-xs text-muted-foreground">{t('chart.drivers')}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-[hsl(0,84%,60%)]"></div>
            <span className="text-xs text-muted-foreground">{t('chart.barriers')}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
