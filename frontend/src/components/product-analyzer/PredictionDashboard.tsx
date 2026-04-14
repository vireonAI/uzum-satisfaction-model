import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import type { PredictionMetric } from '@/types/dashboard';

interface PredictionDashboardProps {
  metrics: PredictionMetric[];
  confidence?: 'High' | 'Medium' | 'Low';
}

export function PredictionDashboard({ metrics, confidence = 'High' }: PredictionDashboardProps) {
  const [showAll, setShowAll] = useState(false);
  
  const visibleMetrics = showAll ? metrics : metrics.slice(0, 4);
  const hiddenCount = metrics.length - 4;

  const getProgressColor = (colorType: PredictionMetric['colorType']) => {
    switch (colorType) {
      case 'success': return 'bg-success';
      case 'primary': return 'bg-primary';
      case 'warning': return 'bg-warning';
      case 'destructive': return 'bg-destructive';
      default: return 'bg-primary';
    }
  };

  const getConfidenceBadgeClass = () => {
    switch (confidence) {
      case 'High': return 'bg-success/20 text-success';
      case 'Medium': return 'bg-warning/20 text-warning';
      case 'Low': return 'bg-destructive/20 text-destructive';
    }
  };

  return (
    <Card className="border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between pb-4">
        <CardTitle className="text-lg font-semibold">Prediction Dashboard</CardTitle>
        <Badge className={getConfidenceBadgeClass()}>
          {confidence} Confidence
        </Badge>
      </CardHeader>
      <CardContent className="space-y-5">
        {visibleMetrics.map((metric) => (
          <div key={metric.id} className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">{metric.label}</span>
              <span className="text-sm font-medium text-foreground">
                {metric.value}{metric.unit || '%'}
              </span>
            </div>
            <div className="relative h-2 overflow-hidden rounded-full bg-muted">
              <div 
                className={`absolute left-0 top-0 h-full rounded-full ${getProgressColor(metric.colorType)}`}
                style={{ width: `${(metric.value / metric.maxValue) * 100}%` }}
              />
            </div>
          </div>
        ))}

        {hiddenCount > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAll(!showAll)}
            className="w-full text-muted-foreground hover:text-foreground"
          >
            {showAll ? (
              <>
                <ChevronUp className="mr-2 h-4 w-4" />
                Show less
              </>
            ) : (
              <>
                <ChevronDown className="mr-2 h-4 w-4" />
                Show {hiddenCount} more factors
              </>
            )}
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
