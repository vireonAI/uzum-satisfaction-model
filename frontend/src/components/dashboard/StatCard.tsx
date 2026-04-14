import { 
  FileOutput, 
  Star, 
  AlertTriangle, 
  ThumbsUp,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { StatCardData } from '@/types/dashboard';

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  FileOutput,
  Star,
  AlertTriangle,
  ThumbsUp,
};

interface StatCardProps {
  data: StatCardData;
}

export function StatCard({ data }: StatCardProps) {
  const Icon = iconMap[data.icon] || FileOutput;

  const getTrendIcon = () => {
    if (data.trendType === 'up') return <TrendingUp className="h-3 w-3" />;
    if (data.trendType === 'down') return <TrendingDown className="h-3 w-3" />;
    return <Minus className="h-3 w-3" />;
  };

  const getTrendColor = () => {
    if (data.trendType === 'up') return 'text-success';
    if (data.trendType === 'down') return 'text-destructive';
    return 'text-muted-foreground';
  };

  const getBadgeVariant = () => {
    switch (data.badgeType) {
      case 'success': return 'default';
      case 'warning': return 'secondary';
      case 'destructive': return 'destructive';
      default: return 'outline';
    }
  };

  return (
    <Card className="border-border bg-card transition-all hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5">
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          {data.trend && (
            <div className={`flex items-center gap-1 text-xs font-medium ${getTrendColor()}`}>
              {getTrendIcon()}
              <span>{data.trend}</span>
            </div>
          )}
          {data.badge && (
            <Badge 
              variant={getBadgeVariant()}
              className={`text-xs ${data.badgeType === 'destructive' ? 'bg-destructive/20 text-destructive' : data.badgeType === 'success' ? 'bg-success/20 text-success' : ''}`}
            >
              {data.badge}
            </Badge>
          )}
        </div>
        <div className="mt-4">
          <p className="text-sm text-muted-foreground">{data.title}</p>
          <p className="mt-1 text-2xl font-bold text-foreground">{data.value}</p>
        </div>
      </CardContent>
    </Card>
  );
}
