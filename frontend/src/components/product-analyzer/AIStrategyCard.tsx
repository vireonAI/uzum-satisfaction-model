import { RefreshCw, TrendingUp, Package, MessageSquare, Zap } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import type { StrategyAction } from '@/types/dashboard';

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  TrendingUp,
  Package,
  MessageSquare,
  Zap,
};

interface AIStrategyCardProps {
  insightText: string;
  actions: StrategyAction[];
  onApplyStrategy?: () => void;
  onRefresh?: () => void;
}

export function AIStrategyCard({ insightText, actions, onApplyStrategy, onRefresh }: AIStrategyCardProps) {
  return (
    <Card className="border-gradient overflow-hidden bg-card">
      <CardContent className="p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            <h3 className="font-semibold text-foreground">AI Strategy Consultant</h3>
          </div>
          <span className="text-xs text-muted-foreground">Powered by Groq</span>
        </div>

        <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
          {insightText}
        </p>

        <div className="mt-5 space-y-3">
          <h4 className="text-sm font-medium text-foreground">Recommended Actions</h4>
          {actions.map((action) => {
            const Icon = iconMap[action.icon] || TrendingUp;
            return (
              <div key={action.id} className="flex items-start gap-3 rounded-lg bg-muted/50 p-3">
                <Icon className="mt-0.5 h-4 w-4 flex-shrink-0 text-primary" />
                <span className="text-sm text-foreground">{action.text}</span>
              </div>
            );
          })}
        </div>

        <div className="mt-5 flex items-center gap-3">
          <Button
            onClick={onApplyStrategy}
            className="flex-1 bg-gradient-mixed text-white hover:opacity-90"
          >
            Apply Strategy
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={onRefresh}
            className="border-border text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
