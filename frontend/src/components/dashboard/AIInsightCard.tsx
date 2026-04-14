import { Bot, ArrowRight } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface AIInsightCardProps {
  insightText: string;
  onViewActionPlan?: () => void;
}

export function AIInsightCard({ insightText, onViewActionPlan }: AIInsightCardProps) {
  return (
    <Card className="overflow-hidden border-0 bg-gradient-to-br from-primary/20 via-accent/10 to-card">
      <CardContent className="p-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-cyan">
            <Bot className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">AI Insight</h3>
            <p className="text-xs text-muted-foreground">Powered by DeepLearn</p>
          </div>
        </div>

        <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
          {insightText}
        </p>

        <Button 
          onClick={onViewActionPlan}
          className="mt-4 w-full bg-gradient-cyan text-primary-foreground hover:opacity-90"
        >
          View Action Plan
          <ArrowRight className="ml-2 h-4 w-4" />
        </Button>
      </CardContent>
    </Card>
  );
}
