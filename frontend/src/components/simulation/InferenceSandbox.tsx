import { History, Sparkles } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import type { DetectedFactor } from '@/types/dashboard';

interface InferenceSandboxProps {
  text: string;
  onTextChange: (text: string) => void;
  onAnalyze: () => void;
  detectedFactors: DetectedFactor[];
  confidence: number;
  sentimentDistribution: { positive: number; neutral: number; negative: number };
  onViewHistory?: () => void;
}

export function InferenceSandbox({
  text,
  onTextChange,
  onAnalyze,
  detectedFactors,
  confidence,
  sentimentDistribution,
  onViewHistory,
}: InferenceSandboxProps) {
  const getFactorBadgeClass = (type: DetectedFactor['type']) => {
    switch (type) {
      case 'positive': return 'bg-success/20 text-success border-success/30';
      case 'negative': return 'bg-destructive/20 text-destructive border-destructive/30';
      default: return 'bg-muted text-muted-foreground border-border';
    }
  };

  return (
    <Card className="border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between pb-4">
        <CardTitle className="text-lg font-semibold">Inference Sandbox</CardTitle>
        <Button
          variant="ghost"
          size="sm"
          onClick={onViewHistory}
          className="text-muted-foreground hover:text-foreground"
        >
          <History className="mr-2 h-4 w-4" />
          History
        </Button>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="relative">
          <Textarea
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            placeholder="Enter review text or product description for analysis..."
            className="min-h-[120px] resize-none border-border bg-muted/30 font-mono text-sm text-foreground placeholder:text-muted-foreground focus:border-primary"
          />
        </div>

        <Button
          onClick={onAnalyze}
          disabled={!text.trim()}
          className="w-full bg-gradient-purple text-white hover:opacity-90"
        >
          <Sparkles className="mr-2 h-4 w-4" />
          Analyze
        </Button>

        {detectedFactors.length > 0 && (
          <>
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-foreground">Detected Factors</h4>
              <div className="flex flex-wrap gap-2">
                {detectedFactors.map((factor) => (
                  <Badge
                    key={factor.id}
                    variant="outline"
                    className={getFactorBadgeClass(factor.type)}
                  >
                    {factor.label}
                  </Badge>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Confidence</span>
                <span className="text-sm font-medium text-foreground">{confidence}%</span>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-medium text-foreground">Sentiment Distribution</h4>
              <div className="flex h-3 overflow-hidden rounded-full">
                <div 
                  className="bg-success transition-all" 
                  style={{ width: `${sentimentDistribution.positive}%` }} 
                />
                <div 
                  className="bg-muted-foreground transition-all" 
                  style={{ width: `${sentimentDistribution.neutral}%` }} 
                />
                <div 
                  className="bg-destructive transition-all" 
                  style={{ width: `${sentimentDistribution.negative}%` }} 
                />
              </div>
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Positive {sentimentDistribution.positive}%</span>
                <span>Neutral {sentimentDistribution.neutral}%</span>
                <span>Negative {sentimentDistribution.negative}%</span>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
