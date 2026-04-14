import { RotateCcw } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import type { SimulationState } from '@/types/dashboard';

interface RatingSimulatorProps {
  state: SimulationState;
  projectedRating: number;
  ratingChange: number;
  onStateChange: (newState: Partial<SimulationState>) => void;
  onReset: () => void;
}

export function RatingSimulator({ 
  state, 
  projectedRating, 
  ratingChange, 
  onStateChange, 
  onReset 
}: RatingSimulatorProps) {
  const ratingPercentage = (projectedRating / 5) * 100;

  return (
    <Card className="border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between pb-4">
        <CardTitle className="text-lg font-semibold">Rating Impact Simulator</CardTitle>
        <Badge variant="outline" className="border-primary/50 text-primary">
          v2.4 Model
        </Badge>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Projected Rating Display */}
        <div className="flex items-center justify-center gap-6">
          <div className="relative h-28 w-28">
            <svg className="h-28 w-28 -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="hsl(222 47% 20%)"
                strokeWidth="8"
              />
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke="hsl(186 100% 50%)"
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${ratingPercentage * 2.83} 283`}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-3xl font-bold text-foreground">{projectedRating.toFixed(1)}</span>
              <span className="text-xs text-muted-foreground">/ 5.0</span>
            </div>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Projected Seller Rating</p>
            <p className={`mt-1 text-lg font-semibold ${ratingChange >= 0 ? 'text-success' : 'text-destructive'}`}>
              {ratingChange >= 0 ? '+' : ''}{ratingChange.toFixed(1)}
            </p>
          </div>
        </div>

        {/* Sliders */}
        <div className="space-y-5">
          {/* Defect Rate */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Defect Rate Adjustment</span>
              <span className="text-sm font-medium text-primary">{state.defectRate > 0 ? '+' : ''}{state.defectRate}%</span>
            </div>
            <Slider
              value={[state.defectRate]}
              onValueChange={([value]) => onStateChange({ defectRate: value })}
              min={-5}
              max={5}
              step={0.5}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>-5%</span>
              <span>+5%</span>
            </div>
          </div>

          {/* Delivery Speed */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Delivery Speed</span>
              <span className="text-sm font-medium text-primary">{state.deliverySpeed} Days</span>
            </div>
            <Slider
              value={[state.deliverySpeed]}
              onValueChange={([value]) => onStateChange({ deliverySpeed: value })}
              min={1}
              max={7}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>1 Day</span>
              <span>7 Days</span>
            </div>
          </div>

          {/* Response Time */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-foreground">Customer Response Time</span>
              <span className="text-sm font-medium text-primary">{state.responseTime}h</span>
            </div>
            <Slider
              value={[state.responseTime]}
              onValueChange={([value]) => onStateChange({ responseTime: value })}
              min={1}
              max={24}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>1h</span>
              <span>24h</span>
            </div>
          </div>
        </div>

        <Button
          variant="outline"
          onClick={onReset}
          className="w-full border-border text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <RotateCcw className="mr-2 h-4 w-4" />
          Reset to Default
        </Button>
      </CardContent>
    </Card>
  );
}
