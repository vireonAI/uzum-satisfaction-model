import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { AlertItem } from '@/types/dashboard';

interface AlertsListProps {
  alerts: AlertItem[];
}

export function AlertsList({ alerts }: AlertsListProps) {
  const getStatusColor = (type: AlertItem['type']) => {
    switch (type) {
      case 'error': return 'bg-destructive';
      case 'warning': return 'bg-warning';
      case 'success': return 'bg-success';
      default: return 'bg-muted-foreground';
    }
  };

  return (
    <Card className="border-border bg-card">
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-semibold">Recent Alerts</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {alerts.map((alert) => (
          <div key={alert.id} className="flex items-start gap-3">
            <div className={`mt-1.5 h-2 w-2 flex-shrink-0 rounded-full ${getStatusColor(alert.type)}`} />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-foreground">{alert.message}</p>
              <p className="mt-0.5 text-xs text-muted-foreground">{alert.timestamp}</p>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
