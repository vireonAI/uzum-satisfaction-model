import { Search, Filter, Download } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { QualityCoefficient } from '@/types/dashboard';

interface QualityMatrixProps {
  data: QualityCoefficient[];
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onFilter?: () => void;
  onDownload?: () => void;
}

export function QualityMatrix({
  data,
  searchQuery,
  onSearchChange,
  onFilter,
  onDownload,
}: QualityMatrixProps) {
  const getStatusBadgeClass = (status: QualityCoefficient['status']) => {
    switch (status) {
      case 'Active': return 'bg-success/20 text-success';
      case 'Pending': return 'bg-warning/20 text-warning';
      case 'Inactive': return 'bg-muted text-muted-foreground';
    }
  };

  const getImpactBarColor = (impact: number) => {
    if (impact < 0) return 'bg-destructive';
    if (impact > 0.7) return 'bg-success';
    if (impact > 0.4) return 'bg-primary';
    return 'bg-warning';
  };

  const filteredData = data.filter((item) =>
    item.factorName.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Card className="border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between pb-4">
        <CardTitle className="text-lg font-semibold">Quality Coefficients Matrix</CardTitle>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              type="text"
              value={searchQuery}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder="Search factors..."
              className="h-9 w-48 border-border bg-muted/50 pl-9 text-sm"
            />
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={onFilter}
            className="h-9 w-9 border-border"
          >
            <Filter className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="icon"
            onClick={onDownload}
            className="h-9 w-9 border-border"
          >
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="rounded-lg border border-border">
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-transparent">
                <TableHead className="text-muted-foreground">Factor Name</TableHead>
                <TableHead className="text-muted-foreground">Weight Impact</TableHead>
                <TableHead className="text-muted-foreground">Baseline Coeff.</TableHead>
                <TableHead className="text-muted-foreground">Simulated Coeff.</TableHead>
                <TableHead className="text-muted-foreground">Status</TableHead>
                <TableHead className="text-muted-foreground">Last Updated</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredData.map((row) => (
                <TableRow key={row.id} className="border-border hover:bg-muted/30">
                  <TableCell className="font-medium text-foreground">{row.factorName}</TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-16 overflow-hidden rounded-full bg-muted">
                        <div
                          className={`h-full rounded-full ${getImpactBarColor(row.weightImpact)}`}
                          style={{ width: `${Math.abs(row.weightImpact) * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {row.weightImpact.toFixed(2)}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{row.baselineCoeff.toFixed(2)}</TableCell>
                  <TableCell className="text-foreground">{row.simulatedCoeff.toFixed(2)}</TableCell>
                  <TableCell>
                    <Badge className={getStatusBadgeClass(row.status)}>{row.status}</Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{row.lastUpdated}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
