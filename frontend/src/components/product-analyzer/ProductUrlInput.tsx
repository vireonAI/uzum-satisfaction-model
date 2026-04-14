import { Link, Search } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

interface ProductUrlInputProps {
  url: string;
  onUrlChange: (url: string) => void;
  onAnalyze: () => void;
  isLoading?: boolean;
}

export function ProductUrlInput({ url, onUrlChange, onAnalyze, isLoading }: ProductUrlInputProps) {
  return (
    <div className="flex gap-3">
      <div className="relative flex-1">
        <Link className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="url"
          value={url}
          onChange={(e) => onUrlChange(e.target.value)}
          placeholder="Enter product URL to analyze..."
          className="h-12 border-border bg-muted/50 pl-10 text-foreground placeholder:text-muted-foreground focus:border-primary focus:ring-primary"
        />
      </div>
      <Button
        onClick={onAnalyze}
        disabled={isLoading || !url}
        className="h-12 bg-gradient-cyan px-6 text-primary-foreground hover:opacity-90"
      >
        <Search className="mr-2 h-4 w-4" />
        Analyze
      </Button>
    </div>
  );
}
