import { NavLink, useLocation } from 'react-router-dom';
import { TrendingUp, LogOut } from 'lucide-react';
import { Button } from '@/components/ui/button';

const navItems = [
  { title: 'Dashboard', url: '/market-overview' },
  { title: 'Analytics', url: '/product-analyzer' },
  { title: 'Simulation', url: '/simulation' },
  { title: 'Model Performance', url: '/model-performance' },
];

export function TopNavigation() {
  const location = useLocation();

  const isActive = (url: string) => location.pathname === url;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center justify-between px-6">
        {/* Logo */}
        <NavLink to="/" className="flex items-center gap-3 group">
          <div className="relative flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 shadow-md group-hover:shadow-cyan-500/25 transition-shadow">
            <TrendingUp className="h-4 w-4 text-white" />
            <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-emerald-400 ring-2 ring-background" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold text-foreground leading-tight">Uzum Intel</span>
            <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider leading-tight">Seller Analytics</span>
          </div>
        </NavLink>

        {/* Navigation */}
        <nav className="flex items-center gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.title}
              to={item.url}
              className={`relative px-4 py-2 text-sm font-medium transition-colors ${isActive(item.url)
                ? 'text-primary'
                : 'text-muted-foreground hover:text-foreground'
                }`}
            >
              {item.title}
              {isActive(item.url) && (
                <span className="absolute bottom-0 left-1/2 h-1 w-1 -translate-x-1/2 rounded-full bg-primary" />
              )}
            </NavLink>
          ))}
        </nav>

        {/* Right section */}
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
            <LogOut className="mr-2 h-4 w-4" />
            Logout
          </Button>
          <div className="h-8 w-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600" />
        </div>
      </div>
    </header>
  );
}
