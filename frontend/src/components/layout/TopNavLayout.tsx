import { ReactNode } from 'react';
import { TopNavigation } from './TopNavigation';

interface TopNavLayoutProps {
  children: ReactNode;
}

export function TopNavLayout({ children }: TopNavLayoutProps) {
  return (
    <div className="min-h-screen w-full">
      <TopNavigation />
      <main className="p-6">
        {children}
      </main>
    </div>
  );
}
