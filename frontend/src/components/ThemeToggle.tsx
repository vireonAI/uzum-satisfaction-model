import { Moon, Sun } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useEffect, useState } from 'react';

export function ThemeToggle() {
    const [isDark, setIsDark] = useState(() => {
        if (typeof window !== 'undefined') {
            return document.documentElement.classList.contains('dark');
        }
        return true; // default dark
    });

    useEffect(() => {
        // Initialize: set dark mode by default if no preference saved
        const saved = localStorage.getItem('theme');
        if (saved === 'light') {
            document.documentElement.classList.remove('dark');
            setIsDark(false);
        } else {
            document.documentElement.classList.add('dark');
            setIsDark(true);
        }
    }, []);

    const toggle = () => {
        const next = !isDark;
        setIsDark(next);
        if (next) {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        }
    };

    return (
        <Button
            variant="ghost"
            onClick={toggle}
            className="w-full justify-between px-3 h-10 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground group"
        >
            <span className="flex items-center gap-2">
                {isDark ? (
                    <Moon className="h-4 w-4 text-muted-foreground" />
                ) : (
                    <Sun className="h-4 w-4 text-yellow-500" />
                )}
                <span className="text-sm font-medium">
                    {isDark ? 'Dark Mode' : 'Light Mode'}
                </span>
            </span>
            <div className={`relative w-10 h-5 rounded-full transition-colors ${isDark ? 'bg-primary/30' : 'bg-muted'}`}>
                <div className={`absolute top-0.5 w-4 h-4 rounded-full transition-all ${isDark ? 'left-5 bg-primary' : 'left-0.5 bg-yellow-500'}`} />
            </div>
        </Button>
    );
}
