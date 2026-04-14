import {
  Globe,
  Package,
  TrendingUp,
  Beaker,
  FlaskConical,
  PlayCircle
} from 'lucide-react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from '@/components/ui/sidebar';
import { useTranslation } from 'react-i18next';
import { LanguageSwitcher } from '@/components/LanguageSwitcher';
import { ThemeToggle } from '@/components/ThemeToggle';


export function AppSidebar() {
  const location = useLocation();
  const { t } = useTranslation();

  const isActive = (url: string) => location.pathname === url;

  const platformItems = [
    { title: t('sidebar.globalInsights'), url: '/market-overview', icon: Globe },
    { title: t('sidebar.myProducts'), url: '/product-analyzer', icon: Package },
    { title: 'Model Performance', url: '/model-performance', icon: Beaker },
    { title: 'Tadqiqot Natijalari', url: '/research-results', icon: FlaskConical },
    { title: 'Simulyatsiya', url: '/simulation', icon: PlayCircle },
  ];

  return (
    <Sidebar className="border-r border-sidebar-border">
      <SidebarHeader className="p-4">
        <NavLink to="/" className="flex items-center gap-3 group">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 shadow-md group-hover:shadow-cyan-500/25 transition-shadow">
            <TrendingUp className="h-5 w-5 text-white" />
            <span className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-emerald-400 ring-2 ring-background" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-foreground leading-tight">Uzum Intel</h1>
            <p className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Seller Analytics</p>
          </div>
        </NavLink>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="text-xs uppercase tracking-wider text-muted-foreground">
            {t('sidebar.platform')}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {platformItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={isActive(item.url)}
                    className="transition-colors hover:bg-sidebar-accent"
                  >
                    <NavLink to={item.url} className="flex items-center gap-3">
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4 space-y-3 mt-auto">
        <ThemeToggle />
        <LanguageSwitcher />
        <div className="flex items-center gap-3 rounded-lg bg-sidebar-accent p-3">
          <div className="h-8 w-8 rounded-full bg-gradient-mixed" />
          <div className="flex-1 overflow-hidden">
            <p className="truncate text-sm font-medium text-foreground">Doniyor</p>
            <p className="truncate text-xs text-muted-foreground">Pro Plan</p>
          </div>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
