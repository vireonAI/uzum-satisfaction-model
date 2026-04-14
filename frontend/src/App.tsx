import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import MarketOverview from "./pages/MarketOverview";
import ProductAnalyzer from "./pages/ProductAnalyzer";
import Simulation from "./pages/Simulation";
import ModelPerformance from "./pages/ModelPerformance";
import ResearchResults from "./pages/ResearchResults";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/market-overview" element={<MarketOverview />} />
          <Route path="/product-analyzer" element={<ProductAnalyzer />} />
          <Route path="/simulation" element={<Simulation />} />
          <Route path="/model-performance" element={<ModelPerformance />} />
          <Route path="/research-results" element={<ResearchResults />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
