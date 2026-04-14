import { useState, useCallback } from 'react';
import type { SimulationState } from '@/types/dashboard';

const DEFAULT_STATE: SimulationState = {
  defectRate: 0,
  deliverySpeed: 3,
  responseTime: 4,
};

export function useSimulation() {
  const [state, setState] = useState<SimulationState>(DEFAULT_STATE);

  const updateState = useCallback((updates: Partial<SimulationState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  }, []);

  const reset = useCallback(() => {
    setState(DEFAULT_STATE);
  }, []);

  // Calculate projected rating based on simulation parameters
  const calculateRating = useCallback(() => {
    const baseRating = 4.6;
    
    // Defect rate impact: each 1% change affects rating by 0.05
    const defectImpact = -state.defectRate * 0.05;
    
    // Delivery speed impact: faster is better (1 day = +0.2, 7 days = -0.2)
    const deliveryImpact = (4 - state.deliverySpeed) * 0.05;
    
    // Response time impact: faster is better (1h = +0.1, 24h = -0.1)
    const responseImpact = (12 - state.responseTime) * 0.01;
    
    const projectedRating = Math.max(1, Math.min(5, baseRating + defectImpact + deliveryImpact + responseImpact));
    const ratingChange = projectedRating - baseRating;
    
    return {
      projectedRating: Math.round(projectedRating * 10) / 10,
      ratingChange: Math.round(ratingChange * 10) / 10,
    };
  }, [state]);

  const { projectedRating, ratingChange } = calculateRating();

  return {
    state,
    updateState,
    reset,
    projectedRating,
    ratingChange,
  };
}
