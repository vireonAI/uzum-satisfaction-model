import { useState, useEffect } from 'react';
import { Play } from 'lucide-react';
import { TopNavLayout } from '@/components/layout/TopNavLayout';
import { RatingSimulator } from '@/components/simulation/RatingSimulator';
import { InferenceSandbox } from '@/components/simulation/InferenceSandbox';
import { QualityMatrix } from '@/components/simulation/QualityMatrix';
import { Button } from '@/components/ui/button';
import { useSimulation as useSimulationState } from '@/hooks/useSimulation';
import { useAnalyzeReview, useSimulation, useCoefficients } from '@/services/api';
import { qualityCoefficients as mockCoefficients } from '@/data/mockData';
import type { DetectedFactor } from '@/types/dashboard';

export default function Simulation() {
  const { state, updateState, reset, projectedRating, ratingChange } = useSimulationState();

  const [inferenceText, setInferenceText] = useState('');
  const [detectedFactors, setDetectedFactors] = useState<DetectedFactor[]>([]);
  const [confidence, setConfidence] = useState(0);
  const [sentiment, setSentiment] = useState({ positive: 0, neutral: 100, negative: 0 });
  const [searchQuery, setSearchQuery] = useState('');

  const analyzeMutation = useAnalyzeReview();
  const simulationMutation = useSimulation();
  const { data: coefficientsData } = useCoefficients();

  const handleAnalyze = async () => {
    if (!inferenceText.trim()) return;

    try {
      const result = await analyzeMutation.mutateAsync({ text: inferenceText });

      // Convert API response to component format
      setDetectedFactors(result.factors.map((f, idx) => ({
        id: String(idx + 1),
        label: f.factor.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        type: f.label,
      })));

      setConfidence(Math.round((result.positive_count / result.total_factors) * 100));

      // Calculate sentiment distribution
      const positive = result.factors.filter(f => f.label === 'positive').length;
      const negative = result.factors.filter(f => f.label === 'negative').length;
      const neutral = result.factors.length - positive - negative;
      const total = result.factors.length;

      setSentiment({
        positive: Math.round((positive / total) * 100),
        neutral: Math.round((neutral / total) * 100),
        negative: Math.round((negative / total) * 100),
      });
    } catch (error) {
      console.error('Analysis failed:', error);
    }
  };

  const handleRunGlobalSimulation = async () => {
    try {
      const result = await simulationMutation.mutateAsync({
        defect_rate: state.defectRate,
        delivery_speed: state.deliverySpeed,
        response_time: state.responseTime,
      });
      console.log('Simulation result:', result);
    } catch (error) {
      console.error('Simulation failed:', error);
    }
  };

  // Use API coefficients if available, otherwise fallback to mock
  const qualityCoefficients = coefficientsData?.coefficients?.map((c: any) => ({
    id: c.id,
    factorName: c.factor_name,
    weightImpact: c.weight_impact,
    baselineCoeff: c.baseline_coeff,
    simulatedCoeff: c.simulated_coeff,
    status: c.status,
    lastUpdated: c.last_updated,
  })) || mockCoefficients;

  return (
    <TopNavLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">
              Interactive Simulation & Sandbox
            </h1>
            <p className="mt-1 text-muted-foreground">
              Test ML model scenarios and analyze review patterns in real-time.
            </p>
          </div>
          <Button
            onClick={handleRunGlobalSimulation}
            className="bg-gradient-cyan text-primary-foreground hover:opacity-90"
            disabled={simulationMutation.isPending}
          >
            <Play className="mr-2 h-4 w-4" />
            {simulationMutation.isPending ? 'Running...' : 'Run Global Simulation'}
          </Button>
        </div>

        {/* Simulator & Sandbox Grid */}
        <div className="grid gap-6 lg:grid-cols-2">
          <RatingSimulator
            state={state}
            projectedRating={projectedRating}
            ratingChange={ratingChange}
            onStateChange={updateState}
            onReset={reset}
          />
          <InferenceSandbox
            text={inferenceText}
            onTextChange={setInferenceText}
            onAnalyze={handleAnalyze}
            detectedFactors={detectedFactors}
            confidence={confidence}
            sentimentDistribution={sentiment}
            onViewHistory={() => console.log('Viewing history...')}
          />
        </div>

        {/* Quality Matrix */}
        <QualityMatrix
          data={qualityCoefficients}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          onFilter={() => console.log('Opening filter...')}
          onDownload={() => console.log('Downloading...')}
        />
      </div>
    </TopNavLayout>
  );
}
