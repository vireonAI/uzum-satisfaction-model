import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { AppSidebar } from "@/components/layout/AppSidebar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Beaker, TrendingUp, Clock, Database, AlertTriangle, CheckCircle } from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface PerFactorMetrics {
    f1: number;
    precision: number;
    recall: number;
    support: number;
    threshold?: number;
}

interface ModelResult {
    model: string;
    per_factor: Record<string, PerFactorMetrics>;
    overall: { macro_f1: number; micro_f1: number; weighted_f1: number };
    train_time_sec: number | string;
    inference_ms: number;
    params: string;
    threshold_method?: string;
}

interface ComparisonData {
    generated_at: string;
    test_set_size: number;
    factors: string[];
    dataset_stats: {
        train_size: number;
        test_size: number;
        class_distribution: Record<string, { positive_rate: number; count: number }>;
    };
    models: ModelResult[];
    thresholds?: Record<string, number>;
}

const FACTOR_LABELS: Record<string, string> = {
    product_quality: "Sifat",
    price_value: "Narx",
    logistics_delivery: "Yetkazish",
    packaging_condition: "Qadoq",
    accuracy_expectation: "Moslik",
    seller_service: "Xizmat",
    specifications: "O'lcham",
    product_defects: "Nuqson",
};

function F1Bar({ value, best }: { value: number; best: boolean }) {
    const pct = Math.round(value * 100);
    const color = best
        ? "bg-emerald-500"
        : value >= 0.5
            ? "bg-blue-500"
            : value >= 0.3
                ? "bg-amber-500"
                : "bg-red-500";

    return (
        <div className="flex items-center gap-2 min-w-0">
            <div className="w-24 h-3 bg-muted rounded-full overflow-hidden flex-shrink-0">
                <div className={`h-full ${color} rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
            </div>
            <span className={`text-xs font-mono tabular-nums ${best ? "font-bold text-emerald-600" : ""}`}>
                {value > 0 ? value.toFixed(3) : "—"}
            </span>
            {best && <CheckCircle className="h-3 w-3 text-emerald-500 flex-shrink-0" />}
        </div>
    );
}

export default function ModelPerformance() {
    const { t } = useTranslation();
    const [data, setData] = useState<ComparisonData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetch(`${API}/api/model-performance`)
            .then((r) => {
                if (!r.ok) throw new Error("Failed to load model comparison data");
                return r.json();
            })
            .then(setData)
            .catch((e) => setError(e.message))
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return (
            <SidebarProvider>
                <AppSidebar />
                <main className="flex-1 flex items-center justify-center">
                    <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
                </main>
            </SidebarProvider>
        );
    }

    if (error || !data) {
        return (
            <SidebarProvider>
                <AppSidebar />
                <main className="flex-1 p-6">
                    <div className="text-destructive">{error || "No data"}</div>
                </main>
            </SidebarProvider>
        );
    }

    const bestModel = data.models.reduce((best, m) =>
        m.overall.macro_f1 > best.overall.macro_f1 ? m : best
    );

    return (
        <SidebarProvider>
            <AppSidebar />
            <main className="flex-1 overflow-auto">
                <div className="p-6 space-y-6">
                    {/* Header */}
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-2">
                            <Beaker className="h-6 w-6 text-primary" />
                            Model Performance
                        </h1>
                        <p className="text-muted-foreground text-sm mt-1">
                            Comparison of {data.models.length} models on {data.test_set_size.toLocaleString()} test samples
                            &nbsp;·&nbsp; Generated {data.generated_at}
                        </p>
                    </div>

                    {/* Model Overview Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {data.models.map((m) => {
                            const isBest = m.model === bestModel.model;
                            return (
                                <Card key={m.model} className={isBest ? "border-emerald-500 border-2" : ""}>
                                    <CardHeader className="pb-2">
                                        <CardTitle className="text-base flex items-center gap-2">
                                            {m.model}
                                            {isBest && <Badge variant="default" className="bg-emerald-500 text-xs">Best</Badge>}
                                        </CardTitle>
                                    </CardHeader>
                                    <CardContent className="space-y-2">
                                        <div className="flex justify-between text-sm">
                                            <span className="text-muted-foreground">Macro F1</span>
                                            <span className="font-mono font-bold text-lg">{m.overall.macro_f1.toFixed(4)}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-muted-foreground">Micro F1</span>
                                            <span className="font-mono">{m.overall.micro_f1.toFixed(4)}</span>
                                        </div>
                                        <div className="flex justify-between text-sm">
                                            <span className="text-muted-foreground">Weighted F1</span>
                                            <span className="font-mono">{m.overall.weighted_f1.toFixed(4)}</span>
                                        </div>
                                        <hr className="my-2" />
                                        <div className="flex justify-between text-xs text-muted-foreground">
                                            <span className="flex items-center gap-1"><Clock className="h-3 w-3" /> {typeof m.inference_ms === "number" ? `${m.inference_ms.toFixed(1)} ms` : m.inference_ms}</span>
                                            <span className="flex items-center gap-1"><Database className="h-3 w-3" /> {m.params}</span>
                                        </div>
                                    </CardContent>
                                </Card>
                            );
                        })}
                    </div>

                    {/* Per-Factor Comparison Table */}
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base flex items-center gap-2">
                                <TrendingUp className="h-4 w-4" />
                                Per-Factor F1 Score Comparison
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b">
                                            <th className="text-left py-2 pr-4 font-medium text-muted-foreground">Factor</th>
                                            <th className="text-left py-2 pr-4 font-medium text-muted-foreground">Support</th>
                                            {data.models.map((m) => (
                                                <th key={m.model} className="text-left py-2 pr-4 font-medium text-muted-foreground text-xs">
                                                    {m.model}
                                                </th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {data.factors.map((factor) => {
                                            const bestF1 = Math.max(...data.models.map((m) => m.per_factor[factor]?.f1 || 0));
                                            return (
                                                <tr key={factor} className="border-b border-muted/30 hover:bg-muted/20">
                                                    <td className="py-2 pr-4">
                                                        <div className="font-medium">{FACTOR_LABELS[factor] || factor}</div>
                                                        <div className="text-xs text-muted-foreground font-mono">{factor}</div>
                                                    </td>
                                                    <td className="py-2 pr-4 text-xs text-muted-foreground">
                                                        {data.dataset_stats.class_distribution[factor]?.count || 0}
                                                        <span className="ml-1">({(data.dataset_stats.class_distribution[factor]?.positive_rate * 100).toFixed(1)}%)</span>
                                                    </td>
                                                    {data.models.map((m) => {
                                                        const f1 = m.per_factor[factor]?.f1 || 0;
                                                        return (
                                                            <td key={m.model} className="py-2 pr-4">
                                                                <F1Bar value={f1} best={f1 === bestF1 && f1 > 0} />
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </CardContent>
                    </Card>

                    {/* V1 → V2 → V3 Progression */}
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base flex items-center gap-2">
                                <TrendingUp className="h-4 w-4 text-emerald-500" />
                                Model Progressiyasi: V1 → V2 → V3
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b text-muted-foreground">
                                            <th className="text-left py-2 pr-6">Versiya</th>
                                            <th className="text-right py-2 pr-6">Macro F1</th>
                                            <th className="text-right py-2 pr-6">Micro F1</th>
                                            <th className="text-left py-2">Asosiy Yangilik</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {[
                                            { v: "V1 — Baseline", macro: 0.4890, micro: 0.5980, note: "XLM-RoBERTa, BCE Loss, threshold=0.5" },
                                            { v: "V2 — Focal Loss", macro: 0.5590, micro: 0.6609, note: "Focal Loss (γ=2), class weighting, threshold opt." },
                                            { v: "V3 — Multi-Task ★", macro: 0.5657, micro: 0.6753, note: "Rating regression head, epoch 12 best checkpoint" },
                                        ].map((row) => (
                                            <tr key={row.v} className={`border-b border-muted/30 hover:bg-muted/20 ${row.v.includes("★") ? "bg-emerald-500/5" : ""}`}>
                                                <td className={`py-2 pr-6 font-medium ${row.v.includes("★") ? "text-emerald-600" : ""}`}>{row.v}</td>
                                                <td className={`py-2 pr-6 text-right font-mono ${row.v.includes("★") ? "font-bold text-emerald-600" : ""}`}>{row.macro.toFixed(4)}</td>
                                                <td className={`py-2 pr-6 text-right font-mono ${row.v.includes("★") ? "font-bold text-emerald-600" : ""}`}>{row.micro.toFixed(4)}</td>
                                                <td className="py-2 text-xs text-muted-foreground">{row.note}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <p className="text-xs text-muted-foreground mt-3">
                                * Barcha versiyalar bir xil test to'plamida (1,550 namuna) baholandi. V3 ham Macro, ham Micro F1 bo'yicha yuqori.
                            </p>
                        </CardContent>
                    </Card>

                    {/* XLM-RoBERTa V3 Analysis */}
                    <Card className="border-amber-500/30">
                        <CardHeader>
                            <CardTitle className="text-base flex items-center gap-2">
                                <AlertTriangle className="h-4 w-4 text-amber-500" />
                                V3 Model Tahlili — Kuchli va Zaif Tomonlar
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-3 text-sm">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <p className="font-semibold text-emerald-600">Kuchli tomonlar</p>
                                    <ul className="space-y-1 text-muted-foreground text-xs">
                                        <li>✓ <strong>product_quality</strong>: F1=0.846 — eng yuqori natija (barcha modellar orasida)</li>
                                        <li>✓ <strong>product_defects</strong>: F1=0.606 — TF-IDF dan +12% yuqori</li>
                                        <li>✓ <strong>packaging_condition</strong>: F1=0.558 — baseline dan +19% yuqori</li>
                                        <li>✓ Ko'p tilli (Uzbek + Rus) bir vaqtda ishlaydi</li>
                                        <li>✓ Real-time inference: 3.86 ms/namuna (GPU)</li>
                                    </ul>
                                </div>
                                <div className="space-y-2">
                                    <p className="font-semibold text-amber-600">Cheklovlar (Data Muammosi)</p>
                                    <ul className="space-y-1 text-muted-foreground text-xs">
                                        <li>✗ <strong>accuracy_expectation</strong>: F1=0.380 — faqat 1,362 ta pozitiv namuna</li>
                                        <li>✗ <strong>specifications</strong>: F1=0.479 — 1,263 ta pozitiv (11:1 imbalance)</li>
                                        <li>✗ Sinflar disbalans: product_quality 63%, specifications 7%</li>
                                        <li>✗ Qisqa sharhlar (&lt;15 belgi) noto'g'ri tasniflangan</li>
                                    </ul>
                                </div>
                            </div>
                            <p className="text-xs text-muted-foreground bg-muted/50 rounded p-2 mt-2">
                                <strong>Ilmiy kontekst:</strong> O'xshash ko'p tilli, ko'p yorliqli NLP vazifalarda (Uzbek/Rus aralash matn)
                                BERT-base modellari odatda Macro F1: 0.45–0.60 oralig'ida natija ko'rsatadi.
                                V3 ning 0.5657 natijasi bu diapazonning yuqori qismiga to'g'ri keladi va
                                data-cheklangan muhit uchun qabul qilinardi hisoblanadi.
                            </p>
                            {data.thresholds && (
                                <div className="mt-3">
                                    <p className="font-medium mb-1 text-xs">Optimallashtirilgan chegara qiymatlari (validation set bo'yicha):</p>
                                    <div className="flex flex-wrap gap-2">
                                        {Object.entries(data.thresholds).map(([f, t]) => (
                                            <Badge key={f} variant="outline" className="font-mono text-xs">
                                                {FACTOR_LABELS[f] || f}: {t}
                                            </Badge>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </CardContent>
                    </Card>

                    {/* Dataset Stats */}
                    <Card>
                        <CardHeader>
                            <CardTitle className="text-base">Dataset Statistics</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-3 gap-4 text-center mb-4">
                                <div>
                                    <div className="text-2xl font-bold">{data.dataset_stats.train_size.toLocaleString()}</div>
                                    <div className="text-xs text-muted-foreground">Training samples</div>
                                </div>
                                <div>
                                    <div className="text-2xl font-bold">{data.dataset_stats.test_size.toLocaleString()}</div>
                                    <div className="text-xs text-muted-foreground">Test samples</div>
                                </div>
                                <div>
                                    <div className="text-2xl font-bold">{data.factors.length}</div>
                                    <div className="text-xs text-muted-foreground">Satisfaction factors</div>
                                </div>
                            </div>
                            <div className="space-y-1">
                                {data.factors.map((f) => {
                                    const stat = data.dataset_stats.class_distribution[f];
                                    const pct = stat ? Math.round(stat.positive_rate * 100) : 0;
                                    return (
                                        <div key={f} className="flex items-center gap-2 text-xs">
                                            <span className="w-28 text-muted-foreground">{FACTOR_LABELS[f] || f}</span>
                                            <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                                <div className="h-full bg-primary/60 rounded-full" style={{ width: `${pct}%` }} />
                                            </div>
                                            <span className="w-10 text-right font-mono">{pct}%</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </main>
        </SidebarProvider>
    );
}
