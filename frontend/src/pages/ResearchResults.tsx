import { useState } from "react";
import { AppSidebar } from "@/components/layout/AppSidebar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Cell
} from "recharts";
import { FlaskConical, TrendingDown, TrendingUp, Database, BarChart2, Star } from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface SatisfactionData {
  baseline_rating: number;
  formula: Array<{ factor: string; weight: number; direction: string; pearson: number; gini_pct: number; perm_pct: number }>;
  category_analysis: Record<string, { baseline: number; count: number; top_factor: string; top_factor_pct: number }>;
  model_r2: number;
  model_mae: number;
}

const FACTOR_LABELS: Record<string, { uz: string; full: string }> = {
  product_defects:      { uz: "Nuqsonlar",    full: "Mahsulot Nuqsonlari" },
  product_quality:      { uz: "Sifat",        full: "Mahsulot Sifati" },
  accuracy_expectation: { uz: "Moslik",       full: "Kutilgan Moslik" },
  seller_service:       { uz: "Xizmat",       full: "Sotuvchi Xizmati" },
  price_value:          { uz: "Narx",         full: "Narx-Qiymat" },
  logistics_delivery:   { uz: "Yetkazish",    full: "Yetkazib Berish" },
  packaging_condition:  { uz: "Qadoq",        full: "Qadoqlash" },
  specifications:       { uz: "Tavsif",       full: "Spesifikasiya" },
};

const CAT_LABELS: Record<string, string> = {
  Elektronika: "Elektronika", Maishiy_Texnika: "Maishiy Texnika",
  Kiyim: "Kiyim", Aksessuarlar: "Aksessuarlar", Uy_Rozgor: "Uy Rozg'or",
  Bolalar_Tovarlari: "Bolalar", Oziq_Ovqat: "Oziq-Ovqat", Kitoblar: "Kitoblar",
};

// Hard-coded from script output (we embed so page works even without live call)
const STATIC_DATA: SatisfactionData = {
  baseline_rating: 4.036,
  model_r2: 0.4546,
  model_mae: 0.7764,
  formula: [
    { factor: "product_defects",      weight: 0.6297, direction: "-", pearson: -0.5603, gini_pct: 66.76, perm_pct: 62.97 },
    { factor: "product_quality",      weight: 0.2605, direction: "+", pearson:  0.4130, gini_pct: 22.45, perm_pct: 26.05 },
    { factor: "accuracy_expectation", weight: 0.0435, direction: "-", pearson: -0.1500, gini_pct:  4.71, perm_pct:  4.35 },
    { factor: "seller_service",       weight: 0.0218, direction: "+", pearson:  0.1486, gini_pct:  1.36, perm_pct:  2.18 },
    { factor: "price_value",          weight: 0.0158, direction: "+", pearson:  0.1915, gini_pct:  1.50, perm_pct:  1.58 },
    { factor: "logistics_delivery",   weight: 0.0108, direction: "+", pearson:  0.1552, gini_pct:  1.09, perm_pct:  1.08 },
    { factor: "packaging_condition",  weight: 0.0090, direction: "-", pearson: -0.0841, gini_pct:  1.25, perm_pct:  0.90 },
    { factor: "specifications",       weight: 0.0089, direction: "-", pearson: -0.0158, gini_pct:  0.88, perm_pct:  0.89 },
  ],
  category_analysis: {
    Elektronika:       { baseline: 3.765, count: 2402,  top_factor: "product_defects", top_factor_pct: 84.51 },
    Maishiy_Texnika:   { baseline: 3.806, count: 3635,  top_factor: "product_defects", top_factor_pct: 71.35 },
    Kiyim:             { baseline: 3.985, count: 2885,  top_factor: "product_quality",  top_factor_pct: 59.15 },
    Aksessuarlar:      { baseline: 3.841, count: 1992,  top_factor: "product_defects", top_factor_pct: 53.03 },
    Uy_Rozgor:         { baseline: 3.964, count: 3805,  top_factor: "product_quality",  top_factor_pct: 48.98 },
    Bolalar_Tovarlari: { baseline: 4.138, count: 3217,  top_factor: "product_defects", top_factor_pct: 57.96 },
    Oziq_Ovqat:        { baseline: 4.261, count: 3813,  top_factor: "product_defects", top_factor_pct: 40.63 },
    Kitoblar:          { baseline: 4.529, count: 2238,  top_factor: "product_defects", top_factor_pct: 51.19 },
  },
};

function WeightBar({ weight, direction, max }: { weight: number; direction: string; max: number }) {
  const pct = (weight / max) * 100;
  const isNeg = direction === "-";
  return (
    <div className="flex items-center gap-3 w-full">
      <div className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-white text-xs font-bold ${isNeg ? "bg-red-500" : "bg-emerald-500"}`}>
        {isNeg ? "−" : "+"}
      </div>
      <div className="flex-1 h-4 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${isNeg ? "bg-gradient-to-r from-red-400 to-red-600" : "bg-gradient-to-r from-emerald-400 to-emerald-600"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-12 text-right text-sm font-mono font-semibold tabular-nums">{(weight * 100).toFixed(1)}%</span>
    </div>
  );
}

export default function ResearchResults() {
  const [data] = useState<SatisfactionData>(STATIC_DATA);

  const maxWeight = Math.max(...data.formula.map(f => f.weight));

  const barData = data.formula.map(f => ({
    name: FACTOR_LABELS[f.factor]?.uz || f.factor,
    fullName: FACTOR_LABELS[f.factor]?.full || f.factor,
    importance: Math.round(f.perm_pct),
    direction: f.direction,
    pearson: Math.abs(f.pearson),
  }));

  const catData = Object.entries(data.category_analysis).map(([cat, v]) => ({
    name: CAT_LABELS[cat] || cat,
    rating: v.baseline,
    count: v.count,
    topFactor: FACTOR_LABELS[v.top_factor]?.uz || v.top_factor,
    topPct: v.top_factor_pct,
  })).sort((a, b) => b.rating - a.rating);

  return (
    <SidebarProvider>
      <AppSidebar />
      <main className="flex-1 overflow-auto">
        <div className="p-6 space-y-6 max-w-7xl mx-auto">

          {/* ── Header ── */}
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <FlaskConical className="h-6 w-6 text-primary" />
              Tadqiqot Natijalari
            </h1>
            <p className="text-muted-foreground text-sm mt-1">
              <em>Elektron tijorat bozorida mijozlar qoniqishining asosiy omillarini aniqlash modeli</em>
              &nbsp;·&nbsp; 23,987 ta sharh tahlili
            </p>
          </div>

          {/* ── Key Finding Banner ── */}
          <div className="rounded-xl border-2 border-primary/30 bg-primary/5 p-5">
            <p className="text-xs font-semibold uppercase tracking-wider text-primary mb-2">Asosiy Ilmiy Natija</p>
            <p className="text-base leading-relaxed">
              Uzum.uz platformasida mijoz qoniqishiga eng kuchli ta'sir etuvchi ikki omil aniqlandi:
              <strong className="text-red-600 mx-1">Mahsulot Nuqsonlari</strong>
              (β = −0.630, Permutation Importance <strong>62.97%</strong>) va
              <strong className="text-emerald-600 mx-1">Mahsulot Sifati</strong>
              (β = +0.261, Permutation Importance <strong>26.05%</strong>).
              Qolgan 6 ta omil birgalikda faqat <strong>11%</strong> ta'sir ko'rsatadi.
            </p>
          </div>

          {/* ── Formula + Stats ── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Satisfaction Formula */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <BarChart2 className="h-4 w-4 text-primary" />
                  Qoniqish Formulasi (Matematikaviy Model)
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="font-mono text-sm bg-muted rounded-lg p-3 space-y-1">
                  <div className="text-muted-foreground text-xs mb-2">Rating = Baseline + Σ(weight × factor)</div>
                  <div className="font-bold">Rating = {data.baseline_rating.toFixed(4)}</div>
                  {data.formula.map(f => (
                    <div key={f.factor} className={`text-xs ${f.direction === "-" ? "text-red-500" : "text-emerald-600"}`}>
                      &nbsp;&nbsp;{f.direction}{(f.weight).toFixed(4)} × {f.factor}
                    </div>
                  ))}
                </div>
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="bg-muted/50 rounded-lg p-3">
                    <div className="text-xl font-bold">{(data.model_r2 * 100).toFixed(1)}%</div>
                    <div className="text-xs text-muted-foreground">R² Score (Explained Variance)</div>
                  </div>
                  <div className="bg-muted/50 rounded-lg p-3">
                    <div className="text-xl font-bold">±{data.model_mae.toFixed(2)}</div>
                    <div className="text-xs text-muted-foreground">MAE (yulduz)</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Factor Impact Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <BarChart2 className="h-4 w-4 text-primary" />
                  Omillar Ta'sir Kuchi (Permutation Importance)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {data.formula.map(f => (
                    <div key={f.factor} className="space-y-1">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span className="font-medium text-foreground">{FACTOR_LABELS[f.factor]?.full}</span>
                        <span className={f.direction === "-" ? "text-red-500" : "text-emerald-500"}>
                          {f.direction === "-" ? <TrendingDown className="inline h-3 w-3 mr-1" /> : <TrendingUp className="inline h-3 w-3 mr-1" />}
                          Pearson: {f.pearson > 0 ? "+" : ""}{f.pearson.toFixed(4)}
                        </span>
                      </div>
                      <WeightBar weight={f.weight} direction={f.direction} max={maxWeight} />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* ── Bar Chart: Importance ── */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Omil Ahamiyati — Vizualizatsiya (Random Forest Permutation Importance)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={barData} layout="vertical" margin={{ left: 20, right: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="name" width={80} tick={{ fontSize: 12 }} />
                  <Tooltip
                    formatter={(v: number, _: string, props: any) => [`${v}%`, props.payload.fullName]}
                    contentStyle={{ fontSize: 12 }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {barData.map((entry, i) => (
                      <Cell key={i} fill={entry.direction === "-" ? "#ef4444" : "#10b981"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="flex gap-4 mt-2 justify-center text-xs text-muted-foreground">
                <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500 inline-block" /> Salbiy ta'sir (reyting pasaytiradi)</span>
                <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-emerald-500 inline-block" /> Ijobiy ta'sir (reyting oshiradi)</span>
              </div>
            </CardContent>
          </Card>

          {/* ── Category Analysis ── */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Database className="h-4 w-4 text-primary" />
                Kategoriya Bo'yicha Tahlil (8 kategoriya, 23,987 sharh)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b text-muted-foreground">
                      <th className="text-left py-2 pr-4">Kategoriya</th>
                      <th className="text-right py-2 pr-4">Sharhlar</th>
                      <th className="text-right py-2 pr-4">O'rtacha Reyting</th>
                      <th className="text-left py-2 pr-4">Asosiy Omil</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {catData.map(row => (
                      <tr key={row.name} className="border-b border-muted/30 hover:bg-muted/20">
                        <td className="py-2 pr-4 font-medium">{row.name}</td>
                        <td className="py-2 pr-4 text-right text-muted-foreground">{row.count.toLocaleString()}</td>
                        <td className="py-2 pr-4 text-right">
                          <span className="flex items-center justify-end gap-1">
                            <Star className="h-3 w-3 text-amber-400 fill-amber-400" />
                            <span className="font-mono font-bold">{row.rating.toFixed(3)}</span>
                          </span>
                        </td>
                        <td className="py-2 pr-4">
                          <Badge variant={row.topFactor === "Nuqsonlar" ? "destructive" : "default"} className="text-xs">
                            {row.topFactor}
                          </Badge>
                        </td>
                        <td className="py-2 text-xs text-muted-foreground">{row.topPct.toFixed(1)}% ahamiyat</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground mt-3 italic">
                * Ahamiyat = Random Forest Feature Importance (kategoriyaga xos model). Asosiy omil o'sha kategoriyada reyting farqini eng ko'p tushuntiruvchi faktor hisoblanadi.
              </p>
            </CardContent>
          </Card>

          {/* ── Research Conclusion ── */}
          <Card className="border-primary/20 bg-primary/5">
            <CardHeader>
              <CardTitle className="text-base">Tadqiqot Xulosasi</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-relaxed">
              <p>
                Ushbu tadqiqot Uzum.uz elektron tijorat platformasida <strong>23,987 ta</strong> mijoz sharhi
                asosida qoniqish omillarini aniqlash maqsadida o'tkazildi. XLM-RoBERTa asosidagi ko'p vazifali
                klassifikatsiya modeli (V3) yordamida har bir sharh 8 ta omil bo'yicha tahlil qilindi.
              </p>
              <p>
                <strong>Asosiy topilmalar:</strong> Regresiya tahlili shuni ko'rsatdiki, <strong>mahsulot
                nuqsonlari</strong> mijoz reytingida eng kuchli (va salbiy) ta'sirga ega — Permutation Importance
                62.97%, Pearson korrelyatsiyasi r = −0.56. <strong>Mahsulot sifati</strong> esa yagona kuchli
                ijobiy omil — 26.05% ahamiyat, r = +0.41. Yetkazib berish, narx va qadoqlash omillari har biri
                2% dan kam ta'sir ko'rsatadi.
              </p>
              <p>
                <strong>Kategoriya farqlari:</strong> Elektronika va Maishiy texnikada nuqsonlar 70–85% ahamiyatga
                ega bo'lsa, Kiyim kategoriyasida sifat muhimroq (59%). Bu sotuvchilarga kategoriyaga mos
                strategiya tanlashga imkon beradi.
              </p>
              <p className="text-muted-foreground italic">
                Random Forest Regressori (200 daraxt, 23,987 namuna): R² = 0.455, MAE = 0.776 yulduz.
                Natijalar 80% train / 20% test split asosida tasdiqlangan.
              </p>
            </CardContent>
          </Card>

        </div>
      </main>
    </SidebarProvider>
  );
}
