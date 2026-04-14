import React from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
    LabelList
} from 'recharts';
import { useTranslation } from 'react-i18next';

interface FactorData {
    name: string;
    value: number;
    fill: string;
}

interface Props {
    factors: FactorData[];
}

const FactorBreakdownChart: React.FC<Props> = ({ factors }) => {
    const { t } = useTranslation();
    const data = factors;

    // Custom label renderer
    const renderCustomLabel = (props: any) => {
        const { x, y, width, value } = props;
        return (
            <text
                x={x + width / 2}
                y={y - 5}
                fill="currentColor"
                textAnchor="middle"
                fontSize={14}
                fontWeight={600}
                className="fill-foreground"
            >
                {value}%
            </text>
        );
    };

    // Custom tooltip - theme-aware
    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const data = payload[0];
            return (
                <div className="bg-popover border border-border p-3 rounded-lg shadow-xl">
                    <p className="text-popover-foreground font-semibold">{data.payload.name}</p>
                    <p className={`text-lg font-bold ${data.payload.fill === '#ef4444' ? 'text-red-500' : 'text-green-500'}`}>
                        {data.value}%
                    </p>
                    <p className="text-muted-foreground text-sm mt-1">
                        {data.payload.fill === '#ef4444' ? '⚠️' : '✅'}
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="w-full h-[400px] bg-card border border-border p-4 rounded-lg">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart
                    data={data}
                    margin={{ top: 30, right: 30, left: 20, bottom: 80 }}
                    barSize={50}
                >
                    <CartesianGrid strokeDasharray="3 3" className="stroke-border" opacity={0.3} />
                    <XAxis
                        dataKey="name"
                        angle={-45}
                        textAnchor="end"
                        height={120}
                        tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12, fontWeight: 500 }}
                        stroke="hsl(var(--border))"
                    />
                    <YAxis
                        tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                        stroke="hsl(var(--border))"
                        label={{ value: t('chart.issue_prevalence'), angle: -90, position: 'insideLeft', fill: 'hsl(var(--muted-foreground))' }}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'hsl(var(--muted) / 0.3)' }} />
                    <Bar
                        dataKey="value"
                        radius={[8, 8, 0, 0]}
                        animationDuration={800}
                    >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                        <LabelList dataKey="value" content={renderCustomLabel} />
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default FactorBreakdownChart;
