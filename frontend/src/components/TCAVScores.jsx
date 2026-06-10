import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

export default function TCAVScores({ scores }) {
  if (!scores || Object.keys(scores).length === 0) return null;

  const data = Object.entries(scores).map(([concept, score]) => ({
    name: concept,
    value: Math.round(score * 100),
  })).sort((a, b) => b.value - a.value);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-950/90 border border-brand-border px-3 py-1.5 rounded-lg text-xs backdrop-blur-sm">
          <p className="font-semibold text-white uppercase tracking-wider">{payload[0].payload.name}</p>
          <p className="text-brand-cyan font-bold mt-0.5">TCAV Score: {payload[0].value}%</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="glass-panel p-5 rounded-xl space-y-4">
      <div>
        <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block">Concept TCAV Scores</span>
        <span className="text-[11px] text-brand-textMuted leading-relaxed block mt-1">
          Testing with Concept Activation Vectors (TCAV) quantifies the positive or negative alignment of high-level visual concepts on model prediction layers.
        </span>
      </div>

      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
          >
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis 
              dataKey="name" 
              type="category" 
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 600 }}
              width={80}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(56, 189, 248, 0.04)' }} />
            <Bar 
              dataKey="value" 
              radius={[0, 4, 4, 0]} 
              barSize={12}
              animationDuration={1200}
            >
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill="url(#tcavGradient)"
                />
              ))}
            </Bar>
            <defs>
              <linearGradient id="tcavGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#0ea5e9" />
                <stop offset="100%" stopColor="#8b5cf6" />
              </linearGradient>
            </defs>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
