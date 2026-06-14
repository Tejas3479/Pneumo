import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';
import { Scale, RefreshCw, Activity, Users, Cpu } from 'lucide-react';

export default function FairnessPage() {
  const { addNotification } = useApp();
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [federatedRunning, setFederatedRunning] = useState(false);

  const fetchFairnessData = async (showNotification = false) => {
    setLoading(true);
    try {
      const result = await api.getFairness();
      setData(result);
      if (showNotification) {
        addNotification('Fairness metrics audited successfully.', 'success');
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to audit fairness: ' + err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFairnessData();
  }, []);

  const handleRunFederated = async () => {
    setFederatedRunning(true);
    try {
      const response = await api.runFederated();
      if (response.task_id) {
        addNotification('Federated training round client enqueued in background thread.', 'success');
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to launch federated training: ' + err.message, 'error');
    } finally {
      setFederatedRunning(false);
    }
  };

  if (loading || !data) {
    return (
      <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center gap-4 text-center">
        <div className="w-10 h-10 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin text-brand-cyan"></div>
        <span className="text-xs text-brand-textMuted">Running serving-side bias audit on validation subset...</span>
      </div>
    );
  }

  // Bar chart data
  const chartData = [
    {
      name: 'Demographic Parity',
      value: data.metrics?.demographic_parity_difference || 0,
      description: 'Difference in positive prediction rates across subgroups'
    },
    {
      name: 'Equal Opportunity',
      value: data.metrics?.equal_opportunity_difference || 0,
      description: 'Difference in true positive rates across subgroups'
    }
  ];

  return (
    <div className="space-y-8 animate-fade-in relative z-0">
      
      {/* Header Panel */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-brand-cyan">
            <Scale className="w-5 h-5" />
            <h2 className="font-heading font-bold text-xl text-white">Demographic Fairness Audit</h2>
          </div>
          <p className="text-xs text-brand-textMuted max-w-xl leading-relaxed">
            Evaluate algorithm bias across demographics (age, sex) on the validation subset. Running continuous federated learning mitigates disparities and aligns weights.
          </p>
        </div>
        <button
          onClick={() => fetchFairnessData(true)}
          className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-brand-border bg-slate-900 text-xs font-semibold text-white hover:bg-slate-800 transition-colors shrink-0 uppercase tracking-wider"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Re-Audit Bias</span>
        </button>
      </div>

      {/* Stats and Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left column: Metrics summary list & Federated trigger */}
        <div className="lg:col-span-1 space-y-6">
          
          {/* Learning DB stats */}
          <div className="glass-panel p-5 rounded-xl space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Users className="w-4 h-4 text-brand-cyan" />
              <span>Continuous Learning Stats</span>
            </h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 bg-slate-950/40 border border-brand-border rounded-lg text-center">
                <span className="font-heading font-black text-2xl text-brand-cyan block">
                  {data.db_stats?.num_flagged || 0}
                </span>
                <span className="text-[9px] font-bold text-brand-textMuted uppercase tracking-wider mt-1 block">
                  Flagged (0.4 - 0.6)
                </span>
              </div>
              <div className="p-4 bg-slate-950/40 border border-brand-border rounded-lg text-center">
                <span className="font-heading font-black text-2xl text-brand-healthy block">
                  {data.db_stats?.num_corrected || 0}
                </span>
                <span className="text-[9px] font-bold text-brand-textMuted uppercase tracking-wider mt-1 block">
                  Clinician Corrected
                </span>
              </div>
            </div>
          </div>

          {/* Demographic Subgroup Metrics */}
          <div className="glass-panel p-5 rounded-xl space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Users className="w-4 h-4 text-brand-cyan" />
              <span>Demographic Subgroup Metrics</span>
            </h3>
            <div className="space-y-4 text-xs">
              <div className="border-b border-brand-border/40 pb-3">
                <h4 className="font-semibold text-slate-200 mb-2 uppercase tracking-wide text-[10px]">True Positive Rate (TPR)</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-slate-950/40 border border-brand-border/40 rounded-lg">
                    <span className="text-[10px] text-brand-textMuted block uppercase font-medium">Subgroup 0 (Male)</span>
                    <span className="text-sm font-bold text-white mt-1 block">{(data.metrics?.tpr_subgroup_0 * 100).toFixed(2)}%</span>
                  </div>
                  <div className="p-3 bg-slate-950/40 border border-brand-border/40 rounded-lg">
                    <span className="text-[10px] text-brand-textMuted block uppercase font-medium">Subgroup 1 (Female)</span>
                    <span className="text-sm font-bold text-white mt-1 block">{(data.metrics?.tpr_subgroup_1 * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="font-semibold text-slate-200 mb-2 uppercase tracking-wide text-[10px]">Selection Rate</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 bg-slate-950/40 border border-brand-border/40 rounded-lg">
                    <span className="text-[10px] text-brand-textMuted block uppercase font-medium">Subgroup 0 (Male)</span>
                    <span className="text-sm font-bold text-white mt-1 block">{(data.metrics?.selection_rate_subgroup_0 * 100).toFixed(2)}%</span>
                  </div>
                  <div className="p-3 bg-slate-950/40 border border-brand-border/40 rounded-lg">
                    <span className="text-[10px] text-brand-textMuted block uppercase font-medium">Subgroup 1 (Female)</span>
                    <span className="text-sm font-bold text-white mt-1 block">{(data.metrics?.selection_rate_subgroup_1 * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Federated training triggers */}
          <div className="glass-panel p-5 rounded-xl space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-brand-violet" />
              <span>Federated Mitigation</span>
            </h3>
            <p className="text-[11px] text-brand-textMuted leading-relaxed">
              Trigger decentralized federated training iterations on locally corrected clinician labels to adjust LoRA projection weights and minimize demographic parity gap.
            </p>
            <button
              onClick={handleRunFederated}
              disabled={federatedRunning}
              className="w-full py-3 px-4 rounded-xl border border-brand-violet/35 hover:bg-brand-violet/10 text-brand-violet bg-brand-violet/5 text-xs font-bold uppercase tracking-wider transition-colors disabled:opacity-50 flex items-center justify-center gap-2 animate-pulse"
            >
              <Activity className={`w-4 h-4 ${federatedRunning ? 'animate-spin' : ''}`} />
              <span>{federatedRunning ? 'Running Round...' : 'Run Federated Learning Round'}</span>
            </button>
          </div>

        </div>

        {/* Right column: Recharts Bar Chart (Span 2) */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-xl flex flex-col justify-between min-h-[350px]">
          <div>
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-1">Algorithmic Parity Gap</h3>
            <span className="text-[11px] text-brand-textMuted font-medium">Disparity scores closer to 0 indicate unbiased classifications across demographics.</span>
          </div>

          <div className="h-60 w-full mt-6">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 10, bottom: 5 }}
              >
                <XAxis 
                  dataKey="name" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 600 }}
                />
                <YAxis 
                  domain={[0, 0.5]}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                />
                <Tooltip 
                  cursor={{ fill: 'rgba(56, 189, 248, 0.02)' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-slate-950/90 border border-brand-border p-3 rounded-lg text-xs max-w-xs leading-relaxed backdrop-blur-sm">
                          <p className="font-semibold text-white uppercase tracking-wider">{payload[0].payload.name}</p>
                          <p className="text-brand-cyan font-bold mt-0.5">Value: {payload[0].value.toFixed(4)}</p>
                          <p className="text-[10px] text-brand-textMuted mt-1">{payload[0].payload.description}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar 
                  dataKey="value" 
                  radius={[4, 4, 0, 0]} 
                  barSize={40}
                  animationDuration={1200}
                >
                  {chartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={index === 0 ? 'url(#parityGradient)' : 'url(#oppGradient)'}
                    />
                  ))}
                </Bar>
                <defs>
                  <linearGradient id="parityGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#0ea5e9" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                  </linearGradient>
                  <linearGradient id="oppGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#8b5cf6" />
                    <stop offset="100%" stopColor="#f43f5e" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

    </div>
  );
}
