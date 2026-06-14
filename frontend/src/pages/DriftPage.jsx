import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';
import { useTaskPolling } from '../hooks/useTaskPolling';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { TrendingUp, AlertTriangle, AlertCircle, CheckCircle, RefreshCw } from 'lucide-react';

export default function DriftPage() {
  const { addNotification } = useApp();
  const [taskId, setTaskId] = useState(null);

  const { status, result, error } = useTaskPolling(taskId);

  const runDriftCheck = async () => {
    setTaskId(null);
    try {
      const response = await api.getDrift();
      if (response.task_id) {
        setTaskId(response.task_id);
        addNotification('Data drift analysis enqueued on Celery queue.', 'info');
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to trigger drift check: ' + err.message, 'error');
    }
  };

  const [dbHistory, setDbHistory] = useState([]);

  const fetchDriftHistory = async () => {
    try {
      const data = await api.getDriftHistory();
      setDbHistory(data || []);
    } catch (err) {
      console.error("Failed to fetch drift history", err);
    }
  };

  useEffect(() => {
    runDriftCheck();
    fetchDriftHistory();
  }, []);

  useEffect(() => {
    if (status === 'SUCCESS') {
      fetchDriftHistory();
    }
  }, [status]);

  const isLoading = status === 'PENDING';

  // Simulated historical data based on current PSI or defaults
  const getHistoricalData = (currentMeanVal, currentStdVal) => {
    const baseMean = currentMeanVal !== undefined ? currentMeanVal : 0.12;
    const baseStd = currentStdVal !== undefined ? currentStdVal : 0.09;
    return [
      { day: 'Mon', meanPsi: baseMean * 0.75, stdPsi: baseStd * 0.8 },
      { day: 'Tue', meanPsi: baseMean * 0.85, stdPsi: baseStd * 0.95 },
      { day: 'Wed', meanPsi: baseMean * 0.95, stdPsi: baseStd * 1.1 },
      { day: 'Thu', meanPsi: baseMean * 0.65, stdPsi: baseStd * 0.7 },
      { day: 'Fri', meanPsi: baseMean * 1.05, stdPsi: baseStd * 1.05 },
      { day: 'Sat', meanPsi: baseMean * 1.15, stdPsi: baseStd * 1.2 },
      { day: 'Sun', meanPsi: baseMean, stdPsi: baseStd },
    ];
  };

  const currentMean = result?.psi_mean || 0.0;
  const currentStd = result?.psi_std || 0.0;
  const driftDetected = result?.drift_detected || false;
  const samplesCount = result?.actual_samples_count || 0;
  const historyData = dbHistory.length >= 2 ? dbHistory : getHistoricalData(currentMean, currentStd);

  // Gauge parameters
  const getOffset = (psiVal) => {
    const cap = Math.min(psiVal, 0.5);
    const fraction = cap / 0.5;
    const circ = 2 * Math.PI * 45;
    return circ - fraction * circ;
  };

  return (
    <div className="space-y-8 animate-fade-in relative z-0">
      
      {/* Header Panel */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-brand-cyan">
            <TrendingUp className="w-5 h-5" />
            <h2 className="font-heading font-bold text-xl text-white">Continuous Data Drift Monitor</h2>
          </div>
          <p className="text-xs text-brand-textMuted max-w-xl leading-relaxed">
            Evaluates statistical distribution differences (Population Stability Index - PSI) of input pixel properties over the last 24 hours against baseline training targets.
          </p>
        </div>
        <button
          onClick={runDriftCheck}
          disabled={isLoading}
          className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-brand-border bg-slate-900 text-xs font-semibold text-white hover:bg-slate-800 transition-colors shrink-0 uppercase tracking-wider disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Analyze Drift</span>
        </button>
      </div>

      {isLoading ? (
        <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center gap-4 text-center">
          <div className="w-10 h-10 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin text-brand-cyan"></div>
          <span className="text-xs text-brand-textMuted">Auditing data drift logs...</span>
        </div>
      ) : status === 'FAILED' ? (
        <div className="glass-panel p-8 rounded-2xl border-brand-pathology/30 bg-brand-pathology/5 text-brand-pathology max-w-xl mx-auto flex items-start gap-4">
          <AlertCircle className="w-6 h-6 shrink-0" />
          <div className="space-y-1 text-xs leading-relaxed">
            <h3 className="font-heading font-bold text-slate-100 text-sm">Drift Check Failed</h3>
            <p className="text-slate-300">{error || 'An error occurred during drift validation.'}</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Columns: Gauge Gauges (Span 1) */}
          <div className="lg:col-span-1 space-y-6">
            
            {/* Status Panel */}
            <div className={`p-5 rounded-xl border flex items-start gap-3 shadow-lg ${driftDetected ? 'bg-brand-pathology/10 border-brand-pathology/30 text-brand-pathology' : 'bg-brand-healthy/10 border-brand-healthy/30 text-brand-healthy'}`}>
              {driftDetected ? (
                <>
                  <AlertCircle className="w-5 h-5 shrink-0 text-brand-pathology" />
                  <div className="text-xs space-y-1">
                    <h3 className="font-heading font-bold uppercase tracking-wider text-slate-200">Drift Detected</h3>
                    <p className="text-slate-300">PSI limits exceeded warning threshold (0.25). Webhook alert pushed. Fine-tuning recommended.</p>
                  </div>
                </>
              ) : (
                <>
                  <CheckCircle className="w-5 h-5 shrink-0 text-brand-healthy" />
                  <div className="text-xs space-y-1">
                    <h3 className="font-heading font-bold uppercase tracking-wider text-slate-200">Distribution Stable</h3>
                    <p className="text-slate-300">Clinical distributions align with model baseline configurations. Samples tested: <span className="font-bold">{samplesCount}</span></p>
                  </div>
                </>
              )}
            </div>

            {/* Mean Gauge */}
            <div className="glass-panel p-6 rounded-xl flex flex-col items-center justify-center gap-4 text-center">
              <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted">PSI Mean Index</span>
              <div className="relative w-28 h-28 flex items-center justify-center">
                <svg className="w-full h-full transform -rotate-90">
                  <circle className="text-slate-900" strokeWidth={6} stroke="currentColor" fill="transparent" r={45} cx={56} cy={56} />
                  <circle className="text-brand-cyan transition-all duration-1000" strokeWidth={6} strokeDasharray={2 * Math.PI * 45} style={{ strokeDashoffset: getOffset(currentMean) }} strokeLinecap="round" stroke="currentColor" fill="transparent" r={45} cx={56} cy={56} />
                </svg>
                <div className="absolute">
                  <span className="font-heading font-black text-xl text-white">{currentMean.toFixed(3)}</span>
                </div>
              </div>
              <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${currentMean > 0.25 ? 'bg-brand-pathology/15 text-brand-pathology' : 'bg-brand-healthy/15 text-brand-healthy'}`}>
                {currentMean > 0.25 ? 'High Shift' : 'Stable'}
              </span>
            </div>

            {/* Std Gauge */}
            <div className="glass-panel p-6 rounded-xl flex flex-col items-center justify-center gap-4 text-center">
              <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted">PSI Std Dev Index</span>
              <div className="relative w-28 h-28 flex items-center justify-center">
                <svg className="w-full h-full transform -rotate-90">
                  <circle className="text-slate-900" strokeWidth={6} stroke="currentColor" fill="transparent" r={45} cx={56} cy={56} />
                  <circle className="text-brand-violet transition-all duration-1000" strokeWidth={6} strokeDasharray={2 * Math.PI * 45} style={{ strokeDashoffset: getOffset(currentStd) }} strokeLinecap="round" stroke="currentColor" fill="transparent" r={45} cx={56} cy={56} />
                </svg>
                <div className="absolute">
                  <span className="font-heading font-black text-xl text-white">{currentStd.toFixed(3)}</span>
                </div>
              </div>
              <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${currentStd > 0.25 ? 'bg-brand-pathology/15 text-brand-pathology' : 'bg-brand-healthy/15 text-brand-healthy'}`}>
                {currentStd > 0.25 ? 'High Shift' : 'Stable'}
              </span>
            </div>

          </div>

          {/* Right Column: Line Chart (Span 2) */}
          <div className="lg:col-span-2 glass-panel p-6 rounded-xl flex flex-col justify-between min-h-[400px]">
            <div>
              <div className="flex justify-between items-start">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">PSI Stability Over Time</h3>
                <span className="text-[9px] font-semibold text-brand-textMuted uppercase border border-brand-border px-2 py-0.5 rounded">Simulated History</span>
              </div>
              <span className="text-[11px] text-brand-textMuted block mt-1">
                Historical PSI tracking visualizes drift trends across validation subsets over the last 7 days.
              </span>
            </div>

            <div className="h-64 w-full mt-8">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                  <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <YAxis domain={[0, 0.4]} axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-slate-950/90 border border-brand-border p-3 rounded-lg text-xs backdrop-blur-sm">
                            <p className="font-semibold text-white uppercase tracking-wider">{payload[0].payload.day}</p>
                            <p className="text-brand-cyan font-semibold mt-0.5">Mean PSI: {payload[0].value.toFixed(3)}</p>
                            <p className="text-brand-violet font-semibold mt-0.5">Std PSI: {payload[1].value.toFixed(3)}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Line type="monotone" dataKey="meanPsi" stroke="#0ea5e9" strokeWidth={2.5} dot={{ fill: '#0ea5e9', strokeWidth: 1 }} activeDot={{ r: 6 }} name="Mean PSI" />
                  <Line type="monotone" dataKey="stdPsi" stroke="#8b5cf6" strokeWidth={2.5} dot={{ fill: '#8b5cf6', strokeWidth: 1 }} activeDot={{ r: 6 }} name="Std PSI" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      )}

    </div>
  );
}
