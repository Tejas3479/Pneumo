import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';
import { Shield, Eye, CheckCircle, RefreshCw, AlertCircle, Inbox } from 'lucide-react';

export default function ActiveLearningPage() {
  const { addNotification } = useApp();
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeSample, setActiveSample] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const fetchFlaggedSamples = async (showNotification = false) => {
    setLoading(true);
    try {
      const data = await api.getFlaggedSamples();
      setSamples(data);
      if (data.length > 0) {
        setActiveSample(data[0]);
      } else {
        setActiveSample(null);
      }
      if (showNotification) {
        addNotification('Flagged active learning queue updated.', 'success');
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to query active learning queue: ' + err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFlaggedSamples();
  }, []);

  const handleLabelCorrection = async (correctedLabel) => {
    if (!activeSample) return;
    setSubmitting(true);
    try {
      await api.submitFeedback(activeSample.ImagePath, correctedLabel);
      addNotification('Active learning feedback submitted. Audit database updated.', 'success');
      
      // Remove from local list
      const updated = samples.filter((s) => s.id !== activeSample.id);
      setSamples(updated);
      if (updated.length > 0) {
        setActiveSample(updated[0]);
      } else {
        setActiveSample(null);
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to record active learning correction: ' + err.message, 'error');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in relative z-0">
      
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-brand-cyan">
            <Shield className="w-5 h-5" />
            <h2 className="font-heading font-bold text-xl text-white">Active Learning Workstation</h2>
          </div>
          <p className="text-xs text-brand-textMuted max-w-xl leading-relaxed">
            Review chest radiographs flagged with high uncertainty (probabilities near 0.5) or manually enqueued. Submitting your clinical correction retrains models via federated loops.
          </p>
        </div>
        <button
          onClick={() => fetchFlaggedSamples(true)}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-brand-border bg-slate-900 text-xs font-semibold text-white hover:bg-slate-800 transition-colors shrink-0 uppercase tracking-wider"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Queue</span>
        </button>
      </div>

      {loading ? (
        <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center gap-4 text-center">
          <div className="w-10 h-10 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin"></div>
          <span className="text-xs text-brand-textMuted">Querying enqueued flagged predictions...</span>
        </div>
      ) : samples.length === 0 ? (
        <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center gap-4 text-center max-w-md mx-auto">
          <div className="w-12 h-12 rounded-full bg-slate-950/60 border border-brand-border text-brand-textMuted flex items-center justify-center shadow-lg">
            <Inbox className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="font-heading font-bold text-slate-200 text-sm">Review Queue Empty</h3>
            <p className="text-xs text-brand-textMuted leading-relaxed">
              No uncertain radiograph predictions are currently flagged for active learning review.
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          
          {/* Left Panel: Flagged Queue List (Span 1) */}
          <div className="xl:col-span-1 glass-panel p-5 rounded-2xl flex flex-col h-[600px]">
            <span className="text-[10px] uppercase font-bold text-brand-textMuted tracking-wider mb-3 block">
              Flagged Cases ({samples.length})
            </span>
            <div className="flex-1 overflow-y-auto space-y-2 pr-1">
              {samples.map((sample) => {
                const isActive = activeSample && activeSample.id === sample.id;
                const dateStr = sample.Timestamp 
                  ? new Date(sample.Timestamp).toLocaleString([], { hour: '2-digit', minute: '2-digit' })
                  : 'Recent';
                return (
                  <button
                    key={sample.id}
                    onClick={() => setActiveSample(sample)}
                    className={`w-full text-left p-3.5 rounded-xl border transition-all flex items-center justify-between ${isActive ? 'border-brand-cyan bg-brand-cyan/5 text-white' : 'border-brand-border bg-slate-950/20 text-slate-400 hover:border-slate-800'}`}
                  >
                    <div className="space-y-1 max-w-[70%]">
                      <span className="text-[11px] font-bold text-slate-200 block truncate">
                        {sample.ImagePath.split('/').pop()}
                      </span>
                      <span className="text-[9px] font-semibold text-brand-textMuted block">
                        Logged: {dateStr}
                      </span>
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] font-bold font-mono text-brand-cyan">
                        {Math.round(sample.Probability * 100)}%
                      </span>
                      <span className="text-[8px] font-semibold uppercase text-brand-textMuted block mt-0.5">
                        Prob
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Right Panel: Interactive Active Review Workspace (Span 2) */}
          {activeSample && (
            <div className="xl:col-span-2 glass-panel p-6 rounded-2xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-start animate-fade-in">
              
              {/* Left Column: Image Viewer */}
              <div className="space-y-3">
                <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block">
                  Radiograph Preview
                </span>
                <div className="relative aspect-square bg-slate-950 rounded-xl overflow-hidden border border-brand-border flex items-center justify-center">
                  <img
                    src={`/rendered-image?path=${activeSample.ImagePath}`}
                    alt="Flagged Chest X-Ray"
                    className="w-full h-full object-contain"
                  />
                  <span className="absolute bottom-3 left-3 px-2 py-1 bg-slate-950/80 backdrop-blur border border-brand-border text-[9px] font-semibold rounded uppercase tracking-wider text-slate-300 flex items-center gap-1">
                    <Eye className="w-3 h-3 text-brand-cyan" />
                    <span>Active Review Viewport</span>
                  </span>
                </div>
              </div>

              {/* Right Column: Metadata and Feedback Actions */}
              <div className="space-y-6">
                
                {/* Stats Header */}
                <div>
                  <h3 className="font-heading font-bold text-slate-100 text-sm">
                    {activeSample.ImagePath.split('/').pop()}
                  </h3>
                  <p className="text-[10px] text-brand-textMuted font-mono mt-1 break-all">
                    Rel Path: {activeSample.ImagePath}
                  </p>
                </div>

                {/* Patient Subgroup Attributes */}
                <div className="grid grid-cols-2 gap-4 bg-slate-950/40 p-4 rounded-xl border border-brand-border">
                  <div className="space-y-0.5 text-left">
                    <span className="text-[9px] uppercase font-bold text-brand-textMuted block">Biological Sex</span>
                    <span className="text-xs font-semibold text-slate-200">
                      {activeSample.Sex === 1 ? 'Female (Subgroup 1)' : 'Male (Subgroup 0)'}
                    </span>
                  </div>
                  <div className="space-y-0.5 text-left">
                    <span className="text-[9px] uppercase font-bold text-brand-textMuted block">Patient Age</span>
                    <span className="text-xs font-semibold text-slate-200">{activeSample.Age} Years</span>
                  </div>
                </div>

                {/* Model Predictions */}
                <div className="p-4 rounded-xl border border-brand-border bg-slate-950/35 space-y-3">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block">
                    Model Assessment on Logging
                  </span>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-300 font-medium">Predicted Class Label:</span>
                    <span className={`px-2 py-0.5 rounded text-[9px] font-extrabold uppercase tracking-wider ${activeSample.Label === 1 ? 'bg-brand-pathology/15 border border-brand-pathology/30 text-brand-pathology' : 'bg-brand-healthy/15 border border-brand-healthy/30 text-brand-healthy'}`}>
                      {activeSample.Label === 1 ? 'Pathology (1)' : 'Healthy (0)'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-slate-300 font-medium">Model Probability:</span>
                    <span className="text-xs font-bold text-brand-cyan font-mono">
                      {(activeSample.Probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Action Section */}
                <div className="p-5 rounded-xl border border-brand-border bg-slate-950/35 space-y-4">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block">
                      Clinical Correction Panel
                    </span>
                    <p className="text-[10px] text-brand-textMuted leading-relaxed mt-1">
                      Choose the correct diagnostic label. Your input resolves the database flag and prepares this dataset for model retraining.
                    </p>
                  </div>

                  <div className="flex flex-col sm:flex-row gap-3">
                    <button
                      onClick={() => handleLabelCorrection(1)}
                      disabled={submitting}
                      className="flex-1 py-3 px-4 rounded-xl border border-brand-pathology/30 bg-brand-pathology/5 text-brand-pathology hover:bg-brand-pathology/10 transition-colors flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider disabled:opacity-50"
                    >
                      <CheckCircle className="w-4 h-4" />
                      <span>Positive (1)</span>
                    </button>
                    <button
                      onClick={() => handleLabelCorrection(0)}
                      disabled={submitting}
                      className="flex-1 py-3 px-4 rounded-xl border border-brand-healthy/30 bg-brand-healthy/5 text-brand-healthy hover:bg-brand-healthy/10 transition-colors flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider disabled:opacity-50"
                    >
                      <CheckCircle className="w-4 h-4" />
                      <span>Negative (0)</span>
                    </button>
                  </div>
                </div>

              </div>

            </div>
          )}

        </div>
      )}

    </div>
  );
}
