import React, { useState, useEffect } from 'react';
import { useApp } from '../context/AppContext';
import { api } from '../api/client';
import { useTaskPolling } from '../hooks/useTaskPolling';
import { Sliders, FileText, Download, HelpCircle, CheckCircle } from 'lucide-react';

export default function SettingsPage() {
  const { settings, setSettings, addNotification } = useApp();
  const [cardTaskId, setCardTaskId] = useState(null);
  const [modelCardPath, setModelCardPath] = useState('');

  const { status, result, error } = useTaskPolling(cardTaskId);

  const handleModelChange = (e) => {
    setSettings((prev) => ({ ...prev, modelType: e.target.value }));
    addNotification('Active model preference saved locally.', 'info');
  };

  const handleServerChange = (e) => {
    setSettings((prev) => ({ ...prev, serverUrl: e.target.value }));
  };

  const handleGenerateModelCard = async () => {
    setCardTaskId(null);
    setModelCardPath('');
    try {
      const response = await api.createModelCard();
      if (response.task_id) {
        setCardTaskId(response.task_id);
        addNotification('Model card generation enqueued on Celery queue.', 'info');
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to generate model card: ' + err.message, 'error');
    }
  };

  useEffect(() => {
    if (status === 'SUCCESS' && result) {
      setModelCardPath(result.path || 'model_card.md');
      setCardTaskId(null);
      addNotification('Model card compiled successfully.', 'success');
    } else if (status === 'FAILED') {
      addNotification('Model card enqueued task failed: ' + error, 'error');
      setCardTaskId(null);
    }
  }, [status, result, error, addNotification]);

  const isGenerating = status === 'PENDING';

  return (
    <div className="space-y-8 animate-fade-in relative z-0">
      
      {/* Overview Panel */}
      <div className="glass-panel p-6 rounded-2xl flex items-center gap-2.5 text-brand-cyan">
        <Sliders className="w-5 h-5" />
        <h2 className="font-heading font-bold text-xl text-white">System Settings & Controls</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Left Column: UI Controls */}
        <div className="space-y-6">
          
          {/* Active Model Engine Selection */}
          <div className="glass-panel p-5 rounded-xl space-y-4">
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-300">Model Engine Select</span>
                
                {/* Tooltip trigger */}
                <div className="group relative cursor-pointer">
                  <HelpCircle className="w-3.5 h-3.5 text-brand-textMuted" />
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 p-2 bg-slate-950 border border-brand-border rounded text-[10px] text-brand-textMuted leading-relaxed scale-0 group-hover:scale-100 transition-all origin-bottom shadow-2xl z-30">
                    Model selection requires server-side support. Contact your administrator.
                  </div>
                </div>
              </div>
              <p className="text-[11px] text-brand-textMuted leading-relaxed mt-1">
                Configure local preferences for predictions analysis routing. Currently, backend models default to active checkpoint configs.
              </p>
            </div>

            <div className="flex gap-4 flex-col sm:flex-row">
              <label className={`flex-1 flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all duration-300 ${settings.modelType === 'vit' ? 'border-brand-cyan bg-brand-cyan/5 text-white' : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'}`}>
                <input 
                  type="radio" 
                  name="modelType" 
                  value="vit" 
                  checked={settings.modelType === 'vit'} 
                  onChange={handleModelChange} 
                  className="accent-brand-cyan"
                />
                <div className="text-left text-xs">
                  <p className="font-bold">Vision Transformer</p>
                  <p className="text-[10px] opacity-70">ViT-B/16 (LoRA weights)</p>
                </div>
              </label>
              
              <label className={`flex-1 flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all duration-300 ${settings.modelType === 'resnet' ? 'border-brand-cyan bg-brand-cyan/5 text-white' : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'}`}>
                <input 
                  type="radio" 
                  name="modelType" 
                  value="resnet" 
                  checked={settings.modelType === 'resnet'} 
                  onChange={handleModelChange} 
                  className="accent-brand-cyan"
                />
                <div className="text-left text-xs">
                  <p className="font-bold">ResNet Classifier</p>
                  <p className="text-[10px] opacity-70">Standard ResNet-50 Pipeline</p>
                </div>
              </label>
            </div>
          </div>

          {/* Server base configs */}
          <div className="glass-panel p-5 rounded-xl space-y-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300 block">Server API Endpoint</span>
              <p className="text-[11px] text-brand-textMuted leading-relaxed mt-1">
                Point custom requests to external endpoints (leave blank to target the local FastAPI server).
              </p>
            </div>
            <input
              type="text"
              value={settings.serverUrl}
              onChange={handleServerChange}
              placeholder="e.g. http://192.168.1.100:8000"
              className="w-full bg-slate-950/60 border border-brand-border rounded-lg px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-cyan transition-colors"
            />

            <div className="pt-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300 block">Server API Key</span>
              <p className="text-[11px] text-brand-textMuted leading-relaxed mt-1">
                Provide the credentials required to authorize access to endpoint methods.
              </p>
            </div>
            <input
              type="password"
              value={settings.apiKey || ''}
              onChange={(e) => setSettings((prev) => ({ ...prev, apiKey: e.target.value }))}
              placeholder="••••••••••••••••"
              className="w-full bg-slate-950/60 border border-brand-border rounded-lg px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-cyan transition-colors"
            />
          </div>

        </div>

        {/* Right Column: Model Card Generator */}
        <div className="space-y-6">
          <div className="glass-panel p-5 rounded-xl space-y-5 flex flex-col justify-between h-full">
            <div className="space-y-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300 block">Model Card Compilation</span>
              <p className="text-[11px] text-brand-textMuted leading-relaxed">
                Generate standard-compliant model specification reports (`model_card.md`) containing parameters, quantitative validation evaluation metrics, constraints, and audit trail details.
              </p>
            </div>

            {isGenerating ? (
              <div className="text-center py-6 space-y-3 bg-slate-950/30 border border-brand-border rounded-xl">
                <div className="w-7 h-7 rounded-full border-2 border-slate-900 border-t-brand-cyan animate-spin mx-auto"></div>
                <p className="text-[11px] text-brand-textMuted">Compiling model parameters on worker...</p>
              </div>
            ) : modelCardPath ? (
              <div className="p-4 rounded-xl border border-brand-healthy/20 bg-brand-healthy/5 space-y-3">
                <div className="flex items-center gap-2 text-xs text-brand-healthy font-semibold">
                  <CheckCircle className="w-4 h-4" />
                  <span>Model Card compiled successfully!</span>
                </div>
                <p className="text-[10px] text-brand-textMuted font-mono break-all leading-normal">
                  Target output: {modelCardPath}
                </p>
                <div className="flex gap-3">
                  <a
                    href={`/static/${modelCardPath.split('/').pop()}`}
                    download
                    className="flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg border border-brand-healthy/30 bg-brand-healthy/5 text-brand-healthy hover:bg-brand-healthy/10 text-[10px] font-bold uppercase tracking-wider transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download MD</span>
                  </a>
                </div>
              </div>
            ) : (
              <button
                onClick={handleGenerateModelCard}
                className="w-full py-3 px-4 rounded-xl bg-slate-950 hover:bg-slate-900 border border-brand-border text-slate-200 font-bold uppercase text-xs tracking-wider flex items-center justify-center gap-2 transition-colors"
              >
                <FileText className="w-4 h-4 text-brand-cyan" />
                <span>Compile Model Card</span>
              </button>
            )}
          </div>
        </div>

      </div>

    </div>
  );
}
