import React, { useState, useEffect, useRef } from 'react';
import { useApp } from '../context/AppContext';
import { api } from '../api/client';
import { useTaskPolling } from '../hooks/useTaskPolling';
import { Sliders, FileText, Download, HelpCircle, CheckCircle, Info } from 'lucide-react';

export default function SettingsPage() {
  const { settings, setSettings, addNotification } = useApp();
  const [cardTaskId, setCardTaskId] = useState(null);
  const [modelCardContent, setModelCardContent] = useState('');
  const [modelCardPath, setModelCardPath] = useState('');
  const serverUrlDebounceRef = useRef(null);

  const { status, result, error } = useTaskPolling(cardTaskId);

  const handleModelChange = (e) => {
    setSettings((prev) => ({ ...prev, modelType: e.target.value }));
    addNotification(`Active model preference set to "${e.target.value}". Saved locally.`, 'info');
  };

  // Debounce server URL changes — 800ms after typing stops
  const handleServerChange = (e) => {
    const val = e.target.value;
    setSettings((prev) => ({ ...prev, serverUrl: val }));
    if (serverUrlDebounceRef.current) clearTimeout(serverUrlDebounceRef.current);
    serverUrlDebounceRef.current = setTimeout(() => {
      if (val) addNotification('Server URL updated. Reconnecting...', 'info');
    }, 800);
  };

  const handleGenerateModelCard = async () => {
    setCardTaskId(null);
    setModelCardPath('');
    setModelCardContent('');
    try {
      const response = await api.createModelCard();
      if (response.task_id) {
        setCardTaskId(response.task_id);
        addNotification('Model card generation enqueued on Celery queue.', 'info');
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to generate model card: ' + (err.response?.data?.detail || err.message), 'error');
    }
  };

  useEffect(() => {
    if (status === 'SUCCESS' && result) {
      // result.path is an absolute path on the server — we use the download endpoint
      // instead of constructing a static URL
      setModelCardPath(result.path || 'model_card.md');
      setModelCardContent(result.content || '');
      setCardTaskId(null);
      addNotification('Model card compiled successfully.', 'success');
    } else if (status === 'FAILED') {
      addNotification('Model card task failed: ' + (error || 'Unknown error'), 'error');
      setCardTaskId(null);
    }
  }, [status, result, error, addNotification]);

  const isGenerating = status === 'PENDING' || status === 'STARTED';

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
                <div className="group relative cursor-pointer">
                  <HelpCircle className="w-3.5 h-3.5 text-brand-textMuted" />
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-60 p-2.5 bg-slate-950 border border-brand-border rounded text-[10px] text-brand-textMuted leading-relaxed scale-0 group-hover:scale-100 transition-all origin-bottom shadow-2xl z-30">
                    ViT-B/16 uses LoRA fine-tuning with XAI heatmaps. ResNet-50 is faster. BioViL-T requires the MedFound checkpoint. All run on CPU via ONNX.
                  </div>
                </div>
              </div>
              <p className="text-[11px] text-brand-textMuted leading-relaxed mt-1">
                Select the inference backbone. Preference is saved locally and sent to the backend with each prediction request.
              </p>
            </div>

            <div className="flex gap-3 flex-col">
              {[
                { value: 'vit', label: 'Vision Transformer', sub: 'ViT-B/16 (LoRA weights)' },
                { value: 'resnet', label: 'ResNet Classifier', sub: 'Standard ResNet-50 Pipeline' },
                { value: 'medfound', label: 'BioViL-T Medical', sub: 'CheXpert Pre-trained Foundation' },
              ].map(({ value, label, sub }) => (
                <label
                  key={value}
                  className={`flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all duration-200 ${settings.modelType === value ? 'border-brand-cyan bg-brand-cyan/5 text-white' : 'border-slate-800 bg-slate-950/40 text-slate-400 hover:border-slate-700'}`}
                >
                  <input
                    type="radio"
                    name="modelType"
                    value={value}
                    checked={settings.modelType === value}
                    onChange={handleModelChange}
                    className="accent-brand-cyan"
                  />
                  <div className="text-left text-xs">
                    <p className="font-bold">{label}</p>
                    <p className="text-[10px] opacity-70">{sub}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Server Base Config */}
          <div className="glass-panel p-5 rounded-xl space-y-4">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300 block">Server API Endpoint</span>
              <p className="text-[11px] text-brand-textMuted leading-relaxed mt-1">
                Point requests to an external endpoint. Leave blank to use the local FastAPI server (relative paths).
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
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-300 block">Server API Key</span>
              </div>
              <p className="text-[11px] text-brand-textMuted leading-relaxed mt-1">
                Required if the server has <code className="font-mono text-[10px] text-brand-cyan">PNEUMO_API_KEY</code> set. Stored in browser localStorage.
              </p>
            </div>
            <div className="relative">
              <input
                type="password"
                value={settings.apiKey || ''}
                onChange={(e) => setSettings((prev) => ({ ...prev, apiKey: e.target.value }))}
                placeholder="••••••••••••••••"
                className="w-full bg-slate-950/60 border border-brand-border rounded-lg px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-cyan transition-colors"
              />
            </div>
            {/* Security notice */}
            <div className="flex items-start gap-2 text-[10px] text-amber-500/80 bg-amber-500/5 border border-amber-500/20 rounded-lg px-3 py-2">
              <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>API key is stored in browser localStorage (plaintext). Do not use sensitive credentials on shared devices.</span>
            </div>
          </div>
        </div>

        {/* Right Column: Model Card Generator */}
        <div className="space-y-6">
          <div className="glass-panel p-5 rounded-xl space-y-5 flex flex-col h-full">
            <div className="space-y-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300 block">Model Card Compilation</span>
              <p className="text-[11px] text-brand-textMuted leading-relaxed">
                Generate standard-compliant model specification reports (<code className="font-mono text-[10px] text-brand-cyan">model_card.md</code>) 
                containing parameters, quantitative validation metrics, constraints, and audit trail details.
              </p>
            </div>

            {isGenerating ? (
              <div className="flex-1 text-center py-6 space-y-3 bg-slate-950/30 border border-brand-border rounded-xl">
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
                  Generated: {modelCardPath.split(/[/\\]/).pop()}
                </p>

                {/* Model card inline preview */}
                {modelCardContent && (
                  <div className="max-h-48 overflow-y-auto bg-slate-950/60 border border-brand-border rounded-lg p-3 text-[10px] font-mono text-slate-400 leading-relaxed whitespace-pre-wrap">
                    {modelCardContent.substring(0, 1200)}
                    {modelCardContent.length > 1200 && '...'}
                  </div>
                )}

                <div className="flex gap-3">
                  {/* Download via /regulatory/model-card/download if available, else direct link */}
                  <a
                    href="/regulatory/model-card/download"
                    download="model_card.md"
                    className="flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg border border-brand-healthy/30 bg-brand-healthy/5 text-brand-healthy hover:bg-brand-healthy/10 text-[10px] font-bold uppercase tracking-wider transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download MD</span>
                  </a>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex flex-col gap-4">
                <button
                  onClick={handleGenerateModelCard}
                  className="w-full py-3 px-4 rounded-xl bg-slate-950 hover:bg-slate-900 border border-brand-border text-slate-200 font-bold uppercase text-xs tracking-wider flex items-center justify-center gap-2 transition-colors"
                >
                  <FileText className="w-4 h-4 text-brand-cyan" />
                  <span>Compile Model Card</span>
                </button>
                <p className="text-[10px] text-brand-textMuted text-center leading-relaxed">
                  Reads <code className="font-mono text-brand-cyan">metrics.json</code> + <code className="font-mono text-brand-cyan">config.json</code> from the models directory and the audit ledger.
                </p>
              </div>
            )}
          </div>
        </div>

      </div>

    </div>
  );
}
