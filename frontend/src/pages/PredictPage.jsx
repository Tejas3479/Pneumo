import React, { useState } from 'react';
import { useApp } from '../context/AppContext';
import { api } from '../api/client';
import { useTaskPolling } from '../hooks/useTaskPolling';
import UploadZone from '../components/UploadZone';
import PredictionCard from '../components/PredictionCard';
import TCAVScores from '../components/TCAVScores';
import { AlertCircle, RefreshCw } from 'lucide-react';

export default function PredictPage() {
  const { addNotification } = useApp();
  const [selectedFile, setSelectedFile] = useState(null);
  const [taskId, setTaskId] = useState(null);

  // Poll prediction task
  const { status, result, error } = useTaskPolling(taskId);

  const handleFileSelected = async (file) => {
    setSelectedFile(file);
    setTaskId(null);

    try {
      const data = await api.predict(file);
      if (data.task_id) {
        setTaskId(data.task_id);
        addNotification('Inference enqueued on Celery queue.', 'info');
      } else {
        throw new Error('Server did not return a task_id.');
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to launch prediction: ' + (err.response?.data?.detail || err.message), 'error');
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setTaskId(null);
  };

  const isLoading = status === 'PENDING';

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Overview Card */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="space-y-1">
          <h2 className="font-heading font-bold text-xl text-white">X-Ray Image Diagnostics</h2>
          <p className="text-xs text-brand-textMuted max-w-xl leading-relaxed">
            Acquire chest radiographs and upload them to run inference on our Vision Transformer (ViT) ensemble pipeline. We evaluate pixel boundaries and self-attentions on the worker.
          </p>
        </div>
        {selectedFile && (
          <button
            onClick={handleReset}
            className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-brand-border bg-slate-900 text-xs font-semibold text-white hover:bg-slate-800 transition-colors shrink-0 uppercase tracking-wider"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>New Diagnostic</span>
          </button>
        )}
      </div>

      {!selectedFile ? (
        <div className="max-w-xl mx-auto py-12">
          <UploadZone onFileSelected={handleFileSelected} isLoading={isLoading} />
        </div>
      ) : (
        <div className="space-y-8">
          
          {/* Active Diagnostic Status */}
          {isLoading && (
            <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center gap-4 text-center max-w-xl mx-auto">
              <div className="w-12 h-12 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin"></div>
              <div className="space-y-1">
                <h3 className="font-heading font-bold text-slate-100 text-sm">Evaluating Deep Neural Networks</h3>
                <p className="text-xs text-brand-textMuted leading-relaxed max-w-xs">
                  Running ensembles and propagating gradients back for heatmaps...
                </p>
              </div>
            </div>
          )}

          {status === 'FAILED' && (
            <div className="glass-panel p-8 rounded-2xl border-brand-pathology/30 bg-brand-pathology/5 text-brand-pathology max-w-xl mx-auto flex items-start gap-4">
              <AlertCircle className="w-6 h-6 shrink-0" />
              <div className="space-y-1 text-xs leading-relaxed">
                <h3 className="font-heading font-bold text-slate-100 text-sm">Analysis Execution Failed</h3>
                <p className="text-slate-300">
                  {error || 'An unexpected error occurred during prediction polling.'}
                </p>
                <button
                  onClick={() => handleFileSelected(selectedFile)}
                  className="mt-3 px-3 py-1.5 rounded bg-brand-pathology/10 border border-brand-pathology/20 text-brand-pathology font-semibold hover:bg-brand-pathology/20 transition-colors uppercase tracking-wider"
                >
                  Retry Analysis
                </button>
              </div>
            </div>
          )}

          {status === 'SUCCESS' && result && (
            <div className="grid grid-cols-1 gap-8">
              {/* Prediction details viewports and metrics */}
              <PredictionCard data={result} file={selectedFile} />
              
              {/* TCAV concepts influence scores */}
              {result.tcav_scores && Object.keys(result.tcav_scores).length > 0 && (
                <TCAVScores scores={result.tcav_scores} />
              )}
            </div>
          )}

        </div>
      )}

    </div>
  );
}
