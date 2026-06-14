import React, { useState, useEffect } from 'react';
import { ThumbsUp, ThumbsDown, Check, AlertTriangle } from 'lucide-react';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';

export default function PredictionCard({ data, file, originalImageUrl }) {
  const { addNotification } = useApp();
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');

  useEffect(() => {
    // Only create blob URL if originalImageUrl is not provided
    if (!originalImageUrl && file && !file.name.toLowerCase().endsWith('.dcm')) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl('');
    }
  }, [file, originalImageUrl]);

  if (!data) return null;

  const isPositive = data.prediction === 'POSITIVE';
  const probability = typeof data.probability === 'number' ? data.probability : 0;
  const percentage = Math.round(probability * 100);
  const uncertainty = data.uncertainty;

  // SVG Gauge calculations — correct CSS rotate-90 is available in Tailwind
  const radius = 70;
  const stroke = 8;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - probability * circumference;

  // Resolve heatmap — handle both base64 (non-DICOM) and URL (DICOM) responses
  const heatmapSrc = data.heatmap_base64
    ? data.heatmap_base64
    : data.heatmap_url
      ? data.heatmap_url
      : null;

  // Resolve counterfactual — same dual-source pattern
  const counterfactualSrc = data.counterfactual_base64
    ? data.counterfactual_base64
    : data.counterfactual_url
      ? data.counterfactual_url
      : null;

  // Feedback can be submitted if we have any path identifier — either saved path or file name
  const feedbackIdentifier = data.image_path || (file ? file.name : null);
  const feedbackEnabled = !!feedbackIdentifier;

  const handleFeedback = async (isCorrect) => {
    if (!feedbackIdentifier) return;
    setSubmittingFeedback(true);

    let clinicianLabel = 0;
    if (data.prediction === 'POSITIVE') {
      clinicianLabel = isCorrect ? 1 : 0;
    } else {
      clinicianLabel = isCorrect ? 0 : 1;
    }

    try {
      await api.submitFeedback(feedbackIdentifier, clinicianLabel);
      setFeedbackSubmitted(true);
      addNotification('Clinician feedback recorded successfully. Active learning weights updated.', 'success');
    } catch (err) {
      console.error(err);
      addNotification('Failed to submit feedback: ' + (err.response?.data?.detail || err.message), 'error');
    } finally {
      setSubmittingFeedback(false);
    }
  };

  return (
    <div className="glass-panel p-6 rounded-2xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-start animate-fade-in">
      
      {/* Left Column: Viewports Comparative Display */}
      <div className="space-y-6">
        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block mb-2">Input Radiograph</span>
          <div className="relative aspect-square bg-slate-950 rounded-xl overflow-hidden border border-brand-border flex items-center justify-center">
            {originalImageUrl ? (
              <img src={originalImageUrl} alt="Input Radiograph" className="w-full h-full object-contain" />
            ) : previewUrl ? (
              <img src={previewUrl} alt="Input Radiograph" className="w-full h-full object-contain" />
            ) : file && file.name.toLowerCase().endsWith('.dcm') ? (
              <div className="text-center p-4 text-brand-textMuted text-xs">
                <p className="font-semibold text-slate-300">DICOM Clinical Radiograph</p>
                <p className="mt-1 font-mono text-[10px] text-slate-500">{data.image_path?.split('/').pop() || file.name}</p>
              </div>
            ) : counterfactualSrc ? (
              <img src={counterfactualSrc} alt="Fallback Preview" className="w-full h-full object-contain opacity-40" />
            ) : (
              <div className="text-center text-xs text-brand-textMuted">No preview available</div>
            )}
            <span className="absolute bottom-3 left-3 px-2 py-1 bg-slate-950/80 backdrop-blur border border-brand-border text-[9px] font-semibold rounded uppercase tracking-wider text-slate-300">Standardized DX</span>
          </div>
        </div>

        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block mb-2">Self-Attention Heatmap</span>
          <div className="relative aspect-square bg-slate-950 rounded-xl overflow-hidden border border-brand-violet/20 flex items-center justify-center">
            {heatmapSrc ? (
              <img src={heatmapSrc} alt="Grad-CAM Heatmap" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center text-xs text-brand-textMuted p-4">
                <AlertTriangle className="w-6 h-6 text-brand-textMuted mx-auto mb-2 opacity-50" />
                <p>Heatmap not available</p>
              </div>
            )}
            <span className="absolute bottom-3 left-3 px-2 py-1 bg-brand-violet/85 backdrop-blur border border-brand-violet/40 text-[9px] font-semibold rounded uppercase tracking-wider text-white">Grad-CAM Overlay</span>
          </div>
        </div>

        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block mb-2">Counterfactual Inpainting</span>
          <div className="relative aspect-square bg-slate-950 rounded-xl overflow-hidden border border-brand-healthy/20 flex items-center justify-center">
            {counterfactualSrc ? (
              <img src={counterfactualSrc} alt="Counterfactual inpainting" className="w-full h-full object-contain" />
            ) : (
              <div className="text-center text-xs text-brand-textMuted p-4">
                <AlertTriangle className="w-6 h-6 text-brand-textMuted mx-auto mb-2 opacity-50" />
                <p>Counterfactual not available</p>
              </div>
            )}
            <span className="absolute bottom-3 left-3 px-2 py-1 bg-brand-healthy/85 backdrop-blur border border-brand-healthy/40 text-[9px] font-semibold rounded uppercase tracking-wider text-white">Anomaly Erased</span>
          </div>
        </div>
      </div>

      {/* Right Column: AI Diagnostic Metrics */}
      <div className="space-y-6">
        
        {/* Classification Banner */}
        <div className={`p-5 rounded-xl text-center flex flex-col items-center justify-center gap-1 shadow-lg ${isPositive ? 'bg-brand-pathology/10 border border-brand-pathology/30 text-brand-pathology shadow-brand-pathology/5' : 'bg-brand-healthy/10 border border-brand-healthy/30 text-brand-healthy shadow-brand-healthy/5'}`}>
          <span className="font-heading font-black text-2xl tracking-widest">{isPositive ? 'PATHOLOGY DETECTED' : 'NO COLLAPSED LUNG'}</span>
          <span className="text-xs uppercase tracking-wider text-white/70 font-semibold">{isPositive ? 'Pneumothorax Positive' : 'Pneumothorax Negative'}</span>
        </div>

        {/* Confidence Gauge */}
        <div className="p-6 rounded-xl border border-brand-border bg-slate-950/35 flex flex-col items-center gap-4 text-center">
          <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted">Confidence Metric</span>
          <div className="relative flex items-center justify-center w-36 h-36">
            {/* Use rotate-90 (standard Tailwind) — SVG starts at 3 o'clock, -90deg rotates to 12 o'clock */}
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 140 140">
              <circle
                className="text-slate-900"
                strokeWidth={stroke}
                stroke="currentColor"
                fill="transparent"
                r={normalizedRadius}
                cx={radius}
                cy={radius}
              />
              <circle
                strokeWidth={stroke}
                strokeDasharray={`${circumference} ${circumference}`}
                style={{ strokeDashoffset, transition: 'stroke-dashoffset 1s ease-out' }}
                strokeLinecap="round"
                stroke="url(#confidenceGradient)"
                fill="transparent"
                r={normalizedRadius}
                cx={radius}
                cy={radius}
              />
              <defs>
                <linearGradient id="confidenceGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#0ea5e9" />
                  <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
              </defs>
            </svg>
            <div className="absolute flex flex-col items-center justify-center">
              <span className="font-heading font-black text-3xl text-white">{percentage}%</span>
              <span className="text-[10px] text-brand-textMuted uppercase font-bold tracking-wider">Probability</span>
            </div>
          </div>
          {uncertainty !== null && uncertainty !== undefined ? (
            <span className="text-xs font-semibold px-2.5 py-1 rounded bg-slate-900 border border-brand-border text-slate-400">
              Ensemble Std: ± {(uncertainty * 100).toFixed(1)}%
            </span>
          ) : (
            <span className="text-xs font-semibold px-2.5 py-1 rounded bg-slate-900 border border-brand-border text-slate-400">
              ± N/A (Single Model)
            </span>
          )}
        </div>

        {/* Clinical Explanatory Narrative */}
        <div className="p-5 rounded-xl border border-brand-border bg-slate-950/35 space-y-2">
          <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block">AI Explanatory Narrative</span>
          <p className="text-xs leading-relaxed text-slate-300 font-medium">
            {data.text_justification || 'Clinical narrative was not returned by the model for this prediction.'}
          </p>
        </div>

        {/* Clinician Feedback */}
        <div className="p-5 rounded-xl border border-brand-border bg-slate-950/35 space-y-4">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block">Clinician Feedback</span>
            <p className="text-[11px] text-brand-textMuted leading-relaxed mt-1">
              Your feedback corrects the diagnostic label and registers instances for active learning retraining.
            </p>
          </div>

          {!feedbackEnabled && (
            <div className="text-[11px] text-amber-500/80 bg-amber-500/5 border border-amber-500/20 rounded-lg px-3 py-2">
              Feedback unavailable — image path not saved. Enable <code className="font-mono text-[10px]">save_image=true</code> in settings.
            </div>
          )}

          {feedbackSubmitted ? (
            <div className="flex items-center gap-2 p-3 rounded-lg border border-brand-healthy/20 bg-brand-healthy/5 text-brand-healthy text-xs font-semibold">
              <Check className="w-4 h-4 shrink-0" />
              <span>Diagnostic feedback submitted successfully!</span>
            </div>
          ) : (
            <div className="flex gap-4">
              <button
                onClick={() => handleFeedback(true)}
                disabled={submittingFeedback || !feedbackEnabled}
                className="flex-1 py-3 px-4 rounded-xl border border-brand-healthy/30 bg-brand-healthy/5 text-brand-healthy hover:bg-brand-healthy/10 transition-colors flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <ThumbsUp className="w-4 h-4" />
                <span>Correct</span>
              </button>
              <button
                onClick={() => handleFeedback(false)}
                disabled={submittingFeedback || !feedbackEnabled}
                className="flex-1 py-3 px-4 rounded-xl border border-brand-pathology/30 bg-brand-pathology/5 text-brand-pathology hover:bg-brand-pathology/10 transition-colors flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <ThumbsDown className="w-4 h-4" />
                <span>Incorrect</span>
              </button>
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
