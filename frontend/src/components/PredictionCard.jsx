import React, { useState, useEffect } from 'react';
import { ThumbsUp, ThumbsDown, Check } from 'lucide-react';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';

export default function PredictionCard({ data, file }) {
  const { addNotification } = useApp();
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');

  useEffect(() => {
    if (file && !file.name.toLowerCase().endsWith('.dcm')) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreviewUrl('');
    }
  }, [file]);

  if (!data) return null;

  const isPositive = data.prediction === 'POSITIVE';
  const probability = data.probability;
  const percentage = Math.round(probability * 100);
  const uncertainty = data.uncertainty;

  // SVG Gauge calculations
  const radius = 70;
  const stroke = 8;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (probability) * circumference;

  const handleFeedback = async (isCorrect) => {
    if (!data.image_path) return;
    setSubmittingFeedback(true);

    let clinicianLabel = 0;
    if (data.prediction === 'POSITIVE') {
      clinicianLabel = isCorrect ? 1 : 0;
    } else {
      clinicianLabel = isCorrect ? 0 : 1;
    }

    try {
      await api.submitFeedback(data.image_path, clinicianLabel);
      setFeedbackSubmitted(true);
      addNotification('Clinician feedback recorded successfully. Database weights updated.', 'success');
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
            {previewUrl ? (
              <img src={previewUrl} alt="Input Radiograph" className="w-full h-full object-contain" />
            ) : file && file.name.toLowerCase().endsWith('.dcm') ? (
              <div className="text-center p-4 text-brand-textMuted text-xs">
                <p className="font-semibold text-slate-300">DICOM Clinical Radiograph</p>
                <p className="mt-1 font-mono text-[10px] text-slate-500">{data.image_path?.split('/').pop() || file.name}</p>
              </div>
            ) : (
              <img src={data.heatmap_base64} alt="Input Radiograph" className="w-full h-full object-contain" />
            )}
            <span className="absolute bottom-3 left-3 px-2 py-1 bg-slate-950/80 backdrop-blur border border-brand-border text-[9px] font-semibold rounded uppercase tracking-wider text-slate-300">Standardized DX</span>
          </div>
        </div>

        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block mb-2">Self-Attention Heatmap</span>
          <div className="relative aspect-square bg-slate-950 rounded-xl overflow-hidden border border-brand-violet/20 flex items-center justify-center">
            <img src={data.heatmap_base64} alt="Grad-CAM Heatmap" className="w-full h-full object-contain" />
            <span className="absolute bottom-3 left-3 px-2 py-1 bg-brand-violet/85 backdrop-blur border border-brand-violet/40 text-[9px] font-semibold rounded uppercase tracking-wider text-white">Grad-CAM Overlay</span>
          </div>
        </div>

        <div>
          <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block mb-2">Counterfactual Inpainting</span>
          <div className="relative aspect-square bg-slate-950 rounded-xl overflow-hidden border border-brand-healthy/20 flex items-center justify-center">
            <img src={data.counterfactual_base64} alt="Counterfactual inpainting" className="w-full h-full object-contain" />
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
            <svg className="w-full h-full transform -rotate-95">
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
                className="text-brand-cyan transition-all duration-1000 ease-out"
                strokeWidth={stroke}
                strokeDasharray={circumference + ' ' + circumference}
                style={{ strokeDashoffset }}
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
            {data.text_justification || 'Generating clinical report narrative...'}
          </p>
        </div>

        {/* Clinician Feedback */}
        <div className="p-5 rounded-xl border border-brand-border bg-slate-950/35 space-y-4">
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-brand-textMuted block">Clinician Feedback</span>
            <p className="text-[11px] text-brand-textMuted leading-relaxed mt-1">
              Your feedback corrects the diagnostic label and registers instances for active learning models.
            </p>
          </div>

          {feedbackSubmitted ? (
            <div className="flex items-center gap-2 p-3 rounded-lg border border-brand-healthy/20 bg-brand-healthy/5 text-brand-healthy text-xs font-semibold">
              <Check className="w-4 h-4 shrink-0" />
              <span>Diagnostic feedback submitted successfully!</span>
            </div>
          ) : (
            <div className="flex gap-4">
              <button
                onClick={() => handleFeedback(true)}
                disabled={submittingFeedback}
                className="flex-1 py-3 px-4 rounded-xl border border-brand-healthy/30 bg-brand-healthy/5 text-brand-healthy hover:bg-brand-healthy/10 transition-colors flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider disabled:opacity-50"
              >
                <ThumbsUp className="w-4.5 h-4.5" />
                <span>Correct</span>
              </button>
              <button
                onClick={() => handleFeedback(false)}
                disabled={submittingFeedback}
                className="flex-1 py-3 px-4 rounded-xl border border-brand-pathology/30 bg-brand-pathology/5 text-brand-pathology hover:bg-brand-pathology/10 transition-colors flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-wider disabled:opacity-50"
              >
                <ThumbsDown className="w-4.5 h-4.5" />
                <span>Incorrect</span>
              </button>
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
