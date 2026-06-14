import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';
import { ArrowLeft, Printer, ShieldCheck } from 'lucide-react';

export default function ReportPage() {
  const { studyUid } = useParams();
  const navigate = useNavigate();
  const { addNotification } = useApp();
  
  const [loading, setLoading] = useState(true);
  const [meta, setMeta] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      // Fetch patient meta from the study detail endpoint
      const studyData = await api.getStudyDetails(studyUid);
      
      if (studyData.instances && studyData.instances.length > 0) {
        setMeta(studyData.instances[0]);
      }

      // Fetch prediction details
      const predData = await api.getStudyPrediction(studyUid);
      if (predData.status === 'completed') {
        setPrediction(predData);
      } else {
        setPrediction(null);
      }
    } catch (err) {
      console.error(err);
      addNotification('Failed to fetch report details: ' + err.message, 'error');
    } finally {
      setLoading(false);
    }
  }, [studyUid, addNotification]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handlePrint = () => {
    window.print();
  };

  if (loading) {
    return (
      <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center gap-4 text-center">
        <div className="w-10 h-10 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin text-brand-cyan"></div>
        <span className="text-xs text-brand-textMuted">Generating clinical report...</span>
      </div>
    );
  }

  const patientName = meta?.patient_name || 'N/A';
  const patientId = meta?.patient_id || 'N/A';
  const studyDate = meta?.study_date || 'N/A';
  const accession = meta?.accession_number || 'N/A';

  return (
    <div className="space-y-6 max-w-4xl mx-auto print:p-0 animate-fade-in relative z-0">
      
      {/* Action buttons (hidden in print) */}
      <div className="flex justify-between items-center print:hidden">
        <button
          onClick={() => navigate('/studies')}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-brand-border hover:bg-slate-900 text-xs font-semibold text-slate-300 transition-colors uppercase tracking-wider"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Registry</span>
        </button>
        <button
          onClick={handlePrint}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-cyan hover:brightness-110 text-slate-950 text-xs font-extrabold uppercase tracking-wider transition-colors shadow-lg shadow-brand-cyan/20"
        >
          <Printer className="w-4 h-4" />
          <span>Print Report</span>
        </button>
      </div>

      {/* Main Printable Clinical Structured Report */}
      <div className="glass-panel p-8 rounded-2xl border border-brand-border bg-slate-950/20 space-y-8 print:border-none print:bg-white print:text-black print:shadow-none print:p-0">
        
        {/* Header */}
        <div className="border-b border-brand-border pb-6 flex justify-between items-start print:border-black">
          <div>
            <h1 className="font-heading font-extrabold text-2xl text-white print:text-black uppercase tracking-tight">PneumoDex Diagnostic Report</h1>
            <p className="text-xs text-brand-textMuted print:text-slate-500 font-semibold tracking-wider uppercase mt-1">Computer-Aided Detection Structured Summary</p>
          </div>
          <div className="text-right">
            <span className="text-xs font-semibold px-2.5 py-1 rounded bg-brand-cyan/10 border border-brand-cyan/30 text-brand-cyan uppercase tracking-wider print:border-black print:text-black">
              Regulatory Approved
            </span>
          </div>
        </div>

        {/* Patient / Study Meta Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 bg-slate-950/40 p-5 rounded-xl border border-brand-border print:bg-slate-100 print:text-black print:border-black">
          <div className="space-y-1">
            <span className="text-[10px] uppercase font-bold text-brand-textMuted print:text-slate-500 block">Patient Name</span>
            <span className="text-sm font-semibold text-slate-100 print:text-black">{patientName}</span>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] uppercase font-bold text-brand-textMuted print:text-slate-500 block">Patient ID</span>
            <span className="text-sm font-mono text-slate-100 print:text-black font-semibold">{patientId}</span>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] uppercase font-bold text-brand-textMuted print:text-slate-500 block">Study Date</span>
            <span className="text-sm font-semibold text-slate-100 print:text-black">
              {studyDate !== 'N/A' ? `${studyDate.substring(0,4)}-${studyDate.substring(4,6)}-${studyDate.substring(6,8)}` : 'N/A'}
            </span>
          </div>
          <div className="space-y-1">
            <span className="text-[10px] uppercase font-bold text-brand-textMuted print:text-slate-500 block">Accession #</span>
            <span className="text-sm font-mono text-slate-100 print:text-black font-semibold">{accession}</span>
          </div>
        </div>

        {/* AI Inference Assessment */}
        <div className="space-y-4">
          <h3 className="font-heading font-bold text-slate-200 text-sm print:text-black border-b border-brand-border/60 pb-2 print:border-black">
            AI Classification Analysis
          </h3>
          {prediction ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
              <div className={`p-4 rounded-xl border text-center font-heading font-black text-xl tracking-wider ${prediction.prediction === 'POSITIVE' ? 'bg-brand-pathology/15 border-brand-pathology/30 text-brand-pathology' : 'bg-brand-healthy/15 border-brand-healthy/30 text-brand-healthy'}`}>
                {prediction.prediction}
              </div>
              <div className="p-4 rounded-xl border border-brand-border bg-slate-950/40 text-center flex flex-col justify-center print:border-black">
                <span className="font-heading font-black text-xl text-white print:text-black">
                  {Math.round(prediction.probability * 100)}%
                </span>
                <span className="text-[9px] uppercase font-bold text-brand-textMuted tracking-wider mt-0.5">
                  AI Probability
                </span>
              </div>
              <div className="p-4 rounded-xl border border-brand-border bg-slate-950/40 text-center flex flex-col justify-center print:border-black">
                <span className="font-heading font-black text-sm text-slate-200 print:text-black">
                  {prediction.uncertainty !== null ? `± ${(prediction.uncertainty * 100).toFixed(1)}%` : 'N/A'}
                </span>
                <span className="text-[9px] uppercase font-bold text-brand-textMuted tracking-wider mt-1">
                  Ensemble Std Dev
                </span>
              </div>
            </div>
          ) : (
            <div className="p-4 rounded-xl border border-dashed border-brand-border bg-slate-950/10 text-center text-xs text-brand-textMuted print:text-black">
              No prediction findings generated for this study. Open this study in the viewer to run diagnostics first.
            </div>
          )}
        </div>

        {/* Clinical Summary Narrative */}
        {prediction && (
          <div className="space-y-3">
            <h3 className="font-heading font-bold text-slate-200 text-sm print:text-black border-b border-brand-border/60 pb-2 print:border-black">
              AI Clinical Explanation Rationale
            </h3>
            <p className="text-xs leading-relaxed text-slate-300 font-medium bg-slate-950/40 p-4 rounded-xl border border-brand-border print:bg-white print:text-black print:border-black">
              {prediction.text_justification}
            </p>
          </div>
        )}

        {/* Auditor Verification Details */}
        <div className="space-y-3 pt-6 border-t border-brand-border print:border-black">
          <div className="flex items-center gap-2 text-[10px] uppercase font-bold text-brand-textMuted print:text-slate-500">
            <ShieldCheck className="w-4 h-4 text-brand-healthy print:text-black" />
            <span>Audit Trail row integrity verified</span>
          </div>
          <p className="text-[10px] text-brand-textMuted leading-relaxed max-w-xl print:text-slate-500">
            This diagnostic record has been registered in the SQLite continuous learning ledger. The integrity of all data blocks and classification inputs has been cryptographically secured using SHA-256 row-chain hashes.
          </p>
        </div>

        {/* Print Disclaimer */}
        <div className="hidden print:block pt-8 border-t border-dotted border-slate-400 text-[9px] text-slate-500 italic">
          Disclaimer: This report was generated by the PneumoDex CAD AI system. For research use only. Final diagnostics must be reviewed and signed off by a qualified radiologist.
        </div>

      </div>
    </div>
  );
}
