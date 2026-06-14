import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';
import { ArrowLeft, Printer, ShieldCheck, AlertCircle } from 'lucide-react';

export default function ReportPage() {
  const { studyUid } = useParams();
  const navigate = useNavigate();
  const { addNotification } = useApp();
  
  const [loading, setLoading] = useState(true);
  const [studyMeta, setStudyMeta] = useState(null);   // Full study metadata from /studies/{uid}
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // /studies/{uid} returns { status, study_instance_uid, instances: [...] }
      // Each instance: { sop_instance_uid, series_instance_uid, modality, file_type, file_path }
      // Patient metadata comes from the QIDO-RS /dicomweb/studies response, but is also stored
      // in the dicom_instances table. We call the dicomweb QIDO-RS to get it:
      let patientName = 'N/A', patientId = 'N/A', studyDate = 'N/A', accession = 'N/A', studyDesc = 'N/A';

      try {
        // Try to get patient-level metadata from QIDO-RS (returns DICOM JSON format)
        const qidoStudies = await api.getStudies();
        const matchingStudy = Array.isArray(qidoStudies)
          ? qidoStudies.find(s => {
              const uid = s['0020000D']?.Value?.[0] || s.studyInstanceUid || s.StudyInstanceUID;
              return uid === studyUid;
            })
          : null;

        if (matchingStudy) {
          // QIDO-RS DICOM JSON uses numeric tag keys
          patientName = matchingStudy['00100010']?.Value?.[0]?.Alphabetic
            || matchingStudy['00100010']?.Value?.[0]
            || 'N/A';
          patientId = matchingStudy['00100020']?.Value?.[0] || 'N/A';
          studyDate = matchingStudy['00080020']?.Value?.[0] || 'N/A';
          accession = matchingStudy['00080050']?.Value?.[0] || 'N/A';
          studyDesc = matchingStudy['00081030']?.Value?.[0] || 'Chest PA';
        }
      } catch (qidoErr) {
        console.warn('[ReportPage] QIDO-RS fetch failed, using fallback:', qidoErr.message);
        // Fallback: try flat /studies/{uid} response
        try {
          const flat = await api.getStudyDetails(studyUid);
          if (flat.instances && flat.instances.length > 0) {
            const inst = flat.instances[0];
            patientName = inst.patient_name || 'N/A';
            patientId = inst.patient_id || 'N/A';
            studyDate = inst.study_date || 'N/A';
            accession = inst.accession_number || 'N/A';
            studyDesc = inst.study_description || 'Chest PA';
          }
        } catch (flatErr) {
          console.warn('[ReportPage] Flat study fetch also failed:', flatErr.message);
        }
      }

      setStudyMeta({ patientName, patientId, studyDate, accession, studyDesc });

      // Fetch cached prediction
      try {
        const predData = await api.getStudyPrediction(studyUid);
        if (predData.status === 'completed') {
          setPrediction(predData);
        } else if (predData.status === 'not_predicted') {
          setPrediction(null);
        }
      } catch (predErr) {
        if (predErr.response?.status !== 404) {
          console.warn('[ReportPage] Prediction fetch failed:', predErr.message);
        }
        setPrediction(null);
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
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

  const formatStudyDate = (dateStr) => {
    if (!dateStr || dateStr === 'N/A' || dateStr.length < 8) return dateStr || 'N/A';
    try {
      return `${dateStr.substring(0,4)}-${dateStr.substring(4,6)}-${dateStr.substring(6,8)}`;
    } catch {
      return dateStr;
    }
  };

  if (loading) {
    return (
      <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center gap-4 text-center">
        <div className="w-10 h-10 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin text-brand-cyan"></div>
        <span className="text-xs text-brand-textMuted">Generating clinical report...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel p-8 rounded-2xl border-brand-pathology/30 bg-brand-pathology/5 flex items-start gap-4 max-w-xl mx-auto">
        <AlertCircle className="w-6 h-6 text-brand-pathology shrink-0" />
        <div className="space-y-2">
          <h3 className="font-heading font-bold text-slate-100">Report Load Failed</h3>
          <p className="text-xs text-slate-300">{error}</p>
          <button onClick={fetchData} className="text-xs text-brand-cyan underline">Retry</button>
        </div>
      </div>
    );
  }

  const { patientName, patientId, studyDate, accession, studyDesc } = studyMeta || {};
  const probability = prediction?.probability;
  const probabilityPct = typeof probability === 'number' && !isNaN(probability)
    ? Math.round(probability * 100) + '%'
    : 'N/A';
  const uncertaintyDisplay = (typeof prediction?.uncertainty === 'number' && prediction.uncertainty !== null)
    ? `± ${(prediction.uncertainty * 100).toFixed(1)}%`
    : 'N/A';

  return (
    <div className="space-y-6 max-w-4xl mx-auto print:p-0 animate-fade-in relative z-0">
      
      {/* Action buttons — hidden in print */}
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

      {/* Printable Clinical Structured Report */}
      <div className="glass-panel p-8 rounded-2xl border border-brand-border bg-slate-950/20 space-y-8 print:border-none print:bg-white print:text-black print:shadow-none print:p-0">
        
        {/* Header */}
        <div className="border-b border-brand-border pb-6 flex justify-between items-start print:border-black">
          <div>
            <h1 className="font-heading font-extrabold text-2xl text-white print:text-black uppercase tracking-tight">PneumoDex Diagnostic Report</h1>
            <p className="text-xs text-brand-textMuted print:text-slate-500 font-semibold tracking-wider uppercase mt-1">
              Computer-Aided Detection Structured Summary
            </p>
            {studyDesc && studyDesc !== 'N/A' && (
              <p className="text-[11px] text-slate-400 mt-1">{studyDesc}</p>
            )}
          </div>
          <div className="text-right space-y-1">
            <span className="text-xs font-semibold px-2.5 py-1 rounded bg-brand-cyan/10 border border-brand-cyan/30 text-brand-cyan uppercase tracking-wider print:border-black print:text-black block">
              For Research Use Only
            </span>
            <span className="text-[10px] text-brand-textMuted font-mono block">{studyUid?.substring(0, 20)}...</span>
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
            <span className="text-sm font-semibold text-slate-100 print:text-black">{formatStudyDate(studyDate)}</span>
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
                <span className="font-heading font-black text-xl text-white print:text-black">{probabilityPct}</span>
                <span className="text-[9px] uppercase font-bold text-brand-textMuted tracking-wider mt-0.5">AI Probability</span>
              </div>
              <div className="p-4 rounded-xl border border-brand-border bg-slate-950/40 text-center flex flex-col justify-center print:border-black">
                <span className="font-heading font-black text-sm text-slate-200 print:text-black">{uncertaintyDisplay}</span>
                <span className="text-[9px] uppercase font-bold text-brand-textMuted tracking-wider mt-1">Ensemble Std Dev</span>
              </div>
            </div>
          ) : (
            <div className="p-6 rounded-xl border border-dashed border-brand-border bg-slate-950/10 text-center space-y-2">
              <p className="text-xs text-brand-textMuted print:text-black">
                No prediction findings generated for this study.
              </p>
              <p className="text-[11px] text-brand-textMuted">
                Open this study in the <button onClick={() => navigate(`/studies/${studyUid}`)} className="text-brand-cyan underline">Study Viewer</button> to run diagnostics first.
              </p>
            </div>
          )}
        </div>

        {/* Clinical Summary Narrative */}
        {prediction?.text_justification && (
          <div className="space-y-3">
            <h3 className="font-heading font-bold text-slate-200 text-sm print:text-black border-b border-brand-border/60 pb-2 print:border-black">
              AI Clinical Explanation Rationale
            </h3>
            <p className="text-xs leading-relaxed text-slate-300 font-medium bg-slate-950/40 p-4 rounded-xl border border-brand-border print:bg-white print:text-black print:border-black">
              {prediction.text_justification}
            </p>
          </div>
        )}

        {/* Audit Trail Row */}
        <div className="space-y-3 pt-6 border-t border-brand-border print:border-black">
          <div className="flex items-center gap-2 text-[10px] uppercase font-bold text-brand-textMuted print:text-slate-500">
            <ShieldCheck className="w-4 h-4 text-brand-healthy print:text-black" />
            <span>Audit Trail — Hash-chain protected record</span>
          </div>
          <p className="text-[10px] text-brand-textMuted leading-relaxed max-w-xl print:text-slate-500">
            This diagnostic record is registered in the SQLite continuous learning ledger. Row-level SHA-256 HMAC 
            chain hashes protect data integrity. Verify the ledger in the <strong className="text-slate-300">Ledger Audit</strong> page.
          </p>
        </div>

        {/* Print Disclaimer */}
        <div className="hidden print:block pt-8 border-t border-dotted border-slate-400 text-[9px] text-slate-500 italic">
          Disclaimer: This report was generated by the PneumoDex CAD AI system. For research use only. 
          Final diagnostics must be reviewed and signed off by a qualified radiologist.
        </div>

      </div>
    </div>
  );
}
