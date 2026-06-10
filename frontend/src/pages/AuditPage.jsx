import React, { useState } from 'react';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';
import { ShieldAlert, ShieldCheck, RefreshCw } from 'lucide-react';

export default function AuditPage() {
  const { addNotification } = useApp();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const runAuditLedgerCheck = async () => {
    setLoading(true);
    setResult(null);
    try {
      const data = await api.verifyLedger();
      setResult(data);
      if (data.valid) {
        addNotification('Ledger hash chain integrity verified.', 'success');
      } else {
        addNotification('Audit ledger tamper mismatch detected!', 'error');
      }
    } catch (err) {
      console.error(err);
      addNotification('Audit ledger verification failed: ' + err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in relative z-0">
      
      {/* Header Panel */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-brand-cyan">
            <ShieldCheck className="w-5 h-5" />
            <h2 className="font-heading font-bold text-xl text-white">Cryptographic Audit Ledger</h2>
          </div>
          <p className="text-xs text-brand-textMuted max-w-xl leading-relaxed">
            Verify the continuous learning database integrity. We check the SHA-256 row-hash chain dynamically, matching each entry's hash to its parent record.
          </p>
        </div>
        <button
          onClick={runAuditLedgerCheck}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-brand-border bg-slate-900 text-xs font-semibold text-white hover:bg-slate-800 transition-colors shrink-0 uppercase tracking-wider disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Verify Ledger</span>
        </button>
      </div>

      {loading ? (
        <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center gap-4 text-center">
          <div className="w-10 h-10 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin text-brand-cyan"></div>
          <span className="text-xs text-brand-textMuted">Auditing SHA-256 row-chains...</span>
        </div>
      ) : result ? (
        <div className="max-w-2xl mx-auto space-y-6">
          
          {/* Main Status Panel */}
          {result.valid ? (
            <div className="glass-panel p-8 rounded-2xl border-brand-healthy/30 bg-brand-healthy/5 text-center space-y-4 flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-brand-healthy/10 border border-brand-healthy/30 text-brand-healthy flex items-center justify-center shadow-lg shadow-brand-healthy/25">
                <ShieldCheck className="w-8 h-8" />
              </div>
              <div className="space-y-1">
                <h3 className="font-heading font-bold text-slate-100 text-lg">Ledger Integrity Secured</h3>
                <p className="text-xs text-slate-300 max-w-md mx-auto leading-relaxed">
                  All active learning feedback records verified successfully. No row tampering or hash mismatches discovered in audit trail.
                </p>
              </div>
            </div>
          ) : (
            <div className="glass-panel p-8 rounded-2xl border-brand-pathology/30 bg-brand-pathology/5 text-center space-y-4 flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-brand-pathology/10 border border-brand-pathology/30 text-brand-pathology flex items-center justify-center shadow-lg shadow-brand-pathology/25">
                <ShieldAlert className="w-8 h-8 text-brand-pathology animate-bounce" />
              </div>
              <div className="space-y-1">
                <h3 className="font-heading font-bold text-slate-100 text-lg">Tampering Discovered</h3>
                <p className="text-xs text-slate-300 max-w-md mx-auto leading-relaxed">
                  The row hash chain integrity checks failed! Mismatched hashes identify row alterations or unauthorized database insertion.
                </p>
              </div>
            </div>
          )}

          {/* Mismatch Ledger details */}
          {!result.valid && result.mismatches && result.mismatches.length > 0 && (
            <div className="glass-panel p-5 rounded-xl space-y-3">
              <span className="text-[10px] uppercase font-bold text-brand-textMuted block">Mismatched Block Registry</span>
              <div className="max-h-60 overflow-y-auto space-y-2 border border-brand-border/40 p-3 rounded-lg bg-slate-950/20">
                {result.mismatches.map((m, idx) => (
                  <div key={idx} className="flex justify-between items-center bg-brand-pathology/5 border border-brand-pathology/20 p-2.5 rounded text-xs text-brand-pathology font-mono">
                    <span>Mismatched Row ID:</span>
                    <span className="font-bold">{m}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      ) : (
        <div className="glass-panel p-12 rounded-2xl text-center text-xs text-brand-textMuted max-w-md mx-auto">
          Click "Verify Ledger" to dynamically run database checks across the feedback registry trail.
        </div>
      )}

    </div>
  );
}
