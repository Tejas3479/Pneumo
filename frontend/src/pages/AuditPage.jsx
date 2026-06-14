import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';
import { ShieldAlert, ShieldCheck, RefreshCw, Lock } from 'lucide-react';

export default function AuditPage() {
  const { addNotification } = useApp();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  // Auto-load on mount for immediate visibility
  useEffect(() => {
    runAuditLedgerCheck();
  }, []);

  const runAuditLedgerCheck = async () => {
    setLoading(true);
    setResult(null);
    try {
      const data = await api.verifyLedger();
      setResult(data);
      if (data.valid) {
        addNotification('Ledger hash chain integrity verified.', 'success');
      } else {
        addNotification(`Audit ledger tamper mismatch detected! ${data.mismatches?.length || 0} row(s) compromised.`, 'error');
      }
    } catch (err) {
      console.error(err);
      addNotification('Audit ledger verification failed: ' + (err.response?.data?.detail || err.message), 'error');
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
            Verify the continuous learning database integrity. We check the SHA-256 row-hash chain dynamically, 
            matching each entry's hash to its parent record to detect any unauthorized data modifications.
          </p>
        </div>
        <button
          onClick={runAuditLedgerCheck}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-brand-border bg-slate-900 text-xs font-semibold text-white hover:bg-slate-800 transition-colors shrink-0 uppercase tracking-wider disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Re-Verify Ledger</span>
        </button>
      </div>

      {loading ? (
        <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center gap-4 text-center">
          <div className="w-10 h-10 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin text-brand-cyan"></div>
          <span className="text-xs text-brand-textMuted">Auditing SHA-256 row-chains across all ledger entries...</span>
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
                <h3 className="font-heading font-bold text-slate-100 text-lg">Ledger Integrity Secured ✓</h3>
                <p className="text-xs text-slate-300 max-w-md mx-auto leading-relaxed">
                  All active learning feedback records verified successfully. No row tampering or hash mismatches 
                  discovered in the audit trail.
                </p>
              </div>
              <div className="flex gap-6 text-xs text-brand-textMuted mt-2">
                <div className="text-center">
                  <span className="font-bold text-brand-healthy text-base block">{result.rows_checked ?? '—'}</span>
                  <span>Rows Verified</span>
                </div>
                <div className="text-center">
                  <span className="font-bold text-brand-healthy text-base block">0</span>
                  <span>Mismatches</span>
                </div>
                <div className="text-center">
                  <span className="font-bold text-brand-healthy text-base block">SHA-256</span>
                  <span>Algorithm</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-panel p-8 rounded-2xl border-brand-pathology/30 bg-brand-pathology/5 text-center space-y-4 flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-brand-pathology/10 border border-brand-pathology/30 text-brand-pathology flex items-center justify-center shadow-lg shadow-brand-pathology/25">
                <ShieldAlert className="w-8 h-8 text-brand-pathology animate-bounce" />
              </div>
              <div className="space-y-1">
                <h3 className="font-heading font-bold text-slate-100 text-lg">Tampering Discovered ⚠</h3>
                <p className="text-xs text-slate-300 max-w-md mx-auto leading-relaxed">
                  The row hash chain integrity checks failed! Mismatched hashes identify row alterations or 
                  unauthorized database insertions. Escalate immediately.
                </p>
              </div>
            </div>
          )}

          {/* Mismatch Ledger Table */}
          {!result.valid && result.mismatches && result.mismatches.length > 0 && (
            <div className="glass-panel p-5 rounded-xl space-y-3">
              <div className="flex items-center gap-2">
                <Lock className="w-4 h-4 text-brand-pathology" />
                <span className="text-[10px] uppercase font-bold text-brand-pathology">
                  {result.mismatches.length} Compromised Block{result.mismatches.length > 1 ? 's' : ''} Detected
                </span>
              </div>
              <div className="border border-brand-pathology/20 rounded-lg overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-brand-pathology/10 border-b border-brand-pathology/20">
                      <th className="text-left px-4 py-2 text-brand-pathology font-bold uppercase tracking-wider">#</th>
                      <th className="text-left px-4 py-2 text-brand-pathology font-bold uppercase tracking-wider">Row ID</th>
                      <th className="text-left px-4 py-2 text-brand-pathology font-bold uppercase tracking-wider">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.mismatches.map((m, idx) => (
                      <tr key={idx} className="border-b border-brand-pathology/10 last:border-0 bg-brand-pathology/5 hover:bg-brand-pathology/10 transition-colors">
                        <td className="px-4 py-2.5 text-slate-400">{idx + 1}</td>
                        <td className="px-4 py-2.5 font-mono font-bold text-brand-pathology">{m}</td>
                        <td className="px-4 py-2.5">
                          <span className="px-2 py-0.5 rounded bg-brand-pathology/15 border border-brand-pathology/30 text-brand-pathology text-[10px] font-semibold uppercase">
                            Hash Mismatch
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Integrity Details */}
          <div className="glass-panel p-5 rounded-xl space-y-3">
            <span className="text-[10px] uppercase font-bold text-brand-textMuted block">Verification Details</span>
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div className="space-y-1">
                <span className="text-brand-textMuted block">Hash Algorithm</span>
                <span className="text-white font-mono font-semibold">HMAC-SHA-256</span>
              </div>
              <div className="space-y-1">
                <span className="text-brand-textMuted block">Database</span>
                <span className="text-white font-mono font-semibold">audit_ledger.db</span>
              </div>
              <div className="space-y-1">
                <span className="text-brand-textMuted block">Chain Type</span>
                <span className="text-white font-semibold">Row-Linked Hash Chain</span>
              </div>
              <div className="space-y-1">
                <span className="text-brand-textMuted block">Integrity Status</span>
                <span className={`font-semibold ${result.valid ? 'text-brand-healthy' : 'text-brand-pathology'}`}>
                  {result.valid ? 'VALID ✓' : 'COMPROMISED ✗'}
                </span>
              </div>
            </div>
          </div>

        </div>
      ) : (
        <div className="glass-panel p-12 rounded-2xl text-center text-xs text-brand-textMuted max-w-md mx-auto">
          <div className="w-8 h-8 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin mx-auto mb-4"></div>
          Loading ledger verification...
        </div>
      )}

    </div>
  );
}
