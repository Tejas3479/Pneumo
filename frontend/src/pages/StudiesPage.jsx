import React, { useState, useEffect } from 'react';
import { api } from '../api/client';
import { useApp } from '../context/AppContext';
import StudyTable from '../components/StudyTable';
import { RefreshCw, Database } from 'lucide-react';

export default function StudiesPage() {
  const { addNotification } = useApp();
  const [studies, setStudies] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);

  const fetchStudies = async () => {
    setLoading(true);
    try {
      const data = await api.getStudies();
      setStudies(data);
    } catch (err) {
      console.error(err);
      addNotification('Failed to query studies from DICOM registry: ' + err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStudies();
  }, []);

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Page Header */}
      <div className="glass-panel p-6 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-brand-cyan">
            <Database className="w-5 h-5" />
            <h2 className="font-heading font-bold text-xl text-white">Acquired DICOMweb Studies</h2>
          </div>
          <p className="text-xs text-brand-textMuted max-w-xl leading-relaxed">
            Query studies indexes retrieved from standard QIDO-RS web queries. Clicking on a study launches the medical-grade Cornerstone viewport or loads SR report summaries.
          </p>
        </div>
        <button
          onClick={fetchStudies}
          disabled={loading}
          className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-brand-border bg-slate-900 text-xs font-semibold text-white hover:bg-slate-800 transition-colors shrink-0 uppercase tracking-wider disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh List</span>
        </button>
      </div>

      {loading ? (
        <div className="glass-panel p-16 rounded-2xl flex flex-col items-center justify-center gap-4 text-center">
          <div className="w-10 h-10 rounded-full border-4 border-slate-900 border-t-brand-cyan animate-spin"></div>
          <span className="text-xs text-brand-textMuted">Querying studies index registry...</span>
        </div>
      ) : (
        <StudyTable 
          studies={studies} 
          searchQuery={searchQuery} 
          onSearchChange={setSearchQuery} 
        />
      )}

    </div>
  );
}
