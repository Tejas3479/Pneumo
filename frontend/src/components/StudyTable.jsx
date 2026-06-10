import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, FileText } from 'lucide-react';

export default function StudyTable({ studies, searchQuery, onSearchChange }) {
  const navigate = useNavigate();

  const filteredStudies = studies.filter((study) => {
    const patientId = study["00100020"]?.["Value"]?.[0]?.toLowerCase() || '';
    const patientNameObj = study["00100010"]?.["Value"]?.[0] || {};
    const patientName = (patientNameObj.Alphabetic || '').toLowerCase();
    const query = searchQuery.toLowerCase();
    return patientId.includes(query) || patientName.includes(query);
  });

  const formatDate = (studyDate) => {
    if (!studyDate || studyDate === 'N/A') return 'N/A';
    try {
      return `${studyDate.substring(0, 4)}-${studyDate.substring(4, 6)}-${studyDate.substring(6, 8)}`;
    } catch (e) {
      return studyDate;
    }
  };

  return (
    <div className="space-y-4">
      {/* Search Filter Header */}
      <div className="relative">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search studies by Patient Name or ID..."
          className="w-full bg-slate-950/60 border border-brand-border rounded-xl pl-5 pr-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-brand-cyan transition-colors"
        />
      </div>

      {/* Grid Study List */}
      <div className="glass-panel rounded-2xl overflow-hidden border border-brand-border">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-brand-border text-brand-textMuted uppercase font-bold tracking-wider bg-slate-950/20">
                <th className="py-4 px-6">Patient Name</th>
                <th className="py-4 px-6">Patient ID</th>
                <th className="py-4 px-6">Study Date</th>
                <th className="py-4 px-6">Modality</th>
                <th className="py-4 px-6 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-brand-border">
              {filteredStudies.length === 0 ? (
                <tr>
                  <td colSpan="5" className="text-center py-12 text-slate-500 font-medium">
                    No matching studies discovered in DICOM registry.
                  </td>
                </tr>
              ) : (
                filteredStudies.map((study, idx) => {
                  const studyUid = study["0020000D"]["Value"][0];
                  const patientId = study["00100020"]?.["Value"]?.[0] || 'N/A';
                  const patientNameObj = study["00100010"]?.["Value"]?.[0] || {};
                  const patientName = patientNameObj.Alphabetic || 'N/A';
                  const studyDate = study["00080020"]?.["Value"]?.[0] || 'N/A';
                  const modality = study["00080060"]?.["Value"]?.[0] || 'DX';

                  return (
                    <tr key={studyUid + '-' + idx} className="hover:bg-slate-900/10 transition-colors group">
                      <td className="py-4 px-6 font-semibold text-slate-100 group-hover:text-brand-cyan transition-colors">
                        {patientName}
                      </td>
                      <td className="py-4 px-6 text-slate-300 font-mono text-[11px]">
                        {patientId}
                      </td>
                      <td className="py-4 px-6 text-slate-300">
                        {formatDate(studyDate)}
                      </td>
                      <td className="py-4 px-6">
                        <span className="px-2.5 py-0.5 rounded-full border border-brand-cyan/25 bg-brand-cyan/5 text-brand-cyan font-bold tracking-wide uppercase text-[9px]">
                          {modality}
                        </span>
                      </td>
                      <td className="py-4 px-6 text-right flex justify-end gap-2.5">
                        <button
                          onClick={() => navigate(`/studies/${studyUid}`)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-brand-cyan/35 hover:bg-brand-cyan/10 text-brand-cyan text-[11px] font-bold uppercase transition-colors"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          <span>Viewer</span>
                        </button>
                        <button
                          onClick={() => navigate(`/reports/${studyUid}`)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-brand-violet/35 hover:bg-brand-violet/10 text-brand-violet text-[11px] font-bold uppercase transition-colors"
                        >
                          <FileText className="w-3.5 h-3.5" />
                          <span>Report</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
