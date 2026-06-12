import React from 'react';
import { Eye, HelpCircle } from 'lucide-react';

function ShipmentTable({ shipments, onSelect }) {
  return (
    <div className="bg-[#1e293b]/80 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden group">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-slate-200 flex items-center gap-2">
          Active Shipments & Risk Status
        </h3>
        <span className="text-xs text-slate-500 font-mono">
          Total Shipments: {shipments.length}
        </span>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full table-auto text-left">
          <thead>
            <tr className="text-slate-400 border-b border-slate-800 text-xs font-semibold uppercase tracking-wider">
              <th className="pb-3 px-3">Shipment ID</th>
              <th className="pb-3 px-3">Route</th>
              <th className="pb-3 px-3">Carrier</th>
              <th className="pb-3 px-3">Category</th>
              <th className="pb-3 px-3 text-right">Predicted Delay</th>
              <th className="pb-3 px-3 text-center">Risk level</th>
              <th className="pb-3 px-3 text-center">Uncertainty Bound (90%)</th>
              <th className="pb-3 px-3 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/40">
            {shipments.length === 0 ? (
              <tr>
                <td colSpan="8" className="py-8 text-center text-slate-500 italic text-sm">
                  No shipments loaded. Please upload a CSV shipment ledger above.
                </td>
              </tr>
            ) : (
              shipments.map(s => {
                let badgeClass = "bg-emerald-950/60 text-emerald-400 border border-emerald-800/50";
                if (s.risk_level === 'High') {
                  badgeClass = "bg-rose-950/60 text-rose-400 border border-rose-800/50";
                } else if (s.risk_level === 'Medium') {
                  badgeClass = "bg-amber-950/60 text-amber-400 border border-amber-800/50";
                }
                
                return (
                  <tr 
                    key={s.shipment_id} 
                    className="hover:bg-slate-950/30 transition-all duration-150 border-b border-slate-800/40 text-sm"
                  >
                    <td className="py-3 px-3 font-mono text-blue-400 font-medium">{s.shipment_id}</td>
                    <td className="py-3 px-3 text-slate-300 font-medium">
                      {s.origin} <span className="text-slate-600">→</span> {s.destination}
                    </td>
                    <td className="py-3 px-3 text-slate-400 text-xs">{s.carrier}</td>
                    <td className="py-3 px-3 text-slate-400 text-xs">{s.product_category}</td>
                    <td className="py-3 px-3 text-right font-bold text-slate-200">
                      {s.predicted_delay.toFixed(1)} <span className="text-[10px] text-slate-500 font-normal">days</span>
                    </td>
                    <td className="py-3 px-3 text-center">
                      <span className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-semibold ${badgeClass}`}>
                        {s.risk_level}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-center text-slate-500 font-mono text-xs">
                      {s.confidence}
                    </td>
                    <td className="py-3 px-3 text-center">
                      <button 
                        onClick={() => onSelect(s)} 
                        className="bg-blue-600/10 hover:bg-blue-600 text-blue-400 hover:text-white border border-blue-600/30 font-medium text-xs px-2.5 py-1 rounded-md transition-all duration-200 flex items-center gap-1 mx-auto"
                      >
                        <Eye size={12} /> Insight
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
  );
}

export default ShipmentTable;
