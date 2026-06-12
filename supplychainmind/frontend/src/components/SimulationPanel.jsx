import React, { useState } from 'react';
import { Play, Loader2, AlertTriangle, Eye } from 'lucide-react';

const PORTS = [
  "Shanghai", "Singapore", "Rotterdam", "Los Angeles", 
  "Hamburg", "Dubai", "New York", "Shenzhen", "Antwerp", "Mumbai"
];

function SimulationPanel({ onSubmit, simulationResults, onSelectShipment }) {
  const [port, setPort] = useState(PORTS[0]);
  const [days, setDays] = useState(3);
  const [isLoading, setIsLoading] = useState(false);

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      await onSubmit(port, days);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-[#1e293b]/80 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden group">
      <div className="absolute top-0 right-0 w-32 h-32 bg-amber-600/5 blur-3xl pointer-events-none rounded-full"></div>
      
      <h3 className="text-lg font-bold text-slate-200 mb-2 flex items-center gap-2">
        <AlertTriangle className="text-amber-400" size={20} /> Port Disruption Simulator
      </h3>
      <p className="text-xs text-slate-400 mb-5">
        Run what-if scenarios by simulating full closures at major shipping terminals. Overrides route congestion to 100%.
      </p>

      <form onSubmit={handleFormSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end mb-6">
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Target Port</label>
          <select 
            value={port} 
            onChange={(e) => setPort(e.target.value)}
            className="bg-slate-900 border border-slate-800 text-slate-300 text-sm p-2.5 rounded-lg w-full focus:outline-none focus:ring-1 focus:ring-amber-500"
          >
            {PORTS.map(p => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Closure Duration (Days)</label>
          <input 
            type="number" 
            min="1" 
            max="30"
            value={days} 
            onChange={(e) => setDays(parseInt(e.target.value) || 1)}
            className="bg-slate-900 border border-slate-800 text-slate-300 text-sm p-2.5 rounded-lg w-full focus:outline-none focus:ring-1 focus:ring-amber-500"
          />
        </div>

        <div>
          <button 
            type="submit" 
            disabled={isLoading}
            className="w-full bg-amber-600 hover:bg-amber-500 disabled:bg-amber-800 text-white font-semibold text-sm py-2.5 px-4 rounded-lg shadow-lg hover:shadow-amber-600/20 transition-all duration-200 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Simulating...
              </>
            ) : (
              <>
                <Play size={16} fill="white" />
                Simulate Disruption
              </>
            )}
          </button>
        </div>
      </form>

      {/* Simulation Results Section */}
      <div className="border-t border-slate-800/60 pt-5">
        <h4 className="text-sm font-bold text-slate-300 mb-3">Simulated Disruption Impact</h4>
        {simulationResults.length === 0 ? (
          <div className="bg-slate-950/30 border border-slate-800/40 rounded-lg p-5 text-center text-xs text-slate-500">
            No disruption simulation active. Run a scenario above to inspect affected transits.
          </div>
        ) : (
          <div className="overflow-x-auto border border-slate-800/60 rounded-lg">
            <table className="min-w-full text-left bg-slate-950/20">
              <thead>
                <tr className="bg-slate-950/40 text-slate-400 text-[11px] font-semibold uppercase tracking-wider border-b border-slate-800">
                  <th className="py-2.5 px-3">Shipment ID</th>
                  <th className="py-2.5 px-3">Route</th>
                  <th className="py-2.5 px-3 text-right">Simulated Delay</th>
                  <th className="py-2.5 px-3 text-center">New Risk</th>
                  <th className="py-2.5 px-3 text-center">Insight</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/40 text-xs">
                {simulationResults.map(s => {
                  let badgeClass = "bg-emerald-950/60 text-emerald-400 border border-emerald-800/50";
                  if (s.risk_level === 'High') {
                    badgeClass = "bg-rose-950/60 text-rose-400 border border-rose-800/50";
                  } else if (s.risk_level === 'Medium') {
                    badgeClass = "bg-amber-950/60 text-amber-400 border border-amber-800/50";
                  }
                  
                  return (
                    <tr key={s.shipment_id} className="hover:bg-slate-950/40 transition-colors">
                      <td className="py-2.5 px-3 font-mono text-blue-400 font-semibold">{s.shipment_id}</td>
                      <td className="py-2.5 px-3 text-slate-300">
                        {s.origin} → {s.destination}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold text-amber-400">
                        {s.predicted_delay.toFixed(1)} days
                      </td>
                      <td className="py-2.5 px-3 text-center">
                        <span className={`inline-block px-2 py-0.5 rounded-full text-[10px] font-bold ${badgeClass}`}>
                          {s.risk_level}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-center">
                        <button 
                          onClick={() => onSelectShipment(s)}
                          className="bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium text-[10px] px-2 py-1 rounded transition-colors"
                        >
                          <Eye size={10} className="inline mr-1" /> View
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default SimulationPanel;
