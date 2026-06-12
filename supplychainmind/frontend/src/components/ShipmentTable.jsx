import React from 'react';

function ShipmentTable({ shipments, onSelect }) {
  return (
    <div className="bg-gray-800/60 backdrop-blur-md border border-gray-700/50 rounded-xl p-6 shadow-2xl mt-6 transition-all duration-300 hover:shadow-cyan-900/20">
      <h3 className="text-xl font-semibold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-4">
        Active Shipments & Risk Status
      </h3>
      <div className="overflow-x-auto">
        <table className="min-w-full table-auto border-collapse text-left">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700/50 text-sm">
              <th className="pb-3 font-medium">ID</th>
              <th className="pb-3 font-medium">Origin</th>
              <th className="pb-3 font-medium">Destination</th>
              <th className="pb-3 font-medium">Delay (days)</th>
              <th className="pb-3 font-medium text-center">Risk Level</th>
              <th className="pb-3 font-medium text-right">Insight</th>
            </tr>
          </thead>
          <tbody>
            {shipments.length === 0 ? (
              <tr>
                <td colSpan="6" className="py-8 text-center text-gray-500 italic text-sm">
                  No shipments loaded. Run a simulation to display results.
                </td>
              </tr>
            ) : (
              shipments.map(s => (
                <tr 
                  key={s.ShipmentID} 
                  className="border-b border-gray-800/40 hover:bg-gray-700/20 transition-all duration-200"
                >
                  <td className="py-3 font-mono text-cyan-400 text-sm">{s.ShipmentID}</td>
                  <td className="py-3 text-sm text-gray-300">{s.Origin}</td>
                  <td className="py-3 text-sm text-gray-300">{s.Destination}</td>
                  <td className="py-3 font-semibold text-sm text-gray-200">{s.PredictedDelay.toFixed(1)}</td>
                  <td className="py-3 text-center">
                    <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold tracking-wide ${
                      s.RiskLevel === 'High' 
                        ? 'bg-rose-950/60 text-rose-400 border border-rose-800/50' 
                        : s.RiskLevel === 'Medium' 
                          ? 'bg-amber-950/60 text-amber-400 border border-amber-800/50' 
                          : 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/50'
                    }`}>
                      {s.RiskLevel}
                    </span>
                  </td>
                  <td className="py-3 text-right">
                    <button 
                      onClick={() => onSelect(s)} 
                      className="bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs px-3 py-1.5 rounded-lg shadow-lg hover:shadow-cyan-600/30 transition-all duration-300 transform active:scale-95"
                    >
                      Explain
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ShipmentTable;
