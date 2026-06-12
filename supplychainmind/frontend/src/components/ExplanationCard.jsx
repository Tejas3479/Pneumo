import React from 'react';

function ExplanationCard({ shipment }) {
  return (
    <div className="bg-gray-800/60 backdrop-blur-md border border-gray-700/50 p-6 rounded-xl shadow-2xl mt-6 transition-all duration-300 hover:shadow-cyan-900/10">
      <div className="flex items-center justify-between border-b border-gray-700/50 pb-3 mb-4">
        <h3 className="text-lg font-semibold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          ML Risk Explanation
        </h3>
        <span className="font-mono text-xs text-gray-400">{shipment.ShipmentID}</span>
      </div>
      <p className="text-gray-300 text-sm leading-relaxed">
        {shipment.Explanation}
      </p>
      <div className="mt-4 pt-3 border-t border-gray-700/30 flex gap-2">
        <span className="text-xs bg-gray-900/60 text-gray-400 px-2 py-1 rounded">
          Origin: {shipment.Origin}
        </span>
        <span className="text-xs bg-gray-900/60 text-gray-400 px-2 py-1 rounded">
          Destination: {shipment.Destination}
        </span>
        <span className={`text-xs px-2 py-1 rounded font-semibold ${
          shipment.RiskLevel === 'High' ? 'bg-rose-950/40 text-rose-400' : shipment.RiskLevel === 'Medium' ? 'bg-amber-950/40 text-amber-400' : 'bg-emerald-950/40 text-emerald-400'
        }`}>
          Risk: {shipment.RiskLevel}
        </span>
      </div>
    </div>
  )
}

export default ExplanationCard;
