import React from 'react';
import { ShieldAlert, Calendar, MapPin, Truck, Box, ShieldCheck, HelpCircle } from 'lucide-react';

function ExplanationCard({ shipment }) {
  if (!shipment) {
    return (
      <div className="bg-[#1e293b]/40 border border-slate-800 border-dashed rounded-xl p-8 text-center text-slate-500 text-xs h-full flex flex-col justify-center items-center gap-2">
        <HelpCircle size={32} className="text-slate-600 animate-pulse" />
        <p>Select a shipment from the ledger or disruption table to view deep ML explanations and recommendations.</p>
      </div>
    );
  }

  const isHighRisk = shipment.risk_level === 'High';
  const isMediumRisk = shipment.risk_level === 'Medium';
  
  let riskColor = 'text-emerald-400 border-emerald-800/40 bg-emerald-950/30';
  if (isHighRisk) {
    riskColor = 'text-rose-400 border-rose-800/40 bg-rose-950/30';
  } else if (isMediumRisk) {
    riskColor = 'text-amber-400 border-amber-800/40 bg-amber-950/30';
  }

  return (
    <div className="bg-[#1e293b]/80 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden group">
      <div className="absolute top-0 right-0 w-32 h-32 bg-blue-600/5 blur-3xl pointer-events-none rounded-full"></div>
      
      {/* Title */}
      <h3 className="text-lg font-bold text-slate-200 mb-4 flex items-center gap-2">
        <ShieldCheck className="text-blue-400" size={20} /> ML Insight Explanation
      </h3>

      {/* Shipment Details Metadata */}
      <div className="grid grid-cols-2 gap-3 mb-5 text-xs text-slate-400 border-b border-slate-800/60 pb-4">
        <div className="flex items-center gap-1.5">
          <MapPin size={12} className="text-slate-500" />
          <span>Route: <strong>{shipment.origin} → {shipment.destination}</strong></span>
        </div>
        <div className="flex items-center gap-1.5">
          <Truck size={12} className="text-slate-500" />
          <span>Carrier: <strong>{shipment.carrier}</strong></span>
        </div>
        <div className="flex items-center gap-1.5">
          <Calendar size={12} className="text-slate-500" />
          <span>Departure: <strong>{shipment.departure_date}</strong></span>
        </div>
        <div className="flex items-center gap-1.5">
          <Box size={12} className="text-slate-500" />
          <span>Category: <strong>{shipment.product_category}</strong></span>
        </div>
      </div>

      {/* Prediction Metrics */}
      <div className="bg-slate-950/40 border border-slate-800/60 rounded-xl p-4 mb-5 flex justify-between items-center">
        <div>
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 block mb-1">
            Predicted Delay
          </span>
          <span className="text-3xl font-extrabold text-slate-100 font-mono">
            {shipment.predicted_delay.toFixed(1)}
            <span className="text-sm font-normal text-slate-500 ml-1">days</span>
          </span>
        </div>
        
        <div className="text-right">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 block mb-1">
            Risk Tier
          </span>
          <span className={`inline-block px-3 py-1 rounded-full text-xs font-bold border ${riskColor}`}>
            {shipment.risk_level} Risk
          </span>
        </div>
      </div>

      {/* Bound details */}
      <div className="mb-5 bg-slate-950/20 rounded-lg p-3 border border-slate-800/30 text-xs">
        <div className="flex justify-between items-center text-slate-400 mb-1">
          <span>Lower Bound (5th percentile):</span>
          <span className="font-mono text-slate-300 font-medium">{shipment.lower_bound.toFixed(1)} days</span>
        </div>
        <div className="flex justify-between items-center text-slate-400">
          <span>Upper Bound (95th percentile):</span>
          <span className="font-mono text-slate-300 font-medium">{shipment.upper_bound.toFixed(1)} days</span>
        </div>
      </div>

      {/* Narrative Explanation Text */}
      <div className="space-y-3">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 block">
          Model Narrative
        </span>
        <div className="p-4 bg-slate-950/50 border border-slate-850 rounded-xl text-sm text-slate-300 leading-relaxed italic">
          "{shipment.explanation}"
        </div>
      </div>
    </div>
  );
}

export default ExplanationCard;
