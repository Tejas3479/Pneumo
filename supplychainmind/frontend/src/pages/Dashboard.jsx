import React, { useState, useEffect } from 'react';
import { uploadCSV, getHeatmap, simulateDisruption } from '../api/client';
import ShipmentTable from '../components/ShipmentTable';
import RiskMap from '../components/RiskMap';
import SimulationPanel from '../components/SimulationPanel';
import ExplanationCard from '../components/ExplanationCard';
import { UploadCloud, ShieldAlert, Sparkles, Activity } from 'lucide-react';

function Dashboard() {
  const [shipments, setShipments] = useState([]);
  const [heatmapData, setHeatmapData] = useState(null);
  const [selectedShipment, setSelectedShipment] = useState(null);
  const [simulationResults, setSimulationResults] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // Fetch heatmap on mount
  useEffect(() => {
    fetchHeatmap();
  }, []);

  const fetchHeatmap = async () => {
    try {
      const data = await getHeatmap();
      setHeatmapData(data);
    } catch (err) {
      console.error('Failed to load heatmap data:', err);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadError(null);
    try {
      const data = await uploadCSV(file);
      if (data && data.shipments) {
        setShipments(data.shipments);
        setSelectedShipment(null); // Clear selected
        setSimulationResults([]); // Reset simulation results on new load
      }
    } catch (err) {
      console.error('File upload failed:', err);
      setUploadError(err.response?.data?.detail || 'Failed to parse CSV or run ML predictions.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleSimulationSubmit = async (port, daysClosed) => {
    try {
      const data = await simulateDisruption(port, daysClosed);
      if (data && data.shipments) {
        setSimulationResults(data.shipments);
      }
    } catch (err) {
      console.error('Simulation failed:', err);
      alert('Simulation failed: ' + (err.response?.data?.detail || err.message));
    }
  };

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-100 p-6 font-sans">
      {/* Header */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between border-b border-slate-800 pb-5 mb-8">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-blue-400 via-indigo-400 to-cyan-400 bg-clip-text text-transparent">
              SupplyChainMind
            </h1>
            <span className="bg-indigo-950 text-indigo-400 text-[10px] font-semibold px-2 py-0.5 rounded-full border border-indigo-800 flex items-center gap-1">
              <Sparkles size={10} /> ML v2.0
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            End-to-End AI Disruption Predictor, Quantile Uncertainty Estimates & GIS Risk Hotspots
          </p>
        </div>
        <div className="flex items-center gap-2 mt-4 md:mt-0">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400 flex items-center gap-1">
            <Activity size={12} /> Live Pipeline Active
          </span>
        </div>
      </header>

      {/* Main Grid */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column (8 spans on lg) */}
        <div className="lg:col-span-8 space-y-8">
          
          {/* CSV Upload Section */}
          <div className="bg-[#1e293b]/80 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden group hover:border-blue-500/50 transition-all duration-300">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-600/5 blur-3xl pointer-events-none rounded-full"></div>
            <h3 className="text-lg font-bold text-slate-200 mb-2 flex items-center gap-2">
              <UploadCloud className="text-blue-400" size={20} /> Load Shipment Ledger
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Upload active shipments in CSV format to trigger XGBoost quantile delay regressions.
            </p>
            
            <div className="flex flex-col items-center justify-center border-2 border-dashed border-slate-800 rounded-lg p-6 bg-slate-950/40 hover:bg-slate-950/80 transition-all cursor-pointer relative">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="absolute inset-0 opacity-0 cursor-pointer"
                disabled={isUploading}
              />
              <UploadCloud className={`text-slate-500 group-hover:text-blue-400 mb-3 transition-colors ${isUploading ? 'animate-bounce' : ''}`} size={40} />
              {isUploading ? (
                <p className="text-sm font-semibold text-blue-400">Processing ML Quantile Regressions...</p>
              ) : (
                <p className="text-sm font-semibold text-slate-300">Click or Drag CSV here to upload</p>
              )}
              <p className="text-xs text-slate-500 mt-1">Requires columns: shipment_id, origin, destination, carrier, departure_date, product_category</p>
            </div>
            
            {uploadError && (
              <div className="mt-4 p-3 bg-red-950/40 border border-red-900/50 rounded-lg text-xs text-red-400 flex items-center gap-2">
                <ShieldAlert size={14} /> {uploadError}
              </div>
            )}
          </div>

          {/* Shipment Table */}
          <ShipmentTable shipments={shipments} onSelect={setSelectedShipment} />

          {/* Simulation Panel & Results */}
          <SimulationPanel 
            onSubmit={handleSimulationSubmit} 
            simulationResults={simulationResults}
            onSelectShipment={setSelectedShipment}
          />

        </div>

        {/* Right Column (4 spans on lg) */}
        <div className="lg:col-span-4 space-y-8">
          
          {/* Geospatial Map */}
          <RiskMap geoJSON={heatmapData} />

          {/* Detailed Narrative Explanation */}
          <ExplanationCard shipment={selectedShipment} />

        </div>

      </main>
    </div>
  );
}

export default Dashboard;
