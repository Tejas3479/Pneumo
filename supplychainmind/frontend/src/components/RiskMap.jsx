import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Compass } from 'lucide-react';

function RiskMap({ geoJSON }) {
  const features = geoJSON?.features || [];

  return (
    <div className="bg-[#1e293b]/80 border border-slate-800 rounded-xl p-5 shadow-xl relative overflow-hidden group">
      <h3 className="text-lg font-bold text-slate-200 mb-2 flex items-center gap-2">
        <Compass className="text-indigo-400 animate-spin-slow" size={20} /> Geospatial Risk Heatmap
      </h3>
      <p className="text-xs text-slate-400 mb-4">
        Interactive GIS overlays showing composite risk index (congestion, weather, geopolitical) for global shipping ports.
      </p>
      
      <div className="h-[380px] rounded-lg overflow-hidden border border-slate-800 relative">
        <MapContainer 
          center={[25, 12]} 
          zoom={2} 
          minZoom={1.5}
          maxZoom={8}
          style={{ height: '100%', width: '100%', background: '#090d16' }}
        >
          {/* CARTO Dark All Basemap */}
          <TileLayer 
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" 
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
          />
          
          {features.map((f, idx) => {
            const lon = f.geometry.coordinates[0];
            const lat = f.geometry.coordinates[1];
            const name = f.properties.name;
            const country = f.properties.country;
            const riskScore = f.properties.risk_score;
            const explanation = f.properties.explanation;
            
            // Marker color based on risk score
            let color = '#10b981'; // Green
            if (riskScore > 0.4) {
              color = '#f59e0b'; // Yellow/Orange
            }
            if (riskScore > 0.65) {
              color = '#ef4444'; // Red
            }
            
            return (
              <CircleMarker
                key={idx}
                center={[lat, lon]}
                radius={8 + riskScore * 10}
                color={color}
                fillColor={color}
                fillOpacity={0.6}
                weight={1.5}
              >
                <Popup className="custom-popup">
                  <div className="text-slate-800 text-xs font-sans p-1">
                    <strong className="text-sm block border-b pb-1 mb-1 text-slate-900">{name} ({country})</strong>
                    <span className="font-semibold block text-slate-700 mt-1">
                      Risk Score: <span className="text-indigo-600">{(riskScore * 100).toFixed(1)}%</span>
                    </span>
                    <p className="text-slate-600 mt-1 leading-relaxed text-[11px]">{explanation}</p>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}

export default RiskMap;
