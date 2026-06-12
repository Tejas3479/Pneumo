import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import { useState, useEffect } from 'react'
import axios from 'axios'

function RiskMap() {
  const [features, setFeatures] = useState([])
  
  useEffect(() => {
    axios.get('/api/heatmap')
      .then(res => setFeatures(res.data.features))
      .catch(err => console.error("Error loading port congestion data:", err))
  }, [])

  return (
    <div className="bg-gray-800/60 backdrop-blur-md border border-gray-700/50 rounded-xl p-5 shadow-2xl overflow-hidden hover:shadow-cyan-900/10 transition-all duration-300">
      <h3 className="text-xl font-semibold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent mb-4">
        Geospatial Disruption Heatmap
      </h3>
      <div className="h-96 rounded-lg overflow-hidden border border-gray-700/30">
        <MapContainer center={[20, 30]} zoom={2} style={{ height: '100%', width: '100%', background: '#111827' }}>
          {/* Custom dark-mode looking tile layer */}
          <TileLayer 
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" 
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
          />
          {features.map((f, i) => (
            <CircleMarker
              key={i}
              center={[f.geometry.coordinates[1], f.geometry.coordinates[0]]}
              radius={10}
              color={
                f.properties.risk_score > 0.7 
                  ? '#f43f5e' 
                  : f.properties.risk_score > 0.4 
                    ? '#f59e0b' 
                    : '#10b981'
              }
              fillColor={
                f.properties.risk_score > 0.7 
                  ? '#f43f5e' 
                  : f.properties.risk_score > 0.4 
                    ? '#f59e0b' 
                    : '#10b981'
              }
              fillOpacity={0.65}
              weight={1.5}
            >
              <Popup>
                <div className="text-gray-900 text-xs font-sans">
                  <strong className="text-sm block border-b pb-1 mb-1">{f.properties.name}</strong>
                  <span className="text-gray-600 block">Country: {f.properties.country}</span>
                  <span className="font-semibold text-rose-600 block mt-1">Risk Score: {(f.properties.risk_score * 100).toFixed(0)}%</span>
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  )
}

export default RiskMap;
