import { useState } from 'react'
import axios from 'axios'

const DEFAULT_SHIPMENTS = [
  {
    "ShipmentID": "SHP00001",
    "Origin": "Shanghai",
    "Destination": "Los Angeles",
    "Carrier": "Maersk",
    "ProductCategory": "Electronics",
    "DepartureDate": "2024-06-01",
    "ExpectedDelivery": "2024-06-15",
    "Weight_kg": 5200.0,
    "WeatherRisk": 0.12,
    "PortCongestion": 0.35,
    "GeopoliticalSentiment": 0.08
  },
  {
    "ShipmentID": "SHP00002",
    "Origin": "Shenzhen",
    "Destination": "Rotterdam",
    "Carrier": "MSC",
    "ProductCategory": "Automotive",
    "DepartureDate": "2024-06-05",
    "ExpectedDelivery": "2024-06-22",
    "Weight_kg": 12500.0,
    "WeatherRisk": 0.28,
    "PortCongestion": 0.42,
    "GeopoliticalSentiment": 0.15
  },
  {
    "ShipmentID": "SHP00003",
    "Origin": "Mumbai",
    "Destination": "Hamburg",
    "Carrier": "COSCO",
    "ProductCategory": "Pharmaceuticals",
    "DepartureDate": "2024-06-10",
    "ExpectedDelivery": "2024-06-25",
    "Weight_kg": 3400.0,
    "WeatherRisk": 0.05,
    "PortCongestion": 0.18,
    "GeopoliticalSentiment": 0.35
  }
];

function SimulationPanel({ onSimulate }) {
  const [port, setPort] = useState('Shanghai')
  const [days, setDays] = useState(3)
  const [shipments, setShipments] = useState(JSON.stringify(DEFAULT_SHIPMENTS, null, 2))

  const handleSimulate = async () => {
    try {
      const parsedShipments = JSON.parse(shipments)
      const res = await axios.post('/api/simulate', {
        affected_port: port,
        delay_days: days,
        shipments: parsedShipments
      })
      onSimulate(res.data.predictions)
    } catch (e) {
      alert('Simulation Input Error: ' + e.message)
    }
  }

  const handlePredictRaw = async () => {
    try {
      const parsedShipments = JSON.parse(shipments)
      const res = await axios.post('/api/predict', parsedShipments)
      onSimulate(res.data.predictions)
    } catch (e) {
      alert('Prediction Error: ' + e.message)
    }
  }

  return (
    <div className="bg-gray-800/60 backdrop-blur-md border border-gray-700/50 p-6 rounded-xl shadow-2xl transition-all duration-300 hover:shadow-cyan-900/10">
      <h3 className="text-xl font-semibold bg-gradient-to-r from-emerald-400 to-teal-500 bg-clip-text text-transparent mb-4">
        What‑If Scenario Simulation
      </h3>
      
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">Affected Port</label>
          <input 
            className="bg-gray-900/80 border border-gray-700 text-gray-200 text-sm p-2.5 rounded-lg w-full focus:outline-none focus:ring-1 focus:ring-emerald-500" 
            placeholder="e.g. Shanghai" 
            value={port} 
            onChange={e => setPort(e.target.value)} 
          />
        </div>
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">Disruption Severity (1-10)</label>
          <input 
            className="bg-gray-900/80 border border-gray-700 text-gray-200 text-sm p-2.5 rounded-lg w-full focus:outline-none focus:ring-1 focus:ring-emerald-500" 
            type="number" 
            min="1"
            max="10"
            value={days} 
            onChange={e => setDays(+e.target.value)} 
          />
        </div>
      </div>

      <div className="mb-4">
        <label className="block text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">Shipments Data payload (JSON)</label>
        <textarea 
          className="bg-gray-900/80 border border-gray-700 text-cyan-400 font-mono text-xs p-3 rounded-lg w-full focus:outline-none focus:ring-1 focus:ring-emerald-500" 
          rows={7} 
          value={shipments} 
          onChange={e => setShipments(e.target.value)} 
        />
      </div>

      <div className="flex gap-4">
        <button 
          onClick={handlePredictRaw} 
          className="flex-1 bg-cyan-700 hover:bg-cyan-600 text-white font-medium text-sm py-2.5 rounded-lg shadow-lg hover:shadow-cyan-750/30 transition-all duration-300"
        >
          Predict Baseline
        </button>
        <button 
          onClick={handleSimulate} 
          className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm py-2.5 rounded-lg shadow-lg hover:shadow-emerald-600/30 transition-all duration-300"
        >
          Inject Disruption
        </button>
      </div>
    </div>
  )
}

export default SimulationPanel;
