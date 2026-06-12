import { useState } from 'react'
import ShipmentTable from './components/ShipmentTable'
import RiskMap from './components/RiskMap'
import SimulationPanel from './components/SimulationPanel'
import ExplanationCard from './components/ExplanationCard'

function App() {
  const [shipments, setShipments] = useState([])
  const [selectedShipment, setSelectedShipment] = useState(null)

  const handleSimulateUpdate = (predictions) => {
    setShipments(predictions)
    setSelectedShipment(null)
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6 antialiased font-sans">
      <header className="max-w-7xl mx-auto flex items-center justify-between border-b border-gray-800 pb-5 mb-8">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-500 bg-clip-text text-transparent">
            SupplyChainMind
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            AI-Powered Supply Chain Disruption Predictor & What-If Simulator
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-xs font-semibold uppercase tracking-wider text-emerald-400">System Online</span>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left column: simulation and table */}
        <section className="lg:col-span-7 space-y-6">
          <SimulationPanel onSimulate={handleSimulateUpdate} />
          <ShipmentTable shipments={shipments} onSelect={setSelectedShipment} />
        </section>

        {/* Right column: map and explanation */}
        <section className="lg:col-span-5 space-y-6">
          <RiskMap />
          {selectedShipment ? (
            <ExplanationCard shipment={selectedShipment} />
          ) : (
            <div className="bg-gray-800/20 border border-gray-800 border-dashed rounded-xl p-8 text-center text-gray-500 text-sm">
              Select a shipment from the table to view the ML explanation breakdown.
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
