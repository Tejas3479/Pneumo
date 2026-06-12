# 📦 SupplyChainMind — AI-Powered Supply Chain Disruption Predictor

SupplyChainMind is an end-to-end supply chain disruption predictor that ingests shipment details (origin, destination, carrier, weights, dates) enriched with simulated external risk parameters (weather risk, port congestion, geopolitical sentiment) to forecast shipment delay days, map risk hotspots on an interactive dashboard, and generate plain-English risk explanations.

## 🚀 Features

- **XGBoost ML Pipeline**: Clean, reproducible dataset generator, Target Encoding + Standard Scaling preprocessing pipeline, and model validation CLI.
- **FastAPI backend**: Exposes endpoints for delay predictions, what-if simulations, explainability, and spatial GeoJSON heatmap rendering.
- **Interactive React frontend**: Glassmorphic dark theme dashboard containing shipment lists, interactive maps, scenario playgrounds, and text explanations.
- **100% Offline Inference**: Operates without external cloud dependency.

---

## 🛠️ Installation & Setup

1. **Prepare Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train Model**:
   Generate mock data and train the baseline XGBoost predictor:
   ```bash
   $env:PYTHONPATH="."  # Set python path
   python train.py --samples 1000
   ```

3. **Evaluate Model**:
   ```bash
   python evaluate.py
   ```

4. **Verify Tests**:
   ```bash
   python -m pytest tests/
   ```

---

## 💻 Running the Application

1. **Build Frontend Static Assets**:
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

2. **Start Serving Backend**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   Open `http://localhost:8000` to interact with the dashboard.
