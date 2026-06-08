# 🫁 PneumoDetect AI — Chest X-Ray Pneumothorax Detection

PneumoDetect AI is a medical-grade binary classification application that detects pneumothorax (collapsed lung) from chest X-ray images. It is built as a reproducible machine learning pipeline (PyTorch Lightning) and an optimized web application served via FastAPI and ONNX Runtime.

## 🚀 Key Features
- **Reproducible Pipeline**: Scripted synthetic DICOM data generator, model training, metric evaluation, and ONNX model export.
- **Explainable Diagnostics**: Native NumPy implementation of Grad-CAM to visualize neural network activation map heatmaps overlaying input X-rays.
- **Production-ready API**: Serves predictions in milliseconds on CPU via ONNX Runtime and FastAPI.
- **Premium User Dashboard**: Beautiful dark-themed, glassmorphic UI supporting drag-and-drop uploads of clinical DICOM images, animated gauge confidence indicators, and side-by-side comparative views.

---

## 🛠️ Installation & Setup

Ensure you are using **Python 3.11**. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## 📈 Running the Machine Learning Pipeline

Follow these steps in order to generate mock data, train the model, evaluate performance, and export it for serving.

### Step 1: Generate Mock Data
Generate a synthetic dataset of 100 chest X-ray images in DICOM format with patient metadata tags and labels biased to mirror real clinical ratios (80% negative class):

```bash
python generate_mock_data.py
```
This writes files to `data/dicoms/` and generates `data/train.csv`.

### Step 2: Train the Model
Train the ResNet-50 model. The script automatically uses PyTorch Lightning with early stopping and checkpoint logging:

```bash
python train.py --epochs 20 --batch_size 32
```
The best checkpoint is written under `models/best.ckpt`.

### Step 3: Evaluate Validation Metrics
Evaluate the trained checkpoint against the validation set to inspect accuracy, AUROC, F1, and the confusion matrix:

```bash
python evaluate.py --checkpoint models/best.ckpt
```

### Step 4: Export to ONNX Runtime
Compile and export the PyTorch model to ONNX. This step wraps the model to output the final fully connected logits and the last bottleneck block (`resnet50.layer4[-1]`) feature maps via forward hooks:

```bash
python export_onnx.py --checkpoint models/best.ckpt
```
Outputs `models/model.onnx` and weights metadata `models/model_weights.npy`.

---

## 🖥️ Running the Application Server

Start the FastAPI serving server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your web browser. Drag-and-drop a DICOM file (e.g. from `data/dicoms/`) or a standard image to view predictions and the Grad-CAM activation heatmap overlay.

---

## 🧪 Running Unit Tests

Run the test suite to verify code correctness, model parity, and API response structures:

```bash
pytest
```

---

## 🐳 Containerization & Cloud Deployment

### Build and Run locally with Docker
```bash
docker build -t pneumodetect-ai .
docker run -p 8000:8000 pneumodetect-ai
```

### Render Deployment
This repository contains a `render.yaml` blueprint. Link this repository to Render, and it will deploy automatically as a Web Service running inside Python 3.11 with the pinned requirements.
