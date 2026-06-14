# Use official stable Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies needed for OpenCV-headless and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire application code into the container
# All modules are needed: src/, app/, mlops/, models/, scripts
COPY src/ ./src/
COPY app/ ./app/
COPY mlops/ ./mlops/
COPY train.py .
COPY export_onnx.py .
COPY train_tcav.py .
COPY generate_mock_data.py .
COPY download_pretrained.py .
COPY evaluate.py .
COPY README.md .

# Copy models directory if it exists (weights, ONNX files)
# In production, mount via volume instead
COPY models/ ./models/

# Expose port 8000 (standard for local and cloud mapping)
EXPOSE 8000

# Health check — FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the FastAPI server on container startup
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
