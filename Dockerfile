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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker caching
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY README.md .

# Expose port 8000 (standard for local and cloud mapping)
EXPOSE 8000

# Run the FastAPI server on container startup
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
