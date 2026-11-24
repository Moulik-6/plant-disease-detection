# Production Dockerfile for AgriScan Plant Disease Detection API
# CPU inference image - no checkpoint weights baked into the image.

FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_DEVICE=cpu \
    APP_DB=/workspace/data/app.db

WORKDIR /workspace

# Install system libraries needed by OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheels followed by app requirements
COPY requirements-app.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install --no-cache-dir -r requirements-app.txt

# Create non-root application user and directories
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    mkdir -p /workspace/data /workspace/app/uploads && \
    chown -R appuser:appuser /workspace

# Copy code modules (excluding large weights and datasets)
COPY --chown=appuser:appuser ml/src /workspace/ml/src
COPY --chown=appuser:appuser ml/analysis /workspace/ml/analysis
COPY --chown=appuser:appuser ml/configs /workspace/ml/configs
COPY --chown=appuser:appuser app /workspace/app

USER appuser

EXPOSE 8000

# Mount checkpoints via volume at runtime: -v $(pwd)/ml/checkpoints:/workspace/ml/checkpoints:ro
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
