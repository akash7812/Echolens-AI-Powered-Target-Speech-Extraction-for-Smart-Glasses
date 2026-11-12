# Echolens — Production Dockerfile
# Optimized for Hugging Face Spaces, Render, and Cloud Containers

FROM python:3.12-slim

# Install system dependencies (libsndfile for audio I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests
COPY requirements.txt pyproject.toml setup.py /app/

# Install PyTorch CPU-only wheels first (fast, ~150MB instead of 3GB CUDA)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, documentation, and model checkpoint
COPY src/ /app/src/
COPY data/ /app/data/
COPY docs/ /app/docs/
COPY models/ /app/models/

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Expose standard application port
EXPOSE 8000

# Optimize CPU threads for container instances (prevents thread contention and memory spikes)
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV ECHOLENS_CHECKPOINT_PATH=/app/models/checkpoint_finetuned.pt
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

# Health check endpoint (fast /health check with generous start period for model pre-load)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start Echolens Web Dashboard (binds to Render's dynamic $PORT or 8000)
CMD ["sh", "-c", "python -m echolens server --host 0.0.0.0 --port ${PORT:-8000}"]
