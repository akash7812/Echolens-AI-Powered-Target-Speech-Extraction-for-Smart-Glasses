# Echolens — Production Ultra-Lightweight Dockerfile
# Optimized for Render Free Tier (Build storage < 400 MB)

FROM python:3.12-slim

# Install only the absolute minimal system library (libsndfile1) and curl for health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy dependency manifests
COPY requirements.txt pyproject.toml setup.py /app/

# Install lightweight PyTorch CPU wheels + minimal requirements, then clean caches
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip /tmp/*

# Copy application source, data, and models
COPY src/ /app/src/
COPY data/ /app/data/
COPY docs/ /app/docs/
COPY models/ /app/models/

# Install package in editable mode
RUN pip install --no-cache-dir -e . \
    && find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

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

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start Echolens Web Dashboard (binds to Render's dynamic $PORT or 8000)
CMD ["sh", "-c", "python -m echolens server --host 0.0.0.0 --port ${PORT:-8000}"]
