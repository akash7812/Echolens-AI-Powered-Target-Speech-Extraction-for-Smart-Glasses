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

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and documentation
COPY src/ /app/src/
COPY data/ /app/data/
COPY docs/ /app/docs/

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Expose standard application port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/ || exit 1

# Start Echolens Web Dashboard (binds to Render's dynamic $PORT or 8000)
CMD ["sh", "-c", "python -m echolens server --host 0.0.0.0 --port ${PORT:-8000}"]
