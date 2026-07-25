# ─────────────────────────────────────────────────────────────────────────────
# SocrAItes Production Dockerfile
# Multi-stage build for optimized image size
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Dependencies
FROM python:3.11-slim AS dependencies

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Model download (cached layer)
FROM dependencies AS model

# Pre-download the embedding model so it's cached in the image
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', cache_dir='/app/models')"

# Stage 3: Production image
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy pre-downloaded model
COPY --from=model /app/models /app/models

# Set HuggingFace cache to use pre-downloaded models
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Copy application code
COPY src/ ./src/
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p /app/data /app/temp_uploads

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
