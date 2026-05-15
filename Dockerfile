# ============================================================
# Dockerfile.backend
# Industrial Energy Efficiency Copilot — Backend
# ============================================================
FROM python:3.12-slim

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    libfreetype6-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY .env* ./

# Copy data indexes directly into the image for ephemeral deployments (like Railway)
# Note: PDFs are excluded as they are too large for GitHub. Indexes are sufficient for serving.
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app/backend

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
