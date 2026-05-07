# ── Build Stage ──────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime Stage ────────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="NNStudio"
LABEL description="NNStudio — browser-based neural network trainer"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd --create-home --shell /bin/bash nnstudio

WORKDIR /app

# Copy application code
COPY app/         ./app/
COPY run.py       .
COPY requirements.txt .

# Create instance directory for SQLite (will be volume-mounted)
RUN mkdir -p /app/instance && chown -R nnstudio:nnstudio /app

USER nnstudio

# Flask runs on port 5000
EXPOSE 5000

# Environment defaults (override via docker-compose or Coolify)
ENV FLASK_APP=app \
    FLASK_ENV=production \
    SECRET_KEY=change-me-in-production \
    DATABASE_URL=sqlite:////app/instance/nnstudio.db

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/login')" || exit 1

# Production WSGI server
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:create_app()"]
