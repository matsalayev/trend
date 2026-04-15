# Trend Robot - Production Dockerfile
# Multi-stage build for smaller image size

# ═══════════════════════════════════════════════════════════════════════════════
#                               BUILD STAGE
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ═══════════════════════════════════════════════════════════════════════════════
#                               PRODUCTION STAGE
# ═══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS production

# Labels
LABEL maintainer="HEMA Team"
LABEL description="Trend Trading Robot"
LABEL version="1.0.0"

# curl for Coolify healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Security: Run as non-root user
RUN groupadd -r trendbot && useradd -r -g trendbot trendbot

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=trendbot:trendbot trend_robot/ ./trend_robot/
COPY --chown=trendbot:trendbot run_server.py .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVER_PORT=8090 \
    BOT_ID=trend-v1 \
    BOT_NAME="Trend Robot" \
    BOT_VERSION=1.0.0

# Persistent state storage
RUN mkdir -p /data/state && chown trendbot:trendbot /data/state
VOLUME /data/state

# Switch to non-root user
USER trendbot

# Expose port
EXPOSE 8090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')" || exit 1

# Run the server
CMD ["python", "run_server.py"]
