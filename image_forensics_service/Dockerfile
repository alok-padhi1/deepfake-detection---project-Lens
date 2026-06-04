# ── BUILD STAGE ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies to local site-packages
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── RUNTIME STAGE ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install runtime system libraries (e.g. for SciPy/OpenCV if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed libraries from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application source code
COPY config.py .
COPY app.py .

# Create non-root system user for secure cluster execution
RUN useradd -u 8888 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Start application server
ENTRYPOINT ["python", "app.py"]
