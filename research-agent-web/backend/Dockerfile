# Multi-stage Docker build for Research Agent Backend
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data storage
RUN mkdir -p data uploads exports

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Development stage
FROM base as development
ENV DEBUG=true
CMD ["python", "run_dev.py"]

# Production stage
FROM base as production
ENV DEBUG=false
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]