# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy the entire application
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Build frontend CSS
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Switch back to app directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p backend/uploads backend/static/css

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV LIME_SAMPLES=400

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]