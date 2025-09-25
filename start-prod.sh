#!/bin/bash

# Production startup script for AgriScan

set -e

echo "🌱 Starting AgriScan Deployment..."

# Check if models directory exists and has files
if [ ! -d "backend/models" ] || [ -z "$(ls -A backend/models)" ]; then
    echo "❌ Error: Model files not found in backend/models/"
    echo "Please ensure the following files exist:"
    echo "  - vit_model.weights.h5"
    echo "  - vitTomato_model.weights.h5"
    echo "  - vitMaize_model.weights.h5"
    exit 1
fi

# Build frontend if needed
echo "🔨 Building frontend..."
cd frontend
npm install
npm run build
cd ..

# Create necessary directories
mkdir -p backend/uploads
mkdir -p backend/static/css

# Set production environment
export PYTHONPATH=$(pwd)
export LIME_SAMPLES=400

echo "🚀 Starting AgriScan with Gunicorn..."

# Start with Gunicorn for production
gunicorn backend.app.main:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --worker-connections 1000 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --timeout 120 \
    --keep-alive 2 \
    --log-level info \
    --access-logfile - \
    --error-logfile -