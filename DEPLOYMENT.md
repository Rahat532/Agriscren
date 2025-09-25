# 🚀 AgriScan Deployment Guide

This guide covers multiple deployment options for the AgriScan application, from local development to production cloud deployment.

## 📋 Pre-deployment Checklist

- [ ] Model files are present in `backend/models/`
- [ ] Frontend is built (`npm run build`)
- [ ] All dependencies are installed
- [ ] Environment variables are configured
- [ ] Upload directories exist

## 🐳 Docker Deployment (Recommended)

### Quick Start with Docker

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build -d
   ```

2. **Access the application:**
   - Application: http://localhost
   - Direct access: http://localhost:8000

### Manual Docker Build

```bash
# Build the Docker image
docker build -t agriscan:latest .

# Run the container
docker run -d \
  --name agriscan \
  -p 8000:8000 \
  -v $(pwd)/backend/uploads:/app/backend/uploads \
  -v $(pwd)/backend/models:/app/backend/models \
  agriscan:latest
```

## 🖥️ Local Production Deployment

### Linux/macOS

```bash
# Make script executable
chmod +x start-prod.sh

# Run production server
./start-prod.sh
```

### Windows

```batch
# Run the batch file
start-prod.bat
```

### Manual Production Setup

```bash
# 1. Install production dependencies
pip install -r backend/requirements-prod.txt

# 2. Build frontend
cd frontend && npm run build && cd ..

# 3. Start with Gunicorn
gunicorn backend.app.main:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker
```

## ☁️ Cloud Deployment Options

### 1. Heroku Deployment

Create `Procfile`:
```
web: gunicorn backend.app.main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

Deploy steps:
```bash
# Install Heroku CLI and login
heroku login

# Create app
heroku create your-agriscan-app

# Set buildpacks
heroku buildpacks:add heroku/nodejs
heroku buildpacks:add heroku/python

# Configure environment
heroku config:set PYTHONPATH=/app
heroku config:set LIME_SAMPLES=400

# Deploy
git push heroku main
```

### 2. AWS EC2 Deployment

```bash
# 1. Launch EC2 instance (Ubuntu 20.04+)
# 2. Install Docker
sudo apt update
sudo apt install docker.io docker-compose -y
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# 3. Clone and deploy
git clone https://github.com/Rahat532/Agriscren.git
cd agri-app
docker-compose up -d --build

# 4. Configure security group (ports 80, 443)
```

### 3. Google Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/agriscan

# Deploy to Cloud Run
gcloud run deploy agriscan \
  --image gcr.io/YOUR_PROJECT_ID/agriscan \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300
```

### 4. DigitalOcean App Platform

Create `app.yaml`:
```yaml
name: agriscan
services:
- name: web
  source_dir: /
  github:
    repo: Rahat532/Agriscren
    branch: main
  run_command: gunicorn backend.app.main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  routes:
  - path: /
  envs:
  - key: PYTHONPATH
    value: /app
  - key: LIME_SAMPLES
    value: "400"
```

## 🔧 Production Configuration

### Environment Variables

```bash
export PYTHONPATH=/app
export LIME_SAMPLES=400
export ENV=production
export HOST=0.0.0.0
export PORT=8000
export WORKERS=4
```

### Nginx Configuration (Optional)

For high-traffic deployments, use Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### SSL/HTTPS Setup

```bash
# Using Certbot for Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 📊 Performance Optimization

### Resource Requirements

| Deployment Type | CPU | RAM | Storage |
|----------------|-----|-----|---------|
| Development    | 1 core | 2GB | 5GB |
| Small Production | 2 cores | 4GB | 20GB |
| Medium Production | 4 cores | 8GB | 50GB |
| Large Production | 8 cores | 16GB | 100GB |

### Optimization Tips

1. **Model Loading**: Models are loaded at startup - use container health checks
2. **Memory Management**: Monitor memory usage with TensorFlow models
3. **Concurrent Requests**: Limit concurrent image processing
4. **File Cleanup**: Implement cleanup for old uploaded files
5. **Caching**: Use Redis for prediction caching if needed

## 🔒 Security Considerations

### Production Security Checklist

- [ ] Use HTTPS in production
- [ ] Set up proper CORS policies
- [ ] Implement rate limiting
- [ ] Validate file uploads strictly
- [ ] Use environment variables for secrets
- [ ] Regular security updates
- [ ] Monitor logs and metrics

### File Upload Security

```python
# Add to main.py for enhanced security
from fastapi import HTTPException

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

async def validate_file_size(file: UploadFile):
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large")
```

## 📈 Monitoring and Logging

### Health Check Endpoint

Add to `main.py`:
```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
```

### Logging Configuration

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce LIME_SAMPLES or increase server memory
2. **Model Loading**: Verify model files exist and are accessible
3. **Permission Errors**: Check file permissions for uploads directory
4. **Port Conflicts**: Ensure port 8000 is available

### Debug Mode

```bash
# Run in debug mode
uvicorn backend.app.main:app --reload --log-level debug
```

## 📞 Support

For deployment issues:
- Check logs: `docker-compose logs -f`
- Monitor resources: `docker stats`
- Validate models: Test prediction endpoints

---

Choose the deployment method that best fits your infrastructure and requirements!