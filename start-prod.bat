@echo off
echo 🌱 Starting AgriScan Deployment...

REM Check if models directory exists
if not exist "backend\models" (
    echo ❌ Error: Model files not found in backend\models\
    echo Please ensure the following files exist:
    echo   - vit_model.weights.h5
    echo   - vitTomato_model.weights.h5  
    echo   - vitMaize_model.weights.h5
    pause
    exit /b 1
)

REM Build frontend
echo 🔨 Building frontend...
cd frontend
call npm install
call npm run build
cd ..

REM Create necessary directories
if not exist "backend\uploads" mkdir backend\uploads
if not exist "backend\static\css" mkdir backend\static\css

REM Set environment variables
set PYTHONPATH=%CD%
set LIME_SAMPLES=400

echo 🚀 Starting AgriScan with Uvicorn...

REM Activate virtual environment and start server
call aenv\Scripts\activate
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1