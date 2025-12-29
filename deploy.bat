@echo off
REM Windows deployment script for Google Cloud Run

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_ID=%GCP_PROJECT_ID%
if "%PROJECT_ID%"=="" set PROJECT_ID=your-project-id

set REGION=%GCP_REGION%
if "%REGION%"=="" set REGION=us-central1

set SERVICE_NAME=%SERVICE_NAME%
if "%SERVICE_NAME%"=="" set SERVICE_NAME=iris-classifier

set IMAGE_NAME=iris-classifier
set ARTIFACT_REGISTRY_REPO=%ARTIFACT_REGISTRY_REPO%
if "%ARTIFACT_REGISTRY_REPO%"=="" set ARTIFACT_REGISTRY_REPO=ml-models

echo ==========================================
echo Google Cloud Deployment Script (Windows)
echo ==========================================
echo Project ID: %PROJECT_ID%
echo Region: %REGION%
echo Service Name: %SERVICE_NAME%
echo ==========================================

REM Check if gcloud is installed
where gcloud >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: gcloud CLI is not installed
    echo Please install it from: https://cloud.google.com/sdk/docs/install
    exit /b 1
)

REM Set project
echo Setting project...
gcloud config set project %PROJECT_ID%

REM Enable required APIs
echo Enabling required APIs...
gcloud services enable artifactregistry.googleapis.com run.googleapis.com cloudbuild.googleapis.com

REM Create Artifact Registry repository
echo Creating Artifact Registry repository...
gcloud artifacts repositories create %ARTIFACT_REGISTRY_REPO% --repository-format=docker --location=%REGION% --description="ML models repository" 2>nul || echo Repository already exists

REM Configure Docker authentication
echo Configuring Docker authentication...
gcloud auth configure-docker %REGION%-docker.pkg.dev

REM Build Docker image
echo Building Docker image...
docker build -t %IMAGE_NAME%:latest .

REM Tag image for Artifact Registry
set FULL_IMAGE_NAME=%REGION%-docker.pkg.dev/%PROJECT_ID%/%ARTIFACT_REGISTRY_REPO%/%IMAGE_NAME%:latest
echo Tagging image as: %FULL_IMAGE_NAME%
docker tag %IMAGE_NAME%:latest %FULL_IMAGE_NAME%

REM Push image to Artifact Registry
echo Pushing image to Artifact Registry...
docker push %FULL_IMAGE_NAME%

REM Deploy to Cloud Run
echo Deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% --image=%FULL_IMAGE_NAME% --platform=managed --region=%REGION% --allow-unauthenticated --memory=2Gi --cpu=2 --port=7860 --max-instances=10 --timeout=300

REM Get service URL
for /f "delims=" %%i in ('gcloud run services describe %SERVICE_NAME% --platform=managed --region=%REGION% --format="value(status.url)"') do set SERVICE_URL=%%i

echo ==========================================
echo Deployment Successful!
echo ==========================================
echo Service URL: %SERVICE_URL%
echo ==========================================
echo.
echo Next steps:
echo 1. Visit the URL above to access your application
echo 2. Monitor logs: gcloud run services logs read %SERVICE_NAME% --region=%REGION%
echo 3. Update service: Re-run this script with new code
echo.

endlocal
