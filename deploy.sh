#!/bin/bash

# Google Cloud Deployment Script for Iris Classification Application
# This script builds and deploys the Docker image to Google Cloud Run

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-iris-classifier}"
IMAGE_NAME="iris-classifier"
ARTIFACT_REGISTRY_REPO="${ARTIFACT_REGISTRY_REPO:-ml-models}"

echo "=========================================="
echo "Google Cloud Deployment Script"
echo "=========================================="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Name: $SERVICE_NAME"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
echo "Checking authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "Error: Not authenticated with gcloud"
    echo "Please run: gcloud auth login"
    exit 1
fi

# Set project
echo "Setting project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create $ARTIFACT_REGISTRY_REPO \
    --repository-format=docker \
    --location=$REGION \
    --description="ML models repository" \
    2>/dev/null || echo "Repository already exists"

# Configure Docker authentication
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:latest .

# Tag image for Artifact Registry
FULL_IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${IMAGE_NAME}:latest"
echo "Tagging image as: $FULL_IMAGE_NAME"
docker tag $IMAGE_NAME:latest $FULL_IMAGE_NAME

# Push image to Artifact Registry
echo "Pushing image to Artifact Registry..."
docker push $FULL_IMAGE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image=$FULL_IMAGE_NAME \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --port=7860 \
    --max-instances=10 \
    --timeout=300

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform=managed \
    --region=$REGION \
    --format="value(status.url)")

echo "=========================================="
echo "Deployment Successful!"
echo "=========================================="
echo "Service URL: $SERVICE_URL"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Visit the URL above to access your application"
echo "2. Monitor logs: gcloud run services logs read $SERVICE_NAME --region=$REGION"
echo "3. Update service: Re-run this script with new code"
echo ""
