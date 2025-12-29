# Iris Classification ML Application - Task Breakdown

## Planning Phase
- [x] Create implementation plan
- [x] Get user approval on architecture and approach

## Project Setup
- [x] Update `pyproject.toml` with all required dependencies
- [x] Create modular project structure (folders and files)
- [x] Create configuration file for hyperparameters and paths

## Core ML Pipeline Components
- [x] Implement `data_ingestion.py` - Load Iris dataset
- [x] Implement `data_preprocessing.py` - Handle missing values, data validation
- [x] Implement `eda.py` - Exploratory data analysis with visualizations
- [x] Implement `data_transformation.py` - Train-test split, scaling
- [x] Implement `feature_engineering.py` - Feature selection/engineering if needed
- [x] Implement `model_building.py` - XGBoost with Optuna + MLFlow integration
- [x] Implement `model_evaluation.py` - Metrics, confusion matrix, classification report

## Application Layer
- [x] Create Gradio UI (`app.py`) for user input and predictions
- [x] Create main training pipeline script
- [x] Create prediction/inference module

## MLFlow Setup
- [x] Configure MLFlow tracking
- [x] Log hyperparameters, metrics, and model artifacts
- [x] Test MLFlow UI locally

## Dockerization
- [x] Create comprehensive `Dockerfile`
- [x] Create `.dockerignore` file
- [x] Build and test Docker image locally

## Google Cloud Deployment
- [x] Create deployment scripts for Artifact Registry
- [x] Create Cloud Run deployment configuration
- [x] Document deployment steps

## Documentation & Testing
- [x] Update README with setup and usage instructions
- [x] Create requirements/dependencies documentation
- [x] Test end-to-end workflow locally
