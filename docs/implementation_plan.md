# Iris Flower Classification ML Application

Build a production-ready ML application that predicts iris flower species using XGBoost classifier with Optuna hyperparameter tuning, MLFlow experiment tracking, Gradio UI, and Google Cloud deployment.

## User Review Required

> [!IMPORTANT]
> **Modular Architecture**: The application will follow a strict modular design with separate Python files for each pipeline stage. This ensures maintainability and follows ML engineering best practices.

> [!IMPORTANT]
> **DVC Integration**: Data Version Control (DVC) will be used for:
> - Pipeline versioning and reproducibility
> - Data versioning and tracking
> - Experiment tracking alongside MLFlow
> - Dependency management between pipeline stages

> [!IMPORTANT]
> **Logging Strategy**: Comprehensive logging will be implemented using Python's `logging` module:
> - Structured logging with different levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
> - Separate log files for training and inference
> - Rotating file handlers to manage log file sizes
> - Console and file output for better debugging
> - Error tracking with stack traces

> [!IMPORTANT]
> **MLFlow Tracking**: All Optuna hyperparameter experiments will be logged to MLFlow running on localhost. You'll need to start the MLFlow UI server (`mlflow ui`) to view experiments.

> [!IMPORTANT]
> **Google Cloud Deployment**: The final deployment requires:
> - Google Cloud Project with billing enabled
> - Artifact Registry API enabled
> - Cloud Run API enabled
> - Proper authentication (`gcloud auth configure-docker`)

---

## Proposed Changes

### Project Structure

The application will be organized as follows:

```
ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py          # Load Iris dataset
│   ├── data_preprocessing.py      # Data cleaning and validation
│   ├── eda.py                     # Exploratory data analysis
│   ├── data_transformation.py     # Train-test split, scaling
│   ├── feature_engineering.py     # Feature selection/engineering
│   ├── model_building.py          # XGBoost + Optuna + MLFlow
│   ├── model_evaluation.py        # Metrics and evaluation
│   ├── prediction.py              # Inference module
│   └── logger.py                  # Logging configuration
├── config/
│   ├── config.yaml                # Configuration parameters
│   └── logging_config.yaml        # Logging configuration
├── models/                        # Saved model artifacts
├── data/                          # Data storage
│   └── .gitkeep
├── logs/                          # Application logs
│   └── .gitkeep
├── notebooks/                     # EDA notebooks (optional)
├── dvc.yaml                       # DVC pipeline definition
├── dvc.lock                       # DVC pipeline lock file
├── .dvc/                          # DVC internal directory
├── .dvcignore                     # DVC ignore patterns
├── app.py                         # Gradio UI application
├── train.py                       # Main training pipeline
├── Dockerfile                     # Docker configuration
├── .dockerignore                  # Docker ignore patterns
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── README.md                      # Documentation
```

---

### Core Components

#### **[NEW]** [config.yaml](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/config/config.yaml)

Central configuration file containing:
- Data paths and parameters
- Model hyperparameter search spaces
- MLFlow tracking URI and experiment name
- Training configuration (test size, random state, etc.)
- Optuna study parameters (n_trials, direction)

#### **[NEW]** [logging_config.yaml](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/config/logging_config.yaml)

Logging configuration file:
- Log levels for different modules
- File and console handlers
- Log format specifications
- Rotating file handler settings
- Separate configurations for training and inference

#### **[NEW]** [logger.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/src/logger.py)

Centralized logging utility module.

**Key Functions:**
- `setup_logger()`: Initializes logger with configuration
- `get_logger()`: Returns configured logger instance
- Automatic log rotation and archival
- Structured logging with timestamps and module names

#### **[NEW]** [dvc.yaml](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/dvc.yaml)

DVC pipeline definition with stages:
- `data_ingestion`: Load Iris dataset
- `data_preprocessing`: Clean and validate data
- `data_transformation`: Split and scale features
- `feature_engineering`: Create additional features
- `model_training`: Train with Optuna + MLFlow
- `model_evaluation`: Generate metrics and plots

Each stage defines:
- Command to execute
- Dependencies (input files/data)
- Outputs (data files, models, metrics)
- Parameters from config.yaml

#### **[NEW]** [.dvcignore](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/.dvcignore)

DVC ignore patterns to exclude:
- Virtual environments
- MLFlow runs
- Logs and temporary files
- IDE configurations

#### **[NEW]** [data_ingestion.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/src/data_ingestion.py)

Loads the Iris dataset from sklearn and returns as pandas DataFrame with proper column names and target labels.

**Key Functions:**
- `load_iris_data()`: Fetches Iris dataset and converts to DataFrame
- `save_raw_data()`: Optionally saves raw data to CSV

#### **[NEW]** [data_preprocessing.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/src/data_preprocessing.py)

Handles data quality checks and preprocessing.

**Key Functions:**
- `check_missing_values()`: Validates data completeness
- `check_duplicates()`: Removes duplicate records
- `validate_data_types()`: Ensures correct data types
- `preprocess_data()`: Main preprocessing pipeline

#### **[NEW]** [eda.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/src/eda.py)

Generates exploratory data analysis visualizations.

**Key Functions:**
- `plot_feature_distributions()`: Histograms and KDE plots
- `plot_correlation_matrix()`: Feature correlation heatmap
- `plot_pairplot()`: Seaborn pairplot by species
- `plot_class_distribution()`: Target class balance
- `generate_eda_report()`: Comprehensive EDA report

#### **[NEW]** [data_transformation.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/src/data_transformation.py)

Prepares data for model training.

**Key Functions:**
- `split_data()`: Train-test split with stratification
- `scale_features()`: StandardScaler for feature normalization
- `transform_data()`: Complete transformation pipeline

#### **[NEW]** [feature_engineering.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/src/feature_engineering.py)

Creates additional features if needed (for Iris, may include polynomial features or feature interactions).

**Key Functions:**
- `create_feature_interactions()`: Polynomial/interaction features
- `select_features()`: Feature selection based on importance
- `engineer_features()`: Main feature engineering pipeline

#### **[NEW]** [model_building.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/src/model_building.py)

Core ML training with Optuna hyperparameter optimization and MLFlow tracking.

**Key Functions:**
- `objective()`: Optuna objective function for XGBoost
- `optimize_hyperparameters()`: Runs Optuna study with MLFlow logging
- `train_best_model()`: Trains final model with best parameters
- `save_model()`: Persists trained model and scaler

**MLFlow Integration:**
- Logs all trial parameters and metrics
- Tracks best model artifacts
- Records feature importance
- Saves model with signature

**Optuna Integration:**
- Hyperparameter search space: learning_rate, max_depth, n_estimators, subsample, colsample_bytree, gamma, min_child_weight
- Uses TPE sampler for efficient search
- Maximizes accuracy (or other specified metric)

#### **[NEW]** [model_evaluation.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/src/model_evaluation.py)

Comprehensive model evaluation and metrics.

**Key Functions:**
- `calculate_metrics()`: Accuracy, precision, recall, F1-score
- `plot_confusion_matrix()`: Confusion matrix visualization
- `generate_classification_report()`: Detailed classification report
- `plot_feature_importance()`: XGBoost feature importance
- `evaluate_model()`: Complete evaluation pipeline

#### **[NEW]** [prediction.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/src/prediction.py)

Inference module for making predictions.

**Key Functions:**
- `load_trained_model()`: Loads saved model and scaler
- `predict_single()`: Predicts single instance
- `predict_batch()`: Batch predictions
- `get_prediction_probabilities()`: Returns class probabilities

---

### Application Layer

#### **[MODIFY]** [app.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/app.py)

Gradio web interface for user interaction.

**Features:**
- Input fields for 4 Iris features (sepal length, sepal width, petal length, petal width)
- Real-time prediction with confidence scores
- Display predicted species with probabilities
- Clean, user-friendly interface
- Error handling for invalid inputs

**UI Components:**
- Number inputs with appropriate ranges
- Prediction button
- Output display with species name and confidence
- Example inputs for quick testing

#### **[NEW]** [train.py](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/train.py)

Main training pipeline orchestrator.

**Workflow:**
1. Load configuration
2. Data ingestion
3. Data preprocessing
4. EDA generation
5. Data transformation
6. Feature engineering
7. Model building with Optuna + MLFlow
8. Model evaluation
9. Save final artifacts

---

### Configuration & Dependencies

#### **[MODIFY]** [pyproject.toml](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/pyproject.toml)

Update dependencies to include:
- `scikit-learn`: Iris dataset and preprocessing
- `xgboost`: Classifier model
- `optuna`: Hyperparameter optimization
- `mlflow`: Experiment tracking
- `gradio`: Web UI
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualizations
- `pyyaml`: Configuration management
- `joblib`: Model serialization
- `dvc`: Data and pipeline versioning
- `dvc[gs]`: Optional Google Cloud Storage support

#### **[NEW]** [requirements.txt](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/requirements.txt)

Generate from `pyproject.toml` for Docker compatibility.

---

### Dockerization

#### **[MODIFY]** [Dockerfile](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/Dockerfile)

Multi-stage Docker build for optimized image size.

**Configuration:**
- Base image: `python:3.12-slim`
- Working directory: `/app`
- Copy requirements and install dependencies
- Copy application code
- Expose port 7860 (Gradio default)
- Health check endpoint
- Run Gradio app with `app.py`

#### **[NEW]** [.dockerignore](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/.dockerignore)

Exclude unnecessary files from Docker context:
- `.venv`, `__pycache__`, `.git`
- `mlruns`, `notebooks`
- Development files

---

### Google Cloud Deployment

#### **[NEW]** [deploy.sh](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/deploy.sh)

Automated deployment script for Google Cloud.

**Steps:**
1. Build Docker image
2. Tag image for Artifact Registry
3. Push to Artifact Registry
4. Deploy to Cloud Run
5. Output public URL

**Environment Variables Required:**
- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `GCP_REGION`: Deployment region (e.g., `us-central1`)
- `SERVICE_NAME`: Cloud Run service name

#### **[NEW]** [cloudbuild.yaml](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/cloudbuild.yaml)

Optional CI/CD configuration for Google Cloud Build.

---

### Documentation

#### **[MODIFY]** [README.md](file:///d:/projects/ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/README.md)

Comprehensive documentation including:
- Project overview and architecture
- Local setup instructions
- Training pipeline usage
- MLFlow UI access
- Gradio app usage
- Docker build and run commands
- Google Cloud deployment guide
- API documentation
- Troubleshooting guide

---

## Verification Plan

### Automated Tests

1. **DVC Pipeline Execution**
   ```bash
   dvc repro
   ```
   - Verify all pipeline stages execute successfully
   - Check DVC tracks data and model artifacts
   - Confirm pipeline dependencies are correct
   - Validate dvc.lock file is updated

2. **Logging Verification**
   - Check log files are created in `logs/` directory
   - Verify different log levels are working
   - Confirm log rotation is functioning
   - Review error logs for proper stack traces
   - Validate both console and file logging

3. **Local Training Pipeline**
   ```bash
   python train.py
   ```
   - Verify data ingestion completes
   - Check EDA plots are generated
   - Confirm Optuna trials run successfully
   - Validate MLFlow logs experiments
   - Ensure model artifacts are saved
   - Review logs for any errors or warnings

4. **MLFlow UI Verification**
   ```bash
   mlflow ui
   ```
   - Access http://localhost:5000
   - Verify experiment tracking
   - Check logged parameters and metrics
   - Confirm model artifacts are stored

5. **Gradio App Testing**
   ```bash
   python app.py
   ```
   - Access http://localhost:7860
   - Test predictions with sample inputs
   - Verify all three species predictions work
   - Check confidence scores are displayed

6. **Docker Build & Run**
   ```bash
   docker build -t iris-classifier .
   docker run -p 7860:7860 iris-classifier
   ```
   - Verify image builds successfully
   - Test containerized app functionality
   - Confirm predictions work in container

### Manual Verification

1. **Google Cloud Deployment**
   - Push image to Artifact Registry
   - Deploy to Cloud Run
   - Access public URL
   - Test predictions from deployed app
   - Verify scalability and performance

2. **End-to-End Workflow**
   - Train model locally
   - Review MLFlow experiments
   - Test Gradio UI locally
   - Build Docker image
   - Deploy to Cloud Run
   - Test production endpoint
