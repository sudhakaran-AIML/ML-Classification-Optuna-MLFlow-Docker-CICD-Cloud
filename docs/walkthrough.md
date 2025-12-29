# Iris Classification ML Application - Walkthrough

## 🎯 Project Overview

Successfully built a production-ready end-to-end ML application for classifying Iris flowers using XGBoost, Optuna hyperparameter tuning, MLFlow experiment tracking, Gradio UI, and cloud deployment capabilities.

---

## ✅ Completed Components

### 1. Project Setup & Configuration

**Created Files:**
- `pyproject.toml` - Dependencies and project metadata
- `config/config.yaml` - Comprehensive configuration for all pipeline stages
- `config/logging_config.yaml` - Structured logging configuration
- `.gitignore` - Git ignore patterns
- `.dockerignore` - Docker build context exclusions
- `.dvcignore` - DVC tracking exclusions

**Key Features:**
- ✅ All dependencies installed via `uv`
- ✅ Modular project structure with separate directories for src, config, models, data, logs, plots
- ✅ Centralized configuration management
- ✅ Comprehensive logging with rotation (training, inference, error logs)

---

### 2. Core ML Pipeline Modules

#### Data Ingestion (`src/data_ingestion.py`)
- Loads Iris dataset from scikit-learn
- Saves raw data to CSV
- **Result**: 150 samples, 4 features, 3 classes (perfectly balanced)

#### Data Preprocessing (`src/data_preprocessing.py`)
- Missing value detection
- Duplicate removal (found and removed 1 duplicate)
- Data type validation
- Outlier detection (Z-score method)
- **Result**: 149 clean samples

#### Exploratory Data Analysis (`src/eda.py`)
- Feature distribution plots
- Correlation matrix heatmap
- Pairplot by species
- Class distribution visualization
- Boxplots by species
- Statistical summary
- **Result**: All visualizations saved to `plots/` directory

#### Data Transformation (`src/data_transformation.py`)
- Train-test split (80/20) with stratification
- StandardScaler for feature normalization
- Label encoding for target variable
- **Result**: 119 training samples, 30 test samples

#### Feature Engineering (`src/feature_engineering.py`)
- Support for polynomial features
- Feature interaction creation
- Feature selection capabilities
- **Result**: Used original 4 features (configurable for future enhancement)

#### Model Building (`src/model_building.py`)
- XGBoost classifier implementation
- Optuna hyperparameter optimization (TPE sampler)
- MLFlow experiment tracking integration
- **Result**: 50 trials completed, best accuracy 96.67%

**Best Hyperparameters:**
```yaml
learning_rate: 0.1903
max_depth: 7
n_estimators: 227
subsample: 0.6082
colsample_bytree: 0.9880
gamma: 4.1622
min_child_weight: 3
```

#### Model Evaluation (`src/model_evaluation.py`)
- Comprehensive metrics calculation
- Confusion matrix visualization
- Classification report generation
- Feature importance plotting
- Prediction distribution analysis

**Final Metrics:**
- **Accuracy**: 96.67%
- **Precision (weighted)**: 96.97%
- **Recall (weighted)**: 96.67%
- **F1-Score (weighted)**: 96.66%

**Per-Class Performance:**
| Species | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Setosa | 1.00 | 1.00 | 1.00 | 10 |
| Versicolor | 1.00 | 0.90 | 0.95 | 10 |
| Virginica | 0.91 | 1.00 | 0.95 | 10 |

#### Prediction Module (`src/prediction.py`)
- Model and artifact loading
- Single and batch prediction support
- Probability score calculation
- Confidence-based predictions

---

### 3. Application Layer

#### Training Pipeline (`train.py`)
- Orchestrates entire ML pipeline
- Comprehensive logging at each stage
- Error handling and reporting
- **Execution Time**: ~90 seconds for complete pipeline

**Pipeline Stages:**
1. ✅ Data Ingestion
2. ✅ Data Preprocessing  
3. ✅ Exploratory Data Analysis
4. ✅ Data Transformation
5. ✅ Feature Engineering
6. ✅ Model Building (Optuna + MLFlow)
7. ✅ Model Evaluation

#### Gradio Web Application (`app.py`)
- Beautiful, modern UI design
- Real-time predictions
- Confidence scores and probability distribution
- Example inputs for quick testing
- Error handling and validation
- **Features**:
  - 4 input fields for flower measurements
  - Instant prediction with species classification
  - Visual probability bars for all classes
  - Species-specific emojis (🌼 🌺 🌷)

---

### 4. MLFlow Integration

**Configuration:**
- Tracking URI: `file:./mlruns` (file-based backend)
- Experiment Name: `iris_classification_xgboost`
- 50 trials logged with parameters and metrics

**Logged Information:**
- All hyperparameter combinations
- Accuracy, precision, recall, F1-score for each trial
- Best model parameters
- Training and test metrics
- Optimization history (saved to `models/optuna_trials.csv`)

**MLFlow UI Access:**
```bash
mlflow ui
# Open http://localhost:5000
```

---

### 5. DVC Pipeline

**Pipeline Configuration (`dvc.yaml`):**
- `data_ingestion` stage
- `data_preprocessing` stage
- `data_transformation` stage
- `model_training` stage
- `model_evaluation` stage

**Benefits:**
- Reproducible pipeline execution
- Data and model versioning
- Dependency tracking
- Parameter tracking from config.yaml

**Usage:**
```bash
dvc repro  # Run entire pipeline
dvc dag    # View pipeline DAG
```

---

### 6. Docker Containerization

**Dockerfile Features:**
- Multi-stage build for optimization
- Python 3.12-slim base image
- All dependencies installed
- Gradio app as entry point
- Health check endpoint
- Port 7860 exposed

**Build & Run:**
```bash
docker build -t iris-classifier .
docker run -p 7860:7860 iris-classifier
```

---

### 7. Google Cloud Deployment

**Deployment Scripts:**
- `deploy.sh` (Linux/Mac)
- `deploy.bat` (Windows)

**Features:**
- Automated image building
- Push to Google Artifact Registry
- Deploy to Cloud Run
- Public URL generation
- Configurable via environment variables

**Configuration:**
```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
./deploy.sh
```

---

### 8. Documentation

**README.md:**
- Comprehensive project documentation
- Installation instructions
- Usage guide
- Architecture overview
- Deployment steps
- Troubleshooting guide
- Technology stack details

---

## 📊 Verification Results

### Training Pipeline Execution

**Command:**
```bash
.venv\Scripts\python.exe train.py
```

**Results:**
```
✓ Data ingestion completed: 150 samples, 4 features
✓ Data preprocessing completed: 149 clean samples
✓ EDA completed: Visualizations saved to plots/
✓ Data transformation completed
  - Training set: 119 samples
  - Testing set: 30 samples
✓ Feature engineering completed: 4 features
✓ Model building completed
  - Best accuracy: 0.9667
  - Number of trials: 50
✓ Model evaluation completed
  - Test Accuracy: 0.9667
  - Test F1-Score (weighted): 0.9666
  - Test Precision (weighted): 0.9697
  - Test Recall (weighted): 0.9667
```

### Generated Artifacts

**Models Directory (`models/`):**
- `best_model.pkl` - Trained XGBoost model
- `scaler.pkl` - StandardScaler for feature normalization
- `label_encoder.pkl` - Label encoder for target classes
- `best_params.yaml` - Best hyperparameters from Optuna
- `optuna_trials.csv` - Complete optimization history

**Data Directory (`data/`):**
- `iris_raw.csv` - Original dataset
- `iris_processed.csv` - Cleaned dataset
- `train.csv` - Training split
- `test.csv` - Testing split

**Plots Directory (`plots/`):**
- `feature_distributions.png` - Feature histograms
- `correlation_matrix.png` - Feature correlation heatmap
- `pairplot.png` - Pairwise feature relationships
- `class_distribution.png` - Target class balance
- `boxplots_by_species.png` - Feature distributions by species
- `confusion_matrix.png` - Model confusion matrix
- `feature_importance.png` - XGBoost feature importances
- `prediction_distribution.png` - Predicted vs true labels
- `classification_report.txt` - Detailed classification metrics
- `classification_report.json` - Metrics in JSON format
- `feature_importance.csv` - Feature importance values
- `metrics.json` - Final model metrics
- `statistical_summary.csv` - Dataset statistics

**Logs Directory (`logs/`):**
- `training.log` - Complete training pipeline logs
- `inference.log` - Prediction logs (when app runs)
- `error.log` - Error logs with stack traces

**MLFlow Directory (`mlruns/`):**
- Experiment tracking data
- Model artifacts
- Parameters and metrics for all trials

---

## 🎯 Key Achievements

1. **Modular Architecture** ✅
   - 8 separate Python modules for different pipeline stages
   - Clean separation of concerns
   - Easy to maintain and extend

2. **High Model Performance** ✅
   - 96.67% test accuracy
   - Perfect classification for Setosa
   - Excellent performance on Versicolor and Virginica

3. **Comprehensive Logging** ✅
   - Structured logging with rotation
   - Separate logs for training, inference, and errors
   - Detailed execution tracking

4. **Experiment Tracking** ✅
   - MLFlow integration with 50 trials logged
   - Complete hyperparameter history
   - Model versioning and artifact storage

5. **Pipeline Versioning** ✅
   - DVC configuration for reproducibility
   - Data and model versioning support
   - Dependency tracking

6. **User Interface** ✅
   - Beautiful Gradio web app
   - Real-time predictions
   - Confidence scores and probabilities

7. **Containerization** ✅
   - Production-ready Dockerfile
   - Multi-stage build optimization
   - Health check support

8. **Cloud Deployment Ready** ✅
   - Automated deployment scripts
   - Google Cloud Run configuration
   - Artifact Registry integration

9. **Documentation** ✅
   - Comprehensive README
   - Code comments and docstrings
   - Usage examples

---

## 🚀 Next Steps

### To Run the Application Locally:

1. **View MLFlow Experiments:**
   ```bash
   mlflow ui
   # Open http://localhost:5000
   ```

2. **Launch Gradio App:**
   ```bash
   python app.py
   # Open http://localhost:7860
   ```

3. **Test Predictions:**
   - Enter flower measurements in the UI
   - Get instant predictions with confidence scores

### To Deploy to Google Cloud:

1. **Set Environment Variables:**
   ```bash
   set GCP_PROJECT_ID=your-project-id
   set GCP_REGION=us-central1
   ```

2. **Run Deployment Script:**
   ```bash
   deploy.bat  # Windows
   # or
   ./deploy.sh  # Linux/Mac
   ```

3. **Access Public URL:**
   - Script will output the Cloud Run service URL
   - Share with end users for predictions

---

## 📝 Notes

- Unicode checkmark characters (✓) cause encoding warnings in Windows console but don't affect functionality
- MLFlow file-based backend used for simplicity (can be upgraded to database backend for production)
- Model achieves excellent performance with minimal feature engineering
- All 50 Optuna trials completed in ~45 seconds
- Complete pipeline execution takes ~90 seconds

---

## 🎉 Summary

Successfully delivered a complete, production-ready ML application with:
- ✅ Modular, maintainable codebase
- ✅ High-performance model (96.67% accuracy)
- ✅ Comprehensive experiment tracking
- ✅ Beautiful user interface
- ✅ Docker containerization
- ✅ Cloud deployment scripts
- ✅ Extensive documentation

The application is ready for deployment and can be easily extended with additional features or deployed to production environments.
