# ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud

🌸 **Iris Flower Classification** - A production-ready ML application with XGBoost, Optuna hyperparameter tuning, MLFlow experiment tracking, Gradio UI, and Google Cloud deployment.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)
[![MLFlow](https://img.shields.io/badge/MLFlow-Tracking-green.svg)](https://mlflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-red.svg)](https://gradio.app/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [MLFlow Tracking](#mlflow-tracking)
- [DVC Pipeline](#dvc-pipeline)
- [Docker Deployment](#docker-deployment)
- [Google Cloud Deployment](#google-cloud-deployment)
- [Configuration](#configuration)
- [Logging](#logging)
- [Contributing](#contributing)

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Implementation Plan](docs/implementation_plan.md)** - Detailed technical plan and architecture decisions
- **[Walkthrough](docs/walkthrough.md)** - Complete walkthrough of the implementation with verification results
- **[Task Breakdown](docs/task.md)** - Development task checklist and progress tracking

These documents provide in-depth information about the project design, implementation details, and verification steps.

---

## 🎯 Overview

This project demonstrates a complete end-to-end machine learning pipeline for classifying Iris flowers into three species (Setosa, Versicolor, Virginica) using their sepal and petal measurements.

**Key Highlights:**
- ✅ Modular, production-ready code architecture
- ✅ XGBoost classifier with Optuna hyperparameter optimization
- ✅ MLFlow experiment tracking and model registry
- ✅ DVC for data and pipeline versioning
- ✅ Comprehensive logging with rotation
- ✅ Interactive Gradio web interface
- ✅ Docker containerization
- ✅ Google Cloud Run deployment ready

---

## ✨ Features

### Machine Learning
- **Dataset:** Iris dataset from scikit-learn (150 samples, 4 features, 3 classes)
- **Model:** XGBoost Classifier
- **Optimization:** Optuna with TPE sampler (50 trials by default)
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Evaluation:** Confusion matrix, classification report, feature importance

### MLOps
- **Experiment Tracking:** MLFlow with local tracking server
- **Pipeline Versioning:** DVC for reproducible pipelines
- **Logging:** Structured logging with rotation (training, inference, error logs)
- **Containerization:** Multi-stage Docker build
- **Deployment:** Google Cloud Run with Artifact Registry

### User Interface
- **Framework:** Gradio
- **Features:** Real-time predictions, confidence scores, probability distribution
- **Design:** Clean, modern UI with example inputs

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  Ingestion → Preprocessing → EDA → Transformation →         │
│  Feature Engineering → Model Building → Evaluation          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Optuna + MLFlow                             │
├─────────────────────────────────────────────────────────────┤
│  • Hyperparameter Optimization (50 trials)                  │
│  • Experiment Tracking                                       │
│  • Model Registry                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Gradio UI                                 │
├─────────────────────────────────────────────────────────────┤
│  • User Input (4 features)                                   │
│  • Real-time Predictions                                     │
│  • Confidence Scores                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Docker + Google Cloud Run                       │
├─────────────────────────────────────────────────────────────┤
│  • Containerized Application                                 │
│  • Scalable Deployment                                       │
│  • Public URL Access                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.12+
- Docker (for containerization)
- Google Cloud SDK (for cloud deployment)
- Git

### Local Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud
```

2. **Install dependencies using uv (recommended):**
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

3. **Initialize DVC (optional):**
```bash
dvc init
```

---

## 💻 Usage

### 1. Train the Model

Run the complete training pipeline:

```bash
python train.py
```

This will:
- Load and preprocess the Iris dataset
- Generate EDA visualizations
- Transform and engineer features
- Run Optuna hyperparameter optimization (50 trials)
- Train the best model
- Evaluate and save metrics
- Log everything to MLFlow

**Expected Output:**
```
✓ Data ingestion completed: 150 samples, 4 features
✓ Data preprocessing completed: 150 clean samples
✓ EDA completed: Visualizations saved to plots/
✓ Data transformation completed
✓ Feature engineering completed: 4 features
✓ Model building completed
  - Best accuracy: 0.9667
  - Number of trials: 50
✓ Model evaluation completed
  - Test Accuracy: 0.9667
```

### 2. View MLFlow Experiments

Start the MLFlow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser to view:
- All experiment runs
- Hyperparameter values
- Metrics and plots
- Model artifacts

### 3. Run the Gradio App

Launch the web interface:

```bash
python app.py
```

Access the app at http://localhost:7860

**Features:**
- Enter flower measurements (sepal length, sepal width, petal length, petal width)
- Get instant predictions with confidence scores
- View probability distribution for all three species

### 4. Run DVC Pipeline

Execute the entire pipeline with DVC:

```bash
dvc repro
```

This ensures reproducibility and tracks all data/model versions.

---

## 📁 Project Structure

```
ML-Classification-Optuna-MLFlow-Docker-CICD-Cloud/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── logger.py                 # Logging utility
│   ├── data_ingestion.py         # Load Iris dataset
│   ├── data_preprocessing.py     # Data cleaning
│   ├── eda.py                    # Exploratory analysis
│   ├── data_transformation.py    # Train-test split, scaling
│   ├── feature_engineering.py    # Feature creation
│   ├── model_building.py         # XGBoost + Optuna + MLFlow
│   ├── model_evaluation.py       # Metrics and plots
│   └── prediction.py             # Inference module
├── config/                       # Configuration files
│   ├── config.yaml               # Main configuration
│   └── logging_config.yaml       # Logging settings
├── models/                       # Saved models
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── data/                         # Data files
│   ├── iris_raw.csv
│   ├── iris_processed.csv
│   ├── train.csv
│   └── test.csv
├── logs/                         # Application logs
│   ├── training.log
│   ├── inference.log
│   └── error.log
├── plots/                        # Visualizations
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── app.py                        # Gradio application
├── train.py                      # Training pipeline
├── Dockerfile                    # Docker configuration
├── dvc.yaml                      # DVC pipeline
├── deploy.sh                     # Linux/Mac deployment
├── deploy.bat                    # Windows deployment
├── pyproject.toml                # Dependencies
└── README.md                     # Documentation
```

---

## 📊 MLFlow Tracking

All experiments are automatically logged to MLFlow:

**Logged Information:**
- Hyperparameters (learning_rate, max_depth, n_estimators, etc.)
- Metrics (accuracy, precision, recall, F1-score)
- Model artifacts
- Feature importance
- Training history

**Access MLFlow UI:**
```bash
mlflow ui
# Open http://localhost:5000
```

---

## 🔄 DVC Pipeline

The DVC pipeline ensures reproducibility:

**Pipeline Stages:**
1. `data_ingestion` - Load Iris dataset
2. `data_preprocessing` - Clean and validate
3. `data_transformation` - Split and scale
4. `model_training` - Train with Optuna
5. `model_evaluation` - Generate metrics

**Run Pipeline:**
```bash
dvc repro
```

**View Pipeline DAG:**
```bash
dvc dag
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t iris-classifier .
```

### Run Container Locally

```bash
docker run -p 7860:7860 iris-classifier
```

Access at http://localhost:7860

### Push to Docker Hub (Optional)

```bash
docker tag iris-classifier:latest yourusername/iris-classifier:latest
docker push yourusername/iris-classifier:latest
```

---

## ☁️ Google Cloud Deployment

### Prerequisites
1. Google Cloud account with billing enabled
2. gcloud CLI installed and authenticated
3. Required APIs enabled (Artifact Registry, Cloud Run)

### Deploy to Cloud Run

**Linux/Mac:**
```bash
chmod +x deploy.sh
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
./deploy.sh
```

**Windows:**
```cmd
set GCP_PROJECT_ID=your-project-id
set GCP_REGION=us-central1
deploy.bat
```

The script will:
1. Build the Docker image
2. Push to Google Artifact Registry
3. Deploy to Cloud Run
4. Provide a public URL

**Manual Deployment:**
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/iris-classifier

# Deploy
gcloud run deploy iris-classifier \
  --image gcr.io/PROJECT_ID/iris-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --port 7860
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

**Data Configuration:**
```yaml
data:
  test_size: 0.2
  random_state: 42
  stratify: true
```

**Optuna Configuration:**
```yaml
optuna:
  n_trials: 50
  direction: "maximize"
  metric: "accuracy"
```

**MLFlow Configuration:**
```yaml
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "iris_classification_xgboost"
```

**Gradio Configuration:**
```yaml
gradio:
  server_name: "0.0.0.0"
  server_port: 7860
```

---

## 📝 Logging

Comprehensive logging with rotation:

**Log Files:**
- `logs/training.log` - Training pipeline logs (DEBUG level)
- `logs/inference.log` - Prediction logs (INFO level)
- `logs/error.log` - Error logs with stack traces (ERROR level)

**Log Rotation:**
- Training logs: 10MB max, 5 backups
- Inference logs: 5MB max, 3 backups
- Error logs: 10MB max, 5 backups

**View Logs:**
```bash
# Training logs
tail -f logs/training.log

# Inference logs
tail -f logs/inference.log

# Error logs
tail -f logs/error.log
```

---

## 🧪 Testing

Test individual modules:

```bash
# Test data ingestion
python -m src.data_ingestion

# Test preprocessing
python -m src.data_preprocessing

# Test prediction
python -m src.prediction
```

---

## 📈 Model Performance

**Expected Results:**
- **Test Accuracy:** ~96-97%
- **Precision:** ~96-97%
- **Recall:** ~96-97%
- **F1-Score:** ~96-97%

**Hyperparameters (typical best values):**
- learning_rate: 0.05-0.15
- max_depth: 3-6
- n_estimators: 100-200
- subsample: 0.8-1.0

---

## 🛠️ Troubleshooting

**Model not found error:**
```bash
# Train the model first
python train.py
```

**MLFlow connection error:**
```bash
# Start MLFlow server
mlflow ui
```

**Port already in use:**
```bash
# Change port in config/config.yaml
gradio:
  server_port: 7861
```

**Docker build fails:**
```bash
# Ensure model is trained first
python train.py
# Then rebuild
docker build -t iris-classifier .
```

---

## 📚 Technologies Used

- **ML Framework:** XGBoost, scikit-learn
- **Optimization:** Optuna
- **Tracking:** MLFlow
- **Versioning:** DVC
- **UI:** Gradio
- **Visualization:** Matplotlib, Seaborn
- **Containerization:** Docker
- **Cloud:** Google Cloud Run, Artifact Registry
- **Language:** Python 3.12
- **Package Manager:** uv

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

Created with ❤️ for demonstrating production-ready ML pipelines

---

## 🙏 Acknowledgments

- Iris dataset from UCI Machine Learning Repository
- XGBoost team for the excellent gradient boosting library
- Optuna team for hyperparameter optimization
- MLFlow team for experiment tracking
- Gradio team for the amazing UI framework

---

**Happy Coding! 🚀**