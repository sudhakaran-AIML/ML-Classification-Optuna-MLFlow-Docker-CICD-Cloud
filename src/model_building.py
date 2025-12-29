"""
Model Building Module
Trains XGBoost classifier with Optuna hyperparameter optimization and MLFlow tracking.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from optuna.samplers import TPESampler
import mlflow
import mlflow.xgboost
from pathlib import Path
import yaml
import joblib
from typing import Dict, Tuple
import warnings

from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Suppress Optuna warnings
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


class ModelBuilding:
    """Handles model training with Optuna optimization and MLFlow tracking."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ModelBuilding with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.best_model = None
        self.best_params = None
        self.study = None
        logger.info("ModelBuilding initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def setup_mlflow(self) -> None:
        """Setup MLFlow tracking."""
        try:
            tracking_uri = self.config['mlflow']['tracking_uri']
            experiment_name = self.config['mlflow']['experiment_name']
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            
            logger.info(f"MLFlow tracking URI: {tracking_uri}")
            logger.info(f"MLFlow experiment: {experiment_name}")
            
        except Exception as e:
            logger.error(f"Error setting up MLFlow: {e}", exc_info=True)
            raise
    
    def create_objective(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: np.ndarray, y_test: np.ndarray):
        """
        Create Optuna objective function.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        
        Returns:
            Objective function for Optuna
        """
        def objective(trial: optuna.Trial) -> float:
            """
            Optuna objective function to maximize accuracy.
            
            Args:
                trial: Optuna trial object
            
            Returns:
                Accuracy score on test set
            """
            # Start MLFlow run for this trial
            with mlflow.start_run(run_name=f"{self.config['mlflow']['run_name_prefix']}_trial_{trial.number}", nested=True):
                
                # Sample hyperparameters from search space
                search_space = self.config['optuna']['search_space']
                
                params = {
                    'objective': self.config['model']['objective'],
                    'num_class': self.config['model']['num_class'],
                    'eval_metric': self.config['model']['eval_metric'],
                    'random_state': self.config['data']['random_state'],
                    'verbosity': 0
                }
                
                # Add hyperparameters from search space
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['low'],
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                
                # Train model
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Log parameters to MLFlow
                mlflow.log_params(params)
                
                # Log metrics to MLFlow
                mlflow.log_metrics({
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'trial_number': trial.number
                })
                
                # Log additional info
                mlflow.set_tag("trial_number", trial.number)
                mlflow.set_tag("optuna_study", "iris_classification")
                
                logger.debug(f"Trial {trial.number}: accuracy={accuracy:.4f}, params={params}")
                
                return accuracy
        
        return objective
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                 y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Run Optuna hyperparameter optimization.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        
        Returns:
            Dictionary with best parameters and study results
        """
        logger.info("Starting Optuna hyperparameter optimization")
        
        try:
            # Setup MLFlow
            self.setup_mlflow()
            
            # Create Optuna study
            sampler = TPESampler(seed=self.config['data']['random_state'])
            
            self.study = optuna.create_study(
                direction=self.config['optuna']['direction'],
                sampler=sampler,
                study_name="iris_xgboost_optimization"
            )
            
            # Create objective function
            objective = self.create_objective(X_train, X_test, y_train, y_test)
            
            # Start parent MLFlow run
            with mlflow.start_run(run_name="optuna_optimization"):
                
                # Run optimization
                n_trials = self.config['optuna']['n_trials']
                timeout = self.config['optuna'].get('timeout', None)
                
                logger.info(f"Running {n_trials} trials with timeout={timeout}s")
                
                self.study.optimize(
                    objective,
                    n_trials=n_trials,
                    timeout=timeout,
                    show_progress_bar=True
                )
                
                # Get best parameters
                self.best_params = self.study.best_params
                best_value = self.study.best_value
                
                logger.info(f"Optimization completed!")
                logger.info(f"Best accuracy: {best_value:.4f}")
                logger.info(f"Best parameters: {self.best_params}")
                
                # Log best parameters to MLFlow
                mlflow.log_params(self.best_params)
                mlflow.log_metric("best_accuracy", best_value)
                mlflow.log_metric("n_trials", n_trials)
                
                # Log optimization history
                trials_df = self.study.trials_dataframe()
                trials_csv_path = Path("models/optuna_trials.csv")
                trials_df.to_csv(trials_csv_path, index=False)
                mlflow.log_artifact(str(trials_csv_path))
                
                logger.info(f"Optimization history saved to {trials_csv_path}")
            
            return {
                'best_params': self.best_params,
                'best_value': best_value,
                'n_trials': len(self.study.trials),
                'study': self.study
            }
            
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {e}", exc_info=True)
            raise
    
    def train_best_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: np.ndarray, y_test: np.ndarray) -> xgb.XGBClassifier:
        """
        Train final model with best hyperparameters.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        
        Returns:
            Trained XGBoost model
        """
        logger.info("Training final model with best hyperparameters")
        
        try:
            if self.best_params is None:
                raise ValueError("Best parameters not found. Run optimize_hyperparameters first.")
            
            # Prepare parameters
            params = {
                **self.best_params,
                'objective': self.config['model']['objective'],
                'num_class': self.config['model']['num_class'],
                'eval_metric': self.config['model']['eval_metric'],
                'random_state': self.config['data']['random_state']
            }
            
            # Train model
            self.best_model = xgb.XGBClassifier(**params)
            self.best_model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
            
            # Make predictions
            y_train_pred = self.best_model.predict(X_train)
            y_test_pred = self.best_model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            logger.info(f"Final model trained successfully")
            logger.info(f"Training accuracy: {train_accuracy:.4f}")
            logger.info(f"Testing accuracy: {test_accuracy:.4f}")
            
            # Log to MLFlow
            with mlflow.start_run(run_name="best_model"):
                mlflow.log_params(params)
                mlflow.log_metrics({
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy
                })
                
                # Log model (with error handling for compatibility issues)
                if self.config['mlflow'].get('log_models', True):
                    try:
                        mlflow.sklearn.log_model(self.best_model, "model")
                        logger.info("Model logged to MLFlow")
                    except Exception as e:
                        logger.warning(f"Could not log model to MLFlow: {e}")
                        logger.info("Model will still be saved locally")
            
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error training best model: {e}", exc_info=True)
            raise
    
    def save_model(self, label_encoder=None) -> None:
        """
        Save trained model and related artifacts.
        
        Args:
            label_encoder: Label encoder to save with model
        """
        try:
            if self.best_model is None:
                raise ValueError("No model to save. Train a model first.")
            
            models_dir = Path(self.config['paths']['models_dir'])
            models_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = Path(self.config['paths']['model_file'])
            joblib.dump(self.best_model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save best parameters
            params_path = models_dir / 'best_params.yaml'
            with open(params_path, 'w') as f:
                yaml.dump(self.best_params, f)
            logger.info(f"Best parameters saved to {params_path}")
            
            # Save label encoder if provided
            if label_encoder is not None:
                encoder_path = models_dir / 'label_encoder.pkl'
                joblib.dump(label_encoder, encoder_path)
                logger.info(f"Label encoder saved to {encoder_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            raise
    
    def run(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
            y_train: np.ndarray, y_test: np.ndarray, 
            label_encoder=None) -> Dict:
        """
        Main execution method for model building.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            label_encoder: Label encoder for saving
        
        Returns:
            Dictionary with model and optimization results
        """
        logger.info("Starting model building process")
        
        # Optimize hyperparameters
        optimization_results = self.optimize_hyperparameters(X_train, X_test, y_train, y_test)
        
        # Train best model
        best_model = self.train_best_model(X_train, X_test, y_train, y_test)
        
        # Save model
        self.save_model(label_encoder)
        
        logger.info("Model building process completed successfully")
        
        return {
            'model': best_model,
            'best_params': self.best_params,
            'best_accuracy': optimization_results['best_value'],
            'n_trials': optimization_results['n_trials'],
            'study': self.study
        }


if __name__ == "__main__":
    # Test model building
    from src.data_ingestion import DataIngestion
    from src.data_preprocessing import DataPreprocessing
    from src.data_transformation import DataTransformation
    from src.feature_engineering import FeatureEngineering
    
    # Load and prepare data
    ingestion = DataIngestion()
    features, target = ingestion.run()
    
    preprocessing = DataPreprocessing()
    features_clean, target_clean = preprocessing.run(features, target)
    
    transformation = DataTransformation()
    transformed_data = transformation.run(features_clean, target_clean)
    
    feature_eng = FeatureEngineering()
    engineered_data = feature_eng.run(
        transformed_data['X_train'],
        transformed_data['X_test'],
        transformed_data['y_train']
    )
    
    # Build model
    model_builder = ModelBuilding()
    results = model_builder.run(
        engineered_data['X_train'],
        engineered_data['X_test'],
        transformed_data['y_train'],
        transformed_data['y_test'],
        transformed_data['label_encoder']
    )
    
    print("\nModel building completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Best parameters: {results['best_params']}")
