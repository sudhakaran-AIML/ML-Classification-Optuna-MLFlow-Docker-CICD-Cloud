"""
Model Evaluation Module
Evaluates trained model and generates comprehensive metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from pathlib import Path
import yaml
import json
from typing import Dict, List

from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class ModelEvaluation:
    """Handles model evaluation and metrics generation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ModelEvaluation with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.plots_dir = Path(self.config['paths'].get('plots_dir', 'plots'))
        self.plots_dir.mkdir(exist_ok=True)
        logger.info("ModelEvaluation initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating classification metrics")
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro'),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
                'recall_macro': recall_score(y_true, y_pred, average='macro'),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
                'f1_macro': f1_score(y_true, y_pred, average='macro'),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted')
            }
            
            logger.info("Metrics calculated successfully")
            for metric_name, metric_value in metrics.items():
                logger.info(f"{metric_name}: {metric_value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}", exc_info=True)
            raise
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
        """
        logger.info("Generating confusion matrix plot")
        
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Count'})
            
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            output_path = self.plots_dir / 'confusion_matrix.png'
            plt.savefig(output_path, dpi=self.config['eda'].get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}", exc_info=True)
            raise
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      class_names: List[str] = None) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
        
        Returns:
            Classification report as string
        """
        logger.info("Generating classification report")
        
        try:
            report = classification_report(y_true, y_pred, target_names=class_names)
            
            logger.info(f"Classification Report:\n{report}")
            
            # Save report to file
            report_path = self.plots_dir / 'classification_report.txt'
            with open(report_path, 'w') as f:
                f.write("Classification Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)
            
            logger.info(f"Classification report saved to {report_path}")
            
            # Also save as JSON
            report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            report_json_path = self.plots_dir / 'classification_report.json'
            with open(report_json_path, 'w') as f:
                json.dump(report_dict, f, indent=4)
            
            logger.info(f"Classification report (JSON) saved to {report_json_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {e}", exc_info=True)
            raise
    
    def plot_feature_importance(self, model, feature_names: List[str], top_n: int = 10) -> None:
        """
        Plot feature importance from XGBoost model.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            top_n: Number of top features to display
        """
        logger.info(f"Plotting top {top_n} feature importances")
        
        try:
            # Get feature importances
            importances = model.feature_importances_
            
            # Create DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Select top N features
            top_features = feature_importance_df.head(top_n)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            
            colors = sns.color_palette("viridis", len(top_features))
            plt.barh(range(len(top_features)), top_features['importance'], color=colors)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            # Save plot
            output_path = self.plots_dir / 'feature_importance.png'
            plt.savefig(output_path, dpi=self.config['eda'].get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to {output_path}")
            
            # Save feature importances to CSV
            csv_path = self.plots_dir / 'feature_importance.csv'
            feature_importance_df.to_csv(csv_path, index=False)
            logger.info(f"Feature importances saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}", exc_info=True)
            raise
    
    def plot_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    class_names: List[str] = None) -> None:
        """
        Plot distribution of predictions vs true labels.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
        """
        logger.info("Plotting prediction distribution")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # True label distribution
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            ax1.bar(unique_true, counts_true, color='skyblue', edgecolor='black')
            ax1.set_title('True Label Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3, axis='y')
            
            if class_names:
                ax1.set_xticks(unique_true)
                ax1.set_xticklabels([class_names[i] for i in unique_true], rotation=45)
            
            # Predicted label distribution
            unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
            ax2.bar(unique_pred, counts_pred, color='lightcoral', edgecolor='black')
            ax2.set_title('Predicted Label Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Count')
            ax2.grid(True, alpha=0.3, axis='y')
            
            if class_names:
                ax2.set_xticks(unique_pred)
                ax2.set_xticklabels([class_names[i] for i in unique_pred], rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            output_path = self.plots_dir / 'prediction_distribution.png'
            plt.savefig(output_path, dpi=self.config['eda'].get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Prediction distribution plot saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error plotting prediction distribution: {e}", exc_info=True)
            raise
    
    def save_metrics(self, metrics: Dict[str, float], filename: str = 'metrics.json') -> None:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        try:
            output_path = self.plots_dir / filename
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Metrics saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}", exc_info=True)
            raise
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: np.ndarray,
                      class_names: List[str] = None, feature_names: List[str] = None) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            class_names: Names of classes
            feature_names: Names of features
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting comprehensive model evaluation")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Generate visualizations
            self.plot_confusion_matrix(y_test, y_pred, class_names)
            self.plot_prediction_distribution(y_test, y_pred, class_names)
            
            # Generate classification report
            report = self.generate_classification_report(y_test, y_pred, class_names)
            
            # Plot feature importance
            if feature_names:
                self.plot_feature_importance(model, feature_names)
            
            # Save metrics
            self.save_metrics(metrics)
            
            logger.info("Model evaluation completed successfully")
            
            return {
                'metrics': metrics,
                'predictions': y_pred,
                'classification_report': report
            }
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}", exc_info=True)
            raise
    
    def run(self, model, X_test: pd.DataFrame, y_test: np.ndarray,
            class_names: List[str] = None, feature_names: List[str] = None) -> Dict:
        """
        Main execution method for model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            class_names: Names of classes
            feature_names: Names of features
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Starting model evaluation process")
        
        results = self.evaluate_model(model, X_test, y_test, class_names, feature_names)
        
        logger.info("Model evaluation process completed successfully")
        logger.info(f"All evaluation artifacts saved to {self.plots_dir}")
        
        return results


if __name__ == "__main__":
    # Test model evaluation
    from src.data_ingestion import DataIngestion
    from src.data_preprocessing import DataPreprocessing
    from src.data_transformation import DataTransformation
    from src.feature_engineering import FeatureEngineering
    from src.model_building import ModelBuilding
    
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
    model_results = model_builder.run(
        engineered_data['X_train'],
        engineered_data['X_test'],
        transformed_data['y_train'],
        transformed_data['y_test'],
        transformed_data['label_encoder']
    )
    
    # Evaluate model
    evaluator = ModelEvaluation()
    eval_results = evaluator.run(
        model_results['model'],
        engineered_data['X_test'],
        transformed_data['y_test'],
        class_names=transformed_data['label_encoder'].classes_.tolist(),
        feature_names=engineered_data['feature_names']
    )
    
    print("\nModel Evaluation Completed!")
    print(f"Test Accuracy: {eval_results['metrics']['accuracy']:.4f}")
