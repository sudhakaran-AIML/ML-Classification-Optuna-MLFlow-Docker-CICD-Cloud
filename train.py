"""
Main Training Pipeline
Orchestrates the entire ML pipeline from data ingestion to model evaluation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.logger import setup_logger
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.eda import EDA
from src.data_transformation import DataTransformation
from src.feature_engineering import FeatureEngineering
from src.model_building import ModelBuilding
from src.model_evaluation import ModelEvaluation

# Initialize logger
logger = setup_logger("train")


def main():
    """Main training pipeline execution."""
    logger.info("=" * 80)
    logger.info("STARTING IRIS CLASSIFICATION TRAINING PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Step 1: Data Ingestion
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("=" * 80)
        ingestion = DataIngestion()
        features, target = ingestion.run()
        logger.info(f"✓ Data ingestion completed: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Step 2: Data Preprocessing
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 80)
        preprocessing = DataPreprocessing()
        features_clean, target_clean = preprocessing.run(features, target)
        logger.info(f"✓ Data preprocessing completed: {features_clean.shape[0]} clean samples")
        
        # Step 3: Exploratory Data Analysis
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 80)
        eda = EDA()
        eda.run(features_clean, target_clean)
        logger.info("✓ EDA completed: Visualizations saved to plots/ directory")
        
        # Step 4: Data Transformation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: DATA TRANSFORMATION")
        logger.info("=" * 80)
        transformation = DataTransformation()
        transformed_data = transformation.run(features_clean, target_clean)
        logger.info(f"✓ Data transformation completed")
        logger.info(f"  - Training set: {transformed_data['X_train'].shape[0]} samples")
        logger.info(f"  - Testing set: {transformed_data['X_test'].shape[0]} samples")
        
        # Step 5: Feature Engineering
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: FEATURE ENGINEERING")
        logger.info("=" * 80)
        feature_eng = FeatureEngineering()
        engineered_data = feature_eng.run(
            transformed_data['X_train'],
            transformed_data['X_test'],
            transformed_data['y_train']
        )
        logger.info(f"✓ Feature engineering completed: {engineered_data['n_features']} features")
        
        # Step 6: Model Building with Optuna + MLFlow
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: MODEL BUILDING (OPTUNA + MLFLOW)")
        logger.info("=" * 80)
        model_builder = ModelBuilding()
        model_results = model_builder.run(
            engineered_data['X_train'],
            engineered_data['X_test'],
            transformed_data['y_train'],
            transformed_data['y_test'],
            transformed_data['label_encoder']
        )
        logger.info(f"✓ Model building completed")
        logger.info(f"  - Best accuracy: {model_results['best_accuracy']:.4f}")
        logger.info(f"  - Number of trials: {model_results['n_trials']}")
        logger.info(f"  - Best parameters: {model_results['best_params']}")
        
        # Step 7: Model Evaluation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: MODEL EVALUATION")
        logger.info("=" * 80)
        evaluator = ModelEvaluation()
        eval_results = evaluator.run(
            model_results['model'],
            engineered_data['X_test'],
            transformed_data['y_test'],
            class_names=transformed_data['label_encoder'].classes_.tolist(),
            feature_names=engineered_data['feature_names']
        )
        logger.info(f"✓ Model evaluation completed")
        logger.info(f"  - Test Accuracy: {eval_results['metrics']['accuracy']:.4f}")
        logger.info(f"  - Test F1-Score (weighted): {eval_results['metrics']['f1_weighted']:.4f}")
        logger.info(f"  - Test Precision (weighted): {eval_results['metrics']['precision_weighted']:.4f}")
        logger.info(f"  - Test Recall (weighted): {eval_results['metrics']['recall_weighted']:.4f}")
        
        # Pipeline Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("\nPipeline Summary:")
        logger.info(f"  - Dataset: Iris (150 samples, 4 features)")
        logger.info(f"  - Model: XGBoost Classifier")
        logger.info(f"  - Hyperparameter Optimization: Optuna ({model_results['n_trials']} trials)")
        logger.info(f"  - Experiment Tracking: MLFlow")
        logger.info(f"  - Final Test Accuracy: {eval_results['metrics']['accuracy']:.4f}")
        logger.info(f"\nArtifacts saved:")
        logger.info(f"  - Model: models/best_model.pkl")
        logger.info(f"  - Scaler: models/scaler.pkl")
        logger.info(f"  - Label Encoder: models/label_encoder.pkl")
        logger.info(f"  - Plots: plots/")
        logger.info(f"  - Logs: logs/")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. View MLFlow experiments: mlflow ui")
        logger.info(f"  2. Run Gradio app: python app.py")
        logger.info(f"  3. Build Docker image: docker build -t iris-classifier .")
        logger.info("=" * 80)
        
        return {
            'success': True,
            'model': model_results['model'],
            'metrics': eval_results['metrics'],
            'best_params': model_results['best_params']
        }
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("TRAINING PIPELINE FAILED!")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    results = main()
    
    if results['success']:
        print("\n✓ Training pipeline completed successfully!")
        print(f"✓ Test Accuracy: {results['metrics']['accuracy']:.4f}")
        sys.exit(0)
    else:
        print("\n✗ Training pipeline failed!")
        print(f"✗ Error: {results['error']}")
        sys.exit(1)
