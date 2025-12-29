"""
Feature Engineering Module
Creates additional features and performs feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from pathlib import Path
import yaml
from typing import Tuple, List

from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class FeatureEngineering:
    """Handles feature engineering and selection."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize FeatureEngineering with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.poly_features = None
        self.feature_selector = None
        logger.info("FeatureEngineering initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def create_polynomial_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                   degree: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create polynomial features.
        
        Args:
            X_train: Training features DataFrame
            X_test: Testing features DataFrame
            degree: Degree of polynomial features
        
        Returns:
            Tuple of (X_train with polynomial features, X_test with polynomial features)
        """
        logger.info(f"Creating polynomial features with degree {degree}")
        
        try:
            self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            
            # Fit on training data and transform both
            X_train_poly = self.poly_features.fit_transform(X_train)
            X_test_poly = self.poly_features.transform(X_test)
            
            # Get feature names
            feature_names = self.poly_features.get_feature_names_out(X_train.columns)
            
            # Convert back to DataFrame
            X_train_poly = pd.DataFrame(X_train_poly, columns=feature_names, index=X_train.index)
            X_test_poly = pd.DataFrame(X_test_poly, columns=feature_names, index=X_test.index)
            
            logger.info(f"Polynomial features created: {X_train_poly.shape[1]} features")
            logger.debug(f"Original features: {X_train.shape[1]}, New features: {X_train_poly.shape[1]}")
            
            return X_train_poly, X_test_poly
            
        except Exception as e:
            logger.error(f"Error creating polynomial features: {e}", exc_info=True)
            raise
    
    def create_feature_interactions(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create interaction features between existing features.
        
        Args:
            X_train: Training features DataFrame
            X_test: Testing features DataFrame
        
        Returns:
            Tuple of (X_train with interactions, X_test with interactions)
        """
        logger.info("Creating feature interactions")
        
        try:
            X_train_new = X_train.copy()
            X_test_new = X_test.copy()
            
            # Get original column names
            columns = X_train.columns.tolist()
            
            # Create pairwise interactions
            interaction_count = 0
            for i in range(len(columns)):
                for j in range(i+1, len(columns)):
                    col1, col2 = columns[i], columns[j]
                    interaction_name = f"{col1}_x_{col2}"
                    
                    X_train_new[interaction_name] = X_train[col1] * X_train[col2]
                    X_test_new[interaction_name] = X_test[col1] * X_test[col2]
                    interaction_count += 1
            
            logger.info(f"Created {interaction_count} interaction features")
            logger.debug(f"Total features after interactions: {X_train_new.shape[1]}")
            
            return X_train_new, X_test_new
            
        except Exception as e:
            logger.error(f"Error creating feature interactions: {e}", exc_info=True)
            raise
    
    def select_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: np.ndarray, k: int = 10, 
                       method: str = 'f_classif') -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Select top k features using statistical tests.
        
        Args:
            X_train: Training features DataFrame
            X_test: Testing features DataFrame
            y_train: Training target
            k: Number of features to select
            method: Selection method ('f_classif' or 'mutual_info')
        
        Returns:
            Tuple of (selected X_train, selected X_test, selected feature names)
        """
        logger.info(f"Selecting top {k} features using {method}")
        
        try:
            # Choose scoring function
            score_func = f_classif if method == 'f_classif' else mutual_info_classif
            
            # Ensure k doesn't exceed number of features
            k = min(k, X_train.shape[1])
            
            self.feature_selector = SelectKBest(score_func=score_func, k=k)
            
            # Fit on training data and transform both
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            X_test_selected = self.feature_selector.transform(X_test)
            
            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            selected_features = X_train.columns[selected_mask].tolist()
            
            # Convert back to DataFrame
            X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
            
            logger.info(f"Selected features: {selected_features}")
            logger.debug(f"Feature scores: {self.feature_selector.scores_[selected_mask]}")
            
            return X_train_selected, X_test_selected, selected_features
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}", exc_info=True)
            raise
    
    def engineer_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                         y_train: np.ndarray = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main feature engineering pipeline.
        
        Args:
            X_train: Training features DataFrame
            X_test: Testing features DataFrame
            y_train: Training target (optional, needed for feature selection)
        
        Returns:
            Tuple of (engineered X_train, engineered X_test)
        """
        logger.info("Starting feature engineering pipeline")
        
        X_train_eng = X_train.copy()
        X_test_eng = X_test.copy()
        
        # Check if feature interactions should be created
        if self.config['features'].get('create_interactions', False):
            X_train_eng, X_test_eng = self.create_feature_interactions(X_train_eng, X_test_eng)
        else:
            logger.info("Feature interactions disabled in configuration")
        
        # Check if polynomial features should be created
        poly_degree = self.config['features'].get('polynomial_degree', 1)
        if poly_degree > 1:
            X_train_eng, X_test_eng = self.create_polynomial_features(X_train_eng, X_test_eng, degree=poly_degree)
        else:
            logger.info("Polynomial features disabled (degree = 1)")
        
        logger.info("Feature engineering pipeline completed")
        logger.info(f"Final feature count: {X_train_eng.shape[1]}")
        
        return X_train_eng, X_test_eng
    
    def run(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
            y_train: np.ndarray = None) -> dict:
        """
        Main execution method for feature engineering.
        
        Args:
            X_train: Training features DataFrame
            X_test: Testing features DataFrame
            y_train: Training target (optional)
        
        Returns:
            Dictionary containing engineered features
        """
        logger.info("Starting feature engineering process")
        
        # Engineer features
        X_train_eng, X_test_eng = self.engineer_features(X_train, X_test, y_train)
        
        logger.info("Feature engineering process completed successfully")
        
        return {
            'X_train': X_train_eng,
            'X_test': X_test_eng,
            'feature_names': X_train_eng.columns.tolist(),
            'n_features': X_train_eng.shape[1]
        }


if __name__ == "__main__":
    # Test feature engineering
    from src.data_ingestion import DataIngestion
    from src.data_preprocessing import DataPreprocessing
    from src.data_transformation import DataTransformation
    
    # Load and preprocess data
    ingestion = DataIngestion()
    features, target = ingestion.run()
    
    preprocessing = DataPreprocessing()
    features_clean, target_clean = preprocessing.run(features, target)
    
    transformation = DataTransformation()
    transformed_data = transformation.run(features_clean, target_clean)
    
    # Engineer features
    feature_eng = FeatureEngineering()
    engineered_data = feature_eng.run(
        transformed_data['X_train'],
        transformed_data['X_test'],
        transformed_data['y_train']
    )
    
    print("\nEngineered features:")
    print(f"X_train shape: {engineered_data['X_train'].shape}")
    print(f"X_test shape: {engineered_data['X_test'].shape}")
    print(f"Feature names: {engineered_data['feature_names']}")
