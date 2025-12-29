"""
Data Transformation Module
Handles train-test split and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from pathlib import Path
import yaml
import joblib
from typing import Tuple, Dict

from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class DataTransformation:
    """Handles data transformation including splitting and scaling."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataTransformation with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.scaler = None
        self.label_encoder = None
        logger.info("DataTransformation initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def split_data(self, df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Features DataFrame
            target: Target Series
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train and test sets")
        
        try:
            test_size = self.config['data']['test_size']
            random_state = self.config['data']['random_state']
            stratify_flag = self.config['data'].get('stratify', True)
            
            stratify_param = target if stratify_flag else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                df, 
                target,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_param
            )
            
            logger.info(f"Train set size: {len(X_train)} samples ({(1-test_size)*100:.1f}%)")
            logger.info(f"Test set size: {len(X_test)} samples ({test_size*100:.1f}%)")
            logger.info(f"Train class distribution:\n{y_train.value_counts()}")
            logger.info(f"Test class distribution:\n{y_test.value_counts()}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}", exc_info=True)
            raise
    
    def encode_target(self, y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode target labels to numeric values.
        
        Args:
            y_train: Training target Series
            y_test: Testing target Series
        
        Returns:
            Tuple of (encoded y_train, encoded y_test)
        """
        logger.info("Encoding target labels")
        
        try:
            self.label_encoder = LabelEncoder()
            
            # Fit on training data and transform both
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Log the mapping
            class_mapping = dict(zip(self.label_encoder.classes_, 
                                    self.label_encoder.transform(self.label_encoder.classes_)))
            logger.info(f"Label encoding mapping: {class_mapping}")
            
            return y_train_encoded, y_test_encoded
            
        except Exception as e:
            logger.error(f"Error encoding target labels: {e}", exc_info=True)
            raise
    
    def get_scaler(self, method: str = "standard"):
        """
        Get scaler instance based on method.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
        
        Returns:
            Scaler instance
        """
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if method not in scalers:
            logger.warning(f"Unknown scaling method '{method}'. Using 'standard' instead.")
            method = 'standard'
        
        logger.info(f"Using {method} scaling")
        return scalers[method]
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified scaling method.
        
        Args:
            X_train: Training features DataFrame
            X_test: Testing features DataFrame
        
        Returns:
            Tuple of (scaled X_train, scaled X_test)
        """
        logger.info("Scaling features")
        
        try:
            scaling_method = self.config['features'].get('scaling_method', 'standard')
            self.scaler = self.get_scaler(scaling_method)
            
            # Fit on training data and transform both
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Convert back to DataFrame to preserve column names
            X_train_scaled = pd.DataFrame(X_train_scaled, 
                                         columns=X_train.columns, 
                                         index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, 
                                        columns=X_test.columns, 
                                        index=X_test.index)
            
            logger.info(f"Features scaled using {scaling_method} scaler")
            logger.debug(f"Training data shape after scaling: {X_train_scaled.shape}")
            logger.debug(f"Testing data shape after scaling: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}", exc_info=True)
            raise
    
    def save_artifacts(self) -> None:
        """Save scaler and label encoder artifacts."""
        try:
            models_dir = Path(self.config['paths']['models_dir'])
            models_dir.mkdir(exist_ok=True)
            
            # Save scaler
            if self.scaler is not None:
                scaler_path = Path(self.config['paths']['scaler_file'])
                joblib.dump(self.scaler, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}")
            
            # Save label encoder
            if self.label_encoder is not None:
                encoder_path = models_dir / 'label_encoder.pkl'
                joblib.dump(self.label_encoder, encoder_path)
                logger.info(f"Label encoder saved to {encoder_path}")
                
        except Exception as e:
            logger.error(f"Error saving artifacts: {e}", exc_info=True)
            raise
    
    def save_split_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series) -> None:
        """
        Save train and test datasets to CSV files.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
        """
        try:
            # Save training data
            train_data = X_train.copy()
            train_data[self.config['data']['target_column']] = y_train
            train_path = Path(self.config['paths']['train_data'])
            train_data.to_csv(train_path, index=False)
            logger.info(f"Training data saved to {train_path}")
            
            # Save testing data
            test_data = X_test.copy()
            test_data[self.config['data']['target_column']] = y_test
            test_path = Path(self.config['paths']['test_data'])
            test_data.to_csv(test_path, index=False)
            logger.info(f"Testing data saved to {test_path}")
            
        except Exception as e:
            logger.error(f"Error saving split data: {e}", exc_info=True)
            raise
    
    def run(self, df: pd.DataFrame, target: pd.Series) -> Dict[str, any]:
        """
        Main execution method for data transformation.
        
        Args:
            df: Features DataFrame
            target: Target Series
        
        Returns:
            Dictionary containing transformed data
        """
        logger.info("Starting data transformation process")
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df, target)
        
        # Encode target labels
        y_train_encoded, y_test_encoded = self.encode_target(y_train, y_test)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Save split data (before scaling for reference)
        self.save_split_data(X_train, X_test, y_train, y_test)
        
        # Save scaler and encoder artifacts
        self.save_artifacts()
        
        logger.info("Data transformation process completed successfully")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_encoded,
            'y_test': y_test_encoded,
            'y_train_original': y_train,
            'y_test_original': y_test,
            'feature_names': X_train.columns.tolist(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }


if __name__ == "__main__":
    # Test data transformation
    from src.data_ingestion import DataIngestion
    from src.data_preprocessing import DataPreprocessing
    
    # Load and preprocess data
    ingestion = DataIngestion()
    features, target = ingestion.run()
    
    preprocessing = DataPreprocessing()
    features_clean, target_clean = preprocessing.run(features, target)
    
    # Transform data
    transformation = DataTransformation()
    transformed_data = transformation.run(features_clean, target_clean)
    
    print("\nTransformed data shapes:")
    print(f"X_train: {transformed_data['X_train'].shape}")
    print(f"X_test: {transformed_data['X_test'].shape}")
    print(f"y_train: {transformed_data['y_train'].shape}")
    print(f"y_test: {transformed_data['y_test'].shape}")
