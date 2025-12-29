"""
Data Ingestion Module
Loads the Iris dataset from sklearn and prepares it for processing.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from pathlib import Path
import yaml
from typing import Tuple

from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class DataIngestion:
    """Handles data loading and initial storage."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataIngestion with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        logger.info("DataIngestion initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.debug(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def load_iris_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load Iris dataset from sklearn.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            logger.info("Loading Iris dataset from sklearn")
            
            # Load iris dataset
            iris = load_iris()
            
            # Create DataFrame with feature names
            feature_names = iris.feature_names
            df = pd.DataFrame(iris.data, columns=feature_names)
            
            # Create target series with species names
            target_names = iris.target_names
            target = pd.Series(iris.target, name=self.config['data']['target_column'])
            
            # Map target integers to species names
            target_mapped = target.map({0: target_names[0], 
                                       1: target_names[1], 
                                       2: target_names[2]})
            
            logger.info(f"Dataset loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
            logger.info(f"Target classes: {target_names}")
            logger.info(f"Class distribution:\n{target_mapped.value_counts()}")
            
            return df, target_mapped
            
        except Exception as e:
            logger.error(f"Error loading Iris dataset: {e}", exc_info=True)
            raise
    
    def save_raw_data(self, df: pd.DataFrame, target: pd.Series) -> None:
        """
        Save raw data to CSV file.
        
        Args:
            df: Features DataFrame
            target: Target Series
        """
        try:
            # Ensure data directory exists
            data_dir = Path(self.config['paths']['data_dir'])
            data_dir.mkdir(exist_ok=True)
            
            # Combine features and target
            data = df.copy()
            data[self.config['data']['target_column']] = target
            
            # Save to CSV
            output_path = Path(self.config['paths']['raw_data'])
            data.to_csv(output_path, index=False)
            
            logger.info(f"Raw data saved to {output_path}")
            logger.debug(f"Saved {data.shape[0]} rows and {data.shape[1]} columns")
            
        except Exception as e:
            logger.error(f"Error saving raw data: {e}", exc_info=True)
            raise
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load raw data from CSV file if it exists.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        try:
            raw_data_path = Path(self.config['paths']['raw_data'])
            
            if not raw_data_path.exists():
                logger.warning(f"Raw data file not found at {raw_data_path}")
                return None, None
            
            logger.info(f"Loading raw data from {raw_data_path}")
            data = pd.read_csv(raw_data_path)
            
            # Split features and target
            target_col = self.config['data']['target_column']
            target = data[target_col]
            features = data.drop(columns=[target_col])
            
            logger.info(f"Raw data loaded: {features.shape[0]} samples, {features.shape[1]} features")
            
            return features, target
            
        except Exception as e:
            logger.error(f"Error loading raw data: {e}", exc_info=True)
            raise
    
    def run(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main execution method for data ingestion.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Starting data ingestion process")
        
        # Load data from sklearn
        df, target = self.load_iris_data()
        
        # Save raw data
        self.save_raw_data(df, target)
        
        logger.info("Data ingestion completed successfully")
        
        return df, target


if __name__ == "__main__":
    # Test data ingestion
    ingestion = DataIngestion()
    features, target = ingestion.run()
    
    print("\nFeatures shape:", features.shape)
    print("\nFeatures head:")
    print(features.head())
    print("\nTarget distribution:")
    print(target.value_counts())
