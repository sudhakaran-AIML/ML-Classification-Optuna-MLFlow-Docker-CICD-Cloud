"""
Data Preprocessing Module
Handles data quality checks, cleaning, and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Tuple

from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class DataPreprocessing:
    """Handles data preprocessing and quality checks."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataPreprocessing with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        logger.info("DataPreprocessing initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check and report missing values in the dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with missing value information
        """
        logger.info("Checking for missing values")
        
        missing_info = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        
        total_missing = missing_info['Missing_Count'].sum()
        
        if total_missing > 0:
            logger.warning(f"Found {total_missing} missing values")
            logger.warning(f"Missing values by column:\n{missing_info[missing_info['Missing_Count'] > 0]}")
        else:
            logger.info("No missing values found")
        
        return missing_info
    
    def check_duplicates(self, df: pd.DataFrame) -> int:
        """
        Check for duplicate rows in the dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Number of duplicate rows
        """
        logger.info("Checking for duplicate rows")
        
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        
        return duplicates
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with duplicates removed
        """
        initial_shape = df.shape[0]
        df_clean = df.drop_duplicates()
        final_shape = df_clean.shape[0]
        
        removed = initial_shape - final_shape
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df_clean
    
    def validate_data_types(self, df: pd.DataFrame) -> bool:
        """
        Validate that all feature columns are numeric.
        
        Args:
            df: Input DataFrame
        
        Returns:
            True if all columns are numeric, False otherwise
        """
        logger.info("Validating data types")
        
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric:
            logger.warning(f"Non-numeric columns found: {non_numeric}")
            return False
        else:
            logger.info("All columns are numeric")
            return True
    
    def check_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> dict:
        """
        Detect outliers using Z-score method.
        
        Args:
            df: Input DataFrame
            threshold: Z-score threshold for outlier detection
        
        Returns:
            Dictionary with outlier information per column
        """
        logger.info(f"Checking for outliers (Z-score threshold: {threshold})")
        
        outlier_info = {}
        
        for column in df.columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = (z_scores > threshold).sum()
            outlier_info[column] = outliers
            
            if outliers > 0:
                logger.debug(f"Column '{column}': {outliers} outliers detected")
        
        total_outliers = sum(outlier_info.values())
        logger.info(f"Total outliers detected: {total_outliers}")
        
        return outlier_info
    
    def get_data_summary(self, df: pd.DataFrame, target: pd.Series) -> dict:
        """
        Generate comprehensive data summary.
        
        Args:
            df: Features DataFrame
            target: Target Series
        
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating data summary")
        
        summary = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'feature_names': df.columns.tolist(),
            'target_name': target.name,
            'target_classes': target.unique().tolist(),
            'class_distribution': target.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        logger.info(f"Data summary: {summary['n_samples']} samples, {summary['n_features']} features")
        logger.info(f"Target classes: {summary['target_classes']}")
        
        return summary
    
    def preprocess_data(self, df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main preprocessing pipeline.
        
        Args:
            df: Features DataFrame
            target: Target Series
        
        Returns:
            Tuple of (preprocessed features, target)
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Check missing values
        missing_info = self.check_missing_values(df)
        
        # Check and remove duplicates
        duplicates = self.check_duplicates(df)
        if duplicates > 0:
            df = self.remove_duplicates(df)
            # Also remove corresponding target values
            target = target[df.index]
        
        # Validate data types
        self.validate_data_types(df)
        
        # Check for outliers (informational only, not removing)
        outlier_info = self.check_outliers(df)
        
        # Generate summary
        summary = self.get_data_summary(df, target)
        
        logger.info("Data preprocessing completed successfully")
        
        return df, target
    
    def save_processed_data(self, df: pd.DataFrame, target: pd.Series) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            df: Processed features DataFrame
            target: Target Series
        """
        try:
            # Combine features and target
            data = df.copy()
            data[self.config['data']['target_column']] = target
            
            # Save to CSV
            output_path = Path(self.config['paths']['processed_data'])
            data.to_csv(output_path, index=False)
            
            logger.info(f"Processed data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}", exc_info=True)
            raise
    
    def run(self, df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Main execution method for data preprocessing.
        
        Args:
            df: Input features DataFrame
            target: Input target Series
        
        Returns:
            Tuple of (preprocessed features, target)
        """
        logger.info("Starting data preprocessing process")
        
        # Preprocess data
        df_processed, target_processed = self.preprocess_data(df, target)
        
        # Save processed data
        self.save_processed_data(df_processed, target_processed)
        
        logger.info("Data preprocessing process completed successfully")
        
        return df_processed, target_processed


if __name__ == "__main__":
    # Test data preprocessing
    from src.data_ingestion import DataIngestion
    
    # Load data
    ingestion = DataIngestion()
    features, target = ingestion.run()
    
    # Preprocess data
    preprocessing = DataPreprocessing()
    features_clean, target_clean = preprocessing.run(features, target)
    
    print("\nPreprocessed features shape:", features_clean.shape)
    print("\nData summary:")
    print(features_clean.describe())
