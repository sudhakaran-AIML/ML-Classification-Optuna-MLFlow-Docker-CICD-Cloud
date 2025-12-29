"""
Prediction Module
Handles model loading and inference for new data.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Union

from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)


class Prediction:
    """Handles model loading and predictions."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Prediction with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.scaler = None
        self.label_encoder = None
        logger.info("Prediction initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def load_model(self) -> None:
        """Load trained model from file."""
        try:
            model_path = Path(self.config['paths']['model_file'])
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
    
    def load_scaler(self) -> None:
        """Load scaler from file."""
        try:
            scaler_path = Path(self.config['paths']['scaler_file'])
            
            if not scaler_path.exists():
                logger.warning(f"Scaler file not found at {scaler_path}")
                return
            
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error loading scaler: {e}", exc_info=True)
            raise
    
    def load_label_encoder(self) -> None:
        """Load label encoder from file."""
        try:
            models_dir = Path(self.config['paths']['models_dir'])
            encoder_path = models_dir / 'label_encoder.pkl'
            
            if not encoder_path.exists():
                logger.warning(f"Label encoder file not found at {encoder_path}")
                return
            
            self.label_encoder = joblib.load(encoder_path)
            logger.info(f"Label encoder loaded from {encoder_path}")
            
        except Exception as e:
            logger.error(f"Error loading label encoder: {e}", exc_info=True)
            raise
    
    def load_artifacts(self) -> None:
        """Load all required artifacts (model, scaler, encoder)."""
        logger.info("Loading all artifacts")
        
        self.load_model()
        self.load_scaler()
        self.load_label_encoder()
        
        logger.info("All artifacts loaded successfully")
    
    def preprocess_input(self, input_data: Union[Dict, pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Input data as dict, DataFrame, or array
        
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Convert input to DataFrame if needed
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, np.ndarray):
                # Assume features are in correct order
                feature_names = ['sepal length (cm)', 'sepal width (cm)', 
                               'petal length (cm)', 'petal width (cm)']
                df = pd.DataFrame(input_data.reshape(1, -1) if input_data.ndim == 1 else input_data, 
                                columns=feature_names)
            elif isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            logger.debug(f"Input data shape: {df.shape}")
            logger.debug(f"Input data:\n{df}")
            
            # Scale features if scaler is available
            if self.scaler is not None:
                df_scaled = pd.DataFrame(
                    self.scaler.transform(df),
                    columns=df.columns,
                    index=df.index
                )
                logger.debug("Features scaled")
                return df_scaled
            else:
                logger.warning("Scaler not available, using unscaled features")
                return df
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {e}", exc_info=True)
            raise
    
    def predict_single(self, input_data: Union[Dict, np.ndarray]) -> str:
        """
        Predict class for a single instance.
        
        Args:
            input_data: Single instance as dict or array
        
        Returns:
            Predicted class name
        """
        logger.info("Making single prediction")
        
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_artifacts() first.")
            
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Make prediction
            y_pred_encoded = self.model.predict(X)[0]
            
            # Decode prediction
            if self.label_encoder is not None:
                y_pred = self.label_encoder.inverse_transform([y_pred_encoded])[0]
            else:
                y_pred = str(y_pred_encoded)
            
            logger.info(f"Prediction: {y_pred}")
            
            return y_pred
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}", exc_info=True)
            raise
    
    def predict_batch(self, input_data: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """
        Predict classes for multiple instances.
        
        Args:
            input_data: Multiple instances as DataFrame or array
        
        Returns:
            List of predicted class names
        """
        logger.info("Making batch predictions")
        
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_artifacts() first.")
            
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Make predictions
            y_pred_encoded = self.model.predict(X)
            
            # Decode predictions
            if self.label_encoder is not None:
                y_pred = self.label_encoder.inverse_transform(y_pred_encoded).tolist()
            else:
                y_pred = y_pred_encoded.tolist()
            
            logger.info(f"Made {len(y_pred)} predictions")
            
            return y_pred
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}", exc_info=True)
            raise
    
    def get_prediction_probabilities(self, input_data: Union[Dict, pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """
        Get prediction probabilities for all classes.
        
        Args:
            input_data: Input data
        
        Returns:
            Dictionary mapping class names to probabilities
        """
        logger.info("Getting prediction probabilities")
        
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_artifacts() first.")
            
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Get probabilities
            probabilities = self.model.predict_proba(X)[0]
            
            # Create dictionary with class names
            if self.label_encoder is not None:
                class_names = self.label_encoder.classes_
            else:
                class_names = [f"Class_{i}" for i in range(len(probabilities))]
            
            prob_dict = {class_name: float(prob) 
                        for class_name, prob in zip(class_names, probabilities)}
            
            logger.info(f"Probabilities: {prob_dict}")
            
            return prob_dict
            
        except Exception as e:
            logger.error(f"Error getting probabilities: {e}", exc_info=True)
            raise
    
    def predict_with_confidence(self, input_data: Union[Dict, np.ndarray]) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict class with confidence score and all probabilities.
        
        Args:
            input_data: Input data
        
        Returns:
            Tuple of (predicted class, confidence, all probabilities)
        """
        logger.info("Making prediction with confidence")
        
        try:
            # Get probabilities
            probabilities = self.get_prediction_probabilities(input_data)
            
            # Get predicted class and confidence
            predicted_class = max(probabilities, key=probabilities.get)
            confidence = probabilities[predicted_class]
            
            logger.info(f"Predicted: {predicted_class} with confidence {confidence:.4f}")
            
            return predicted_class, confidence, probabilities
            
        except Exception as e:
            logger.error(f"Error making prediction with confidence: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    # Test prediction module
    predictor = Prediction()
    predictor.load_artifacts()
    
    # Test single prediction
    test_input = {
        'sepal length (cm)': 5.1,
        'sepal width (cm)': 3.5,
        'petal length (cm)': 1.4,
        'petal width (cm)': 0.2
    }
    
    print("\nTest Input:", test_input)
    
    # Get prediction with confidence
    predicted_class, confidence, probabilities = predictor.predict_with_confidence(test_input)
    
    print(f"\nPredicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print("\nAll Probabilities:")
    for class_name, prob in probabilities.items():
        print(f"  {class_name}: {prob:.4f}")
