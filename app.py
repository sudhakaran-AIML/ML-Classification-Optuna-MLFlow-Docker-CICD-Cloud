"""
Gradio Web Application for Iris Flower Classification
Provides a user-friendly interface for making predictions.
"""

import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.prediction import Prediction
from src.logger import setup_logger

# Initialize logger
logger = setup_logger("app")

# Initialize predictor
logger.info("Initializing prediction module")
predictor = Prediction()

try:
    predictor.load_artifacts()
    logger.info("All artifacts loaded successfully")
except Exception as e:
    logger.error(f"Error loading artifacts: {e}")
    logger.warning("App will start but predictions may fail until model is trained")


def predict_iris(sepal_length: float, sepal_width: float, 
                 petal_length: float, petal_width: float) -> tuple:
    """
    Predict iris species from flower measurements.
    
    Args:
        sepal_length: Sepal length in cm
        sepal_width: Sepal width in cm
        petal_length: Petal length in cm
        petal_width: Petal width in cm
    
    Returns:
        Tuple of (prediction text, confidence scores)
    """
    try:
        logger.info(f"Received prediction request: SL={sepal_length}, SW={sepal_width}, PL={petal_length}, PW={petal_width}")
        
        # Validate inputs
        if any(val <= 0 for val in [sepal_length, sepal_width, petal_length, petal_width]):
            error_msg = "❌ Error: All measurements must be positive numbers!"
            logger.warning(error_msg)
            return error_msg, {}
        
        # Prepare input
        input_data = {
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
        }
        
        # Get prediction with confidence
        predicted_class, confidence, probabilities = predictor.predict_with_confidence(input_data)
        
        # Format result
        result_text = f"""
### 🌸 Prediction Result

**Predicted Species:** `{predicted_class}`

**Confidence:** {confidence * 100:.2f}%

---

### 📊 Class Probabilities

"""
        
        # Add emoji for each species
        species_emoji = {
            'setosa': '🌼',
            'versicolor': '🌺',
            'virginica': '🌷'
        }
        
        for species, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            emoji = species_emoji.get(species, '🌸')
            bar_length = int(prob * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            result_text += f"{emoji} **{species.capitalize()}:** {prob * 100:.2f}% {bar}\n\n"
        
        logger.info(f"Prediction successful: {predicted_class} ({confidence:.4f})")
        
        return result_text, probabilities
        
    except Exception as e:
        error_msg = f"❌ Error making prediction: {str(e)}\n\nPlease ensure the model is trained first by running: `python train.py`"
        logger.error(f"Prediction error: {e}", exc_info=True)
        return error_msg, {}


# Create Gradio interface
def create_app():
    """Create and configure Gradio application."""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .output-markdown {
        font-size: 16px;
    }
    """
    
    with gr.Blocks(title="Iris Flower Classification", theme=gr.themes.Soft(), css=custom_css) as app:
        
        gr.Markdown("""
        # 🌸 Iris Flower Classification
        
        ### Predict the species of Iris flower based on its measurements
        
        This application uses a **XGBoost classifier** trained with **Optuna hyperparameter optimization** 
        and tracked with **MLFlow** to classify Iris flowers into three species:
        - 🌼 **Setosa**
        - 🌺 **Versicolor**
        - 🌷 **Virginica**
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📏 Enter Flower Measurements (in cm)")
                
                sepal_length = gr.Number(
                    label="Sepal Length (cm)",
                    value=5.1,
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    info="Typical range: 4.3 - 7.9 cm"
                )
                
                sepal_width = gr.Number(
                    label="Sepal Width (cm)",
                    value=3.5,
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    info="Typical range: 2.0 - 4.4 cm"
                )
                
                petal_length = gr.Number(
                    label="Petal Length (cm)",
                    value=1.4,
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    info="Typical range: 1.0 - 6.9 cm"
                )
                
                petal_width = gr.Number(
                    label="Petal Width (cm)",
                    value=0.2,
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    info="Typical range: 0.1 - 2.5 cm"
                )
                
                predict_btn = gr.Button("🔮 Predict Species", variant="primary", size="lg")
                
                gr.Markdown("""
                ---
                ### 📝 Example Measurements
                
                **Setosa:** SL=5.1, SW=3.5, PL=1.4, PW=0.2
                
                **Versicolor:** SL=6.0, SW=2.7, PL=5.1, PW=1.6
                
                **Virginica:** SL=6.3, SW=3.3, PL=6.0, PW=2.5
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Prediction Results")
                
                output_text = gr.Markdown(
                    value="*Enter measurements and click 'Predict Species' to see results*",
                    label="Prediction"
                )
                
                output_plot = gr.Label(
                    label="Confidence Scores",
                    num_top_classes=3
                )
        
        # Set up prediction on button click
        predict_btn.click(
            fn=predict_iris,
            inputs=[sepal_length, sepal_width, petal_length, petal_width],
            outputs=[output_text, output_plot]
        )
        
        gr.Markdown("""
        ---
        
        ### ℹ️ About
        
        This application demonstrates an end-to-end machine learning pipeline:
        - **Data:** Iris dataset from scikit-learn
        - **Model:** XGBoost Classifier
        - **Optimization:** Optuna for hyperparameter tuning
        - **Tracking:** MLFlow for experiment management
        - **Deployment:** Docker + Google Cloud Run
        
        **Tech Stack:** Python • XGBoost • Optuna • MLFlow • Gradio • Docker • DVC
        
        ---
        
        Made with ❤️ using Gradio
        """)
    
    return app


if __name__ == "__main__":
    logger.info("Starting Gradio application")
    
    # Create app
    app = create_app()
    
    # Launch app
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        logger.error(f"Error launching app: {e}", exc_info=True)
        raise
