"""
Exploratory Data Analysis (EDA) Module
Generates visualizations and statistical analysis of the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from typing import Tuple

from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


class EDA:
    """Handles exploratory data analysis and visualization."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize EDA with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.plots_dir = Path(self.config['paths'].get('plots_dir', 'plots'))
        self.plots_dir.mkdir(exist_ok=True)
        logger.info("EDA initialized")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def plot_feature_distributions(self, df: pd.DataFrame, target: pd.Series) -> None:
        """
        Plot distribution of each feature.
        
        Args:
            df: Features DataFrame
            target: Target Series
        """
        logger.info("Generating feature distribution plots")
        
        try:
            n_features = len(df.columns)
            n_cols = 2
            n_rows = (n_features + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
            axes = axes.flatten()
            
            for idx, column in enumerate(df.columns):
                ax = axes[idx]
                
                # Histogram with KDE
                df[column].hist(bins=30, alpha=0.7, ax=ax, edgecolor='black')
                ax.set_title(f'Distribution of {column}', fontsize=12, fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for idx in range(n_features, len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            output_path = self.plots_dir / 'feature_distributions.png'
            plt.savefig(output_path, dpi=self.config['eda'].get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature distribution plots saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating feature distribution plots: {e}", exc_info=True)
            raise
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        """
        Plot correlation matrix heatmap.
        
        Args:
            df: Features DataFrame
        """
        logger.info("Generating correlation matrix")
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                       fmt='.2f', vmin=-1, vmax=1)
            
            plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            output_path = self.plots_dir / 'correlation_matrix.png'
            plt.savefig(output_path, dpi=self.config['eda'].get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation matrix saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {e}", exc_info=True)
            raise
    
    def plot_pairplot(self, df: pd.DataFrame, target: pd.Series) -> None:
        """
        Create pairplot showing relationships between features colored by species.
        
        Args:
            df: Features DataFrame
            target: Target Series
        """
        logger.info("Generating pairplot")
        
        try:
            # Combine features and target for pairplot
            data = df.copy()
            data['species'] = target
            
            # Create pairplot
            pairplot = sns.pairplot(data, hue='species', diag_kind='kde', 
                                   plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'},
                                   height=2.5)
            
            pairplot.fig.suptitle('Pairplot of Features by Species', 
                                 y=1.02, fontsize=16, fontweight='bold')
            
            output_path = self.plots_dir / 'pairplot.png'
            plt.savefig(output_path, dpi=self.config['eda'].get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Pairplot saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating pairplot: {e}", exc_info=True)
            raise
    
    def plot_class_distribution(self, target: pd.Series) -> None:
        """
        Plot distribution of target classes.
        
        Args:
            target: Target Series
        """
        logger.info("Generating class distribution plot")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Count plot
            class_counts = target.value_counts()
            colors = sns.color_palette("husl", len(class_counts))
            
            ax1.bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
            ax1.set_title('Class Distribution (Count)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Species')
            ax1.set_ylabel('Count')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (idx, val) in enumerate(class_counts.items()):
                ax1.text(i, val + 1, str(val), ha='center', va='bottom', fontweight='bold')
            
            # Pie chart
            ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, explode=[0.05] * len(class_counts))
            ax2.set_title('Class Distribution (Percentage)', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            output_path = self.plots_dir / 'class_distribution.png'
            plt.savefig(output_path, dpi=self.config['eda'].get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Class distribution plot saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating class distribution plot: {e}", exc_info=True)
            raise
    
    def plot_boxplots(self, df: pd.DataFrame, target: pd.Series) -> None:
        """
        Create boxplots for each feature grouped by species.
        
        Args:
            df: Features DataFrame
            target: Target Series
        """
        logger.info("Generating boxplots by species")
        
        try:
            n_features = len(df.columns)
            n_cols = 2
            n_rows = (n_features + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
            axes = axes.flatten()
            
            # Combine data
            data = df.copy()
            data['species'] = target
            
            for idx, column in enumerate(df.columns):
                ax = axes[idx]
                sns.boxplot(data=data, x='species', y=column, ax=ax, palette='Set2')
                ax.set_title(f'{column} by Species', fontsize=12, fontweight='bold')
                ax.set_xlabel('Species')
                ax.set_ylabel(column)
                ax.grid(True, alpha=0.3, axis='y')
            
            # Remove empty subplots
            for idx in range(n_features, len(axes)):
                fig.delaxes(axes[idx])
            
            plt.tight_layout()
            
            output_path = self.plots_dir / 'boxplots_by_species.png'
            plt.savefig(output_path, dpi=self.config['eda'].get('dpi', 300), bbox_inches='tight')
            plt.close()
            
            logger.info(f"Boxplots saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating boxplots: {e}", exc_info=True)
            raise
    
    def generate_statistical_summary(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """
        Generate comprehensive statistical summary.
        
        Args:
            df: Features DataFrame
            target: Target Series
        
        Returns:
            DataFrame with statistical summary
        """
        logger.info("Generating statistical summary")
        
        try:
            summary = df.describe().T
            summary['skewness'] = df.skew()
            summary['kurtosis'] = df.kurtosis()
            
            logger.info("Statistical summary generated")
            logger.debug(f"\n{summary}")
            
            # Save to CSV
            output_path = self.plots_dir / 'statistical_summary.csv'
            summary.to_csv(output_path)
            logger.info(f"Statistical summary saved to {output_path}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating statistical summary: {e}", exc_info=True)
            raise
    
    def run(self, df: pd.DataFrame, target: pd.Series) -> None:
        """
        Main execution method for EDA.
        
        Args:
            df: Features DataFrame
            target: Target Series
        """
        logger.info("Starting EDA process")
        
        if not self.config['eda'].get('generate_plots', True):
            logger.info("Plot generation is disabled in configuration")
            return
        
        # Generate all visualizations
        self.plot_feature_distributions(df, target)
        self.plot_correlation_matrix(df)
        self.plot_pairplot(df, target)
        self.plot_class_distribution(target)
        self.plot_boxplots(df, target)
        
        # Generate statistical summary
        self.generate_statistical_summary(df, target)
        
        logger.info("EDA process completed successfully")
        logger.info(f"All plots saved to {self.plots_dir}")


if __name__ == "__main__":
    # Test EDA
    from src.data_ingestion import DataIngestion
    from src.data_preprocessing import DataPreprocessing
    
    # Load and preprocess data
    ingestion = DataIngestion()
    features, target = ingestion.run()
    
    preprocessing = DataPreprocessing()
    features_clean, target_clean = preprocessing.run(features, target)
    
    # Run EDA
    eda = EDA()
    eda.run(features_clean, target_clean)
    
    print("\nEDA completed. Check the 'plots' directory for visualizations.")
