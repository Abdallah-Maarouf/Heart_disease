"""
Exploratory Data Analysis (EDA) Generator Module

This module provides comprehensive EDA functionality including statistical analysis,
visualizations, and automated report generation for the heart disease dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class EDAGenerator:
    """
    Comprehensive Exploratory Data Analysis generator for heart disease dataset.
    
    This class provides methods for statistical analysis, visualization generation,
    and automated EDA report creation.
    """
    
    def __init__(self, data_path: str = None, output_dir: str = "results/eda"):
        """
        Initialize EDA Generator.
        
        Args:
            data_path (str): Path to the cleaned dataset
            output_dir (str): Directory to save EDA outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.data = None
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        self.feature_descriptions = {
            'age': 'Age in years',
            'sex': 'Sex (1 = male, 0 = female)',
            'cp': 'Chest pain type (1-4)',
            'trestbps': 'Resting blood pressure (mm Hg)',
            'chol': 'Serum cholesterol (mg/dl)',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
            'restecg': 'Resting ECG results (0-2)',
            'thalach': 'Maximum heart rate achieved',
            'exang': 'Exercise induced angina (1 = yes, 0 = no)',
            'oldpeak': 'ST depression induced by exercise',
            'slope': 'Slope of peak exercise ST segment (1-3)',
            'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
            'thal': 'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)',
            'target': 'Heart disease presence (0 = no, 1 = yes)'
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the cleaned heart disease dataset.
        
        Args:
            data_path (str): Path to the dataset file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("Data path must be provided")
            
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            return self.data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def generate_statistical_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary of the dataset.
        
        Returns:
            Dict[str, Any]: Statistical summary including descriptive stats,
                          missing values, and data types
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'basic_info': {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'data_types': self.data.dtypes.to_dict()
            },
            'descriptive_stats': self.data.describe().to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'unique_values': {col: self.data[col].nunique() for col in self.data.columns},
            'value_counts': {}
        }
        
        # Add value counts for categorical variables
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
        for col in categorical_cols:
            if col in self.data.columns:
                summary['value_counts'][col] = self.data[col].value_counts().to_dict()
        
        return summary
    
    def plot_feature_distributions(self, save_plots: bool = True) -> None:
        """
        Create histograms for all numerical features.
        
        Args:
            save_plots (bool): Whether to save plots to file
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(numerical_cols):
            if col in self.data.columns:
                axes[i].hist(self.data[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribution of {col}\n({self.feature_descriptions[col]})', fontsize=12)
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(numerical_cols) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.output_dir}/feature_distributions_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_correlation_heatmap(self, save_plots: bool = True) -> None:
        """
        Create correlation heatmap using seaborn.
        
        Args:
            save_plots (bool): Whether to save plots to file
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Calculate correlation matrix
        correlation_matrix = self.data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.output_dir}/correlation_heatmap_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_boxplots(self, save_plots: bool = True) -> None:
        """
        Generate boxplots for outlier visualization and feature analysis.
        
        Args:
            save_plots (bool): Whether to save plots to file
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(numerical_cols):
            if col in self.data.columns:
                axes[i].boxplot(self.data[col], patch_artist=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2))
                axes[i].set_title(f'Boxplot of {col}\n({self.feature_descriptions[col]})', fontsize=12)
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(numerical_cols) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.output_dir}/boxplots_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_target_distribution(self, save_plots: bool = True) -> None:
        """
        Create target distribution plot for class balance analysis.
        
        Args:
            save_plots (bool): Whether to save plots to file
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if 'target' not in self.data.columns:
            raise ValueError("Target column not found in dataset")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        target_counts = self.data['target'].value_counts()
        ax1.bar(target_counts.index, target_counts.values, 
               color=['lightcoral', 'lightblue'], alpha=0.7, edgecolor='black')
        ax1.set_title('Target Distribution (Count)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Heart Disease (0 = No, 1 = Yes)')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(target_counts.values):
            ax1.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        labels = ['No Heart Disease', 'Heart Disease']
        colors = ['lightcoral', 'lightblue']
        ax2.pie(target_counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=(0.05, 0.05))
        ax2.set_title('Target Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.output_dir}/target_distribution_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print class balance information
        print(f"Class Balance Analysis:")
        print(f"No Heart Disease (0): {target_counts[0]} ({target_counts[0]/len(self.data)*100:.1f}%)")
        print(f"Heart Disease (1): {target_counts[1]} ({target_counts[1]/len(self.data)*100:.1f}%)")
    
    def feature_target_relationships(self, save_plots: bool = True) -> None:
        """
        Analyze bivariate relationships between features and target.
        
        Args:
            save_plots (bool): Whether to save plots to file
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(numerical_cols):
            if col in self.data.columns:
                # Create violin plots for numerical features vs target
                sns.violinplot(data=self.data, x='target', y=col, ax=axes[i])
                axes[i].set_title(f'{col} vs Heart Disease\n({self.feature_descriptions[col]})', 
                                fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Heart Disease (0 = No, 1 = Yes)')
                axes[i].set_ylabel(col)
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(numerical_cols) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.output_dir}/feature_target_relationships_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Create categorical feature analysis
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(categorical_cols):
            if col in self.data.columns and i < len(axes):
                # Create stacked bar plots for categorical features
                crosstab = pd.crosstab(self.data[col], self.data['target'], normalize='index')
                crosstab.plot(kind='bar', stacked=True, ax=axes[i], 
                            color=['lightcoral', 'lightblue'], alpha=0.7)
                axes[i].set_title(f'{col} vs Heart Disease\n({self.feature_descriptions[col]})', 
                                fontsize=10, fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Proportion')
                axes[i].legend(['No Disease', 'Disease'], loc='upper right')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.output_dir}/categorical_target_relationships_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_pairplot(self, save_plots: bool = True) -> None:
        """
        Generate comprehensive pairplot for feature relationships.
        
        Args:
            save_plots (bool): Whether to save plots to file
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Select key numerical features for pairplot (to avoid overcrowding)
        key_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
        subset_data = self.data[key_features].copy()
        
        # Create pairplot
        plt.figure(figsize=(15, 12))
        pairplot = sns.pairplot(subset_data, hue='target', diag_kind='hist', 
                               plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.7})
        pairplot.fig.suptitle('Pairplot of Key Features by Heart Disease Status', 
                             fontsize=16, fontweight='bold', y=1.02)
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.output_dir}/pairplot_{timestamp}.png", 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_comprehensive_eda(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate complete EDA analysis with all visualizations and statistics.
        
        Args:
            save_plots (bool): Whether to save all plots to files
            
        Returns:
            Dict[str, Any]: Complete EDA results including statistics and file paths
        """
        print("Starting Comprehensive EDA Analysis...")
        print("=" * 50)
        
        # Load data if not already loaded
        if self.data is None:
            if self.data_path:
                self.load_data()
            else:
                raise ValueError("Data path must be provided")
        
        # Generate statistical summary
        print("1. Generating statistical summary...")
        stats_summary = self.generate_statistical_summary()
        
        # Generate all visualizations
        print("2. Creating feature distribution plots...")
        self.plot_feature_distributions(save_plots=save_plots)
        
        print("3. Creating correlation heatmap...")
        self.create_correlation_heatmap(save_plots=save_plots)
        
        print("4. Generating boxplots for outlier analysis...")
        self.generate_boxplots(save_plots=save_plots)
        
        print("5. Analyzing target distribution...")
        self.plot_target_distribution(save_plots=save_plots)
        
        print("6. Analyzing feature-target relationships...")
        self.feature_target_relationships(save_plots=save_plots)
        
        print("7. Creating comprehensive pairplot...")
        self.generate_pairplot(save_plots=save_plots)
        
        print("=" * 50)
        print("EDA Analysis Complete!")
        
        return {
            'statistical_summary': stats_summary,
            'output_directory': self.output_dir,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
    
    def generate_eda_report_html(self, stats_summary: Dict[str, Any] = None) -> str:
        """
        Generate HTML report with EDA results.
        
        Args:
            stats_summary (Dict[str, Any]): Statistical summary results
            
        Returns:
            str: Path to generated HTML report
        """
        if stats_summary is None:
            stats_summary = self.generate_statistical_summary()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.output_dir}/eda_report_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Heart Disease EDA Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary-box {{ background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Heart Disease Dataset - Exploratory Data Analysis Report</h1>
            <p><strong>Generated on:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary-box">
                <h2>Dataset Overview</h2>
                <p><strong>Shape:</strong> {stats_summary['basic_info']['shape'][0]} rows × {stats_summary['basic_info']['shape'][1]} columns</p>
                <p><strong>Features:</strong> {', '.join(stats_summary['basic_info']['columns'])}</p>
            </div>
            
            <h2>Descriptive Statistics</h2>
            <table>
                <tr><th>Feature</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>25%</th><th>50%</th><th>75%</th><th>Max</th></tr>
        """
        
        for feature, stats in stats_summary['descriptive_stats'].items():
            if isinstance(stats, dict):
                count_val = stats.get('count', 'N/A')
                mean_val = f"{stats.get('mean', 0):.2f}" if isinstance(stats.get('mean'), (int, float)) else 'N/A'
                std_val = f"{stats.get('std', 0):.2f}" if isinstance(stats.get('std'), (int, float)) else 'N/A'
                min_val = f"{stats.get('min', 0):.2f}" if isinstance(stats.get('min'), (int, float)) else 'N/A'
                q25_val = f"{stats.get('25%', 0):.2f}" if isinstance(stats.get('25%'), (int, float)) else 'N/A'
                q50_val = f"{stats.get('50%', 0):.2f}" if isinstance(stats.get('50%'), (int, float)) else 'N/A'
                q75_val = f"{stats.get('75%', 0):.2f}" if isinstance(stats.get('75%'), (int, float)) else 'N/A'
                max_val = f"{stats.get('max', 0):.2f}" if isinstance(stats.get('max'), (int, float)) else 'N/A'
                
                html_content += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{count_val}</td>
                    <td>{mean_val}</td>
                    <td>{std_val}</td>
                    <td>{min_val}</td>
                    <td>{q25_val}</td>
                    <td>{q50_val}</td>
                    <td>{q75_val}</td>
                    <td>{max_val}</td>
                </tr>
                """
        
        html_content += """
            </table>
            
            <h2>Missing Values Analysis</h2>
            <table>
                <tr><th>Feature</th><th>Missing Values</th></tr>
        """
        
        for feature, missing_count in stats_summary['missing_values'].items():
            html_content += f"<tr><td>{feature}</td><td>{missing_count}</td></tr>"
        
        html_content += """
            </table>
            
            <h2>Feature Descriptions</h2>
            <table>
                <tr><th>Feature</th><th>Description</th><th>Unique Values</th></tr>
        """
        
        for feature in stats_summary['basic_info']['columns']:
            description = self.feature_descriptions.get(feature, 'No description available')
            unique_count = stats_summary['unique_values'].get(feature, 'N/A')
            html_content += f"<tr><td>{feature}</td><td>{description}</td><td>{unique_count}</td></tr>"
        
        html_content += """
            </table>
            
            <div class="summary-box">
                <h2>Key Findings</h2>
                <ul>
                    <li>Dataset contains information about heart disease prediction with 14 features</li>
                    <li>Target variable shows class distribution for heart disease presence/absence</li>
                    <li>Numerical features include age, blood pressure, cholesterol, and heart rate measurements</li>
                    <li>Categorical features include chest pain type, ECG results, and other clinical indicators</li>
                </ul>
            </div>
            
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {report_path}")
        return report_path


if __name__ == "__main__":
    # Example usage
    eda = EDAGenerator(data_path="data/processed/heart_disease_cleaned.csv")
    eda.load_data()
    results = eda.generate_comprehensive_eda(save_plots=True)
    eda.generate_eda_report_html(results['statistical_summary'])